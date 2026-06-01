"""
utils.py
--------
Shared utility functions for the spatio-visual UAV navigation framework.

Implements (matching paper sections 3.3 / 3.4 / 4):
  - spatial_area_pooling()    : 256x256 prob-map -> 5x5 occupancy matrix  (eq. 4)
  - build_state()             : [V_visual || dx || dy || dz] ∈ R^28         (eq. 7)
  - exponential_smoothing()   : first-order low-pass EMA filter             (eq. 23)
  - jerk_penalty()            : ||v_target - v_smooth||^2                   (eq. 24)
  - compute_reward()          : full multi-modal reward  (Section 4)
  - scale_actions()           : [-1,1]^3 -> physical UAV commands           (eq. 10-12)
  - obstacle_reward()         : quadrant-based spatial visual cost           (eq. 25-27)
"""

import cv2
import numpy as np
import math


# ─────────────────────────────────────────────────────────────────────────────
# Constants (Section 3.4)
# ─────────────────────────────────────────────────────────────────────────────
V_MAX      = 5.0    # m/s  – max forward / lateral velocity
OMEGA_MAX  = 5.0    # deg/s – max yaw rate
H_TARGET   = 10.0  # m    – constant altitude hold (NED negative)
ALPHA      = 0.05  # default smoothing coefficient


# ─────────────────────────────────────────────────────────────────────────────
# 1. Spatial Area Pooling  (Section 3.3, eq. 4)
# ─────────────────────────────────────────────────────────────────────────────
def spatial_area_pooling(prob_map: np.ndarray) -> np.ndarray:
    """
    Compress dense obstacle probability tensor M_prob ∈ R^{256×256}
    into a lightweight 5×5 spatial occupancy matrix G using
    cv2.INTER_AREA interpolation.

    Args:
        prob_map: np.ndarray of shape (256, 256) or (256, 256, 1),
                  values in [0, 1].

    Returns:
        G: np.ndarray of shape (5, 5), values in [0, 1].
    """
    if prob_map.ndim == 3:
        prob_map = prob_map[:, :, 0]
    prob_map_u8 = (prob_map * 255).astype(np.float32)
    G = cv2.resize(prob_map_u8, (5, 5), interpolation=cv2.INTER_AREA)
    return G / 255.0   # normalize back to [0,1]


# ─────────────────────────────────────────────────────────────────────────────
# 2. State Construction  (Section 3.3, eq. 5-7)
# ─────────────────────────────────────────────────────────────────────────────
def build_state(G: np.ndarray, uav_pos: np.ndarray, goal_pos: np.ndarray) -> np.ndarray:
    """
    Construct the compact 28-dimensional RL state:
        s_t = [V_visual || dx || dy || dz]

    Args:
        G        : (5, 5) spatial occupancy matrix.
        uav_pos  : (3,) array [x, y, z] in AirSim NED frame.
        goal_pos : (3,) array [xg, yg, zg] in AirSim NED frame.

    Returns:
        state: np.ndarray of shape (28,).
    """
    V_visual = G.flatten()                      # shape (25,)
    d_rel    = goal_pos - uav_pos               # eq. 6: [dx, dy, dz]
    state    = np.concatenate([V_visual, d_rel])  # eq. 7: (28,)
    return state.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Exponential Velocity Smoothing  (Section 4, eq. 23)
# ─────────────────────────────────────────────────────────────────────────────
def exponential_smoothing(v_smooth_prev: np.ndarray,
                          v_target: np.ndarray,
                          alpha: float = ALPHA) -> np.ndarray:
    """
    First-order low-pass exponential moving average:
        v_smooth,t = (1 - α) * v_smooth,t-1 + α * v_target,t

    Args:
        v_smooth_prev: (3,) previous smoothed velocity [vf, vl, ω].
        v_target     : (3,) current raw velocity target from RL policy.
        alpha        : smoothing responsiveness coefficient (default 0.05).

    Returns:
        v_smooth: (3,) updated smoothed velocity.
    """
    return (1.0 - alpha) * v_smooth_prev + alpha * v_target


# ─────────────────────────────────────────────────────────────────────────────
# 4. Jerk Regularization Penalty  (Section 4, eq. 24)
# ─────────────────────────────────────────────────────────────────────────────
def jerk_penalty(v_target: np.ndarray, v_smooth: np.ndarray) -> float:
    """
    R_jerk = -0.05 * ||v_target - v_smooth||^2
    """
    return -0.05 * float(np.sum((v_target - v_smooth) ** 2))


# ─────────────────────────────────────────────────────────────────────────────
# 5. Action Scaling  (Section 3.4, eq. 10-12)
# ─────────────────────────────────────────────────────────────────────────────
def scale_actions(action: np.ndarray):
    """
    Map normalised RL action a ∈ [-1,1]^3 to physical UAV commands.

    Args:
        action: (3,) array [a_forward, a_lateral, a_yaw] in [-1, 1].

    Returns:
        v_forward (m/s), v_lateral (m/s), omega_yaw (deg/s)
    """
    v_forward  = float(action[0]) * V_MAX
    v_lateral  = float(action[1]) * V_MAX
    omega_yaw  = float(action[2]) * OMEGA_MAX
    return v_forward, v_lateral, omega_yaw


# ─────────────────────────────────────────────────────────────────────────────
# 6. Altitude Correction  (Section 3.4, eq. 13)
# ─────────────────────────────────────────────────────────────────────────────
def altitude_correction(z_current: float, h_target: float = H_TARGET) -> float:
    """
    Vertical stabilization command:
        v_z = -(z_current + H_target)
    (AirSim NED: altitude is negative z)
    """
    return -(z_current + h_target)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Obstacle Reward  (Section 4, eq. 25-27)
# ─────────────────────────────────────────────────────────────────────────────
def obstacle_reward(G: np.ndarray, v_lateral: float) -> float:
    """
    Spatial visual quadrant obstacle mitigation reward.

    Frontal safety column: rows 0-1, cols 2-3  (eq. 25).
    Global occupancy: all 25 cells.

    R_obs = -4.0 * O_front - 1.0 * O_global
    + |v_lateral| if O_front > 0.3   (lateral sidestep escape incentive, eq.27)
    """
    O_front  = G[0:2, 2:4].mean()   # eq. 25 (1/4 * sum over 4 cells)
    O_global = G.mean()              # eq. 25 (1/25 * sum over 25 cells)

    R_obs = -4.0 * O_front - 1.0 * O_global   # eq. 26

    if O_front > 0.3:                           # eq. 27 – lateral escape incentive
        R_obs += abs(v_lateral)

    return float(R_obs)


# ─────────────────────────────────────────────────────────────────────────────
# 8. Yaw Alignment Reward  (Section 4, eq. 19-20)
# ─────────────────────────────────────────────────────────────────────────────
def yaw_alignment_reward(theta_target_deg: float,
                         theta_yaw_deg: float,
                         omega_yaw: float) -> float:
    """
    R_yaw = 0.2 * (1 - e_θ / 180) - 0.4 * |ω_yaw|

    e_θ = |(θ_target - θ_yaw + 180) mod 360 - 180|
    """
    e_theta = abs((theta_target_deg - theta_yaw_deg + 180.0) % 360.0 - 180.0)
    R_yaw = 0.2 * (1.0 - e_theta / 180.0) - 0.4 * abs(omega_yaw)
    return float(R_yaw)


# ─────────────────────────────────────────────────────────────────────────────
# 9. Full Multi-Modal Reward  (Section 4, eq. 16-28)
# ─────────────────────────────────────────────────────────────────────────────
def compute_reward(
    dist_prev: float,
    dist_curr: float,
    G: np.ndarray,
    v_forward: float,
    v_lateral: float,
    v_smooth: np.ndarray,
    v_target: np.ndarray,
    theta_target_deg: float,
    theta_yaw_deg: float,
    omega_yaw: float,
    collision: bool,
    success: bool,
    timeout: bool,
    step: int,
) -> tuple:
    """
    Compute full reward:
        R_total = R_progress + R_yaw + R_motion + R_obs + R_jerk + R_terminal

    Returns:
        (total_reward, reward_breakdown_dict, done)
    """
    # R_progress  (eq. 17)
    dt = dist_curr + 1e-6
    R_progress = (dist_prev - dist_curr) * (20.0 + 100.0 / dt)

    # R_yaw  (eq. 20)
    R_yaw = yaw_alignment_reward(theta_target_deg, theta_yaw_deg, omega_yaw)

    # R_motion  (eq. 21)
    v_smooth_mag = float(np.linalg.norm(v_smooth[:2]))
    R_motion = 0.3 * v_forward + 0.4 * v_smooth_mag

    # R_obs  (eq. 25-27)
    R_obs = obstacle_reward(G, v_lateral)

    # R_jerk  (eq. 24)
    R_jerk = jerk_penalty(v_target, v_smooth)

    # R_time per-step penalty
    R_time = -0.02

    # R_terminal  (eq. 28)
    done = False
    R_terminal = 0.0
    if success:
        R_terminal = +2000.0
        done = True
    elif collision:
        R_terminal = -100.0
        done = True
    elif timeout:
        R_terminal = -100.0
        done = True

    R_total = R_progress + R_yaw + R_motion + R_obs + R_jerk + R_time + R_terminal

    breakdown = dict(
        R_progress=R_progress,
        R_yaw=R_yaw,
        R_motion=R_motion,
        R_obs=R_obs,
        R_jerk=R_jerk,
        R_time=R_time,
        R_terminal=R_terminal,
        R_total=R_total,
    )
    return R_total, breakdown, done


# ─────────────────────────────────────────────────────────────────────────────
# 10. Misc helpers
# ─────────────────────────────────────────────────────────────────────────────
def horizontal_distance(pos: np.ndarray, goal: np.ndarray) -> float:
    """Euclidean distance in the XY plane (ignores altitude)."""
    return float(math.sqrt((pos[0] - goal[0])**2 + (pos[1] - goal[1])**2))


def target_yaw_deg(pos: np.ndarray, goal: np.ndarray) -> float:
    """Heading angle (degrees) from UAV position toward goal in NED XY."""
    dx = goal[0] - pos[0]
    dy = goal[1] - pos[1]
    return float(math.degrees(math.atan2(dy, dx)))


if __name__ == "__main__":
    # Quick self-test
    G_test = np.random.rand(5, 5).astype(np.float32)
    pos    = np.array([0.0, 0.0, -10.0])
    goal   = np.array([50.0, 30.0, -10.0])
    state  = build_state(G_test, pos, goal)
    print(f"State shape: {state.shape}")   # (28,)

    v_sm   = np.zeros(3)
    v_tgt  = np.array([3.0, 0.5, 1.0])
    v_sm   = exponential_smoothing(v_sm, v_tgt)
    jp     = jerk_penalty(v_tgt, v_sm)
    print(f"Jerk penalty: {jp:.4f}")

    r, breakdown, done = compute_reward(
        dist_prev=60.0, dist_curr=57.0, G=G_test,
        v_forward=3.0, v_lateral=0.5, v_smooth=v_sm, v_target=v_tgt,
        theta_target_deg=30.0, theta_yaw_deg=25.0, omega_yaw=0.1,
        collision=False, success=False, timeout=False, step=1
    )
    print(f"Total reward: {r:.4f}, Done: {done}")
    for k, v in breakdown.items():
        print(f"  {k}: {v:.4f}")