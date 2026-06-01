import warnings
warnings.filterwarnings("ignore")

import airsim
import keyboard
import time
import math

# Connect to AirSim
client = airsim.MultirotorClient()
client.confirmConnection()

# Enable API control
client.enableApiControl(True)
client.armDisarm(True)
client.takeoffAsync().join()

print("✅ Connected to AirSim")
print("🚁 Keyboard Drone Control")
print("====================================")
print("W → Forward | S → Backward")
print("A → Left    | D → Right")
print("L → Up      | K → Down")
print("ESC → Quit")
print("====================================")
print("📍 Printing X, Y, Z coordinates (NED frame)\n")

while True:
    vx = vy = vz = 0

    # -------- Keyboard Controls --------
    if keyboard.is_pressed('w'): vx = 2
    if keyboard.is_pressed('s'): vx = -2
    if keyboard.is_pressed('a'): vy = -2
    if keyboard.is_pressed('d'): vy = 2
    if keyboard.is_pressed('l'): vz = -2   # UP (negative Z in NED)
    if keyboard.is_pressed('k'): vz = 2    # DOWN

    if keyboard.is_pressed('esc'):
        print("\n🛑 Exiting control loop...")
        break

    # -------- Send Velocity Command --------
    client.moveByVelocityAsync(vx, vy, vz, 0.1)

    # -------- Read Drone State --------
    drone_state = client.getMultirotorState()
    pos = drone_state.kinematics_estimated.position
    vel = drone_state.kinematics_estimated.linear_velocity

    x = pos.x_val   # North
    y = pos.y_val   # East
    z = pos.z_val   # Down

    speed = math.sqrt(vel.x_val**2 + vel.y_val**2 + vel.z_val**2)

    print(
        f"X (North): {x:8.2f} m | "
        f"Y (East): {y:8.2f} m | "
        f"Z (Down): {z:6.2f} m | "
        f"Speed: {speed:5.2f} m/s",
        end="\r"
    )

    time.sleep(0.05)

# -------- Safe Landing --------
client.hoverAsync().join()
client.landAsync().join()
client.armDisarm(False)
client.enableApiControl(False)
print("\n✅ Drone landed and API released")