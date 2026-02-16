import airsim
import time

# Connect to AirSim
client = airsim.CarClient()
client.confirmConnection()

print("✅ Connected to AirSim")
print("🚗 Drive the car MANUALLY (keyboard / joystick)")
print("📍 Printing X, Y coordinates (NED frame)\n")

try:
    while True:
        car_state = client.getCarState()

        pos = car_state.kinematics_estimated.position
        x = pos.x_val  # North
        y = pos.y_val  # East
        z = pos.z_val  # Down

        speed = car_state.speed

        print(
            f"X (North): {x:8.2f} m | "
            f"Y (East): {y:8.2f} m | "
            f"Z (Down): {z:6.2f} m | "
            f"Speed: {speed:5.2f} m/s"
        )

        time.sleep(0.2)

except KeyboardInterrupt:
    print("\n🛑 Stopped by user")
