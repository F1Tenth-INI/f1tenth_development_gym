# IMU Sensor Simulation

This document describes the enhanced IMU (Inertial Measurement Unit) sensor simulation implemented for the F1Tenth development gym.

## Overview

The IMU simulator provides realistic sensor data including:
- **Accelerometer data**: Linear acceleration in the car's coordinate frame (X and Y axes)
- **Gyroscope data**: Angular velocity around the Z-axis (yaw rate)
- **Sensor noise**: Gaussian noise with configurable standard deviation
- **Bias drift**: Random walk bias that changes over time
- **Temperature effects**: Temperature-dependent bias variations
- **Coordinate transformations**: Proper conversion from global to car frame

## Features

### Realistic Sensor Characteristics
- **Noise modeling**: Configurable Gaussian noise for both accelerometer and gyroscope
- **Bias drift**: Random walk bias that simulates real sensor drift
- **Temperature effects**: Temperature-dependent bias variations
- **First-order dynamics**: Proper handling of initial conditions

### Easy Integration
- **Automatic updates**: IMU data is automatically updated in the car system
- **Dictionary format**: Easy access to sensor data via dictionary keys
- **Array format**: Direct access to raw sensor arrays
- **Reset functionality**: Proper reset when car state is reset

## Usage

### Basic Usage

```python
from utilities.imu_simulator import IMUSimulator

# Create IMU simulator with default settings
imu = IMUSimulator()

# Or with custom noise levels
imu = IMUSimulator(noise_level=0.02, bias_std=0.005)

# Update with car state
car_state = np.array([x, y, theta, vx, vy, omega, slip])
imu_data = imu.update_car_state(car_state, delta_time=0.01)

# Access data
accel_x = imu_data[0]  # Forward acceleration
accel_y = imu_data[1]  # Lateral acceleration
omega_z = imu_data[2]  # Yaw rate
```

### Dictionary Format

```python
# Convert to dictionary
imu_dict = imu.array_to_dict(imu_data)
print(imu_dict['imu_a_x'])  # Forward acceleration
print(imu_dict['imu_a_y'])  # Lateral acceleration
print(imu_dict['imu_av_z'])  # Yaw rate
```

### Integration with Car System

The IMU is automatically integrated into the car system and updated every simulation step:

```python
# IMU data is automatically available in the car system
driver = CarSystem()
imu_data = driver.current_imu_dict
```

## Configuration

### Constructor Parameters

```python
IMUSimulator(noise_level=0.01, bias_std=0.001, temperature_coeff=0.0001)
```

- **noise_level**: Standard deviation of Gaussian noise (default: 0.01)
- **bias_std**: Standard deviation of bias drift (default: 0.001)
- **temperature_coeff**: Temperature coefficient for bias (default: 0.0001)

### Sensor Information

```python
# Get current sensor information
info = imu.get_sensor_info()
print(f"Temperature: {info['temperature']}°C")
print(f"Accel bias X: {info['accel_bias_x']}")
print(f"Accel bias Y: {info['accel_bias_y']}")
print(f"Gyro bias Z: {info['gyro_bias_z']}")
```

## Examples

### Basic Test

Run the test script to see the IMU in action:

```bash
python test_imu_simulation.py
```

This will generate plots showing:
- Car trajectory
- IMU acceleration measurements
- Angular velocity measurements
- Comparison between low and high noise settings

### Controller Integration

See `examples/imu_controller_example.py` for a complete example of using IMU data in a controller.

### Key Features Demonstrated

1. **Velocity estimation** from acceleration data
2. **Skid detection** using lateral acceleration
3. **Control logic** based on IMU measurements
4. **Sensor fusion** concepts

## Data Format

### IMU Array Format
```python
imu_data = [accel_x, accel_y, angular_vel_z]
```

### IMU Dictionary Format
```python
imu_dict = {
    'imu_a_x': accel_x,      # Forward acceleration (m/s²)
    'imu_a_y': accel_y,      # Lateral acceleration (m/s²)
    'imu_av_z': angular_vel_z  # Yaw rate (rad/s)
}
```

## Coordinate System

The IMU uses the car's coordinate system:
- **X-axis**: Forward direction (positive = forward)
- **Y-axis**: Left direction (positive = left)
- **Z-axis**: Up direction (positive = up, for angular velocity)

## Calibration

The IMU simulator includes basic calibration functionality:

```python
# Perform calibration (reduces bias)
imu.calibrate(num_samples=100)

# Reset to initial state
imu.reset()
```

## Integration Points

The IMU is integrated at the following points:

1. **Car System Initialization**: IMU is created during `CarSystem.__init__()`
2. **State Updates**: IMU is updated in `update_driver_state()` in `run_simulation.py`
3. **Data Recording**: IMU data is automatically recorded via the `Recorder` class
4. **Reset**: IMU is reset when the car system is reset

## Performance Considerations

- **Computational overhead**: Minimal - only a few calculations per update
- **Memory usage**: Low - stores only previous state and sensor parameters
- **Update frequency**: Matches the control frequency (typically 100 Hz)

## Future Enhancements

Potential improvements for the IMU simulation:

1. **Magnetometer support**: Add compass/heading data
2. **Advanced noise models**: More realistic noise characteristics
3. **Sensor fusion**: Integration with other sensors (GPS, wheel encoders)
4. **Calibration algorithms**: More sophisticated bias estimation
5. **Temperature modeling**: More detailed temperature effects

## Troubleshooting

### Common Issues

1. **First update returns zeros**: This is normal - the first update initializes the sensor
2. **Large bias values**: Use calibration or reduce `bias_std` parameter
3. **Noisy data**: Reduce `noise_level` parameter for cleaner data
4. **Integration drift**: Use proper sensor fusion techniques in your controller

### Debug Information

```python
# Get detailed sensor information
info = imu.get_sensor_info()
print("Sensor Info:", info)

# Check if first update
if imu.first_update:
    print("IMU not yet initialized")
```
