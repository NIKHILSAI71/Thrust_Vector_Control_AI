"""
Intelligent Ascent TVC System - Real-Time Deployment Interface
=============================================================

Hardware abstraction layer and real-time control interface for deploying
the trained TD3 model on microcontrollers and embedded systems.

Author: AI Assistant
Date: August 2025
"""

import numpy as np
import time
import os
import json
from typing import Dict, List, Optional, Tuple, Any
import threading
import queue
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    logging.warning("PySerial not available - hardware interface disabled")

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logging.warning("ONNX Runtime not available - optimized inference disabled")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SensorData:
    """Data structure for sensor readings."""
    timestamp: float
    accelerometer: Tuple[float, float, float]  # m/s² (x, y, z)
    gyroscope: Tuple[float, float, float]      # rad/s (x, y, z)
    magnetometer: Tuple[float, float, float]   # μT (x, y, z)
    barometer: float                           # Pa
    gps_position: Optional[Tuple[float, float, float]] = None  # lat, lon, alt
    thrust_sensor: float = 0.0                 # N
    fuel_level: float = 1.0                    # fraction remaining


@dataclass
class ControlCommand:
    """Data structure for control commands."""
    timestamp: float
    gimbal_x: float      # radians (pitch)
    gimbal_y: float      # radians (yaw)
    throttle: float = 1.0  # fraction (if throttle control available)


@dataclass
class SystemState:
    """Complete system state vector."""
    position: np.ndarray          # [x, y, z] in meters
    orientation: np.ndarray       # [roll, pitch, yaw] in radians
    velocity: np.ndarray          # [vx, vy, vz] in m/s
    angular_velocity: np.ndarray  # [wx, wy, wz] in rad/s
    thrust_magnitude: float       # N
    mass: float                   # kg
    center_of_mass_offset: float  # m
    gimbal_angles: np.ndarray     # [pitch, yaw] in radians
    fuel_remaining: float         # fraction
    wind_velocity: np.ndarray     # [wx, wy, wz] in m/s (estimated)


class SensorInterface(ABC):
    """Abstract base class for sensor interfaces."""
    
    @abstractmethod
    def read_sensors(self) -> SensorData:
        """Read current sensor data."""
        pass
    
    @abstractmethod
    def calibrate(self) -> bool:
        """Calibrate sensors."""
        pass
    
    @abstractmethod
    def is_healthy(self) -> bool:
        """Check if sensors are functioning properly."""
        pass


class ActuatorInterface(ABC):
    """Abstract base class for actuator interfaces."""
    
    @abstractmethod
    def send_command(self, command: ControlCommand) -> bool:
        """Send control command to actuators."""
        pass
    
    @abstractmethod
    def get_actuator_status(self) -> Dict[str, Any]:
        """Get current actuator status."""
        pass
    
    @abstractmethod
    def emergency_stop(self) -> bool:
        """Emergency stop all actuators."""
        pass


class ArduinoInterface(SensorInterface, ActuatorInterface):
    """
    Interface for Arduino-based flight computer communication.
    
    Communication protocol:
    - Sensor data: JSON format over Serial
    - Control commands: Binary format for low latency
    """
    
    def __init__(self, port: str = "COM3", baudrate: int = 115200, timeout: float = 0.1):
        """
        Initialize Arduino interface.
        
        Args:
            port: Serial port (e.g., "COM3" on Windows, "/dev/ttyUSB0" on Linux)
            baudrate: Communication speed
            timeout: Serial timeout in seconds
        """
        if not SERIAL_AVAILABLE:
            raise ImportError("PySerial is required for Arduino interface")
        
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial_conn = None
        self.is_connected = False
        
        # Sensor calibration data
        self.accel_offset = np.zeros(3)
        self.gyro_offset = np.zeros(3)
        self.mag_offset = np.zeros(3)
        
        # Connect to Arduino
        self._connect()
    
    def _connect(self) -> bool:
        """Establish serial connection with Arduino."""
        try:
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                bytesize=serial.EIGHTBITS
            )
            
            # Wait for Arduino to initialize
            time.sleep(2.0)
            
            # Send handshake
            self.serial_conn.write(b"HANDSHAKE\n")
            response = self.serial_conn.readline().decode().strip()
            
            if response == "ACK":
                self.is_connected = True
                logger.info(f"Arduino connected on {self.port}")
                return True
            else:
                logger.error(f"Arduino handshake failed: {response}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to connect to Arduino: {e}")
            return False
    
    def read_sensors(self) -> SensorData:
        """Read sensor data from Arduino."""
        if not self.is_connected:
            raise ConnectionError("Arduino not connected")
        
        try:
            # Request sensor data
            self.serial_conn.write(b"READ_SENSORS\n")
            
            # Read JSON response
            response = self.serial_conn.readline().decode().strip()
            data = json.loads(response)
            
            # Parse sensor data
            sensor_data = SensorData(
                timestamp=time.time(),
                accelerometer=(
                    data['accel']['x'] - self.accel_offset[0],
                    data['accel']['y'] - self.accel_offset[1],
                    data['accel']['z'] - self.accel_offset[2]
                ),
                gyroscope=(
                    data['gyro']['x'] - self.gyro_offset[0],
                    data['gyro']['y'] - self.gyro_offset[1],
                    data['gyro']['z'] - self.gyro_offset[2]
                ),
                magnetometer=(
                    data['mag']['x'] - self.mag_offset[0],
                    data['mag']['y'] - self.mag_offset[1],
                    data['mag']['z'] - self.mag_offset[2]
                ),
                barometer=data['baro'],
                thrust_sensor=data.get('thrust', 0.0),
                fuel_level=data.get('fuel', 1.0)
            )
            
            return sensor_data
            
        except Exception as e:
            logger.error(f"Failed to read sensors: {e}")
            raise
    
    def send_command(self, command: ControlCommand) -> bool:
        """Send control command to Arduino."""
        if not self.is_connected:
            return False
        
        try:
            # Convert angles to servo microseconds (1000-2000 μs)
            gimbal_x_us = int(1500 + (command.gimbal_x / np.pi) * 500)
            gimbal_y_us = int(1500 + (command.gimbal_y / np.pi) * 500)
            
            # Clamp values to safe range
            gimbal_x_us = max(1000, min(2000, gimbal_x_us))
            gimbal_y_us = max(1000, min(2000, gimbal_y_us))
            
            # Send binary command for low latency
            cmd_bytes = bytearray([
                0xFF,  # Start byte
                (gimbal_x_us >> 8) & 0xFF,  # High byte gimbal X
                gimbal_x_us & 0xFF,         # Low byte gimbal X
                (gimbal_y_us >> 8) & 0xFF,  # High byte gimbal Y
                gimbal_y_us & 0xFF,         # Low byte gimbal Y
                int(command.throttle * 255), # Throttle (0-255)
                0xFE   # End byte
            ])
            
            self.serial_conn.write(cmd_bytes)
            return True
            
        except Exception as e:
            logger.error(f"Failed to send command: {e}")
            return False
    
    def calibrate(self) -> bool:
        """Calibrate sensors by averaging readings."""
        logger.info("Starting sensor calibration...")
        
        if not self.is_connected:
            return False
        
        try:
            # Collect calibration samples
            samples = 100
            accel_sum = np.zeros(3)
            gyro_sum = np.zeros(3)
            mag_sum = np.zeros(3)
            
            for i in range(samples):
                # Read raw sensor data
                self.serial_conn.write(b"READ_SENSORS\n")
                response = self.serial_conn.readline().decode().strip()
                data = json.loads(response)
                
                accel_sum += np.array([data['accel']['x'], data['accel']['y'], data['accel']['z']])
                gyro_sum += np.array([data['gyro']['x'], data['gyro']['y'], data['gyro']['z']])
                mag_sum += np.array([data['mag']['x'], data['mag']['y'], data['mag']['z']])
                
                time.sleep(0.01)  # 10ms sampling
            
            # Calculate offsets
            self.accel_offset = accel_sum / samples
            self.accel_offset[2] -= 9.81  # Remove gravity from Z-axis
            
            self.gyro_offset = gyro_sum / samples
            self.mag_offset = mag_sum / samples
            
            logger.info("Sensor calibration completed")
            logger.info(f"Accel offset: {self.accel_offset}")
            logger.info(f"Gyro offset: {self.gyro_offset}")
            logger.info(f"Mag offset: {self.mag_offset}")
            
            return True
            
        except Exception as e:
            logger.error(f"Calibration failed: {e}")
            return False
    
    def get_actuator_status(self) -> Dict[str, Any]:
        """Get actuator status from Arduino."""
        if not self.is_connected:
            return {"connected": False}
        
        try:
            self.serial_conn.write(b"GET_STATUS\n")
            response = self.serial_conn.readline().decode().strip()
            status = json.loads(response)
            return status
            
        except Exception as e:
            logger.error(f"Failed to get actuator status: {e}")
            return {"error": str(e)}
    
    def emergency_stop(self) -> bool:
        """Send emergency stop command."""
        if not self.is_connected:
            return False
        
        try:
            self.serial_conn.write(b"EMERGENCY_STOP\n")
            response = self.serial_conn.readline().decode().strip()
            return response == "STOPPED"
            
        except Exception as e:
            logger.error(f"Emergency stop failed: {e}")
            return False
    
    def is_healthy(self) -> bool:
        """Check if Arduino interface is healthy."""
        return self.is_connected and self.serial_conn.is_open
    
    def disconnect(self):
        """Disconnect from Arduino."""
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
        self.is_connected = False


class StateEstimator:
    """
    Kalman filter-based state estimator for rocket state estimation.
    
    Fuses IMU, GPS, and barometer data to estimate full 6-DOF state.
    """
    
    def __init__(self):
        """Initialize state estimator."""
        # State vector: [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, 
        #                roll, pitch, yaw, omega_x, omega_y, omega_z]
        self.state_dim = 12
        self.state = np.zeros(self.state_dim)
        
        # Covariance matrix
        self.P = np.eye(self.state_dim) * 0.1
        
        # Process noise
        self.Q = np.eye(self.state_dim) * 0.01
        
        # Measurement noise
        self.R_imu = np.eye(6) * 0.1  # IMU measurement noise
        self.R_gps = np.eye(3) * 1.0  # GPS measurement noise
        self.R_baro = 0.5              # Barometer noise
        
        # Last update time
        self.last_update = time.time()
        
        # Reference altitude (ground level)
        self.reference_altitude = None
    
    def predict(self, dt: float, thrust_vector: np.ndarray = None):
        """Predict state forward in time."""
        # State transition matrix (simplified)
        F = np.eye(self.state_dim)
        
        # Position = position + velocity * dt
        F[0:3, 3:6] = np.eye(3) * dt
        
        # Orientation integration (simplified)
        F[6:9, 9:12] = np.eye(3) * dt
        
        # Predict state
        self.state = F @ self.state
        
        # Add gravity effect
        self.state[5] -= 9.81 * dt  # Z-velocity (downward gravity)
        
        # Add thrust effect if provided
        if thrust_vector is not None:
            # Convert thrust to acceleration and integrate
            mass = 2.0  # Approximate rocket mass (kg)
            accel = thrust_vector / mass
            self.state[3:6] += accel * dt
        
        # Predict covariance
        self.P = F @ self.P @ F.T + self.Q
    
    def update_imu(self, accel: np.ndarray, gyro: np.ndarray):
        """Update state with IMU measurements."""
        # Measurement vector
        z = np.concatenate([accel, gyro])
        
        # Measurement matrix
        H = np.zeros((6, self.state_dim))
        H[3:6, 9:12] = np.eye(3)  # Gyro measures angular velocity directly
        
        # Accelerometer measurement model is more complex (includes gravity and motion)
        # Simplified version here
        H[0:3, 3:6] = np.eye(3)  # Accel measures acceleration (velocity derivative)
        
        # Kalman update
        y = z - H @ self.state  # Innovation
        S = H @ self.P @ H.T + self.R_imu  # Innovation covariance
        K = self.P @ H.T @ np.linalg.inv(S)  # Kalman gain
        
        self.state = self.state + K @ y
        self.P = (np.eye(self.state_dim) - K @ H) @ self.P
    
    def update_barometer(self, pressure: float):
        """Update altitude estimate with barometer."""
        # Convert pressure to altitude (simplified)
        if self.reference_altitude is None:
            self.reference_altitude = pressure
        
        # Barometric altitude formula (simplified)
        altitude = 44330 * (1 - (pressure / self.reference_altitude) ** 0.1903)
        
        # Measurement matrix
        H = np.zeros((1, self.state_dim))
        H[0, 2] = 1.0  # Measures Z position
        
        # Kalman update
        y = altitude - H @ self.state
        S = H @ self.P @ H.T + self.R_baro
        K = self.P @ H.T / S
        
        self.state = self.state + K * y
        self.P = self.P - np.outer(K, K) * S
    
    def get_system_state(self, thrust: float, mass: float, 
                        gimbal_angles: np.ndarray, fuel_remaining: float) -> SystemState:
        """Convert internal state to SystemState format."""
        return SystemState(
            position=self.state[0:3].copy(),
            orientation=self.state[6:9].copy(),
            velocity=self.state[3:6].copy(),
            angular_velocity=self.state[9:12].copy(),
            thrust_magnitude=thrust,
            mass=mass,
            center_of_mass_offset=0.0,  # Would need separate estimation
            gimbal_angles=gimbal_angles.copy(),
            fuel_remaining=fuel_remaining,
            wind_velocity=np.zeros(3)  # Would need separate estimation
        )


class RealTimeController:
    """
    Real-time control system for rocket TVC using trained TD3 model.
    """
    
    def __init__(self, model_path: str, sensor_interface: SensorInterface,
                 actuator_interface: ActuatorInterface, use_onnx: bool = True):
        """
        Initialize real-time controller.
        
        Args:
            model_path: Path to trained model (ONNX or PyTorch)
            sensor_interface: Sensor interface implementation
            actuator_interface: Actuator interface implementation
            use_onnx: Whether to use ONNX runtime for inference
        """
        self.sensor_interface = sensor_interface
        self.actuator_interface = actuator_interface
        self.state_estimator = StateEstimator()
        
        # Load model
        if use_onnx and ONNX_AVAILABLE and model_path.endswith('.onnx'):
            self.model = ort.InferenceSession(model_path)
            self.model_type = 'onnx'
            logger.info(f"Loaded ONNX model: {model_path}")
        else:
            # Fallback to PyTorch model
            import torch
            from rocket_env import TD3Agent
            
            # Initialize agent and load state
            self.agent = TD3Agent(21, 2, np.radians(7.0))
            self.agent.load(model_path)
            self.agent.actor.eval()
            self.model_type = 'pytorch'
            logger.info(f"Loaded PyTorch model: {model_path}")
        
        # Control parameters
        self.max_gimbal_angle = np.radians(7.0)  # 7 degrees max
        self.control_frequency = 100.0  # Hz
        self.control_period = 1.0 / self.control_frequency
        
        # Safety parameters
        self.max_attitude_error = np.radians(45)  # 45 degrees
        self.max_angular_rate = np.radians(180)   # 180 deg/s
        self.emergency_stop_triggered = False
        
        # Data logging
        self.log_data = []
        self.logging_enabled = True
        
        # Control loop thread
        self.control_thread = None
        self.running = False
        
        logger.info("Real-time controller initialized")
    
    def predict_action(self, system_state: SystemState) -> np.ndarray:
        """Predict control action using trained model."""
        # Convert system state to model input format
        state_vector = np.array([
            *system_state.position,           # position_x, y, z
            *system_state.orientation,        # roll, pitch, yaw
            *system_state.velocity,           # velocity_x, y, z
            *system_state.angular_velocity,   # angular_velocity_x, y, z
            system_state.thrust_magnitude,    # thrust_magnitude
            system_state.mass,                # mass
            system_state.center_of_mass_offset,  # center_of_mass_offset
            *system_state.gimbal_angles,      # current_gimbal_x, y
            system_state.fuel_remaining,      # fuel_remaining
            *system_state.wind_velocity       # wind_x, y, z
        ], dtype=np.float32)
        
        # Predict action
        if self.model_type == 'onnx':
            # ONNX inference
            input_dict = {self.model.get_inputs()[0].name: state_vector.reshape(1, -1)}
            action = self.model.run(None, input_dict)[0][0]
        else:
            # PyTorch inference
            action = self.agent.select_action(state_vector, noise_scale=0.0)
        
        # Clip action to safe range
        action = np.clip(action, -self.max_gimbal_angle, self.max_gimbal_angle)
        
        return action
    
    def safety_check(self, system_state: SystemState) -> bool:
        """Perform safety checks on system state."""
        # Check attitude error
        attitude_magnitude = np.linalg.norm(system_state.orientation)
        if attitude_magnitude > self.max_attitude_error:
            logger.error(f"Attitude error too large: {np.degrees(attitude_magnitude):.1f} deg")
            return False
        
        # Check angular rate
        angular_rate_magnitude = np.linalg.norm(system_state.angular_velocity)
        if angular_rate_magnitude > self.max_angular_rate:
            logger.error(f"Angular rate too high: {np.degrees(angular_rate_magnitude):.1f} deg/s")
            return False
        
        # Check altitude (don't go below ground)
        if system_state.position[2] < -1.0:  # 1m below reference
            logger.error("Altitude below ground level")
            return False
        
        return True
    
    def control_loop(self):
        """Main real-time control loop."""
        logger.info("Starting real-time control loop")
        
        last_time = time.time()
        
        while self.running:
            try:
                current_time = time.time()
                dt = current_time - last_time
                
                # Read sensors
                sensor_data = self.sensor_interface.read_sensors()
                
                # Update state estimation
                self.state_estimator.predict(dt)
                self.state_estimator.update_imu(
                    np.array(sensor_data.accelerometer),
                    np.array(sensor_data.gyroscope)
                )
                self.state_estimator.update_barometer(sensor_data.barometer)
                
                # Get current system state
                system_state = self.state_estimator.get_system_state(
                    thrust=sensor_data.thrust_sensor,
                    mass=2.0,  # Would need mass estimation
                    gimbal_angles=np.zeros(2),  # Would track last command
                    fuel_remaining=sensor_data.fuel_level
                )
                
                # Safety check
                if not self.safety_check(system_state):
                    self.emergency_stop()
                    break
                
                # Predict control action
                action = self.predict_action(system_state)
                
                # Send control command
                command = ControlCommand(
                    timestamp=current_time,
                    gimbal_x=action[0],
                    gimbal_y=action[1]
                )
                
                success = self.actuator_interface.send_command(command)
                if not success:
                    logger.error("Failed to send control command")
                
                # Log data
                if self.logging_enabled:
                    log_entry = {
                        'timestamp': current_time,
                        'sensor_data': sensor_data,
                        'system_state': system_state,
                        'action': action,
                        'command_success': success
                    }
                    self.log_data.append(log_entry)
                
                # Maintain control frequency
                elapsed = time.time() - current_time
                sleep_time = max(0, self.control_period - elapsed)
                time.sleep(sleep_time)
                
                last_time = current_time
                
            except Exception as e:
                logger.error(f"Control loop error: {e}")
                self.emergency_stop()
                break
        
        logger.info("Control loop stopped")
    
    def start(self):
        """Start the real-time control system."""
        if self.running:
            logger.warning("Controller already running")
            return
        
        # Check sensor and actuator health
        if not self.sensor_interface.is_healthy():
            raise RuntimeError("Sensor interface not healthy")
        
        # Calibrate sensors
        if not self.sensor_interface.calibrate():
            raise RuntimeError("Sensor calibration failed")
        
        # Start control loop in separate thread
        self.running = True
        self.control_thread = threading.Thread(target=self.control_loop)
        self.control_thread.start()
        
        logger.info("Real-time controller started")
    
    def stop(self):
        """Stop the real-time control system."""
        if not self.running:
            return
        
        self.running = False
        
        if self.control_thread:
            self.control_thread.join(timeout=5.0)
        
        logger.info("Real-time controller stopped")
    
    def emergency_stop(self):
        """Emergency stop procedure."""
        if self.emergency_stop_triggered:
            return
        
        self.emergency_stop_triggered = True
        logger.critical("EMERGENCY STOP TRIGGERED!")
        
        # Stop actuators
        self.actuator_interface.emergency_stop()
        
        # Stop control loop
        self.running = False
        
        # Save flight data
        self.save_flight_data("emergency_flight_log.json")
    
    def save_flight_data(self, filename: str):
        """Save logged flight data to file."""
        if not self.log_data:
            return
        
        # Convert to JSON-serializable format
        serializable_data = []
        for entry in self.log_data:
            # Convert numpy arrays and custom objects to lists/dicts
            serializable_entry = {
                'timestamp': entry['timestamp'],
                'action': entry['action'].tolist(),
                'command_success': entry['command_success']
                # Add more fields as needed
            }
            serializable_data.append(serializable_entry)
        
        with open(filename, 'w') as f:
            json.dump(serializable_data, f, indent=2)
        
        logger.info(f"Flight data saved to {filename}")


def main():
    """Main deployment function."""
    parser = argparse.ArgumentParser(description='Real-time TVC deployment')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model file')
    parser.add_argument('--port', type=str, default='COM3',
                       help='Serial port for Arduino communication')
    parser.add_argument('--test-mode', action='store_true',
                       help='Run in test mode (no hardware)')
    
    args = parser.parse_args()
    
    try:
        if args.test_mode:
            logger.info("Running in test mode - no hardware interface")
            # Create mock interfaces for testing
            from unittest.mock import MagicMock
            sensor_interface = MagicMock(spec=SensorInterface)
            actuator_interface = MagicMock(spec=ActuatorInterface)
            
            # Configure mock behavior
            sensor_interface.is_healthy.return_value = True
            sensor_interface.calibrate.return_value = True
            sensor_interface.read_sensors.return_value = SensorData(
                timestamp=time.time(),
                accelerometer=(0, 0, 9.81),
                gyroscope=(0, 0, 0),
                magnetometer=(0, 25, 0),
                barometer=101325,
                thrust_sensor=10.0,
                fuel_level=0.8
            )
            actuator_interface.send_command.return_value = True
            actuator_interface.emergency_stop.return_value = True
        else:
            # Create hardware interfaces
            arduino = ArduinoInterface(port=args.port)
            sensor_interface = arduino
            actuator_interface = arduino
        
        # Initialize controller
        controller = RealTimeController(
            model_path=args.model,
            sensor_interface=sensor_interface,
            actuator_interface=actuator_interface
        )
        
        # Start control system
        controller.start()
        
        # Run for a test period (or until interrupted)
        try:
            if args.test_mode:
                logger.info("Test mode: running for 10 seconds")
                time.sleep(10)
            else:
                logger.info("Real-time control active. Press Ctrl+C to stop.")
                while True:
                    time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutdown requested by user")
        
        # Stop controller
        controller.stop()
        controller.save_flight_data("flight_log.json")
        
        # Disconnect hardware
        if not args.test_mode:
            arduino.disconnect()
        
        logger.info("Deployment completed successfully")
        
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        raise


if __name__ == "__main__":
    main()
