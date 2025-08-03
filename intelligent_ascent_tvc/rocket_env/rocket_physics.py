"""
Intelligent Ascent TVC System - Physics Simulation Environment
==============================================================

This module implements a high-fidelity 6-DOF physics simulation environment for model rockets
with thrust vector control capabilities. The environment serves as the training ground for
reinforcement learning agents.

Author: AI Assistant
Date: August 2025
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import yaml
import os
from typing import Dict, List, Tuple, Optional, Any
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RocketDynamics:
    """
    High-fidelity 6-DOF rocket dynamics model with thrust vector control.
    
    Implements the theoretical framework from the documentation including:
    - Variable mass and center of gravity calculations
    - Moment of inertia updates
    - Aerodynamic forces and torques
    - Thrust vectoring mechanics
    - Environmental factors (wind, gravity, air density)
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rocket_params = config['rocket_params']
        self.engine_params = config['engine_params']
        self.world_params = config['world_params']
        
        # Load thrust curve data
        self.thrust_curve = self._load_thrust_curve()
        
        # Initialize rocket state
        self.reset()
        
    def _load_thrust_curve(self) -> interp1d:
        """Load and interpolate thrust curve data from CSV file."""
        thrust_file = os.path.join(
            os.path.dirname(__file__), 
            '..', 
            self.engine_params['thrust_curve_file']
        )
        
        try:
            thrust_data = pd.read_csv(thrust_file)
            time_points = thrust_data['time'].values
            thrust_points = thrust_data['thrust'].values
            
            # Create interpolation function with extrapolation handling
            thrust_interp = interp1d(
                time_points, 
                thrust_points, 
                kind='linear', 
                fill_value=0.0, 
                bounds_error=False
            )
            
            logger.info(f"Loaded thrust curve from {thrust_file}")
            return thrust_interp
            
        except Exception as e:
            logger.error(f"Failed to load thrust curve: {e}")
            # Fallback to constant thrust
            return lambda t: 6.0 if 0 <= t <= 1.67 else 0.0
    
    def reset(self):
        """Reset rocket to initial conditions."""
        # Mass properties
        self.dry_mass = self.rocket_params['dry_mass']
        self.propellant_mass = self.rocket_params['propellant_mass']
        self.total_mass = self.dry_mass + self.propellant_mass
        self.initial_mass = self.total_mass
        
        # Position and orientation (6-DOF state)
        self.position = np.array(self.rocket_params['initial_position'], dtype=np.float64)
        self.velocity = np.array(self.rocket_params['initial_velocity'], dtype=np.float64)
        self.orientation = np.array(self.rocket_params['initial_orientation'], dtype=np.float64)  # [roll, pitch, yaw]
        self.angular_velocity = np.array(self.rocket_params['initial_angular_velocity'], dtype=np.float64)
        
        # TVC system state
        self.gimbal_angles = np.array([0.0, 0.0])  # [pitch, yaw] gimbal angles in radians
        self.gimbal_rates = np.array([0.0, 0.0])   # Gimbal angular rates
        
        # Flight parameters
        self.time = 0.0
        self.fuel_consumed = 0.0
        self.thrust_magnitude = 0.0
        
        # Center of gravity and moments of inertia
        self._update_mass_properties()
        
        # Wind state
        self.wind_velocity = np.array(self.world_params['wind_base_velocity'])
        
        logger.info("Rocket dynamics reset to initial conditions")
    
    def _update_mass_properties(self):
        """Update center of gravity and moments of inertia based on current fuel consumption."""
        # Calculate current propellant mass
        remaining_propellant = max(0.0, self.propellant_mass - self.fuel_consumed)
        self.total_mass = self.dry_mass + remaining_propellant
        
        # Calculate center of gravity shift
        # Assume propellant is stored in lower 40% of rocket body
        propellant_cg_offset = 0.2 * self.rocket_params['body_length']  # From nose
        dry_cg_offset = 0.6 * self.rocket_params['body_length']         # From nose
        
        if remaining_propellant > 0:
            self.center_of_gravity = (
                (self.dry_mass * dry_cg_offset + remaining_propellant * propellant_cg_offset) / 
                self.total_mass
            )
        else:
            self.center_of_gravity = dry_cg_offset
        
        # Calculate moments of inertia (simplified model)
        body_length = self.rocket_params['body_length']
        body_radius = self.rocket_params['body_diameter'] / 2
        
        # Longitudinal moment of inertia (roll axis)
        self.I_xx = 0.5 * self.total_mass * body_radius**2
        
        # Transverse moments of inertia (pitch and yaw axes)
        # Using thin rod approximation with point mass correction
        I_rod = (1/12) * self.total_mass * body_length**2
        I_point_mass = self.total_mass * (self.center_of_gravity - body_length/2)**2
        self.I_yy = self.I_zz = I_rod + I_point_mass
        
        self.inertia_matrix = np.diag([self.I_xx, self.I_yy, self.I_zz])
    
    def step(self, dt: float, gimbal_command: np.ndarray, wind_disturbance: Optional[np.ndarray] = None):
        """
        Advance rocket dynamics by one time step.
        
        Args:
            dt: Time step (seconds)
            gimbal_command: [pitch_angle, yaw_angle] in radians
            wind_disturbance: Optional wind velocity perturbation [vx, vy, vz]
        """
        self.time += dt
        
        # Update wind conditions
        if wind_disturbance is not None:
            self.wind_velocity += wind_disturbance
        
        # Apply gimbal rate limits
        max_gimbal_rate = np.radians(self.engine_params['gimbal_rate_limit_deg_s'])
        gimbal_command = np.clip(gimbal_command, 
                                np.radians(-self.engine_params['gimbal_limit_deg']),
                                np.radians(self.engine_params['gimbal_limit_deg']))
        
        desired_gimbal_rates = (gimbal_command - self.gimbal_angles) / dt
        self.gimbal_rates = np.clip(desired_gimbal_rates, -max_gimbal_rate, max_gimbal_rate)
        self.gimbal_angles += self.gimbal_rates * dt
        
        # Get thrust magnitude from thrust curve
        self.thrust_magnitude = float(self.thrust_curve(self.time))
        
        # Calculate fuel consumption
        if self.thrust_magnitude > 0:
            # Simplified fuel consumption model
            fuel_flow_rate = self.propellant_mass / self.engine_params['burn_time']
            self.fuel_consumed += fuel_flow_rate * dt
            self.fuel_consumed = min(self.fuel_consumed, self.propellant_mass)
        
        # Update mass properties
        self._update_mass_properties()
        
        # Calculate forces and torques
        forces = self._calculate_forces()
        torques = self._calculate_torques(forces)
        
        # Integrate equations of motion
        self._integrate_motion(dt, forces, torques)
    
    def _calculate_forces(self) -> Dict[str, np.ndarray]:
        """Calculate all forces acting on the rocket in world frame."""
        forces = {}
        
        # Gravity force (world frame)
        forces['gravity'] = np.array([0, 0, -self.total_mass * self.world_params['gravity']])
        
        # Thrust force (body frame -> world frame)
        if self.thrust_magnitude > 0:
            # Thrust direction in body frame (considering gimbal angles)
            thrust_body = np.array([
                self.thrust_magnitude * np.sin(self.gimbal_angles[1]),  # Yaw deflection
                self.thrust_magnitude * np.sin(self.gimbal_angles[0]),  # Pitch deflection
                self.thrust_magnitude * np.cos(np.linalg.norm(self.gimbal_angles))
            ])
            
            # Transform to world frame
            rotation_matrix = R.from_euler('xyz', self.orientation).as_matrix()
            forces['thrust'] = rotation_matrix @ thrust_body
        else:
            forces['thrust'] = np.zeros(3)
        
        # Aerodynamic forces
        forces['aerodynamic'] = self._calculate_aerodynamic_forces()
        
        return forces
    
    def _calculate_aerodynamic_forces(self) -> np.ndarray:
        """Calculate aerodynamic forces (drag and normal forces)."""
        # Relative velocity (rocket velocity - wind velocity)
        relative_velocity = self.velocity - self.wind_velocity
        relative_speed = np.linalg.norm(relative_velocity)
        
        if relative_speed < 0.1:  # Avoid division by zero
            return np.zeros(3)
        
        # Air density (simplified - assume constant)
        air_density = self.world_params['air_density_sl']
        
        # Reference area (cross-sectional area)
        reference_area = np.pi * (self.rocket_params['body_diameter'] / 2)**2
        
        # Dynamic pressure
        q = 0.5 * air_density * relative_speed**2
        
        # Drag force (opposite to velocity direction)
        drag_magnitude = self.rocket_params['drag_coefficient'] * reference_area * q
        drag_direction = -relative_velocity / relative_speed
        drag_force = drag_magnitude * drag_direction
        
        # Normal forces (simplified - assume small angle of attack)
        # Transform relative velocity to body frame
        rotation_matrix = R.from_euler('xyz', self.orientation).as_matrix()
        relative_velocity_body = rotation_matrix.T @ relative_velocity
        
        # Angle of attack components
        if abs(relative_velocity_body[2]) > 0.1:
            alpha_y = np.arctan2(relative_velocity_body[1], relative_velocity_body[2])
            alpha_z = np.arctan2(relative_velocity_body[0], relative_velocity_body[2])
        else:
            alpha_y = alpha_z = 0.0
        
        # Normal force coefficients
        CN_alpha = self.rocket_params['normal_force_coefficient']
        normal_force_y = CN_alpha * alpha_y * reference_area * q
        normal_force_z = CN_alpha * alpha_z * reference_area * q
        
        # Normal forces in body frame
        normal_force_body = np.array([normal_force_z, normal_force_y, 0])
        
        # Transform to world frame
        normal_force_world = rotation_matrix @ normal_force_body
        
        return drag_force + normal_force_world
    
    def _calculate_torques(self, forces: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Calculate all torques acting on the rocket about center of gravity."""
        torques = {}
        
        # Thrust torque (due to gimbal deflection and offset from CG)
        if self.thrust_magnitude > 0:
            # Gimbal pivot location relative to CG
            gimbal_offset = np.array([0, 0, self.engine_params['gimbal_pivot_offset']])
            gimbal_position_cg = gimbal_offset - np.array([0, 0, self.center_of_gravity])
            
            # Thrust force in body frame
            thrust_body = np.array([
                self.thrust_magnitude * np.sin(self.gimbal_angles[1]),
                self.thrust_magnitude * np.sin(self.gimbal_angles[0]),
                self.thrust_magnitude * np.cos(np.linalg.norm(self.gimbal_angles))
            ])
            
            # Torque = r × F
            torques['thrust'] = np.cross(gimbal_position_cg, thrust_body)
        else:
            torques['thrust'] = np.zeros(3)
        
        # Aerodynamic torque (about center of pressure)
        cp_offset = self.rocket_params['center_of_pressure_offset']
        cp_position_cg = np.array([0, 0, cp_offset - self.center_of_gravity])
        
        # Transform aerodynamic force to body frame
        rotation_matrix = R.from_euler('xyz', self.orientation).as_matrix()
        aero_force_body = rotation_matrix.T @ forces['aerodynamic']
        
        torques['aerodynamic'] = np.cross(cp_position_cg, aero_force_body)
        
        return torques
    
    def _integrate_motion(self, dt: float, forces: Dict[str, np.ndarray], torques: Dict[str, np.ndarray]):
        """Integrate equations of motion using Euler integration."""
        # Total force and torque
        total_force = sum(forces.values())
        total_torque = sum(torques.values())
        
        # Linear motion (Newton's second law)
        acceleration = total_force / self.total_mass
        self.velocity += acceleration * dt
        self.position += self.velocity * dt
        
        # Angular motion (Euler's equation for rigid body rotation)
        # Transform torque to body frame
        rotation_matrix = R.from_euler('xyz', self.orientation).as_matrix()
        torque_body = rotation_matrix.T @ total_torque
        
        # Angular acceleration in body frame
        angular_acceleration_body = np.linalg.solve(self.inertia_matrix, 
                                                   torque_body - np.cross(self.angular_velocity, 
                                                                         self.inertia_matrix @ self.angular_velocity))
        
        # Update angular velocity in body frame
        self.angular_velocity += angular_acceleration_body * dt
        
        # Update orientation (convert angular velocity to Euler angle rates)
        # Simplified approach for small angles
        self.orientation += self.angular_velocity * dt
        
        # Normalize angles to [-π, π]
        self.orientation = np.mod(self.orientation + np.pi, 2*np.pi) - np.pi
    
    def get_state_vector(self) -> np.ndarray:
        """Get complete state vector for RL agent."""
        # Calculate remaining fuel fraction
        fuel_remaining = max(0.0, (self.propellant_mass - self.fuel_consumed) / self.propellant_mass)
        
        # Calculate center of mass offset from initial position
        cg_offset = self.center_of_gravity - 0.6 * self.rocket_params['body_length']
        
        # Construct state vector as specified in requirements
        state = np.concatenate([
            self.position,                    # position_x, position_y, position_z
            self.orientation,                 # roll, pitch, yaw
            self.velocity,                    # velocity_x, velocity_y, velocity_z
            self.angular_velocity,            # angular_velocity_x, angular_velocity_y, angular_velocity_z
            [self.thrust_magnitude,           # thrust_magnitude
             self.total_mass,                 # mass
             cg_offset,                       # center_of_mass_offset
             self.gimbal_angles[0],           # current_gimbal_x (pitch)
             self.gimbal_angles[1],           # current_gimbal_y (yaw)
             fuel_remaining],                 # fuel_remaining
            self.wind_velocity                # wind_x, wind_y, wind_z
        ])
        
        return state.astype(np.float32)


class RocketTVCEnvironment(gym.Env):
    """
    Gymnasium environment for rocket thrust vector control training.
    
    This environment provides the interface between the rocket dynamics
    and the reinforcement learning agent.
    """
    
    def __init__(self, config_path: str = None):
        super().__init__()
        
        # Load configuration
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'environment.yaml')
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize rocket dynamics
        self.rocket = RocketDynamics(self.config)
        
        # Environment parameters
        self.dt = 0.01  # 100 Hz simulation
        self.max_episode_time = 10.0  # Maximum episode duration (seconds)
        self.episode_step = 0
        self.max_steps = int(self.max_episode_time / self.dt)
        
        # Define observation and action spaces
        self._setup_spaces()
        
        # Performance tracking
        self.episode_rewards = []
        self.episode_info = {}
        
        logger.info("Rocket TVC Environment initialized")
    
    def _setup_spaces(self):
        """Setup observation and action spaces for the RL agent."""
        # State space (21 dimensions as specified)
        state_size = 21
        obs_low = np.array([
            -1000, -1000, 0,        # position (x, y, z) - z >= 0 (above ground)
            -np.pi, -np.pi, -np.pi,  # orientation (roll, pitch, yaw)
            -100, -100, -100,        # velocity
            -10, -10, -10,           # angular velocity
            0, 0.1, -0.5,            # thrust, mass, cg_offset
            -np.pi/4, -np.pi/4,      # gimbal angles
            0,                       # fuel_remaining
            -20, -20, -20            # wind velocity
        ], dtype=np.float32)
        
        obs_high = np.array([
            1000, 1000, 5000,        # position
            np.pi, np.pi, np.pi,     # orientation
            100, 100, 100,           # velocity
            10, 10, 10,              # angular velocity
            20, 10, 0.5,             # thrust, mass, cg_offset
            np.pi/4, np.pi/4,        # gimbal angles
            1.0,                     # fuel_remaining
            20, 20, 20               # wind velocity
        ], dtype=np.float32)
        
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        
        # Action space: [gimbal_pitch, gimbal_yaw] in radians
        gimbal_limit = np.radians(self.config['engine_params']['gimbal_limit_deg'])
        self.action_space = spaces.Box(
            low=np.array([-gimbal_limit, -gimbal_limit], dtype=np.float32),
            high=np.array([gimbal_limit, gimbal_limit], dtype=np.float32),
            dtype=np.float32
        )
    
    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
        
        # Reset rocket dynamics
        self.rocket.reset()
        
        # Add some randomization to initial conditions if specified
        if options and options.get('randomize_initial_conditions', False):
            self._randomize_initial_conditions()
        
        # Reset episode tracking
        self.episode_step = 0
        self.episode_info = {}
        
        # Get initial observation
        observation = self.rocket.get_state_vector()
        info = self._get_info()
        
        return observation, info
    
    def _randomize_initial_conditions(self):
        """Add random perturbations to initial conditions for robustness."""
        # Small random perturbations to orientation (±2 degrees)
        orientation_noise = np.random.uniform(-np.radians(2), np.radians(2), 3)
        self.rocket.orientation += orientation_noise
        
        # Small random perturbations to angular velocity (±0.1 rad/s)
        angular_velocity_noise = np.random.uniform(-0.1, 0.1, 3)
        self.rocket.angular_velocity += angular_velocity_noise
        
        # Random wind perturbation
        wind_noise = np.random.uniform(-2.0, 2.0, 3)
        self.rocket.wind_velocity += wind_noise
    
    def step(self, action):
        """Execute one step in the environment."""
        # Clip action to valid range
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Add random wind gusts
        wind_disturbance = self._generate_wind_disturbance()
        
        # Step rocket dynamics
        self.rocket.step(self.dt, action, wind_disturbance)
        
        # Get observation
        observation = self.rocket.get_state_vector()
        
        # Calculate reward
        reward = self._calculate_reward(action)
        
        # Check termination conditions
        terminated = self._check_termination()
        truncated = self.episode_step >= self.max_steps
        
        # Update episode tracking
        self.episode_step += 1
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _generate_wind_disturbance(self) -> np.ndarray:
        """Generate random wind gusts based on configuration."""
        if not self.config['world_params']['wind_enabled']:
            return np.zeros(3)
        
        gust_probability = self.config['world_params']['wind_gust_probability']
        gust_magnitude = self.config['world_params']['wind_gust_magnitude']
        
        if np.random.random() < gust_probability:
            # Generate random gust direction and magnitude
            gust_direction = np.random.uniform(-1, 1, 3)
            gust_direction /= np.linalg.norm(gust_direction)
            gust_strength = np.random.uniform(0, gust_magnitude)
            return gust_direction * gust_strength
        
        return np.zeros(3)
    
    def _calculate_reward(self, action: np.ndarray) -> float:
        """Calculate reward based on rocket performance and control objectives."""
        reward = 0.0
        
        # Attitude stability reward (penalize large attitude errors)
        attitude_error = np.linalg.norm(self.rocket.orientation)
        attitude_reward = -10.0 * attitude_error**2
        
        # Angular stability reward (penalize high angular rates)
        angular_rate_magnitude = np.linalg.norm(self.rocket.angular_velocity)
        angular_stability_reward = -5.0 * angular_rate_magnitude**2
        
        # Altitude reward (encourage upward flight)
        if self.rocket.position[2] > 0:
            altitude_reward = 1.0 * self.rocket.position[2]
        else:
            altitude_reward = -100.0  # Heavy penalty for going underground
        
        # Control effort penalty (encourage fuel efficiency)
        control_effort = np.linalg.norm(action)
        control_penalty = -0.1 * control_effort**2
        
        # Velocity reward (encourage controlled ascent)
        if self.rocket.velocity[2] > 0:  # Upward velocity
            velocity_reward = 0.5 * self.rocket.velocity[2]
        else:
            velocity_reward = -1.0 * abs(self.rocket.velocity[2])  # Penalize downward velocity
        
        # Safety penalties
        safety_penalty = 0.0
        
        # Excessive attitude penalty
        max_attitude = np.radians(45)  # 45 degrees
        if attitude_error > max_attitude:
            safety_penalty -= 50.0
        
        # Excessive angular rate penalty
        max_angular_rate = np.radians(180)  # 180 deg/s
        if angular_rate_magnitude > max_angular_rate:
            safety_penalty -= 50.0
        
        # Combine all reward components
        reward = (attitude_reward + angular_stability_reward + altitude_reward + 
                 control_penalty + velocity_reward + safety_penalty)
        
        return float(reward)
    
    def _check_termination(self) -> bool:
        """Check if episode should be terminated."""
        # Ground impact
        if self.rocket.position[2] <= 0:
            return True
        
        # Excessive attitude error (complete loss of control)
        attitude_error = np.linalg.norm(self.rocket.orientation)
        if attitude_error > np.radians(90):  # 90 degrees
            return True
        
        # Fuel exhausted and very low altitude with downward velocity
        fuel_remaining = (self.rocket.propellant_mass - self.rocket.fuel_consumed) / self.rocket.propellant_mass
        if (fuel_remaining <= 0 and 
            self.rocket.position[2] < 10 and 
            self.rocket.velocity[2] < -5):
            return True
        
        return False
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional information about the environment state."""
        fuel_remaining = max(0.0, (self.rocket.propellant_mass - self.rocket.fuel_consumed) / self.rocket.propellant_mass)
        
        info = {
            'time': self.rocket.time,
            'altitude': self.rocket.position[2],
            'velocity': np.linalg.norm(self.rocket.velocity),
            'attitude_error': np.linalg.norm(self.rocket.orientation),
            'angular_rate': np.linalg.norm(self.rocket.angular_velocity),
            'fuel_remaining': fuel_remaining,
            'thrust': self.rocket.thrust_magnitude,
            'mass': self.rocket.total_mass,
            'gimbal_angles': self.rocket.gimbal_angles.copy()
        }
        
        return info
    
    def render(self, mode='human'):
        """Render the environment (placeholder for future visualization)."""
        if mode == 'human':
            print(f"Time: {self.rocket.time:.2f}s, "
                  f"Alt: {self.rocket.position[2]:.1f}m, "
                  f"Attitude: [{np.degrees(self.rocket.orientation[0]):.1f}, "
                  f"{np.degrees(self.rocket.orientation[1]):.1f}, "
                  f"{np.degrees(self.rocket.orientation[2]):.1f}] deg")
    
    def close(self):
        """Clean up environment resources."""
        pass


if __name__ == "__main__":
    # Test the environment
    env = RocketTVCEnvironment()
    obs, info = env.reset()
    
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial info: {info}")
    
    # Run a few test steps
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: reward={reward:.2f}, altitude={info['altitude']:.1f}m")
        
        if terminated or truncated:
            break
    
    env.close()
