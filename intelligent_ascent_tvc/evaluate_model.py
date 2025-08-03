"""
Intelligent Ascent TVC System - Model Evaluation and Testing
===========================================================

Comprehensive evaluation script for trained TD3 models with detailed
performance analysis, Monte Carlo testing, and visualization.

Author: AI Assistant
Date: August 2025
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
import yaml
import torch
from typing import Dict, List, Tuple, Any, Optional
import argparse
from tqdm import tqdm
import logging

# Add rocket_env to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rocket_env import RocketTVCEnvironment, TD3Agent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive evaluation framework for trained TD3 TVC models.
    """
    
    def __init__(self, model_path: str, config_path: str = "configs/environment.yaml"):
        """
        Initialize model evaluator.
        
        Args:
            model_path: Path to trained TD3 model
            config_path: Path to environment configuration
        """
        self.model_path = model_path
        self.config_path = config_path
        
        # Load environment configuration
        with open(config_path, 'r') as f:
            self.env_config = yaml.safe_load(f)
        
        # Initialize environment
        self.env = RocketTVCEnvironment(config_path)
        
        # Get environment dimensions
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.max_action = float(self.env.action_space.high[0])
        
        # Initialize TD3 agent
        self.agent = TD3Agent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            max_action=self.max_action
        )
        
        # Load trained model
        if not self.agent.load(model_path):
            raise FileNotFoundError(f"Could not load model from {model_path}")
        
        # Set agent to evaluation mode
        self.agent.actor.eval()
        
        # Results storage
        self.evaluation_results = []
        self.flight_trajectories = []
        
        logger.info(f"Model evaluator initialized with model: {model_path}")
    
    def evaluate_single_flight(self, scenario_config: Dict[str, Any] = None, 
                              render: bool = False, record_trajectory: bool = True) -> Dict[str, Any]:
        """
        Evaluate a single flight scenario.
        
        Args:
            scenario_config: Custom scenario configuration
            render: Whether to render the flight
            record_trajectory: Whether to record full trajectory data
            
        Returns:
            Flight performance metrics
        """
        # Apply scenario configuration if provided
        if scenario_config:
            self._apply_scenario_config(scenario_config)
        
        # Reset environment
        state, info = self.env.reset()
        
        # Initialize trajectory recording
        trajectory = [] if record_trajectory else None
        episode_reward = 0
        step = 0
        
        # Flight simulation loop
        done = False
        while not done:
            # Get action from trained agent (deterministic)
            action = self.agent.select_action(state, noise_scale=0.0)
            
            # Execute action
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            step += 1
            
            # Record trajectory data
            if record_trajectory:
                trajectory_point = {
                    'time': info['time'],
                    'step': step,
                    'position_x': state[0],
                    'position_y': state[1],
                    'position_z': state[2],
                    'roll': np.degrees(state[3]),
                    'pitch': np.degrees(state[4]),
                    'yaw': np.degrees(state[5]),
                    'velocity_x': state[6],
                    'velocity_y': state[7],
                    'velocity_z': state[8],
                    'angular_vel_x': np.degrees(state[9]),
                    'angular_vel_y': np.degrees(state[10]),
                    'angular_vel_z': np.degrees(state[11]),
                    'thrust': state[12],
                    'mass': state[13],
                    'gimbal_x': np.degrees(action[0]),
                    'gimbal_y': np.degrees(action[1]),
                    'fuel_remaining': state[17],
                    'reward': reward,
                    'altitude': info['altitude'],
                    'attitude_error': np.degrees(info['attitude_error']),
                    'angular_rate': np.degrees(info['angular_rate'])
                }
                trajectory.append(trajectory_point)
            
            # Render if requested
            if render:
                self.env.render()
            
            state = next_state
        
        # Calculate performance metrics
        metrics = self._calculate_performance_metrics(trajectory, info)
        
        if record_trajectory:
            self.flight_trajectories.append(trajectory)
        
        return metrics
    
    def monte_carlo_evaluation(self, num_trials: int = 100, 
                             parameter_variations: Dict[str, Tuple[float, float]] = None) -> Dict[str, Any]:
        """
        Perform Monte Carlo evaluation with parameter variations.
        
        Args:
            num_trials: Number of Monte Carlo trials
            parameter_variations: Dictionary of parameter ranges for variation
                                 Format: {'param_name': (min_value, max_value)}
            
        Returns:
            Statistical analysis of performance across trials
        """
        logger.info(f"Starting Monte Carlo evaluation with {num_trials} trials")
        
        # Default parameter variations if not specified
        if parameter_variations is None:
            parameter_variations = {
                'dry_mass': (1.3, 1.7),  # ±20% variation
                'wind_base_velocity_y': (0.0, 5.0),  # 0-5 m/s wind
                'initial_orientation_pitch': (-0.1, 0.1),  # ±5.7 degrees
                'initial_orientation_yaw': (-0.1, 0.1)
            }
        
        trial_results = []
        
        for trial in tqdm(range(num_trials), desc="Monte Carlo Trials"):
            # Generate random parameter variations
            scenario_config = self._generate_random_scenario(parameter_variations)
            
            # Evaluate flight
            try:
                metrics = self.evaluate_single_flight(
                    scenario_config=scenario_config,
                    record_trajectory=False  # Save memory for large trials
                )
                metrics['trial'] = trial
                metrics['scenario_config'] = scenario_config
                trial_results.append(metrics)
                
            except Exception as e:
                logger.warning(f"Trial {trial} failed: {e}")
                continue
        
        # Statistical analysis
        mc_analysis = self._analyze_monte_carlo_results(trial_results)
        
        # Save results
        self._save_monte_carlo_results(trial_results, mc_analysis)
        
        logger.info(f"Monte Carlo evaluation completed. Success rate: {mc_analysis['success_rate']:.2%}")
        
        return mc_analysis
    
    def stress_test_evaluation(self) -> Dict[str, Any]:
        """
        Perform stress testing with extreme conditions.
        
        Returns:
            Stress test results
        """
        logger.info("Starting stress test evaluation")
        
        stress_scenarios = [
            {
                'name': 'High Wind',
                'config': {'wind_base_velocity_y': 15.0, 'wind_gust_magnitude': 10.0}
            },
            {
                'name': 'Heavy Rocket',
                'config': {'dry_mass': 2.5, 'propellant_mass': 0.8}
            },
            {
                'name': 'Light Rocket',
                'config': {'dry_mass': 0.8, 'propellant_mass': 0.3}
            },
            {
                'name': 'Initial Tilt',
                'config': {'initial_orientation_pitch': 0.3, 'initial_orientation_yaw': 0.2}
            },
            {
                'name': 'Low Thrust',
                'config': {'thrust_curve_scale': 0.7}  # 30% thrust reduction
            },
            {
                'name': 'Gusty Conditions',
                'config': {'wind_gust_probability': 0.5, 'wind_gust_magnitude': 8.0}
            }
        ]
        
        stress_results = []
        
        for scenario in stress_scenarios:
            logger.info(f"Testing scenario: {scenario['name']}")
            
            # Run multiple trials per scenario
            scenario_trials = []
            for _ in range(20):  # 20 trials per stress scenario
                try:
                    metrics = self.evaluate_single_flight(
                        scenario_config=scenario['config'],
                        record_trajectory=False
                    )
                    scenario_trials.append(metrics)
                except Exception as e:
                    logger.warning(f"Stress test trial failed: {e}")
            
            if scenario_trials:
                # Aggregate scenario results
                scenario_summary = {
                    'scenario': scenario['name'],
                    'config': scenario['config'],
                    'num_trials': len(scenario_trials),
                    'success_rate': np.mean([t['mission_success'] for t in scenario_trials]),
                    'mean_reward': np.mean([t['total_reward'] for t in scenario_trials]),
                    'mean_max_altitude': np.mean([t['max_altitude'] for t in scenario_trials]),
                    'mean_attitude_error': np.mean([t['mean_attitude_error'] for t in scenario_trials]),
                    'control_stability': np.mean([t['control_stability'] for t in scenario_trials])
                }
                stress_results.append(scenario_summary)
        
        # Save stress test results
        stress_df = pd.DataFrame(stress_results)
        stress_df.to_csv("results/csv_logs/stress_test_results.csv", index=False)
        
        return {'stress_scenarios': stress_results}
    
    def _apply_scenario_config(self, config: Dict[str, Any]):
        """Apply scenario-specific configuration to environment."""
        for key, value in config.items():
            if key in self.env.config['rocket_params']:
                self.env.config['rocket_params'][key] = value
            elif key in self.env.config['world_params']:
                self.env.config['world_params'][key] = value
            elif key in self.env.config['engine_params']:
                self.env.config['engine_params'][key] = value
    
    def _generate_random_scenario(self, parameter_variations: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
        """Generate random scenario configuration within specified ranges."""
        scenario = {}
        
        for param, (min_val, max_val) in parameter_variations.items():
            scenario[param] = np.random.uniform(min_val, max_val)
        
        return scenario
    
    def _calculate_performance_metrics(self, trajectory: List[Dict], final_info: Dict) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics from trajectory data."""
        if not trajectory:
            return {}
        
        df = pd.DataFrame(trajectory)
        
        # Basic flight metrics
        max_altitude = df['altitude'].max()
        flight_time = df['time'].iloc[-1]
        final_altitude = df['altitude'].iloc[-1]
        
        # Attitude control performance
        attitude_errors = np.sqrt(df['roll']**2 + df['pitch']**2 + df['yaw']**2)
        mean_attitude_error = attitude_errors.mean()
        max_attitude_error = attitude_errors.max()
        
        # Angular rate analysis
        angular_rates = np.sqrt(df['angular_vel_x']**2 + df['angular_vel_y']**2 + df['angular_vel_z']**2)
        mean_angular_rate = angular_rates.mean()
        max_angular_rate = angular_rates.max()
        
        # Control effort analysis
        control_effort = np.sqrt(df['gimbal_x']**2 + df['gimbal_y']**2)
        mean_control_effort = control_effort.mean()
        
        # Fuel efficiency
        fuel_efficiency = df['fuel_remaining'].iloc[-1]
        
        # Stability metrics
        altitude_stability = 1.0 / (1.0 + np.std(df['altitude'].diff().fillna(0)))
        attitude_stability = 1.0 / (1.0 + np.std(attitude_errors))
        control_stability = 1.0 / (1.0 + np.std(control_effort))
        
        # Mission success criteria
        mission_success = (
            max_altitude > 50 and  # Minimum altitude achieved
            mean_attitude_error < 10 and  # Attitude control quality
            max_attitude_error < 45 and  # No catastrophic attitude loss
            fuel_efficiency > 0.1  # Some fuel remaining or efficient use
        )
        
        metrics = {
            # Basic flight performance
            'total_reward': df['reward'].sum(),
            'flight_time': flight_time,
            'max_altitude': max_altitude,
            'final_altitude': final_altitude,
            
            # Attitude control
            'mean_attitude_error': mean_attitude_error,
            'max_attitude_error': max_attitude_error,
            'attitude_stability': attitude_stability,
            
            # Angular rate control
            'mean_angular_rate': mean_angular_rate,
            'max_angular_rate': max_angular_rate,
            
            # Control system performance
            'mean_control_effort': mean_control_effort,
            'control_stability': control_stability,
            
            # Fuel efficiency
            'fuel_efficiency': fuel_efficiency,
            
            # Overall stability
            'altitude_stability': altitude_stability,
            
            # Mission success
            'mission_success': mission_success
        }
        
        return metrics
    
    def _analyze_monte_carlo_results(self, trial_results: List[Dict]) -> Dict[str, Any]:
        """Analyze Monte Carlo trial results statistically."""
        if not trial_results:
            return {}
        
        df = pd.DataFrame(trial_results)
        
        # Success rate analysis
        success_rate = df['mission_success'].mean()
        
        # Performance statistics
        performance_stats = {}
        key_metrics = ['total_reward', 'max_altitude', 'mean_attitude_error', 
                      'control_stability', 'fuel_efficiency']
        
        for metric in key_metrics:
            if metric in df.columns:
                performance_stats[metric] = {
                    'mean': df[metric].mean(),
                    'std': df[metric].std(),
                    'min': df[metric].min(),
                    'max': df[metric].max(),
                    'q25': df[metric].quantile(0.25),
                    'q50': df[metric].quantile(0.50),
                    'q75': df[metric].quantile(0.75)
                }
        
        # Failure analysis
        failures = df[~df['mission_success']]
        failure_modes = {}
        
        if len(failures) > 0:
            failure_modes = {
                'low_altitude_failures': len(failures[failures['max_altitude'] < 50]),
                'attitude_control_failures': len(failures[failures['max_attitude_error'] > 45]),
                'fuel_efficiency_failures': len(failures[failures['fuel_efficiency'] < 0.1])
            }
        
        analysis = {
            'num_trials': len(trial_results),
            'success_rate': success_rate,
            'performance_statistics': performance_stats,
            'failure_analysis': failure_modes
        }
        
        return analysis
    
    def _save_monte_carlo_results(self, trial_results: List[Dict], analysis: Dict[str, Any]):
        """Save Monte Carlo results to files."""
        # Save individual trial results
        if trial_results:
            trial_df = pd.DataFrame(trial_results)
            trial_df.to_csv("results/csv_logs/monte_carlo_trials.csv", index=False)
        
        # Save analysis summary
        with open("results/csv_logs/monte_carlo_analysis.yaml", 'w') as f:
            yaml.dump(analysis, f, default_flow_style=False)
        
        logger.info("Monte Carlo results saved to CSV and YAML files")
    
    def visualize_flight_trajectory(self, trajectory_index: int = -1, save_path: str = None):
        """
        Create comprehensive visualization of a flight trajectory.
        
        Args:
            trajectory_index: Index of trajectory to visualize (-1 for latest)
            save_path: Path to save the visualization
        """
        if not self.flight_trajectories:
            logger.warning("No trajectories available for visualization")
            return
        
        trajectory = self.flight_trajectories[trajectory_index]
        df = pd.DataFrame(trajectory)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        
        # 3D trajectory plot
        ax1 = fig.add_subplot(3, 4, 1, projection='3d')
        ax1.plot(df['position_x'], df['position_y'], df['position_z'], 'b-', linewidth=2)
        ax1.scatter(df['position_x'].iloc[0], df['position_y'].iloc[0], df['position_z'].iloc[0], 
                   color='green', s=100, label='Start')
        ax1.scatter(df['position_x'].iloc[-1], df['position_y'].iloc[-1], df['position_z'].iloc[-1], 
                   color='red', s=100, label='End')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Altitude (m)')
        ax1.set_title('3D Flight Trajectory')
        ax1.legend()
        
        # Altitude vs time
        ax2 = fig.add_subplot(3, 4, 2)
        ax2.plot(df['time'], df['altitude'], 'b-', linewidth=2)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Altitude (m)')
        ax2.set_title('Altitude Profile')
        ax2.grid(True)
        
        # Attitude angles
        ax3 = fig.add_subplot(3, 4, 3)
        ax3.plot(df['time'], df['roll'], label='Roll', linewidth=2)
        ax3.plot(df['time'], df['pitch'], label='Pitch', linewidth=2)
        ax3.plot(df['time'], df['yaw'], label='Yaw', linewidth=2)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Angle (deg)')
        ax3.set_title('Attitude Angles')
        ax3.legend()
        ax3.grid(True)
        
        # Velocities
        ax4 = fig.add_subplot(3, 4, 4)
        ax4.plot(df['time'], df['velocity_x'], label='Vx', linewidth=2)
        ax4.plot(df['time'], df['velocity_y'], label='Vy', linewidth=2)
        ax4.plot(df['time'], df['velocity_z'], label='Vz', linewidth=2)
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Velocity (m/s)')
        ax4.set_title('Velocity Components')
        ax4.legend()
        ax4.grid(True)
        
        # Angular rates
        ax5 = fig.add_subplot(3, 4, 5)
        ax5.plot(df['time'], df['angular_vel_x'], label='ωx', linewidth=2)
        ax5.plot(df['time'], df['angular_vel_y'], label='ωy', linewidth=2)
        ax5.plot(df['time'], df['angular_vel_z'], label='ωz', linewidth=2)
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Angular Rate (deg/s)')
        ax5.set_title('Angular Rates')
        ax5.legend()
        ax5.grid(True)
        
        # Control commands
        ax6 = fig.add_subplot(3, 4, 6)
        ax6.plot(df['time'], df['gimbal_x'], label='Gimbal X', linewidth=2)
        ax6.plot(df['time'], df['gimbal_y'], label='Gimbal Y', linewidth=2)
        ax6.set_xlabel('Time (s)')
        ax6.set_ylabel('Gimbal Angle (deg)')
        ax6.set_title('TVC Commands')
        ax6.legend()
        ax6.grid(True)
        
        # Thrust and mass
        ax7 = fig.add_subplot(3, 4, 7)
        ax7_twin = ax7.twinx()
        ax7.plot(df['time'], df['thrust'], 'r-', linewidth=2, label='Thrust')
        ax7_twin.plot(df['time'], df['mass'], 'b-', linewidth=2, label='Mass')
        ax7.set_xlabel('Time (s)')
        ax7.set_ylabel('Thrust (N)', color='r')
        ax7_twin.set_ylabel('Mass (kg)', color='b')
        ax7.set_title('Thrust and Mass')
        ax7.grid(True)
        
        # Fuel remaining
        ax8 = fig.add_subplot(3, 4, 8)
        ax8.plot(df['time'], df['fuel_remaining'], 'g-', linewidth=2)
        ax8.set_xlabel('Time (s)')
        ax8.set_ylabel('Fuel Remaining Fraction')
        ax8.set_title('Fuel Consumption')
        ax8.grid(True)
        
        # Reward progression
        ax9 = fig.add_subplot(3, 4, 9)
        ax9.plot(df['time'], df['reward'], 'purple', linewidth=2)
        ax9.set_xlabel('Time (s)')
        ax9.set_ylabel('Instantaneous Reward')
        ax9.set_title('Reward Signal')
        ax9.grid(True)
        
        # Attitude error
        ax10 = fig.add_subplot(3, 4, 10)
        ax10.plot(df['time'], df['attitude_error'], 'orange', linewidth=2)
        ax10.set_xlabel('Time (s)')
        ax10.set_ylabel('Attitude Error (deg)')
        ax10.set_title('Attitude Control Error')
        ax10.grid(True)
        
        # 2D trajectory (top view)
        ax11 = fig.add_subplot(3, 4, 11)
        ax11.plot(df['position_x'], df['position_y'], 'b-', linewidth=2)
        ax11.scatter(df['position_x'].iloc[0], df['position_y'].iloc[0], 
                    color='green', s=100, label='Start')
        ax11.scatter(df['position_x'].iloc[-1], df['position_y'].iloc[-1], 
                    color='red', s=100, label='End')
        ax11.set_xlabel('X Position (m)')
        ax11.set_ylabel('Y Position (m)')
        ax11.set_title('Ground Track (Top View)')
        ax11.legend()
        ax11.grid(True)
        ax11.axis('equal')
        
        # Performance summary (text)
        ax12 = fig.add_subplot(3, 4, 12)
        ax12.axis('off')
        
        # Calculate summary metrics
        max_alt = df['altitude'].max()
        flight_time = df['time'].iloc[-1]
        final_fuel = df['fuel_remaining'].iloc[-1]
        mean_attitude_error = df['attitude_error'].mean()
        
        summary_text = f"""
        FLIGHT PERFORMANCE SUMMARY
        
        Max Altitude: {max_alt:.1f} m
        Flight Time: {flight_time:.2f} s
        Final Fuel: {final_fuel:.1%}
        Mean Attitude Error: {mean_attitude_error:.2f}°
        
        Final Position:
        X: {df['position_x'].iloc[-1]:.1f} m
        Y: {df['position_y'].iloc[-1]:.1f} m
        Z: {df['position_z'].iloc[-1]:.1f} m
        
        Total Reward: {df['reward'].sum():.1f}
        """
        
        ax12.text(0.1, 0.9, summary_text, transform=ax12.transAxes, 
                 fontsize=12, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Flight trajectory visualization saved to {save_path}")
        
        plt.show()
    
    def generate_performance_report(self, output_path: str = "results/performance_report.html"):
        """Generate comprehensive HTML performance report."""
        # This would create a detailed HTML report with all evaluation results
        # For now, we'll create a simplified version
        
        report_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Rocket TVC Performance Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #2c3e50; color: white; padding: 20px; }}
                .section {{ margin: 20px 0; padding: 15px; border-left: 4px solid #3498db; }}
                .metric {{ background-color: #ecf0f1; padding: 10px; margin: 5px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Intelligent Ascent TVC System</h1>
                <h2>Performance Evaluation Report</h2>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h3>Model Information</h3>
                <div class="metric">Model Path: {self.model_path}</div>
                <div class="metric">Environment Config: {self.config_path}</div>
                <div class="metric">State Dimension: {self.state_dim}</div>
                <div class="metric">Action Dimension: {self.action_dim}</div>
            </div>
            
            <div class="section">
                <h3>Evaluation Summary</h3>
                <p>This report contains detailed performance analysis of the trained TD3 model.</p>
                <p>Check the results/csv_logs/ directory for detailed numerical data.</p>
                <p>Check the results/plots/ directory for visualization charts.</p>
            </div>
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Performance report generated: {output_path}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate trained TD3 TVC model')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model file')
    parser.add_argument('--config', type=str, default='configs/environment.yaml',
                       help='Path to environment configuration')
    parser.add_argument('--single', action='store_true',
                       help='Run single flight evaluation with visualization')
    parser.add_argument('--monte-carlo', type=int, default=0,
                       help='Number of Monte Carlo trials (0 to skip)')
    parser.add_argument('--stress-test', action='store_true',
                       help='Run stress testing scenarios')
    parser.add_argument('--render', action='store_true',
                       help='Render flight during evaluation')
    
    args = parser.parse_args()
    
    try:
        # Initialize evaluator
        evaluator = ModelEvaluator(args.model, args.config)
        
        # Create results directory
        os.makedirs("results/csv_logs", exist_ok=True)
        os.makedirs("results/plots", exist_ok=True)
        
        # Single flight evaluation
        if args.single:
            logger.info("Running single flight evaluation...")
            metrics = evaluator.evaluate_single_flight(render=args.render)
            
            print("\n" + "="*50)
            print("SINGLE FLIGHT EVALUATION RESULTS")
            print("="*50)
            for key, value in metrics.items():
                if isinstance(value, float):
                    print(f"{key}: {value:.3f}")
                else:
                    print(f"{key}: {value}")
            
            # Visualize trajectory
            if evaluator.flight_trajectories:
                evaluator.visualize_flight_trajectory(
                    save_path="results/plots/single_flight_trajectory.png"
                )
        
        # Monte Carlo evaluation
        if args.monte_carlo > 0:
            logger.info(f"Running Monte Carlo evaluation with {args.monte_carlo} trials...")
            mc_results = evaluator.monte_carlo_evaluation(args.monte_carlo)
            
            print("\n" + "="*50)
            print("MONTE CARLO EVALUATION RESULTS")
            print("="*50)
            print(f"Success Rate: {mc_results['success_rate']:.2%}")
            print(f"Number of Trials: {mc_results['num_trials']}")
            
            # Print key performance statistics
            if 'performance_statistics' in mc_results:
                print("\nPerformance Statistics:")
                for metric, stats in mc_results['performance_statistics'].items():
                    print(f"  {metric}:")
                    print(f"    Mean: {stats['mean']:.3f} ± {stats['std']:.3f}")
                    print(f"    Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
        
        # Stress testing
        if args.stress_test:
            logger.info("Running stress test evaluation...")
            stress_results = evaluator.stress_test_evaluation()
            
            print("\n" + "="*50)
            print("STRESS TEST RESULTS")
            print("="*50)
            for scenario in stress_results['stress_scenarios']:
                print(f"{scenario['scenario']}: {scenario['success_rate']:.2%} success rate")
        
        # Generate performance report
        evaluator.generate_performance_report()
        
        logger.info("Evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    from datetime import datetime
    main()
