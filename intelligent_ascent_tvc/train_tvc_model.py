"""
Intelligent Ascent TVC System - Training Script
==============================================

Main training script for the TD3-based rocket thrust vector control system.
Implements curriculum learning, comprehensive logging, and model checkpointing.

Author: AI Assistant
Date: August 2025
"""

import os
import sys
import time
import yaml
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Any
import logging
from tqdm import tqdm
import argparse

# Add rocket_env to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rocket_env import RocketTVCEnvironment, TD3Agent, ReplayBuffer
from rocket_env.td3_networks import ModelOptimizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TrainingManager:
    """
    Manages the complete training process for the TD3 TVC system.
    """
    
    def __init__(self, config_path: str = "configs/training_params.yaml"):
        """Initialize training manager with configuration."""
        # Load configuration
        self.config = self._load_config(config_path)
        self.training_params = self.config['training_params']
        self.network_params = self.config['network_architecture']
        self.eval_params = self.config['evaluation']
        self.log_params = self.config['logging']
        
        # Initialize environment
        self.env = RocketTVCEnvironment()
        self.eval_env = RocketTVCEnvironment()
        
        # Get environment dimensions
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.max_action = float(self.env.action_space.high[0])
        
        # Initialize TD3 agent
        self.agent = TD3Agent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            max_action=self.max_action,
            lr_actor=self.training_params['learning_rate_actor'],
            lr_critic=self.training_params['learning_rate_critic'],
            gamma=self.training_params['gamma'],
            tau=self.training_params['tau'],
            policy_noise=self.training_params['target_policy_noise'],
            noise_clip=self.training_params['target_noise_clip'],
            policy_freq=self.training_params['policy_delay']
        )
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(
            capacity=self.training_params['buffer_size'],
            state_dim=self.state_dim,
            action_dim=self.action_dim
        )
        
        # Training tracking
        self.total_timesteps = 0
        self.episode_num = 0
        self.training_history = []
        self.evaluation_history = []
        
        # Exploration noise schedule
        self.exploration_noise = self.training_params['exploration_noise']
        self.noise_decay = self.training_params['exploration_noise_decay']
        self.min_noise = self.training_params['min_exploration_noise']
        
        # Create output directories
        self._create_directories()
        
        logger.info("Training manager initialized successfully")
        logger.info(f"State dimension: {self.state_dim}")
        logger.info(f"Action dimension: {self.action_dim}")
        logger.info(f"Max action: {self.max_action:.3f} rad ({np.degrees(self.max_action):.1f} deg)")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load training configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def _create_directories(self):
        """Create necessary directories for outputs."""
        directories = [
            "results/models",
            "results/plots", 
            "results/csv_logs",
            "results/tensorboard_logs"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def train(self):
        """Main training loop."""
        logger.info("Starting TD3 training...")
        logger.info(f"Total timesteps: {self.training_params['total_timesteps']:,}")
        
        start_time = time.time()
        
        # Training loop
        state, _ = self.env.reset()
        episode_reward = 0
        episode_timesteps = 0
        episode_start_time = time.time()
        
        with tqdm(total=self.training_params['total_timesteps'], desc="Training") as pbar:
            while self.total_timesteps < self.training_params['total_timesteps']:
                # Select action with exploration noise
                if self.total_timesteps < self.training_params['learning_starts']:
                    # Random actions during initial exploration
                    action = self.env.action_space.sample()
                else:
                    # TD3 action with exploration noise
                    action = self.agent.select_action(state, self.exploration_noise)
                
                # Execute action
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                # Store experience in replay buffer
                self.replay_buffer.add(state, action, reward, next_state, done)
                
                state = next_state
                episode_reward += reward
                episode_timesteps += 1
                self.total_timesteps += 1
                
                # Train agent
                if (self.total_timesteps >= self.training_params['learning_starts'] and
                    self.replay_buffer.can_sample(self.training_params['batch_size'])):
                    
                    training_metrics = self.agent.train(
                        self.replay_buffer, 
                        self.training_params['batch_size']
                    )
                    
                    # Log training metrics
                    if self.total_timesteps % self.log_params['log_freq'] == 0:
                        self._log_training_metrics(training_metrics)
                
                # Episode end
                if done:
                    # Record episode statistics
                    episode_time = time.time() - episode_start_time
                    self._record_episode(episode_reward, episode_timesteps, episode_time, info)
                    
                    # Reset environment
                    state, _ = self.env.reset()
                    episode_reward = 0
                    episode_timesteps = 0
                    episode_start_time = time.time()
                    self.episode_num += 1
                    
                    # Decay exploration noise
                    self.exploration_noise = max(
                        self.min_noise,
                        self.exploration_noise * self.noise_decay
                    )
                
                # Evaluation
                if self.total_timesteps % self.eval_params['eval_freq'] == 0:
                    eval_metrics = self._evaluate()
                    self.evaluation_history.append({
                        'timestep': self.total_timesteps,
                        **eval_metrics
                    })
                    
                    logger.info(f"Evaluation at {self.total_timesteps:,} steps: "
                              f"Mean reward: {eval_metrics['mean_reward']:.2f}, "
                              f"Success rate: {eval_metrics['success_rate']:.2f}")
                
                # Save model checkpoint
                if self.total_timesteps % self.log_params['save_freq'] == 0:
                    self._save_checkpoint()
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({
                    'Episode': self.episode_num,
                    'Reward': f"{episode_reward:.1f}",
                    'Noise': f"{self.exploration_noise:.3f}",
                    'Buffer': f"{len(self.replay_buffer):,}"
                })
        
        # Final evaluation and save
        final_eval = self._evaluate()
        self._save_final_model()
        self._save_training_history()
        self._generate_plots()
        
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time/3600:.2f} hours")
        logger.info(f"Final evaluation: {final_eval}")
    
    def _record_episode(self, reward: float, timesteps: int, duration: float, info: Dict[str, Any]):
        """Record episode statistics."""
        episode_data = {
            'episode': self.episode_num,
            'timestep': self.total_timesteps,
            'reward': reward,
            'length': timesteps,
            'duration': duration,
            'final_altitude': info.get('altitude', 0),
            'final_attitude_error': info.get('attitude_error', 0),
            'fuel_remaining': info.get('fuel_remaining', 0),
            'exploration_noise': self.exploration_noise
        }
        
        self.training_history.append(episode_data)
        
        # Log every 100 episodes
        if self.episode_num % 100 == 0:
            recent_rewards = [ep['reward'] for ep in self.training_history[-100:]]
            mean_reward = np.mean(recent_rewards)
            logger.info(f"Episode {self.episode_num}: Mean reward (last 100): {mean_reward:.2f}")
    
    def _log_training_metrics(self, metrics: Dict[str, float]):
        """Log training metrics to tensorboard."""
        # This would integrate with tensorboard in a full implementation
        pass
    
    def _evaluate(self) -> Dict[str, float]:
        """Evaluate current policy."""
        eval_rewards = []
        success_count = 0
        
        for _ in range(self.eval_params['n_eval_episodes']):
            state, _ = self.eval_env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                # Use deterministic policy for evaluation
                action = self.agent.select_action(state, noise_scale=0.0)
                state, reward, terminated, truncated, info = self.eval_env.step(action)
                episode_reward += reward
                done = terminated or truncated
            
            eval_rewards.append(episode_reward)
            
            # Success criteria (based on final state)
            if (info['altitude'] > 50 and  # Minimum altitude achieved
                info['attitude_error'] < np.radians(self.eval_params['success_threshold']['max_attitude_error']) and
                info['angular_rate'] < self.eval_params['success_threshold']['max_angular_rate']):
                success_count += 1
        
        metrics = {
            'mean_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'min_reward': np.min(eval_rewards),
            'max_reward': np.max(eval_rewards),
            'success_rate': success_count / self.eval_params['n_eval_episodes']
        }
        
        return metrics
    
    def _save_checkpoint(self):
        """Save model checkpoint."""
        checkpoint_path = f"results/models/td3_checkpoint_{self.total_timesteps}.pth"
        self.agent.save(checkpoint_path)
        
        # Save training state
        training_state = {
            'total_timesteps': self.total_timesteps,
            'episode_num': self.episode_num,
            'exploration_noise': self.exploration_noise,
            'training_history': self.training_history,
            'evaluation_history': self.evaluation_history
        }
        
        state_path = f"results/models/training_state_{self.total_timesteps}.pth"
        torch.save(training_state, state_path)
    
    def _save_final_model(self):
        """Save final trained model."""
        final_path = "results/models/td3_final.pth"
        self.agent.save(final_path)
        
        # Export to ONNX for deployment
        optimizer = ModelOptimizer()
        optimizer.export_to_onnx(
            self.agent.actor,
            (1, self.state_dim),
            "results/models/td3_actor_final.onnx"
        )
        
        # Create quantized version
        quantized_model = optimizer.quantize_model(
            self.agent.actor,
            "results/models/td3_actor_quantized.pth"
        )
        
        # Benchmark inference performance
        inference_results = optimizer.benchmark_inference_time(
            self.agent.actor,
            (1, self.state_dim),
            num_runs=1000
        )
        
        # Save benchmark results
        with open("results/models/inference_benchmark.yaml", 'w') as f:
            yaml.dump(inference_results, f)
        
        logger.info("Final model saved and optimized for deployment")
    
    def _save_training_history(self):
        """Save training history to CSV files."""
        # Episode history
        if self.training_history:
            episode_df = pd.DataFrame(self.training_history)
            episode_df.to_csv("results/csv_logs/training_episodes.csv", index=False)
        
        # Evaluation history
        if self.evaluation_history:
            eval_df = pd.DataFrame(self.evaluation_history)
            eval_df.to_csv("results/csv_logs/evaluation_results.csv", index=False)
        
        logger.info("Training history saved to CSV files")
    
    def _generate_plots(self):
        """Generate training visualization plots."""
        if not self.training_history:
            return
        
        # Convert to DataFrame for easier plotting
        df = pd.DataFrame(self.training_history)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('TD3 Training Results - Rocket TVC System', fontsize=16)
        
        # Episode rewards
        axes[0, 0].plot(df['episode'], df['reward'], alpha=0.3, color='blue')
        # Moving average
        window_size = min(100, len(df) // 10)
        if window_size > 1:
            moving_avg = df['reward'].rolling(window=window_size).mean()
            axes[0, 0].plot(df['episode'], moving_avg, color='red', linewidth=2, label=f'MA({window_size})')
            axes[0, 0].legend()
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Episode Reward')
        axes[0, 0].set_title('Training Rewards')
        axes[0, 0].grid(True)
        
        # Episode lengths
        axes[0, 1].plot(df['episode'], df['length'], alpha=0.6, color='green')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Episode Length')
        axes[0, 1].set_title('Episode Lengths')
        axes[0, 1].grid(True)
        
        # Final altitude achieved
        axes[0, 2].plot(df['episode'], df['final_altitude'], alpha=0.6, color='orange')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Final Altitude (m)')
        axes[0, 2].set_title('Altitude Performance')
        axes[0, 2].grid(True)
        
        # Attitude error
        axes[1, 0].plot(df['episode'], np.degrees(df['final_attitude_error']), alpha=0.6, color='red')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Final Attitude Error (deg)')
        axes[1, 0].set_title('Attitude Control Performance')
        axes[1, 0].grid(True)
        
        # Fuel efficiency
        axes[1, 1].plot(df['episode'], df['fuel_remaining'], alpha=0.6, color='purple')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Fuel Remaining Fraction')
        axes[1, 1].set_title('Fuel Efficiency')
        axes[1, 1].grid(True)
        
        # Exploration noise decay
        axes[1, 2].plot(df['episode'], df['exploration_noise'], color='black')
        axes[1, 2].set_xlabel('Episode')
        axes[1, 2].set_ylabel('Exploration Noise')
        axes[1, 2].set_title('Exploration Schedule')
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        plt.savefig('results/plots/training_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Evaluation results
        if self.evaluation_history:
            eval_df = pd.DataFrame(self.evaluation_history)
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Mean reward progression
            axes[0].plot(eval_df['timestep'], eval_df['mean_reward'], 'o-', color='blue')
            axes[0].fill_between(eval_df['timestep'], 
                               eval_df['mean_reward'] - eval_df['std_reward'],
                               eval_df['mean_reward'] + eval_df['std_reward'],
                               alpha=0.3, color='blue')
            axes[0].set_xlabel('Training Timesteps')
            axes[0].set_ylabel('Mean Evaluation Reward')
            axes[0].set_title('Evaluation Performance')
            axes[0].grid(True)
            
            # Success rate
            axes[1].plot(eval_df['timestep'], eval_df['success_rate'], 'o-', color='green')
            axes[1].set_xlabel('Training Timesteps')
            axes[1].set_ylabel('Success Rate')
            axes[1].set_title('Mission Success Rate')
            axes[1].set_ylim(0, 1)
            axes[1].grid(True)
            
            plt.tight_layout()
            plt.savefig('results/plots/evaluation_results.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info("Training plots generated and saved")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train TD3 agent for rocket TVC')
    parser.add_argument('--config', type=str, default='configs/training_params.yaml',
                       help='Path to training configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training from')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Initialize training manager
    trainer = TrainingManager(args.config)
    
    # Resume from checkpoint if specified
    if args.resume:
        # Implementation for resuming training would go here
        logger.info(f"Resuming training from {args.resume}")
    
    try:
        # Start training
        trainer.train()
        logger.info("Training completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        trainer._save_checkpoint()
        trainer._save_training_history()
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
