"""
Intelligent Ascent TVC System - TD3 Network Architecture
=======================================================

This module implements the Twin Delayed Deep Deterministic Policy Gradient (TD3)
networks for continuous control of rocket thrust vector control systems.

Author: AI Assistant
Date: August 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from typing import Tuple, Dict, Any
import onnx
import onnxruntime as ort
from torch.quantization import quantize_dynamic


class ActorNetwork(nn.Module):
    """
    Actor network for TD3 algorithm - outputs continuous TVC commands.
    
    Architecture:
    - Input: State vector (21 dimensions)
    - Hidden layers: 256 -> 128 -> 64 neurons with ReLU activation
    - Output: Gimbal angles [pitch, yaw] with tanh activation
    """
    
    def __init__(self, state_dim: int, action_dim: int, max_action: float, hidden_dims: list = [256, 128, 64]):
        super(ActorNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        
        # Build network layers
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim) if len(hidden_dims) > 1 else nn.Identity(),
            ])
            input_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(input_dim, action_dim))
        layers.append(nn.Tanh())
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights using Xavier uniform
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights using Xavier uniform initialization."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through actor network."""
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        
        action = self.network(state)
        return self.max_action * action
    
    def get_action(self, state: np.ndarray, noise_scale: float = 0.0) -> np.ndarray:
        """Get action from state with optional exploration noise."""
        # Set to evaluation mode for inference
        was_training = self.training
        self.eval()
        
        try:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            # Move tensor to the same device as the model
            device = next(self.parameters()).device
            state_tensor = state_tensor.to(device)
            
            with torch.no_grad():
                action = self.forward(state_tensor).detach().cpu().numpy()[0]
            
            if noise_scale > 0:
                noise = np.random.normal(0, noise_scale, size=action.shape)
                action = np.clip(action + noise, -self.max_action, self.max_action)
            
            return action
        finally:
            # Restore original training mode
            self.train(was_training)


class CriticNetwork(nn.Module):
    """
    Critic network for TD3 algorithm - estimates Q-values for state-action pairs.
    
    Architecture:
    - Input: State vector (21 dim) + Action vector (2 dim) = 23 dimensions
    - Hidden layers: 256 -> 128 -> 64 neurons with ReLU activation
    - Output: Single Q-value
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list = [256, 128, 64]):
        super(CriticNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Build network layers
        layers = []
        input_dim = state_dim + action_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim) if len(hidden_dims) > 1 else nn.Identity(),
            ])
            input_dim = hidden_dim
        
        # Output layer (single Q-value)
        layers.append(nn.Linear(input_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights using Xavier uniform initialization."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass through critic network."""
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        if len(action.shape) == 1:
            action = action.unsqueeze(0)
        
        state_action = torch.cat([state, action], dim=1)
        q_value = self.network(state_action)
        return q_value


class TD3Agent:
    """
    Twin Delayed Deep Deterministic Policy Gradient (TD3) Agent.
    
    Implements the three key innovations of TD3:
    1. Twin Critics (Clipped Double Q-Learning)
    2. Delayed Policy Updates
    3. Target Policy Smoothing
    """
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 max_action: float,
                 lr_actor: float = 1e-4,
                 lr_critic: float = 1e-3,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 policy_noise: float = 0.2,
                 noise_clip: float = 0.5,
                 policy_freq: int = 2,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        
        # Initialize networks
        self.actor = ActorNetwork(state_dim, action_dim, max_action).to(device)
        self.actor_target = ActorNetwork(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        # Twin critics
        self.critic_1 = CriticNetwork(state_dim, action_dim).to(device)
        self.critic_2 = CriticNetwork(state_dim, action_dim).to(device)
        self.critic_target_1 = CriticNetwork(state_dim, action_dim).to(device)
        self.critic_target_2 = CriticNetwork(state_dim, action_dim).to(device)
        
        # Copy parameters to target networks
        self.critic_target_1.load_state_dict(self.critic_1.state_dict())
        self.critic_target_2.load_state_dict(self.critic_2.state_dict())
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer_1 = torch.optim.Adam(self.critic_1.parameters(), lr=lr_critic)
        self.critic_optimizer_2 = torch.optim.Adam(self.critic_2.parameters(), lr=lr_critic)
        
        # Training step counter
        self.total_it = 0
        
        print(f"TD3 Agent initialized on device: {device}")
        print(f"Actor parameters: {sum(p.numel() for p in self.actor.parameters()):,}")
        print(f"Critic parameters: {sum(p.numel() for p in self.critic_1.parameters()):,}")
    
    def select_action(self, state: np.ndarray, noise_scale: float = 0.0) -> np.ndarray:
        """Select action using current policy with optional exploration noise."""
        return self.actor.get_action(state, noise_scale)
    
    def train(self, replay_buffer, batch_size: int = 256) -> Dict[str, float]:
        """Train the TD3 agent on a batch of experiences."""
        self.total_it += 1
        
        # Sample batch from replay buffer
        batch = replay_buffer.sample(batch_size)
        state = torch.FloatTensor(batch['state']).to(self.device)
        action = torch.FloatTensor(batch['action']).to(self.device)
        next_state = torch.FloatTensor(batch['next_state']).to(self.device)
        reward = torch.FloatTensor(batch['reward']).to(self.device)
        done = torch.BoolTensor(batch['done']).to(self.device)
        
        # Compute target Q-values
        with torch.no_grad():
            # Target policy smoothing: add clipped noise to target actions
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip)
            
            next_action = (self.actor_target(next_state) + noise).clamp(
                -self.max_action, self.max_action)
            
            # Clipped Double Q-Learning: take minimum of twin target Q-values
            target_q1 = self.critic_target_1(next_state, next_action)
            target_q2 = self.critic_target_2(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            
            # Compute target value
            target_q = reward + (1 - done.float()) * self.gamma * target_q
        
        # Update critics
        current_q1 = self.critic_1(state, action)
        current_q2 = self.critic_2(state, action)
        
        critic_loss_1 = F.mse_loss(current_q1, target_q)
        critic_loss_2 = F.mse_loss(current_q2, target_q)
        
        self.critic_optimizer_1.zero_grad()
        critic_loss_1.backward()
        self.critic_optimizer_1.step()
        
        self.critic_optimizer_2.zero_grad()
        critic_loss_2.backward()
        self.critic_optimizer_2.step()
        
        # Delayed policy updates
        actor_loss = torch.tensor(0.0)
        if self.total_it % self.policy_freq == 0:
            # Update actor
            actor_action = self.actor(state)
            actor_loss = -self.critic_1(state, actor_action).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Soft update target networks
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic_1, self.critic_target_1)
            self._soft_update(self.critic_2, self.critic_target_2)
        
        # Return training metrics
        return {
            'critic_loss_1': critic_loss_1.item(),
            'critic_loss_2': critic_loss_2.item(),
            'actor_loss': actor_loss.item(),
            'q1_mean': current_q1.mean().item(),
            'q2_mean': current_q2.mean().item(),
            'target_q_mean': target_q.mean().item()
        }
    
    def _soft_update(self, source: nn.Module, target: nn.Module):
        """Soft update target network parameters."""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1 - self.tau) * target_param.data
            )
    
    def save(self, filepath: str):
        """Save agent state to file."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_1_state_dict': self.critic_1.state_dict(),
            'critic_2_state_dict': self.critic_2.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_1_optimizer_state_dict': self.critic_optimizer_1.state_dict(),
            'critic_2_optimizer_state_dict': self.critic_optimizer_2.state_dict(),
            'total_it': self.total_it,
            'hyperparameters': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'max_action': self.max_action,
                'gamma': self.gamma,
                'tau': self.tau,
                'policy_noise': self.policy_noise,
                'noise_clip': self.noise_clip,
                'policy_freq': self.policy_freq
            }
        }, filepath)
        print(f"Agent saved to {filepath}")
    
    def load(self, filepath: str):
        """Load agent state from file."""
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device)
            
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic_1.load_state_dict(checkpoint['critic_1_state_dict'])
            self.critic_2.load_state_dict(checkpoint['critic_2_state_dict'])
            
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            self.critic_optimizer_1.load_state_dict(checkpoint['critic_1_optimizer_state_dict'])
            self.critic_optimizer_2.load_state_dict(checkpoint['critic_2_optimizer_state_dict'])
            
            self.total_it = checkpoint['total_it']
            
            print(f"Agent loaded from {filepath}")
            return True
        else:
            print(f"No checkpoint found at {filepath}")
            return False


class ModelOptimizer:
    """
    Model optimization utilities for deployment on microcontrollers.
    """
    
    @staticmethod
    def export_to_onnx(model: nn.Module, 
                      input_shape: Tuple[int, ...], 
                      output_path: str,
                      opset_version: int = 11):
        """Export PyTorch model to ONNX format."""
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(*input_shape)
        
        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        
        print(f"Model exported to ONNX: {output_path}")
        
        # Verify ONNX model
        try:
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            print("ONNX model verification successful")
        except Exception as e:
            print(f"ONNX model verification failed: {e}")
    
    @staticmethod
    def quantize_model(model: nn.Module, output_path: str = None) -> nn.Module:
        """Apply dynamic quantization to reduce model size."""
        quantized_model = quantize_dynamic(
            model,
            {nn.Linear, nn.Conv2d},
            dtype=torch.qint8
        )
        
        if output_path:
            torch.save(quantized_model.state_dict(), output_path)
            print(f"Quantized model saved to: {output_path}")
        
        # Calculate size reduction
        original_size = sum(p.numel() * 4 for p in model.parameters()) / 1024  # KB (float32)
        quantized_size = sum(p.numel() for p in quantized_model.parameters()) / 1024  # KB (int8)
        compression_ratio = original_size / quantized_size
        
        print(f"Model size reduction: {original_size:.1f} KB -> {quantized_size:.1f} KB "
              f"({compression_ratio:.1f}x compression)")
        
        return quantized_model
    
    @staticmethod
    def benchmark_inference_time(model: nn.Module, 
                                input_shape: Tuple[int, ...], 
                                num_runs: int = 1000) -> Dict[str, float]:
        """Benchmark model inference time."""
        model.eval()
        
        # Warm up
        dummy_input = torch.randn(*input_shape)
        for _ in range(10):
            with torch.no_grad():
                _ = model(dummy_input)
        
        # Benchmark
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        
        if torch.cuda.is_available():
            start_time.record()
            for _ in range(num_runs):
                with torch.no_grad():
                    _ = model(dummy_input)
            end_time.record()
            torch.cuda.synchronize()
            total_time = start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
        else:
            import time
            start = time.time()
            for _ in range(num_runs):
                with torch.no_grad():
                    _ = model(dummy_input)
            total_time = time.time() - start
        
        avg_time = total_time / num_runs
        fps = 1.0 / avg_time
        
        results = {
            'avg_inference_time_ms': avg_time * 1000,
            'avg_inference_time_us': avg_time * 1000000,
            'fps': fps,
            'total_time_s': total_time,
            'num_runs': num_runs
        }
        
        print(f"Inference benchmark results:")
        print(f"  Average inference time: {results['avg_inference_time_ms']:.3f} ms")
        print(f"  Average inference time: {results['avg_inference_time_us']:.1f} μs")
        print(f"  Throughput: {results['fps']:.1f} FPS")
        
        return results


if __name__ == "__main__":
    # Test the networks
    state_dim = 21
    action_dim = 2
    max_action = np.radians(7.0)  # 7 degrees in radians
    
    # Create agent
    agent = TD3Agent(state_dim, action_dim, max_action)
    
    # Test forward pass
    test_state = np.random.randn(state_dim)
    action = agent.select_action(test_state)
    print(f"Test action: {np.degrees(action)} degrees")
    
    # Test model optimization
    optimizer = ModelOptimizer()
    
    # Benchmark inference time
    results = optimizer.benchmark_inference_time(
        agent.actor, (1, state_dim), num_runs=1000
    )
    
    # Export to ONNX (for deployment)
    optimizer.export_to_onnx(
        agent.actor, 
        (1, state_dim), 
        "actor_model.onnx"
    )
    
    print("Network testing completed successfully!")
