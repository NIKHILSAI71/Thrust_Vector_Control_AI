"""
Intelligent Ascent TVC System - Experience Replay Buffer
=======================================================

High-performance experience replay buffer for TD3 training with
optimized sampling and memory management.

Author: AI Assistant
Date: August 2025
"""

import numpy as np
import torch
from typing import Dict, Any, Optional
import random
from collections import deque


class ReplayBuffer:
    """
    Experience replay buffer for off-policy reinforcement learning.
    
    Features:
    - Efficient circular buffer implementation
    - Random sampling with replacement
    - Batch sampling for training
    - Memory-efficient storage
    """
    
    def __init__(self, capacity: int, state_dim: int, action_dim: int, device: str = "cpu"):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            device: Device for tensor operations
        """
        self.capacity = capacity
        self.device = device
        self.position = 0
        self.size = 0
        
        # Preallocate memory for efficiency
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.bool_)
        
        print(f"Replay buffer initialized with capacity {capacity:,}")
    
    def add(self, state: np.ndarray, action: np.ndarray, reward: float, 
            next_state: np.ndarray, done: bool):
        """Add a new experience to the buffer."""
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Sample a batch of experiences."""
        if self.size < batch_size:
            raise ValueError(f"Not enough experiences in buffer. Current size: {self.size}, "
                           f"requested batch size: {batch_size}")
        
        # Random sampling without replacement
        indices = np.random.choice(self.size, batch_size, replace=False)
        
        batch = {
            'state': self.states[indices],
            'action': self.actions[indices],
            'reward': self.rewards[indices].flatten(),
            'next_state': self.next_states[indices],
            'done': self.dones[indices].flatten()
        }
        
        return batch
    
    def can_sample(self, batch_size: int) -> bool:
        """Check if buffer has enough experiences for sampling."""
        return self.size >= batch_size
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return self.size
    
    def clear(self):
        """Clear all experiences from buffer."""
        self.position = 0
        self.size = 0
        print("Replay buffer cleared")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        if self.size == 0:
            return {"size": 0, "capacity": self.capacity, "utilization": 0.0}
        
        valid_indices = slice(0, self.size)
        
        stats = {
            "size": self.size,
            "capacity": self.capacity,
            "utilization": self.size / self.capacity,
            "reward_mean": float(np.mean(self.rewards[valid_indices])),
            "reward_std": float(np.std(self.rewards[valid_indices])),
            "reward_min": float(np.min(self.rewards[valid_indices])),
            "reward_max": float(np.max(self.rewards[valid_indices])),
            "done_rate": float(np.mean(self.dones[valid_indices]))
        }
        
        return stats


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay (PER) buffer implementation.
    
    Samples experiences based on their temporal difference (TD) error,
    giving higher priority to experiences with larger learning potential.
    """
    
    def __init__(self, capacity: int, state_dim: int, action_dim: int, 
                 alpha: float = 0.6, beta: float = 0.4, device: str = "cpu"):
        """
        Initialize prioritized replay buffer.
        
        Args:
            capacity: Maximum number of experiences
            state_dim: State space dimension
            action_dim: Action space dimension
            alpha: Prioritization exponent (0 = uniform, 1 = full prioritization)
            beta: Importance sampling exponent (0 = no correction, 1 = full correction)
            device: Device for tensor operations
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.device = device
        self.position = 0
        self.size = 0
        
        # Experience storage
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.bool_)
        
        # Priority storage
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.max_priority = 1.0
        
        print(f"Prioritized replay buffer initialized with capacity {capacity:,}")
    
    def add(self, state: np.ndarray, action: np.ndarray, reward: float,
            next_state: np.ndarray, done: bool, td_error: Optional[float] = None):
        """Add experience with priority."""
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done
        
        # Set priority based on TD error or use maximum priority for new experiences
        if td_error is not None:
            priority = (abs(td_error) + 1e-6) ** self.alpha
        else:
            priority = self.max_priority
        
        self.priorities[self.position] = priority
        self.max_priority = max(self.max_priority, priority)
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Dict[str, Any]:
        """Sample batch with prioritized sampling."""
        if self.size < batch_size:
            raise ValueError(f"Not enough experiences in buffer")
        
        # Calculate sampling probabilities
        valid_priorities = self.priorities[:self.size]
        sampling_probs = valid_priorities / np.sum(valid_priorities)
        
        # Sample indices based on priorities
        indices = np.random.choice(self.size, batch_size, p=sampling_probs, replace=False)
        
        # Calculate importance sampling weights
        weights = (self.size * sampling_probs[indices]) ** (-self.beta)
        weights = weights / np.max(weights)  # Normalize weights
        
        batch = {
            'state': self.states[indices],
            'action': self.actions[indices],
            'reward': self.rewards[indices].flatten(),
            'next_state': self.next_states[indices],
            'done': self.dones[indices].flatten(),
            'indices': indices,
            'weights': weights
        }
        
        return batch
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities based on new TD errors."""
        priorities = (np.abs(td_errors) + 1e-6) ** self.alpha
        self.priorities[indices] = priorities
        self.max_priority = max(self.max_priority, np.max(priorities))
    
    def can_sample(self, batch_size: int) -> bool:
        """Check if buffer can provide a batch."""
        return self.size >= batch_size
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return self.size


class MultiStepBuffer:
    """
    Multi-step return buffer for n-step learning.
    
    Computes n-step returns for more efficient learning by looking ahead
    multiple steps into the future.
    """
    
    def __init__(self, n_steps: int = 3, gamma: float = 0.99):
        """
        Initialize multi-step buffer.
        
        Args:
            n_steps: Number of steps for n-step returns
            gamma: Discount factor
        """
        self.n_steps = n_steps
        self.gamma = gamma
        self.buffer = deque(maxlen=n_steps)
        
    def add(self, state: np.ndarray, action: np.ndarray, reward: float,
            next_state: np.ndarray, done: bool) -> Optional[Dict[str, Any]]:
        """
        Add experience and return n-step transition if available.
        
        Returns:
            n-step transition dictionary or None if not enough steps accumulated
        """
        self.buffer.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        })
        
        if len(self.buffer) < self.n_steps:
            return None
        
        # Calculate n-step return
        n_step_return = 0.0
        for i, experience in enumerate(self.buffer):
            n_step_return += (self.gamma ** i) * experience['reward']
            if experience['done']:
                break
        
        # Create n-step transition
        first_experience = self.buffer[0]
        last_experience = self.buffer[-1]
        
        n_step_transition = {
            'state': first_experience['state'],
            'action': first_experience['action'],
            'n_step_return': n_step_return,
            'next_state': last_experience['next_state'],
            'done': last_experience['done'],
            'n_steps': len(self.buffer)
        }
        
        return n_step_transition
    
    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()


if __name__ == "__main__":
    # Test replay buffer
    state_dim = 21
    action_dim = 2
    capacity = 10000
    
    # Test regular replay buffer
    buffer = ReplayBuffer(capacity, state_dim, action_dim)
    
    # Add some random experiences
    for i in range(1000):
        state = np.random.randn(state_dim)
        action = np.random.randn(action_dim)
        reward = np.random.randn()
        next_state = np.random.randn(state_dim)
        done = np.random.random() < 0.1
        
        buffer.add(state, action, reward, next_state, done)
    
    # Test sampling
    if buffer.can_sample(256):
        batch = buffer.sample(256)
        print(f"Sampled batch with keys: {batch.keys()}")
        print(f"Batch shapes: {[(k, v.shape) for k, v in batch.items()]}")
    
    # Print statistics
    stats = buffer.get_statistics()
    print(f"Buffer statistics: {stats}")
    
    # Test prioritized replay buffer
    per_buffer = PrioritizedReplayBuffer(capacity, state_dim, action_dim)
    
    # Add experiences with random TD errors
    for i in range(500):
        state = np.random.randn(state_dim)
        action = np.random.randn(action_dim)
        reward = np.random.randn()
        next_state = np.random.randn(state_dim)
        done = np.random.random() < 0.1
        td_error = np.random.randn()
        
        per_buffer.add(state, action, reward, next_state, done, td_error)
    
    # Test prioritized sampling
    if per_buffer.can_sample(128):
        per_batch = per_buffer.sample(128)
        print(f"PER batch keys: {per_batch.keys()}")
        print(f"Importance weights range: [{per_batch['weights'].min():.3f}, {per_batch['weights'].max():.3f}]")
    
    print("Replay buffer testing completed!")
