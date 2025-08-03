"""
Intelligent Ascent TVC System - Rocket Environment Package
=========================================================

This package contains the complete rocket simulation environment and 
TD3 implementation for thrust vector control systems.
"""

from .rocket_physics import RocketTVCEnvironment, RocketDynamics
from .td3_networks import TD3Agent, ActorNetwork, CriticNetwork, ModelOptimizer

__version__ = "1.0.0"
__author__ = "AI Assistant"

__all__ = [
    'RocketTVCEnvironment',
    'RocketDynamics', 
    'TD3Agent',
    'ActorNetwork',
    'CriticNetwork',
    'ModelOptimizer'
]
