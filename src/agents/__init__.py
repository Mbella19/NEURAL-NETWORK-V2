"""
RL agents module for the EURUSD trading system.

This module provides the PPO-based Sniper Agent that learns to trade
using context vectors from a frozen Market Analyst model.

Main Components:
    SniperAgent: PPO wrapper using Stable Baselines 3
    create_agent: Factory function with config-based initialization
"""

from .sniper_agent import SniperAgent, create_agent

__all__ = ['SniperAgent', 'create_agent']
