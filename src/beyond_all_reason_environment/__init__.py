"""Headless test harness and reinforcement learning environment for Beyond All Reason."""

from .batch import BatchConfiguration
from .match import MatchResult, run_match
from .start_script import ArtificialIntelligencePlayer, MatchConfiguration

__all__ = ["ArtificialIntelligencePlayer", "BatchConfiguration", "MatchConfiguration", "MatchResult", "run_match"]
