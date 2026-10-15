"""Headless test harness and reinforcement learning environment for Beyond All Reason."""

from .match import MatchResult, run_match
from .start_script import ArtificialIntelligencePlayer, MatchConfiguration

__all__ = ["ArtificialIntelligencePlayer", "MatchConfiguration", "MatchResult", "run_match"]
