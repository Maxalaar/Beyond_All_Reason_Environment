"""Headless test harness and reinforcement learning environment for Beyond All Reason."""

from pathlib import Path

from .batch import BatchConfiguration
from .match import MatchResult, run_match
from .scenario import AllyteamConfiguration, ScenarioConfiguration
from .start_script import ArtificialIntelligencePlayer, MatchConfiguration


def results_browser_path() -> Path:
    """The Streamlit script of the web interface (not imported: Streamlit runs it as a page)."""
    return Path(__file__).parent / "results_browser.py"


__all__ = [
    "AllyteamConfiguration",
    "ArtificialIntelligencePlayer",
    "BatchConfiguration",
    "MatchConfiguration",
    "MatchResult",
    "results_browser_path",
    "run_match",
    "ScenarioConfiguration",
]
