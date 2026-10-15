"""Headless test harness and reinforcement learning environment for Beyond All Reason."""

from pathlib import Path

from .batch import BatchConfiguration
from .match import MatchResult, run_match
from .start_script import ArtificialIntelligencePlayer, MatchConfiguration


def results_browser_path() -> Path:
    """The Streamlit script of the web interface (not imported: Streamlit runs it as a page)."""
    return Path(__file__).parent / "results_browser.py"


__all__ = [
    "ArtificialIntelligencePlayer",
    "BatchConfiguration",
    "MatchConfiguration",
    "MatchResult",
    "results_browser_path",
    "run_match",
]
