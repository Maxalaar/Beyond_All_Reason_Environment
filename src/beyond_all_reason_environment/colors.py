"""Colors shared by the chart files (plots.py) and the web interface (results_browser.py).

Categorical slots in fixed order (validated colorblind-safe on the light surface): a series keeps its
color whatever else is shown.
"""

PLAYER_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
# Factions always get the same slot, so Armada is blue and Cortex orange in every view.
SIDE_COLORS = {"Armada": PLAYER_COLORS[0], "Cortex": PLAYER_COLORS[1], "Legion": PLAYER_COLORS[2]}
NEUTRAL_COLOR = "#8a8984"
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e6e5e0"
