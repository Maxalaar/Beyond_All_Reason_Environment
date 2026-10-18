"""Web interface to browse the results of all runs, from their Parquet tables (tables.py).

Started by scripts/browse_results.py (``streamlit run`` on this file).
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from beyond_all_reason_environment.analysis import MINIMUM_MATCHES_RUNNING, aggregate_over_time, win_rates
from beyond_all_reason_environment.colors import GRID, NEUTRAL_COLOR, PLAYER_COLORS, SIDE_COLORS
from beyond_all_reason_environment.metrics import KEY_METRICS, METRICS, METRICS_BY_NAME
from beyond_all_reason_environment.tables import experiment_and_version, list_versions, read_tables

GROUPINGS = {
    "player": "Player (one color per entry of players in the configuration)",
    "side": "Faction",
    "artificial_intelligence": "Artificial intelligence",
}
# Final statistics offered in the "End of match" tab, in this order when present.
FINAL_STATISTICS = [
    "metal_produced",
    "energy_produced",
    "damage_dealt",
    "damage_received",
    "units_produced",
    "units_killed",
    "units_died",
    "units_alive",
    "metal_excess",
    "energy_excess",
]
PLOT_LAYOUT = {
    "template": "plotly_white",
    "margin": {"l": 10, "r": 10, "t": 40, "b": 10},
    "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    "hovermode": "closest",
}


def version_label(path: Path) -> str:
    experiment, version = experiment_and_version(path)
    return f"{experiment} / version {version}"


@st.cache_data(show_spinner="Reading the tables...")
def load_tables(paths: tuple[str, ...], modification_times: tuple[float, ...]) -> dict[str, pd.DataFrame]:
    """Cached on the folders and their modification times, so a new run is picked up after a refresh."""
    del modification_times  # only part of the cache key
    return read_tables([Path(path) for path in paths])


def keep_maps(tables: dict[str, pd.DataFrame], maps: list[str]) -> dict[str, pd.DataFrame]:
    """The rows of the matches played on these maps, in every table (all of them have match_id)."""
    matches = tables["matches"]
    kept = list(matches.loc[matches["map_name"].isin(maps), "match_id"])
    return {name: frame.loc[frame["match_id"].isin(kept)].copy() for name, frame in tables.items()}


def add_group_column(tables: dict[str, pd.DataFrame], grouping: str) -> None:
    """Add a "group" column to the players and timeseries tables, following the chosen comparison."""
    players = tables["players"]
    several_experiments = players["experiment"].nunique() > 1
    if grouping == "player":
        entries = players.groupby(["experiment", "player"], as_index=False).agg(
            side=("side", lambda values: "/".join(sorted(set(values)))),
            name=("artificial_intelligence", "first"),
        )
        labels = {}
        columns = (entries["experiment"], entries["player"], entries["name"], entries["side"])
        for experiment, player, name, side in zip(*columns, strict=True):
            prefix = f"{experiment} · " if several_experiments else ""
            labels[(experiment, player)] = f"{prefix}player {player}: {name} ({side})"
        for table in (players, tables["timeseries"]):
            if not table.empty:
                table["group"] = [labels[key] for key in zip(table["experiment"], table["player"], strict=True)]
    else:
        for table in (players, tables["timeseries"]):
            if not table.empty:
                table["group"] = table[grouping].astype(str)


def group_colors(groups: list[str], grouping: str) -> dict[str, str]:
    """Factions keep their own color everywhere; other groups take the categorical slots in order."""
    if grouping == "side":
        return {group: SIDE_COLORS.get(group, NEUTRAL_COLOR) for group in groups}
    return {group: PLAYER_COLORS[index % len(PLAYER_COLORS)] for index, group in enumerate(sorted(groups))}


def with_alpha(color: str, alpha: float) -> str:
    red, green, blue = (int(color[index : index + 2], 16) for index in (1, 3, 5))
    return f"rgba({red},{green},{blue},{alpha})"


def winner_groups(players: pd.DataFrame) -> dict[str, str]:
    """match_id -> group of its winner (matches without a winner are absent)."""
    won = players[players["won"]]
    return dict(zip(won["match_id"].astype(str), won["group"].astype(str), strict=True))


# --- Tabs ----------------------------------------------------------------------------------------------------


def summary_tab(tables: dict[str, pd.DataFrame], colors: dict[str, str]) -> None:
    matches, players = tables["matches"], tables["players"]
    columns = st.columns(5)
    columns[0].metric("Matches", len(matches))
    columns[1].metric("Victories", int((matches["outcome"] == "victory").sum()))
    columns[2].metric("Time limit", int((matches["outcome"] == "timeout").sum()))
    columns[3].metric("Crashes", int((matches["outcome"] == "crash").sum()))
    columns[4].metric("Mean duration", f"{matches['game_minutes'].mean():.1f} min")

    st.subheader("Win rate, with its 95 % confidence interval")
    decisive_only = st.toggle(
        "Only decisive matches (leave out time limits and draws)",
        value=True,
        help="Otherwise a match stopped by the time limit counts as not won for both players.",
    )
    rates = win_rates(players, "group")
    prefix = "decisive_win_rate" if decisive_only else "win_rate"
    figure = go.Figure()
    for _, row in rates.iterrows():
        rate, low, high = row[prefix], row[f"{prefix}_low"], row[f"{prefix}_high"]
        played = row["decisive_matches"] if decisive_only else row["matches"]
        figure.add_trace(
            go.Bar(
                x=[row["group"]],
                y=[rate],
                marker_color=colors[str(row["group"])],
                error_y={"type": "data", "symmetric": False, "array": [high - rate], "arrayminus": [rate - low]},
                text=[f"{row['wins']} / {played}"],
                textposition="inside",
                hovertemplate=f"{row['group']}<br>{rate:.0%} won ({row['wins']} / {played})"
                f"<br>95 % interval: {low:.0%} to {high:.0%}<extra></extra>",
                name=row["group"],
                showlegend=False,
            )
        )
    figure.update_layout(**PLOT_LAYOUT, yaxis={"tickformat": ".0%", "range": [0, 1], "gridcolor": GRID}, height=380)
    st.plotly_chart(figure, width="stretch")
    st.caption(
        "The interval is where the true win rate lies with 95 % confidence (Wilson). "
        "When the intervals of two groups overlap a lot, the difference may be chance: play more matches."
    )
    st.dataframe(rates.round(3), hide_index=True, width="stretch")
    if players["map_name"].nunique() > 1:
        st.subheader("Wins per map")
        by_map = players.pivot_table(index="map_name", columns="group", values="won", aggfunc=["sum", "count"])
        wins_per_map = pd.DataFrame(index=by_map.index)
        for group in sorted(players["group"].unique()):
            wins_per_map[group] = [
                f"{int(wins)} / {int(played)} ({wins / played:.0%})"
                for wins, played in zip(by_map[("sum", group)], by_map[("count", group)], strict=True)
            ]
        timeouts = (
            players.drop_duplicates("match_id")
            .groupby("map_name")["outcome"]
            .apply(lambda outcomes: int((outcomes == "timeout").sum()))
        )
        wins_per_map["time limit"] = timeouts
        st.dataframe(wins_per_map, width="stretch")
        st.caption("Each cell: wins / matches played on that map. Use the Maps filter to see every tab for one map.")

    left, right = st.columns(2)
    with left:
        st.subheader("Duration of each match")
        winners = winner_groups(players)
        figure = go.Figure()
        for outcome_group, frame in matches.assign(
            winner_group=[winners.get(match_id, "no winner") for match_id in matches["match_id"]]
        ).groupby("winner_group"):
            figure.add_trace(
                go.Scatter(
                    x=frame["match_id"],
                    y=frame["game_minutes"],
                    mode="markers",
                    marker={"size": 10, "color": colors.get(str(outcome_group), NEUTRAL_COLOR)},
                    name=f"won by {outcome_group}" if outcome_group != "no winner" else "no winner",
                    hovertemplate="%{x}<br>%{y:.1f} game minutes<extra></extra>",
                )
            )
        figure.update_layout(**PLOT_LAYOUT, yaxis={"title": "game minutes", "rangemode": "tozero"}, height=380)
        figure.update_xaxes(showticklabels=False, title="match")
        st.plotly_chart(figure, width="stretch")
    with right:
        st.subheader("Start positions and outcome")
        located = players.dropna(subset=["start_x", "start_z"])
        if located.empty:
            st.info("Start positions are recorded for matches played from now on; these matches predate it.")
        else:
            figure = go.Figure()
            for _, frame in located.groupby(["group", "won"]):
                group, won = str(frame["group"].iloc[0]), bool(frame["won"].iloc[0])
                figure.add_trace(
                    go.Scatter(
                        x=frame["start_x"],
                        y=frame["start_z"],
                        mode="markers",
                        marker={
                            "size": 11,
                            "color": colors[group],
                            "symbol": "circle" if won else "x",
                            "line": {"width": 1, "color": "white"},
                        },
                        name=f"{group}, {'won' if won else 'not won'}",
                        text=frame["match_id"] + " (" + frame["map_name"] + ")",
                        hovertemplate="%{text}<br>x %{x:.0f}, z %{y:.0f}<extra></extra>",
                    )
                )
            figure.update_layout(**PLOT_LAYOUT, height=380)
            figure.update_yaxes(autorange="reversed", scaleanchor="x", title="z")
            figure.update_xaxes(title="x")
            st.plotly_chart(figure, width="stretch")
            st.caption(
                "Circle: won, cross: not won. If one area wins more often whatever the group, the map is biased."
            )


def over_time_tab(tables: dict[str, pd.DataFrame], colors: dict[str, str]) -> None:
    timeseries = tables["timeseries"]
    if timeseries.empty:
        st.info("No statistics over time in the selected versions.")
        return
    left, middle, right = st.columns([2, 2, 1])
    names = [metric.name for metric in METRICS]
    metric = left.selectbox(
        "Metric",
        names,
        index=names.index(KEY_METRICS[0]),
        format_func=lambda name: f"{METRICS_BY_NAME[name].title}  ({METRICS_BY_NAME[name].category})",
    )
    relative = (
        middle.radio("Time", ["game minutes", "share of each match's duration"], horizontal=True) != "game minutes"
    )
    show_matches = right.toggle("One line per match", value=True)

    aggregated = aggregate_over_time(timeseries, metric, relative, "group")
    x_column = "relative_time" if relative else "game_minutes"
    figure = go.Figure()
    for group in sorted(aggregated["group"].unique()):
        color = colors[group]
        if show_matches:
            for _, series in timeseries[timeseries["group"] == group].groupby(["match_id", "player"]):
                match_id = series["match_id"].iloc[0]
                figure.add_trace(
                    go.Scatter(
                        x=series[x_column],
                        y=series[metric],
                        mode="lines",
                        line={"width": 0.8, "color": with_alpha(color, 0.3)},
                        hovertemplate=f"{match_id}<br>%{{y:,.1f}}<extra>{group}</extra>",
                        showlegend=False,
                    )
                )
        frame = aggregated[aggregated["group"] == group]
        figure.add_trace(
            go.Scatter(
                x=frame["x"], y=frame["high"], mode="lines", line={"width": 0}, hoverinfo="skip", showlegend=False
            )
        )
        figure.add_trace(
            go.Scatter(
                x=frame["x"],
                y=frame["low"],
                mode="lines",
                line={"width": 0},
                fill="tonexty",
                fillcolor=with_alpha(color, 0.15),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        figure.add_trace(
            go.Scatter(
                x=frame["x"],
                y=frame["median"],
                mode="lines",
                line={"width": 3, "color": color},
                name=group,
                customdata=frame[["low", "high", "running"]],
                hovertemplate="%{y:,.1f} (25-75 %: %{customdata[0]:,.1f} to %{customdata[1]:,.1f})"
                "<br>%{customdata[2]} series<extra>" + group + "</extra>",
            )
        )
    metric_definition = METRICS_BY_NAME[metric]
    figure.update_layout(**PLOT_LAYOUT, height=560, title=metric_definition.title)
    figure.update_xaxes(
        title="share of the match duration" if relative else "game minutes",
        tickformat=".0%" if relative else None,
        gridcolor=GRID,
    )
    figure.update_yaxes(
        rangemode="tozero", tickformat=".0%" if metric_definition.kind == "share" else None, gridcolor=GRID
    )
    st.plotly_chart(figure, width="stretch")
    if relative:
        st.caption(
            "Thick line: median, band: 25th to 75th percentile. Every match counts at every point (no survivor bias)."
        )
    else:
        st.caption(
            "Thick line: median, band: 25th to 75th percentile. A curve stops when fewer than "
            f"{MINIMUM_MATCHES_RUNNING} matches of its group are still running: late values only describe the "
            "longest matches. Switch to the share of each match's duration to compare without this bias."
        )


def end_of_match_tab(tables: dict[str, pd.DataFrame], colors: dict[str, str]) -> None:
    players = tables["players"]
    available = [column for column in FINAL_STATISTICS if column in players.columns]
    statistic = st.selectbox("Statistic at the end of the match", available)
    figure = go.Figure()
    for group in sorted(players["group"].unique()):
        frame = players[players["group"] == group]
        figure.add_trace(
            go.Box(
                y=frame[statistic],
                name=group,
                marker_color=colors[group],
                boxpoints="all",
                jitter=0.4,
                pointpos=0,
                text=frame["match_id"],
                hovertemplate="%{text}<br>%{y:,.0f}<extra></extra>",
            )
        )
    figure.update_layout(**PLOT_LAYOUT, height=520, showlegend=False)
    figure.update_yaxes(rangemode="tozero", gridcolor=GRID)
    st.plotly_chart(figure, width="stretch")
    st.caption("One dot per match; the box spans the 25th to 75th percentile, the line inside is the median.")


def matches_tab(tables: dict[str, pd.DataFrame], colors: dict[str, str]) -> None:
    matches, players, timeseries = tables["matches"], tables["players"], tables["timeseries"]
    winners = winner_groups(players)
    table = matches.assign(winner=[winners.get(match_id, "-") for match_id in matches["match_id"]])
    st.dataframe(
        table[["match_id", "seed", "map_name", "outcome", "winner", "game_minutes", "has_replay"]].round(1),
        hide_index=True,
        width="stretch",
    )
    match_id = st.selectbox("Match", table["match_id"])
    match = table.set_index("match_id").loc[match_id]
    st.markdown(
        f"**{match_id}**: {match['outcome']}, winner {match['winner']}, {match['game_minutes']:.1f} game minutes"
    )

    series = timeseries[timeseries["match_id"] == match_id]
    columns = st.columns(len(KEY_METRICS))
    for column, metric in zip(columns, KEY_METRICS, strict=True):
        figure = go.Figure()
        for group, frame in series.groupby("group"):
            figure.add_trace(
                go.Scatter(
                    x=frame["game_minutes"],
                    y=frame[metric],
                    mode="lines",
                    line={"width": 2.5, "color": colors[str(group)]},
                    name=str(group),
                )
            )
        figure.update_layout(**PLOT_LAYOUT, height=340, title=METRICS_BY_NAME[metric].title, showlegend=False)
        figure.update_xaxes(title="game minutes", gridcolor=GRID)
        figure.update_yaxes(rangemode="tozero", gridcolor=GRID)
        column.plotly_chart(figure, width="stretch")

    run_directory = Path(match["run_directory"])
    if match["has_replay"]:
        st.markdown("Watch the replay:")
        st.code(f"python scripts/watch_replay.py {run_directory}", language="bash")
    overview = run_directory / "overview.png"
    if overview.exists():
        with st.expander("All the charts of this match (overview.png)"):
            st.image(str(overview))
    st.dataframe(players[players["match_id"] == match_id].drop(columns=["group"]), hide_index=True)


# --- Page ----------------------------------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="Beyond All Reason results", layout="wide")
    st.title("Beyond All Reason results")

    versions = list_versions()
    if not versions:
        st.info("No results yet. Run for example: python scripts/run.py configurations/runs/barb_versus_barb.py")
        return
    experiments = sorted({experiment_and_version(path)[0] for path in versions})
    latest = max(versions, key=lambda path: path.stat().st_mtime)

    with st.sidebar:
        chosen_experiments = st.multiselect("Experiments", experiments, default=[experiment_and_version(latest)[0]])
        candidates = [path for path in versions if experiment_and_version(path)[0] in chosen_experiments]
        chosen_versions = st.multiselect(
            "Versions (pooled together)", candidates, default=candidates, format_func=version_label
        )
        grouping = st.radio("Compare by", list(GROUPINGS), format_func=GROUPINGS.get)
        st.caption("Tables: runs/<experiment>/version_<n>/tables/*.parquet")
    if not chosen_versions:
        st.info("Choose at least one version in the sidebar.")
        return

    tables = load_tables(
        tuple(str(path) for path in chosen_versions), tuple(path.stat().st_mtime for path in chosen_versions)
    )
    tables = {name: frame.copy() for name, frame in tables.items()}
    maps = sorted(tables["matches"]["map_name"].unique())
    chosen_maps = st.sidebar.multiselect("Maps", maps, default=maps)
    tables = keep_maps(tables, chosen_maps)
    if tables["matches"].empty:
        st.info("Choose at least one map in the sidebar.")
        return
    add_group_column(tables, grouping)
    colors = group_colors(list(tables["players"]["group"].unique()), grouping)

    summary, over_time, end_of_match, matches = st.tabs(["Summary", "Over time", "End of match", "Matches"])
    with summary:
        summary_tab(tables, colors)
    with over_time:
        over_time_tab(tables, colors)
    with end_of_match:
        end_of_match_tab(tables, colors)
    with matches:
        matches_tab(tables, colors)


main()
