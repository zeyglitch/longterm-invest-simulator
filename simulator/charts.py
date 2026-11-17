"""Plotly figures. The only module of the package that draws: each function takes DataFrames."""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# one palette for the whole app: the same asset or curve keeps its colour on every page
PALETTE = ["#5b8def", "#f2a93b", "#3dbe8b", "#b07cf0", "#ef6f6c", "#4fc3d9", "#c7b94a", "#8a94a6"]
VALUE_COLOR = PALETTE[0]
INVESTED_COLOR = "#8a94a6"
DEPOSIT_COLOR = PALETTE[2]
WITHDRAWAL_COLOR = PALETTE[4]
# one colour per scenario, in the order pessimistic, central, optimistic
SCENARIO_COLORS = [PALETTE[1], PALETTE[0], PALETTE[2]]
LIVRET_COLOR = PALETTE[3]

TRANSPARENT = "rgba(0,0,0,0)"


def _base_figure(**layout) -> go.Figure:
    """A figure with the dark look shared by all charts. Separators give 12 345.67."""
    figure = go.Figure()
    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor=TRANSPARENT,
        plot_bgcolor=TRANSPARENT,
        separators=". ",
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        **layout,
    )
    return figure


def allocation_chart(table: pd.DataFrame) -> go.Figure:
    """Two rings: actual weights and target weights.

    `table` has the columns `name`, `weight` and `target_weight` (decimals, empty if unknown).
    Each asset keeps the same colour in both rings. The target ring is left out when no asset
    has a target.
    """
    colours = {name: PALETTE[i % len(PALETTE)] for i, name in enumerate(table["name"])}
    actual = table[table["weight"].fillna(0) > 0]
    target = table[table["target_weight"].fillna(0) > 0]
    rings = [("Actual", actual, "weight")]
    if not target.empty:
        rings.append(("Target", target, "target_weight"))

    figure = make_subplots(rows=1, cols=len(rings), specs=[[{"type": "domain"}] * len(rings)])
    for column, (title, rows, field) in enumerate(rings, start=1):
        figure.add_trace(
            go.Pie(
                labels=rows["name"],
                values=rows[field],
                hole=0.55,
                sort=False,
                marker={"colors": [colours[name] for name in rows["name"]]},
                texttemplate="%{percent:.1%}",
                hovertemplate="%{label}: %{percent:.1%}<extra></extra>",
                title={"text": title},
            ),
            row=1,
            col=column,
        )
    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor=TRANSPARENT,
        plot_bgcolor=TRANSPARENT,
        separators=". ",
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        legend={"orientation": "h", "y": -0.05},
    )
    return figure


def history_chart(history: pd.DataFrame) -> go.Figure:
    """Portfolio value against the amount invested over time (`date`, `value`, `invested`)."""
    figure = _base_figure(
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.1, "x": 0},
        yaxis={"ticksuffix": " €", "tickformat": ",.0f"},
    )
    figure.add_trace(
        go.Scatter(
            x=history["date"],
            y=history["value"],
            name="Value",
            line={"color": VALUE_COLOR, "width": 2},
            hovertemplate="%{y:,.2f} €",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=history["date"],
            y=history["invested"],
            name="Invested",
            # the invested amount only changes on a trade: it stays flat in between
            line={"color": INVESTED_COLOR, "width": 2, "shape": "hv"},
            hovertemplate="%{y:,.2f} €",
        )
    )
    return figure


def contributions_chart(flows: pd.Series, one_offs: pd.DataFrame) -> go.Figure:
    """Monthly deposit of the plan over time, with a marker for each one-off event.

    `flows` holds the regular deposit of each month (index: dates). `one_offs` has the columns
    `date`, `amount` and `label`. A marker sits on the bar of its month, so that a large one-off
    does not flatten the bars; its amount is in the hover text.
    """
    figure = _base_figure(
        legend={"orientation": "h", "y": 1.1, "x": 0},
        yaxis={"ticksuffix": " €", "tickformat": ",.0f"},
        bargap=0,
    )
    figure.add_trace(
        go.Bar(
            x=flows.index,
            y=flows.values,
            name="Monthly deposit",
            marker={"color": VALUE_COLOR, "line": {"width": 0}},
            hovertemplate="%{x|%b %Y}: %{y:,.2f} €<extra></extra>",
        )
    )
    kinds = [
        ("One-off deposit", one_offs[one_offs["amount"] > 0], DEPOSIT_COLOR, "triangle-up"),
        ("Withdrawal", one_offs[one_offs["amount"] < 0], WITHDRAWAL_COLOR, "triangle-down"),
    ]
    for name, rows, colour, symbol in kinds:
        if rows.empty:
            continue
        figure.add_trace(
            go.Scatter(
                x=rows["date"],
                y=[flows.get(day, 0.0) for day in rows["date"]],
                mode="markers",
                name=name,
                marker={"color": colour, "size": 13, "symbol": symbol},
                customdata=rows[["label", "amount"]],
                hovertemplate="%{customdata[0]}: %{customdata[1]:,.2f} €<extra></extra>",
            )
        )
    return figure


def projection_chart(
    curves: pd.DataFrame, scenarios: list[str], target: float | None = None
) -> go.Figure:
    """Value over time for each scenario, with the cumulated deposits and an optional Livret A.

    `curves` has a `date` column, one column per scenario name, a `Contributions` column and,
    if wanted, a `Livret A` column. `target` draws a horizontal line.
    """
    figure = _base_figure(
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.1, "x": 0},
        yaxis={"ticksuffix": " €", "tickformat": ",.0f"},
    )
    for number, name in enumerate(scenarios):
        colour = SCENARIO_COLORS[number % len(SCENARIO_COLORS)]
        figure.add_trace(
            go.Scatter(
                x=curves["date"],
                y=curves[name],
                name=name,
                line={"color": colour, "width": 2},
                hovertemplate="%{y:,.0f} €",
            )
        )
    extras = [("Contributions", INVESTED_COLOR, "dash"), ("Livret A", LIVRET_COLOR, "dot")]
    for name, colour, dash in extras:
        if name not in curves:
            continue
        figure.add_trace(
            go.Scatter(
                x=curves["date"],
                y=curves[name],
                name=name,
                line={"color": colour, "width": 2, "dash": dash},
                hovertemplate="%{y:,.0f} €",
            )
        )
    if target:
        figure.add_hline(
            y=target,
            line={"color": INVESTED_COLOR, "width": 1, "dash": "dot"},
            annotation_text="Target",
            annotation_position="top left",
        )
    return figure


def comparison_chart(curves: pd.DataFrame) -> go.Figure:
    """Several values over time, to compare them: one line per column of `curves`.

    `curves` has a `date` column and one column per envelope or strategy. A `Contributions`
    column, if there is one, is drawn dashed in grey.
    """
    figure = _base_figure(
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.1, "x": 0},
        yaxis={"ticksuffix": " €", "tickformat": ",.0f"},
    )
    names = [name for name in curves.columns if name not in ("date", "Contributions")]
    for number, name in enumerate(names):
        figure.add_trace(
            go.Scatter(
                x=curves["date"],
                y=curves[name],
                name=name,
                line={"color": PALETTE[number % len(PALETTE)], "width": 2},
                hovertemplate="%{y:,.0f} €",
            )
        )
    if "Contributions" in curves:
        figure.add_trace(
            go.Scatter(
                x=curves["date"],
                y=curves["Contributions"],
                name="Contributions",
                line={"color": INVESTED_COLOR, "width": 2, "dash": "dash"},
                hovertemplate="%{y:,.0f} €",
            )
        )
    return figure


def gains_chart(table: pd.DataFrame) -> go.Figure:
    """Deposits and gains in the final value of each scenario, as stacked bars.

    `table` has the columns `scenario`, `contributions` and `gains`.
    """
    figure = _base_figure(
        barmode="stack",
        legend={"orientation": "h", "y": 1.15, "x": 0},
        yaxis={"ticksuffix": " €", "tickformat": ",.0f"},
    )
    parts = [("Deposits", "contributions", INVESTED_COLOR), ("Gains", "gains", DEPOSIT_COLOR)]
    for name, column, colour in parts:
        figure.add_trace(
            go.Bar(
                x=table["scenario"],
                y=table[column],
                name=name,
                marker={"color": colour},
                hovertemplate=name + ": %{y:,.0f} €<extra></extra>",
            )
        )
    return figure


def fan_chart(bands: pd.DataFrame, samples: pd.DataFrame) -> go.Figure:
    """Spread of the random paths over time: median, 25-75 % band and 10-90 % band.

    `bands` has the columns `date`, 10, 25, 50, 75, 90 (percentiles) and, if wanted,
    `contributions`.
    `samples` holds a few single paths, one per column, drawn as thin lines behind.
    """
    figure = _base_figure(
        legend={"orientation": "h", "y": 1.1, "x": 0},
        yaxis={"ticksuffix": " €", "tickformat": ",.0f"},
    )
    for column in samples:
        figure.add_trace(
            go.Scatter(
                x=bands["date"],
                y=samples[column],
                line={"color": "rgba(138, 148, 166, 0.25)", "width": 1},
                hoverinfo="skip",
                showlegend=False,
            )
        )
    # each band is two lines, the second one filled down to the first
    for low, high, colour, name in [
        (10, 90, "rgba(91, 141, 239, 0.18)", "10-90 %"),
        (25, 75, "rgba(91, 141, 239, 0.35)", "25-75 %"),
    ]:
        figure.add_trace(
            go.Scatter(
                x=bands["date"], y=bands[low], line={"width": 0}, hoverinfo="skip", showlegend=False
            )
        )
        figure.add_trace(
            go.Scatter(
                x=bands["date"],
                y=bands[high],
                line={"width": 0},
                fill="tonexty",
                fillcolor=colour,
                name=name,
                hoverinfo="skip",
            )
        )
    figure.add_trace(
        go.Scatter(
            x=bands["date"],
            y=bands[50],
            name="Median",
            line={"color": VALUE_COLOR, "width": 2},
            hovertemplate="%{x|%b %Y}: %{y:,.0f} €<extra>Median</extra>",
        )
    )
    if "contributions" in bands:
        figure.add_trace(
            go.Scatter(
                x=bands["date"],
                y=bands["contributions"],
                name="Contributions",
                line={"color": INVESTED_COLOR, "width": 2, "dash": "dash"},
                hovertemplate="%{y:,.0f} €<extra>Contributions</extra>",
            )
        )
    return figure


def histogram_chart(final_values: pd.Series, target: float | None = None) -> go.Figure:
    """How the final values of the paths are spread. `target` draws a vertical line.

    The 1 % highest values are grouped out of sight: a few huge paths would flatten the rest.
    """
    shown = final_values[final_values <= final_values.quantile(0.99)]
    figure = _base_figure(
        xaxis={"ticksuffix": " €", "tickformat": ",.0f"},
        yaxis={"title": "Number of paths"},
        bargap=0.05,
    )
    figure.add_trace(
        go.Histogram(
            x=shown,
            nbinsx=60,
            marker={"color": VALUE_COLOR},
            hovertemplate="%{x}: %{y} paths<extra></extra>",
        )
    )
    figure.add_vline(
        x=final_values.median(),
        line={"color": DEPOSIT_COLOR, "width": 2},
        annotation_text="Median",
        annotation_position="top",
    )
    if target:
        figure.add_vline(
            x=target,
            line={"color": INVESTED_COLOR, "width": 2, "dash": "dot"},
            annotation_text="Target",
            annotation_position="top right",
        )
    return figure


def drawdown_chart(falls: pd.Series) -> go.Figure:
    """How far the price is below its previous high, over time. `falls` holds 0 at a high and
    negative shares below it (-0.25 for 25 %), indexed by date."""
    figure = _base_figure(yaxis={"tickformat": ".2~p"})
    figure.add_trace(
        go.Scatter(
            x=falls.index,
            y=falls.values,
            fill="tozeroy",
            line={"color": WITHDRAWAL_COLOR, "width": 1},
            fillcolor="rgba(239, 111, 108, 0.3)",
            hovertemplate="%{x|%d/%m/%Y}: %{y:.1%}<extra></extra>",
        )
    )
    return figure
