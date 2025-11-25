import plotly.express as px
import pandas as pd

# ---------- Helper: add Program column ----------
def _with_program(df, program_label):
    out = df.copy()
    out["Program"] = program_label
    return out

# Long-form data for XM30 vs SEP
makes_by_org = pd.concat(
    [
        _with_program(xm30_make_org, "XM30"),
        _with_program(sep_make_org, "SEP"),
    ],
    ignore_index=True,
)

buys_by_org = pd.concat(
    [
        _with_program(xm30_buy_org, "XM30"),
        _with_program(sep_buy_org, "SEP"),
    ],
    ignore_index=True,
)

# ---------- Helper: consistent bar-chart styling ----------
def make_presentation_bar(df, title, y_col, yaxis_title):
    fig = px.bar(
        df,
        x="Org",
        y=y_col,
        color="Program",
        barmode="group",
        text=y_col,              # label bars with values
    )

    # Bigger, clearer labels on top of bars
    fig.update_traces(
        texttemplate="%{text:,}",   # thousands separator
        textposition="outside",
        cliponaxis=False,           # allow labels above plot area
        marker_line_width=1.2,
    )

    fig.update_layout(
        title=title,
        title_x=0.5,
        width=1000,
        height=500,
        bargap=0.25,
        plot_bgcolor="white",
        xaxis_title="Org",
        yaxis_title=yaxis_title,
        legend_title="Program",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
        font=dict(size=18),         # overall font size
        margin=dict(l=70, r=40, t=80, b=80),
    )

    fig.update_xaxes(tickfont=dict(size=16), showline=True, linewidth=1)
    fig.update_yaxes(
        tickfont=dict(size=16),
        showgrid=True,
        gridcolor="lightgray",
        zeroline=True,
        zerolinewidth=1,
    )

    return fig

# ---------- Charts for the slide ----------

fig_makes = make_presentation_bar(
    makes_by_org,
    title="Makes by Org – XM30 vs SEP",
    y_col="Num_Parts",
    yaxis_title="Number of Make Parts",
)
fig_makes.show()

fig_buys = make_presentation_bar(
    buys_by_org,
    title="Buys by Org – XM30 vs SEP",
    y_col="Num_Parts",
    yaxis_title="Number of Buy Parts",
)
fig_buys.show()

# (Optional) SEP hours by Org in same style
fig_sep_hours = px.bar(
    sep_hours_org,
    x="Org",
    y="SEP_Hours",
    text="SEP_Hours",
    title="SEP Total Labor Hours (CWS) by Org",
)

fig_sep_hours.update_traces(
    texttemplate="%{text:,.0f}",
    textposition="outside",
    cliponaxis=False,
    marker_line_width=1.2,
)
fig_sep_hours.update_layout(
    width=1000,
    height=500,
    plot_bgcolor="white",
    xaxis_title="Org",
    yaxis_title="Labor Hours (SEP)",
    font=dict(size=18),
    margin=dict(l=70, r=40, t=80, b=80),
)
fig_sep_hours.update_xaxes(tickfont=dict(size=16), showline=True, linewidth=1)
fig_sep_hours.update_yaxes(
    tickfont=dict(size=16),
    showgrid=True,
    gridcolor="lightgray",
    zeroline=True,
    zerolinewidth=1,
)

fig_sep_hours.show()