# streamlit_dashboard.py

import json
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# === Data inlezen ===
@st.cache_data
def load_data(path: str):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    records = raw["data"] if isinstance(raw, dict) and "data" in raw else raw
    df = pd.json_normalize(records)

    # Datum parsing
    try:
        df["date"] = pd.to_datetime(df["date"])
    except Exception:
        df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")

    def make_scaled(src, dest, divisor=10):
        if src in df.columns:
            df[dest] = pd.to_numeric(df[src], errors="coerce") / divisor

    make_scaled("TG", "TG_C")
    make_scaled("TN", "TN_C")
    make_scaled("TX", "TX_C")
    make_scaled("RH", "RH_mm")
    make_scaled("SQ", "SQ_h")

    # Windsnelheid berekenen (FG = in tienden m/s bij KNMI)
    if "FG" in df.columns:
        df["FG_ms"] = pd.to_numeric(df["FG"], errors="coerce") / 10.0

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["week"] = df["date"].dt.isocalendar().week

    def season(m):
        return "winter" if m in [12, 1, 2] else "lente" if m in [3, 4, 5] else "zomer" if m in [6, 7, 8] else "herfst"
    df["season"] = df["month"].apply(season)
    return df

# === Sidebar ===
st.sidebar.title("Navigation")

datasets = {
    "2021–2022": "amsterdam_2021_2022.json",
    "2022–2023": "amsterdam_2022_2023.json",
    "2023–2024": "amsterdam_2023_2024.json",
}
dataset_choice = st.sidebar.selectbox("📅 Kies een dataset:", list(datasets.keys()))
df = load_data(datasets[dataset_choice])

page = st.sidebar.radio(
    "Select a page",
    ["Overzicht", "Temperatuur Trends", "Neerslag & Zon", "Windtrends & Topdagen"]
)


# === Pagina's ===if page == "Overzicht":
if page == "Overzicht":
    st.header("🌍 Amsterdam: Het Weer in Verandering")
    st.subheader("Warmer – Droger – Zonniger (jaarvergelijking)")

    # === Data voorbereiden ===
    agg_dict = {"TG_C": "mean", "RH_mm": "sum"}
    if "SQ_h" in df.columns:  # alleen zonuren meenemen als het bestaat
        agg_dict["SQ_h"] = "sum"

    yearly = df.groupby("year").agg(agg_dict).reset_index()

    if "SQ_h" in yearly.columns:
        scale_factor = 200
        yearly["SQ_scaled"] = yearly["SQ_h"] / scale_factor
    else:
        yearly["SQ_scaled"] = 0

    # === Consistente layout-stijl ===
    layout_style = dict(
        font=dict(family="Arial, sans-serif", size=14, color="#ffffff"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, zeroline=False, linecolor="grey"),
        yaxis=dict(showgrid=True, gridcolor="rgba(200,200,200,0.2)"),
        legend=dict(
            orientation="h", y=1.15, x=0.5, xanchor="center",
            font=dict(size=12, color="#ffffff")
        )
    )

    # === 1. Jaarvergelijking ===
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=yearly["year"], y=yearly["TG_C"],
        name="Gem. Temp (°C)", marker_color="#e74c3c",
        hovertemplate="Gem. Temp: %{y:.1f} °C<br>Jaar: %{x}<extra></extra>"
    ))

    if "SQ_h" in df.columns:
        fig.add_trace(go.Bar(
            x=yearly["year"], y=yearly["SQ_scaled"],
            name=f"Zonuren (x{scale_factor}h)", marker_color="#f1c40f",
            hovertemplate="Zonuren: %{customdata} uur<br>Jaar: %{x}<extra></extra>",
            customdata=yearly["SQ_h"]
        ))

    fig.add_trace(go.Scatter(
        x=yearly["year"], y=yearly["RH_mm"],
        name="Neerslag (mm)", mode="lines+markers",
        yaxis="y2", line=dict(color="#3498db", width=3),
        hovertemplate="Neerslag: %{y:.0f} mm<br>Jaar: %{x}<extra></extra>"
    ))

     # Eerst specifieke instellingen
    fig.update_layout(
        title="📊 Vergelijking per jaar: Temperatuur, Neerslag en Zonuren",
        xaxis_title="Jaar",
        yaxis=dict(title="Temp (°C) & Zonuren (geschaald)", side="left"),
        yaxis2=dict(title="Neerslag (mm)", overlaying="y", side="right"),
        barmode="group"
    )
    # Daarna de uniforme stijl toepassen
    fig.update_layout(**layout_style)
    
    st.plotly_chart(fig, use_container_width=True)

    # Praktische conclusie
    if len(yearly) >= 2:
        diff_temp = yearly["TG_C"].iloc[-1] - yearly["TG_C"].iloc[-2]
        diff_rain = yearly["RH_mm"].iloc[-2] - yearly["RH_mm"].iloc[-1]
        diff_sun = yearly["SQ_h"].iloc[-1] - yearly["SQ_h"].iloc[-2] if "SQ_h" in yearly.columns else 0
        st.info(
            f"In {yearly['year'].iloc[-1]} was het gemiddeld {diff_temp:.1f}°C warmer, "
            f"viel er {diff_rain:.0f} mm minder regen en scheen de zon {diff_sun:.0f} uur langer "
            f"dan in {yearly['year'].iloc[-2]}."
        )

    # === 2. Professionele lange termijn trend ===
    avg_yearly_temp = df.groupby("year")["TG_C"].mean().reset_index()
    fig_trend = px.line(
        avg_yearly_temp, x="year", y="TG_C", markers=True,
        title="📈 Lange termijn trend: Gemiddelde jaartemperatuur",
        labels={"TG_C": "Gemiddelde Temp (°C)", "year": "Jaar"}
    )
    fig_trend.update_traces(line=dict(color="#e74c3c", width=4))
    fig_trend.update_layout(**layout_style)
    st.plotly_chart(fig_trend, use_container_width=True)

    # === 3. Professionele seizoensgemiddelden ===
    season_temp = df.groupby(["year", "season"])["TG_C"].mean().reset_index()
    season_order = ["winter", "lente", "zomer", "herfst"]
    season_colors = {"winter": "#3498db", "lente": "#2ecc71", "zomer": "#f1c40f", "herfst": "#e67e22"}

    fig_season = px.bar(
        season_temp, x="year", y="TG_C", color="season",
        title="🌦️ Gemiddelde temperatuur per seizoen",
        labels={"TG_C": "Gemiddelde Temp (°C)", "year": "Jaar", "season": "Seizoen"},
        category_orders={"season": season_order},
        color_discrete_map=season_colors,
        barmode="group"
    )
    fig_season.update_layout(**layout_style)
    st.plotly_chart(fig_season, use_container_width=True)

elif page == "Temperatuur Trends":
    st.header("🌡️ Temperatuur Trends")
    use_cols = [c for c in ["TN_C", "TG_C", "TX_C"] if c in df.columns]

    if use_cols:
        label_map = {"TN_C": "Min temp", "TG_C": "Gem temp", "TX_C": "Max temp"}
        temp = df[["date"] + use_cols].melt("date", var_name="type", value_name="temp_C")
        temp["type"] = temp["type"].replace(label_map)

        fig = px.line(
            temp, x="date", y="temp_C", color="type",
            title="Dagelijkse temperatuur (min, gem, max)",
            labels={"temp_C": "Temperatuur (°C)", "date": "Datum", "type": "Type"},
        )
        st.plotly_chart(fig, use_container_width=True)

        # Boxplot temperatuur per maand
        df["month_name"] = df["date"].dt.month_name()

        fig_box = px.box(
            df, x="month_name", y="TG_C",
            category_orders={"month_name": [
                "January","February","March","April","May","June",
                "July","August","September","October","November","December"
            ]},
            title="📦 Verdeling van gemiddelde temperatuur per maand",
            labels={"month_name": "Maand", "TG_C": "Gemiddelde temperatuur (°C)"},
            color="month_name"
        )
        fig_box.update_traces(line_width=2)
        st.plotly_chart(fig_box, use_container_width=True)

        # Gemiddelde temperatuur per seizoen
        season_temp = df.groupby("season")["TG_C"].mean().reset_index()
        season_colors = {"winter": "#3498db", "lente": "#2ecc71", "zomer": "#f1c40f", "herfst": "#e67e22"}

        fig_season = px.bar(
            season_temp, x="season", y="TG_C", color="season",
            title="🌦️ Gemiddelde temperatuur per seizoen",
            labels={"season": "Seizoen", "TG_C": "Gemiddelde Temp (°C)"},
            color_discrete_map=season_colors
        )
        st.plotly_chart(fig_season, use_container_width=True)


elif page == "Neerslag & Zon":
    st.header("☔ Neerslag vs. Zon")

    if "RH_mm" in df.columns and "SQ_h" in df.columns:
        bins = [0, 1, 5, 10, 50]
        labels = ["0 mm", "0–5 mm", "5–10 mm", "10+ mm"]
        df["rain_cat"] = pd.cut(df["RH_mm"], bins=bins, labels=labels, include_lowest=True)

        ordered_cats = ["0 mm", "0–5 mm", "5–10 mm", "10+ mm"]
        df["rain_cat"] = pd.Categorical(df["rain_cat"], categories=ordered_cats, ordered=True)

        color_map = {
            "0 mm": "#d7263d",
            "0–5 mm": "#0e6eb8",
            "5–10 mm": "#74a9cf",
            "10+ mm": "#eab0b6"
        }

        fig_box = px.box(
            df, x="rain_cat", y="SQ_h",
            color="rain_cat",
            category_orders={"rain_cat": ordered_cats},
            color_discrete_map=color_map,
            title="📦 Verdeling zonuren per neerslagcategorie",
            labels={"SQ_h": "Zonuren", "rain_cat": "Neerslagcategorie"},
            points="all"
        )
        st.plotly_chart(fig_box, use_container_width=True)

        rain_bins = pd.cut(
            df["RH_mm"], bins=[0, 1, 5, 10, 20, 50],
            include_lowest=True,
            labels=["0–1 mm", "1–5 mm", "5–10 mm", "10–20 mm", "20+ mm"]
        )
        avg_temp_rain = df.groupby(rain_bins)["TG_C"].mean().reset_index()

        fig_temp_rain = px.bar(
            avg_temp_rain, x="RH_mm", y="TG_C",
            title="🌧️ Gemiddelde temperatuur bij toenemende regenval",
            labels={"RH_mm": "Neerslagcategorie (mm per dag)", "TG_C": "Gemiddelde temperatuur (°C)"},
            text_auto=".1f",
            color="TG_C",
            color_continuous_scale="RdYlBu_r"
        )
        fig_temp_rain.update_layout(
            xaxis_title="Neerslagcategorie (mm per dag)",
            yaxis_title="Gemiddelde temperatuur (°C)",
            showlegend=False
        )
        st.plotly_chart(fig_temp_rain, use_container_width=True)

elif page == "Windtrends & Topdagen":
    st.header("📊 Windtrends & Topdagen")

    # === 1. Kalender-heatmap temperatuur ===
    if "TG_C" in df.columns and "date" in df.columns:
        df["day"] = df["date"].dt.day
        pivot = df.pivot_table(index="month", columns="day", values="TG_C", aggfunc="mean")
        month_names = {
            1: "Januari", 2: "Februari", 3: "Maart", 4: "April",
            5: "Mei", 6: "Juni", 7: "Juli", 8: "Augustus",
            9: "September", 10: "Oktober", 11: "November", 12: "December"
        }
        pivot.index = pivot.index.map(month_names)

        fig_heatmap = px.imshow(
            pivot,
            color_continuous_scale="RdBu_r",
            origin="upper",
            aspect="auto",
            labels=dict(color="Temperatuur (°C)", x="Dag van de maand", y="Maand")
        )
        fig_heatmap.update_xaxes(title="Dag van de maand", tickmode="linear")
        fig_heatmap.update_yaxes(title="Maand", tickmode="array",
                                 tickvals=list(pivot.index), ticktext=list(pivot.index))
        fig_heatmap.update_layout(title="📅 Kalender-heatmap: gemiddelde temperatuur per dag", title_x=0.5)

        st.plotly_chart(fig_heatmap, use_container_width=True)

    # === 2. Windroos ===
    if "FG_ms" in df.columns and "DDVEC" in df.columns:
        import matplotlib.pyplot as plt
        from windrose import WindroseAxes

        w = df[["DDVEC", "FG_ms"]].dropna()

        fig_wind, ax = plt.subplots(subplot_kw={"projection": "windrose"}, figsize=(6,6))
        ax.bar(
            w["DDVEC"],
            w["FG_ms"],
            normed=True,
            opening=0.8,
            bins=[0, 2, 4, 6, 8, 10, 12],
            edgecolor="white"
        )
        ax.set_title("Windroos — richting & snelheid (gemiddeld)", pad=20)
        ax.set_legend(title="m/s", loc="center left", bbox_to_anchor=(1.1, 0.5))

        st.pyplot(fig_wind)

    # === 3. Boxplot windsnelheid per seizoen ===
    if "FG_ms" in df.columns and "date" in df.columns:
        st.subheader("📦 Verdeling van windsnelheid per seizoen")

        def get_season(date):
            m = date.month
            if m in [3, 4, 5]:
                return "Lente"
            elif m in [6, 7, 8]:
                return "Zomer"
            elif m in [9, 10, 11]:
                return "Herfst"
            else:
                return "Winter"

        df['season'] = df['date'].apply(get_season)

        season_order = ["Lente", "Zomer", "Herfst", "Winter"]
        season_colors = {
            "Lente": "#2ecc71",
            "Zomer": "#f1c40f",
            "Herfst": "#e67e22",
            "Winter": "#3498db"
        }

        fig_box = px.box(
            df,
            x="season",
            y="FG_ms",
            color="season",
            category_orders={"season": season_order},
            color_discrete_map=season_colors,
            points="all"
        )
        fig_box.update_traces(line_width=3)
        fig_box.update_layout(
            title="📦 Verdeling windsnelheid per seizoen",
            xaxis_title="Seizoen",
            yaxis_title="Gemiddelde windsnelheid (m/s)",
            title_x=0.5,
            boxmode="group"
        )

        st.plotly_chart(fig_box, use_container_width=True)# streamlit_dashboard.py

