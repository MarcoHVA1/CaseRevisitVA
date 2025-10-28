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

# === KPI-tegels ===
avg_temp = df["TG_C"].mean().round(1) if "TG_C" in df else None
total_rain = df["RH_mm"].sum().round(1) if "RH_mm" in df else None
total_sun = df["SQ_h"].sum().round(1) if "SQ_h" in df else None

kpi1, kpi2, kpi3 = st.columns(3)
if avg_temp: kpi1.metric("🌡️ Gemiddelde Temp (°C)", avg_temp)
if total_rain: kpi2.metric("🌧️ Totale Neerslag (mm)", total_rain)
if total_sun: kpi3.metric("☀️ Totale Zonuren", total_sun)


# === PAGINA 1: Overzicht ===
# === PAGINA: Kaart ===
elif page == "Kaart":
    import re
    from pathlib import Path

    st.header("🗺️ Interactieve Kaart – Temperatuur, Neerslag & Zonuren")

    # === 1) Bekende stations met coördinaten ===
    STATIONS_META = {
        "amsterdam":  {"name": "Amsterdam",  "lat": 52.3676, "lon": 4.9041},
        "de_bilt":    {"name": "De Bilt",    "lat": 52.1010, "lon": 5.1790},
        "eelde":      {"name": "Eelde",      "lat": 53.1250, "lon": 6.5833},
        "eindhoven":  {"name": "Eindhoven",  "lat": 51.4500, "lon": 5.3740},
        "ijmuiden":   {"name": "IJmuiden",   "lat": 52.4600, "lon": 4.6100},
        "maastricht": {"name": "Maastricht", "lat": 50.8510, "lon": 5.6910},
        "twente":     {"name": "Twente",     "lat": 52.2700, "lon": 6.9000},
        "vlissingen": {"name": "Vlissingen", "lat": 51.4420, "lon": 3.5730},
    }

    # === 2) Automatisch JSON-bestanden detecteren ===
    files = sorted(Path(".").glob("*.json"))
    pat = re.compile(r"^(amsterdam|de_bilt|eelde|eindhoven|ijmuiden|maastricht|twente|vlissingen)_(\d{4}_\d{4})\.json$", re.I)
    found = []
    for f in files:
        m = pat.match(f.name)
        if m:
            station_key, period = m.group(1).lower(), m.group(2)
            found.append((station_key, period, str(f)))

    if not found:
        st.warning("Geen dataset-bestanden gevonden (zoals 'amsterdam_2021_2022.json').")
        st.stop()

    # === 3) Selectie-opties ===
    all_periods = sorted({p for _, p, _ in found})
    sel_periods = st.multiselect("📅 Kies jaren:", all_periods, default=all_periods)

    month_names = [
        "Alle", "01 - Januari", "02 - Februari", "03 - Maart", "04 - April",
        "05 - Mei", "06 - Juni", "07 - Juli", "08 - Augustus",
        "09 - September", "10 - Oktober", "11 - November", "12 - December"
    ]
    sel_month = st.selectbox("📆 Maand:", month_names)

    metric = st.selectbox(
        "📊 Variabele:",
        ["TG_C", "RH_mm", "SQ_h"],
        format_func=lambda k: {
            "TG_C": "Gemiddelde temperatuur (°C)",
            "RH_mm": "Neerslag (mm)",
            "SQ_h": "Zonuren (h)",
        }[k]
    )

    # === 4) Data combineren ===
    frames = []
    for station_key, period, path_str in found:
        if period not in sel_periods:
            continue
        dfp = load_data(path_str)
        if "date" not in dfp.columns:
            continue
        dfp["station_key"] = station_key
        dfp["station"] = STATIONS_META[station_key]["name"]
        dfp["period"] = period
        frames.append(dfp)

    if not frames:
        st.warning("Geen gegevens voor de geselecteerde combinatie.")
        st.stop()

    df_all = pd.concat(frames, ignore_index=True)

    # Filter maand (indien niet 'Alle')
    if sel_month != "Alle":
        month_idx = month_names.index(sel_month)
        df_all = df_all[df_all["date"].dt.month == month_idx]

    # === 5) Aggregatie per station ===
    df_all[metric] = pd.to_numeric(df_all[metric], errors="coerce")
    agg = (
        df_all.groupby(["station_key", "station", "period"], as_index=False)[metric]
        .mean()
        .dropna(subset=[metric])
    )

    # Voeg coördinaten toe
    agg["lat"] = agg["station_key"].map(lambda k: STATIONS_META[k]["lat"])
    agg["lon"] = agg["station_key"].map(lambda k: STATIONS_META[k]["lon"])

    # === 6) Interactieve kaart ===
    color_scale = (
        "RdYlBu_r" if metric == "TG_C"
        else ("Blues" if metric == "RH_mm" else "YlOrBr")
    )

    fig_map = px.scatter_mapbox(
        agg,
        lat="lat",
        lon="lon",
        color=metric,
        size=metric,
        size_max=30,
        hover_name="station",
        hover_data={"period": True, "lat": False, "lon": False},
        zoom=6,
        height=520,
        color_continuous_scale=color_scale,
    )
    fig_map.update_layout(
        mapbox_style="carto-darkmatter",
        margin=dict(l=0, r=0, t=30, b=0),
        title=f"Kaart — {metric} per station (kleur = laag→hoog, grootte = waarde)",
        coloraxis_colorbar=dict(title=metric),
    )
    st.plotly_chart(fig_map, use_container_width=True)

    # === 7) Lijnvergelijking per station ===
    st.subheader("📈 Trend per station over jaren")
    trend = (
        df_all.groupby(["station", "period"], as_index=False)[metric]
        .mean()
        .sort_values(["station", "period"])
    )

    fig_line = px.line(
        trend,
        x="period",
        y=metric,
        color="station",
        markers=True,
        labels={"period": "Jaarperiode", metric: metric},
        title=f"{metric} – gemiddelde trend per station ({sel_month})",
    )
    st.plotly_chart(fig_line, use_container_width=True)

    # === 8) Uitleg ===
    st.caption(
        "💡 Kies bovenin de variabele (temperatuur, neerslag of zonuren) en filter op maand of jaar. "
        "De kaart toont per station de waarde als kleur en grootte, en de grafiek toont trends over jaren."
    )

elif page == "Temperatuur Trends":
    st.header("🌡️ Temperatuur Trendss")
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
