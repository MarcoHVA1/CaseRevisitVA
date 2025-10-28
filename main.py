# streamlit_dashboard.py
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# === App-config ===
st.set_page_config(page_title="Weer Dashboard NL", layout="wide")

# === Data inlezen ===
@st.cache_data
def load_data(path: str):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    records = raw["data"] if isinstance(raw, dict) and "data" in raw else raw
    df = pd.json_normalize(records)

    # Datum parsing
    if "date" in df.columns:
        try:
            df["date"] = pd.to_datetime(df["date"])
        except Exception:
            # Val terug op yyyymmdd formaten
            df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d", errors="coerce")

    # Helper om geschaalde velden te maken
    def make_scaled(src, dest, divisor=10):
        if src in df.columns:
            df[dest] = pd.to_numeric(df[src], errors="coerce") / divisor

    make_scaled("TG", "TG_C")   # Gemiddelde temperatuur (tienden °C)
    make_scaled("TN", "TN_C")   # Minimum temp (tienden °C)
    make_scaled("TX", "TX_C")   # Maximum temp (tienden °C)
    make_scaled("RH", "RH_mm")  # Neerslag (tienden mm)
    make_scaled("SQ", "SQ_h")   # Zonuren (tienden uur)

    # Windsnelheid berekenen (FG = in tienden m/s bij KNMI)
    if "FG" in df.columns:
        df["FG_ms"] = pd.to_numeric(df["FG"], errors="coerce") / 10.0

    # Afgeleide datumvelden
    if "date" in df.columns:
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        df["week"] = df["date"].dt.isocalendar().week

        def season(m):
            return (
                "winter" if m in [12, 1, 2]
                else "lente" if m in [3, 4, 5]
                else "zomer" if m in [6, 7, 8]
                else "herfst"
            )
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
    ["Overzicht", "Temperatuur Trends", "Neerslag & Zon", "Windtrends & Topdagen", "Voorspellingsmodel"]
)

# === KPI-tegels ===
avg_temp = df["TG_C"].mean().round(1) if "TG_C" in df else None
total_rain = df["RH_mm"].sum().round(1) if "RH_mm" in df else None
total_sun = df["SQ_h"].sum().round(1) if "SQ_h" in df else None

kpi1, kpi2, kpi3 = st.columns(3)
if avg_temp is not None:
    kpi1.metric("🌡️ Gemiddelde Temp (°C)", avg_temp)
if total_rain is not None:
    kpi2.metric("🌧️ Totale Neerslag (mm)", total_rain)
if total_sun is not None:
    kpi3.metric("☀️ Totale Zonuren", total_sun)

# === Stationmetadata (hergebruikt) ===
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

# === PAGINA 1: Overzicht ===
if page == "Overzicht":
    st.header("🗺️ Interactieve kaart • Temperatuur, Neerslag & Zonuren")
    st.caption("Kies jaar, maand en variabelen. De kaart toont waarden per KNMI-station; onderaan zie je de correlatie tussen twee variabelen.")

    # 1) JSON-bestanden detecteren
    files = sorted(Path(".").glob("*.json"))
    pat = re.compile(
        r"^(amsterdam|de_bilt|eelde|eindhoven|ijmuiden|maastricht|twente|vlissingen)_(\d{4}_\d{4})\.json$",
        re.I
    )
    found = []
    for f in files:
        m = pat.match(f.name)
        if m:
            station_key, period = m.group(1).lower(), m.group(2)
            found.append((station_key, period, str(f)))

    if not found:
        st.warning("Geen JSON-data gevonden (zoals 'Maastricht_2023_2024.json').")
        st.stop()

    # 2) Filters
    all_periods = sorted({p for _, p, _ in found})
    sel_periods = st.multiselect("📅 Kies jaarperiodes:", all_periods, default=all_periods)

    month_names = [
        "Alle", "01 - Januari", "02 - Februari", "03 - Maart", "04 - April",
        "05 - Mei", "06 - Juni", "07 - Juli", "08 - Augustus",
        "09 - September", "10 - Oktober", "11 - November", "12 - December"
    ]
    colA, colB, colC = st.columns([1, 1, 1])
    sel_month = colA.selectbox("📆 Maand:", month_names)

    map_var = colB.selectbox(
        "🗺️ Variabele op kaart",
        ["TG_C", "RH_mm", "SQ_h"],
        format_func=lambda k: {"TG_C": "🌡️ Temperatuur (°C)", "RH_mm": "🌧️ Neerslag (mm)", "SQ_h": "☀️ Zonuren (h)"}[k]
    )

    agg_choice = colC.radio("Aggregatie", ["Gemiddelde", "Som"], horizontal=True)
    agg_func = "mean" if agg_choice == "Gemiddelde" else "sum"

    # 3) Data samenvoegen
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
        st.warning("Geen data voor de gekozen filters.")
        st.stop()

    df_all = pd.concat(frames, ignore_index=True)

    # Maandfilter
    if sel_month != "Alle":
        month_idx = month_names.index(sel_month)
        df_all = df_all[df_all["date"].dt.month == month_idx]

    # 4) Numeriek maken
    for c in ["TG_C", "RH_mm", "SQ_h"]:
        if c in df_all.columns:
            df_all[c] = pd.to_numeric(df_all[c], errors="coerce")

    # 5) Aggregatie per station
    agg_df = (
        df_all.groupby(["station_key", "station"], as_index=False)
        .agg({map_var: agg_func, "TG_C": "mean", "RH_mm": "sum", "SQ_h": "sum"})
    )

    # Coördinaten toevoegen
    agg_df["lat"] = agg_df["station_key"].map(lambda k: STATIONS_META[k]["lat"])
    agg_df["lon"] = agg_df["station_key"].map(lambda k: STATIONS_META[k]["lon"])

    # 6) Data schoonmaken voor de kaart
    agg_df = agg_df.replace([np.inf, -np.inf], np.nan)
    agg_df = agg_df.dropna(subset=["lat", "lon", map_var])

    if agg_df.empty:
        st.info("Geen geldige waarden om op de kaart te tonen voor de gekozen filters.")
    else:
        color_scale = "RdYlBu_r" if map_var == "TG_C" else ("Blues" if map_var == "RH_mm" else "YlOrBr")
        map_title = {"TG_C": "Temperatuur (°C)", "RH_mm": "Neerslag (mm)", "SQ_h": "Zonuren (h)"}[map_var]

        # Alleen size gebruiken als alle waarden niet-negatief zijn
        size_kwargs = {}
        if (agg_df[map_var] >= 0).all():
            size_kwargs = {"size": map_var, "size_max": 28}

        fig_map = px.scatter_mapbox(
            agg_df,
            lat="lat",
            lon="lon",
            color=map_var,
            hover_name="station",
            hover_data={"lat": False, "lon": False, "TG_C": True, "RH_mm": True, "SQ_h": True},
            color_continuous_scale=color_scale,
            zoom=6,
            height=520,
            **size_kwargs
        )
        fig_map.update_layout(
            mapbox_style="open-street-map",
            margin=dict(l=0, r=0, t=10, b=0),
            coloraxis_colorbar=dict(title=map_title),
        )
        st.plotly_chart(fig_map, use_container_width=True)

# === PAGINA 2: Temperatuur Trends ===
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
                "January", "February", "March", "April", "May", "June",
                "July", "August", "September", "October", "November", "December"
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

# === PAGINA 3: Neerslag & Zon ===
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

# === PAGINA 4: Windtrends & Topdagen ===
elif page == "Windtrends & Topdagen":
    st.header("📊 Windtrends & Topdagen")

    # 1. Kalender-heatmap temperatuur
    if "TG_C" in df.columns and "date" in df.columns:
        df["day"] = df["date"].dt.day
        pivot = df.pivot_table(index="month", columns="day", values="TG_C", aggfunc="mean")
        month_names_map = {
            1: "Januari", 2: "Februari", 3: "Maart", 4: "April",
            5: "Mei", 6: "Juni", 7: "Juli", 8: "Augustus",
            9: "September", 10: "Oktober", 11: "November", 12: "December"
        }
        pivot.index = pivot.index.map(month_names_map)

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

    # 2. Windroos
    if "FG_ms" in df.columns and "DDVEC" in df.columns:
        try:
            import matplotlib.pyplot as plt
            from windrose import WindroseAxes  # vereist het 'windrose' pakket

            w = df[["DDVEC", "FG_ms"]].dropna()

            fig_wind, ax = plt.subplots(subplot_kw={"projection": "windrose"}, figsize=(6, 6))
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
        except Exception as e:
            st.info("Kon de windroos niet tekenen. Zorg dat het 'windrose' pakket is geïnstalleerd.\nFout: %s" % e)

    # 3. Boxplot windsnelheid per seizoen
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

        df["season_box"] = df["date"].apply(get_season)

        season_order = ["Lente", "Zomer", "Herfst", "Winter"]
        season_colors = {
            "Lente": "#2ecc71",
            "Zomer": "#f1c40f",
            "Herfst": "#e67e22",
            "Winter": "#3498db"
        }

        fig_box = px.box(
            df,
            x="season_box",
            y="FG_ms",
            color="season_box",
            category_orders={"season_box": season_order},
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

        st.plotly_chart(fig_box, use_container_width=True)

# === PAGINA 5: Voorspellingsmodel ===
elif page == "Voorspellingsmodel":
    st.header("🧠 Voorspellingsmodel — waar matcht jouw weer het beste?")
    st.caption(
        "Kies maand, dag, neerslag en windsnelheid. "
        "We berekenen per station een ‘matchscore’ op basis van historische dagen die hierop lijken."
    )

    # JSON-bestanden detecteren
    files = sorted(Path(".").glob("*.json"))
    pat = re.compile(
        r"^(amsterdam|de_bilt|eelde|eindhoven|ijmuiden|maastricht|twente|vlissingen)_(\d{4}_\d{4})\.json$",
        re.I
    )
    found = []
    for f in files:
        m = pat.match(f.name)
        if m:
            station_key, period = m.group(1).lower(), m.group(2)
            found.append((station_key, period, str(f)))

    if not found:
        st.warning("Geen JSON-data gevonden (bijv. 'amsterdam_2023_2024.json').")
        st.stop()

    # Periodes & maand/dagfilter UI
    all_periods = sorted({p for _, p, _ in found})
    with st.expander("Filters", expanded=True):
        sel_periods = st.multiselect("📅 Kies jaarperiodes:", all_periods, default=all_periods)

        # Maand-selectie (1..12 + 'Alle')
        month_names = [
            "Alle", "01 - Januari", "02 - Februari", "03 - Maart", "04 - April",
            "05 - Mei", "06 - Juni", "07 - Juli", "08 - Augustus",
            "09 - September", "10 - Oktober", "11 - November", "12 - December"
        ]
        sel_month_label = st.selectbox("📆 Maand:", month_names, index=0)

        # Dag-selectie (1..31 of 'Alle') — pas bereik dynamisch aan nadat data is geladen
        sel_day_default = "Alle"
        sel_day = st.selectbox("📅 Dag:", [sel_day_default] + [str(d) for d in range(1, 32)], index=0)

    # Data laden & samenvoegen
    frames = []
    for station_key, period, path_str in found:
        if period not in sel_periods:
            continue
        dfp = load_data(path_str)
        if "date" not in dfp.columns:
            continue

        # Numeriek maken wat we gebruiken
        for c in ["RH_mm", "FG_ms"]:
            if c in dfp.columns:
                dfp[c] = pd.to_numeric(dfp[c], errors="coerce")

        dfp["station_key"] = station_key
        dfp["station"] = STATIONS_META[station_key]["name"]
        dfp["period"] = period

        # Zorg voor month/day kolommen voor filtering
        dfp["month"] = dfp["date"].dt.month
        dfp["day"] = dfp["date"].dt.day

        frames.append(dfp)

    if not frames:
        st.warning("Geen data voor de gekozen filters.")
        st.stop()

    df_all = pd.concat(frames, ignore_index=True)

    # Maandfilter toepassen
    if sel_month_label != "Alle":
        month_idx = month_names.index(sel_month_label)   # 1..12
        df_all = df_all[df_all["month"] == month_idx]

    # Dagfilter toepassen (alleen als niet 'Alle')
    if sel_day != "Alle":
        try:
            day_idx = int(sel_day)
            df_all = df_all[df_all["day"] == day_idx]
        except ValueError:
            pass  # blijft 'Alle'

    # Als de gebruiker 'Alle' koos maar de data beperkt is, pas de dag-keuze aan
    # (optioneel: deze diagnose helpt de gebruiker)
    with st.expander("Beschikbaarheid"):
        if sel_month_label == "Alle":
            st.write("Beschikbare maanden in de data:", sorted(df_all["month"].dropna().unique().tolist()))
        else:
            st.write("Beschikbare dagen in maand", sel_month_label, ":", sorted(df_all["day"].dropna().unique().tolist()))

    # Benodigde kolommen aanwezig?
    need_cols = ["station_key", "station", "RH_mm", "FG_ms"]
    if not set(need_cols).issubset(df_all.columns):
        st.error("Benodigde kolommen ontbreken in de data (RH_mm, FG_ms).")
        st.stop()

    # Sliders voor gewenste omstandigheden (geen temperatuur, geen tolerantie-sliders)
    col2, col3 = st.columns(2)

    # Dynamisch bereik uit data
    r_min, r_max = float(np.nanmin(df_all["RH_mm"])), float(np.nanmax(df_all["RH_mm"]))
    w_min, w_max = float(np.nanmin(df_all["FG_ms"])), float(np.nanmax(df_all["FG_ms"]))

    # Zorg voor nette randen
    r_min, r_max = np.floor(min(0.0, r_min)), np.ceil(max(5.0, r_max))
    w
