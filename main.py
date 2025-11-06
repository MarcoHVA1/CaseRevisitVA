import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# === App-config ===
st.set_page_config(page_title="Weer Dashboard NL", layout="wide")

# -----------------------------------------------------------------------------
# Helpers & Data loading
# -----------------------------------------------------------------------------
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

STATIONS_META = {
    "amsterdam":  {"name": "Amsterdam",  "lat": 52.3676, "lon": 4.9041},
    "de_bilt":    {"name": "De Bilt",    "lat": 52.1010, "lon": 5.1790},
    "eelde":      {"name": "Eelde",      "lat": 53.1250, "lon": 6.5833},
    "eindhoven":  {"name": "Eindhoven",  "lat": 51.4500, "lon": 5.3740},
    "ijmuiden":   {"name": "IJmuiden",   "lat": 52.4600, "lon": 4.6100},
    "maastricht": {"name": "Maastricht", "lat": 50.8510, "lon": 5.6910},
    "twente":     {"name": "Twente",     "lat": 52.2700, "lon": 6.9000},
    "vlissingen": {"name": "Vlissingen", "lat": 51.4420, "lon": 3.5730}
}

# ---- Discover all JSON files (case-insensitive, incl. IJmuiden) ----
@st.cache_data
def discover_files():
    files = sorted(Path('.').glob('*.json'))
    pat = re.compile(r'^(amsterdam|de_bilt|eelde|eindhoven|ijmuiden|maastricht|twente|vlissingen)_(\d{4}_\d{4})\.json$', re.I)
    found = []
    for f in files:
        m = pat.match(f.name)
        if m:
            station_key, period = m.group(1).lower(), m.group(2)
            found.append((station_key, period, str(f)))
    return found

# ---- Build combined dataframe based on selection ----
@st.cache_data
def build_dataset(selected_periods: tuple, selected_stations: tuple):
    found = discover_files()
    frames = []
    for station_key, period, path_str in found:
        if selected_periods and period not in selected_periods:
            continue
        if selected_stations and station_key not in selected_stations:
            continue
        dfp = load_data(path_str)
        if "date" not in dfp.columns:
            continue
        for c in ["TG_C", "TN_C", "TX_C", "RH_mm", "SQ_h", "FG_ms"]:
            if c in dfp.columns:
                dfp[c] = pd.to_numeric(dfp[c], errors="coerce")
        dfp["station_key"] = station_key
        dfp["station"] = STATIONS_META[station_key]["name"]
        dfp["period"] = period
        frames.append(dfp)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

# ---- UI block for global filters per page ----
def selection_controls(key_prefix: str = ""):
    found = discover_files()
    all_periods = sorted({p for _, p, _ in found})
    all_station_keys = [k for k in STATIONS_META.keys()]

    st.subheader("⚙️ Filters")
    col1, col2 = st.columns([2, 3])
    with col1:
        sel_periods = st.multiselect("📅 Jaarperiodes", options=all_periods, default=all_periods, key=f"{key_prefix}_periods")
    with col2:
        mode = st.selectbox(
            "📍 Locatiekeuze",
            ["Alle locaties (geaggregeerd)", "Enkele locatie", "Vergelijk locaties"],
            index=0,
            key=f"{key_prefix}_mode"
        )

    if mode == "Enkele locatie":
        station = st.selectbox("Station", options=all_station_keys, format_func=lambda k: STATIONS_META[k]["name"], key=f"{key_prefix}_single")
        stations = [station]
    elif mode == "Vergelijk locaties":
        stations = st.multiselect(
            "Stations",
            options=all_station_keys,
            default=["amsterdam", "ijmuiden"],
            format_func=lambda k: STATIONS_META[k]["name"],
            key=f"{key_prefix}_multi"
        )
        if not stations:
            st.warning("Kies minimaal één station.")
    else:
        stations = all_station_keys

    df_all = build_dataset(tuple(sel_periods), tuple(stations))

    # Aggregatie: voor "Alle locaties" middelen we per datum over stations (i.p.v. sommeren)
    if mode == "Alle locaties (geaggregeerd)" and not df_all.empty:
        num_cols = [c for c in ["TN_C", "TG_C", "TX_C", "RH_mm", "SQ_h", "FG_ms"] if c in df_all.columns]
        keep_cols = ["date"] + num_cols
        g = (
            df_all[keep_cols + ["station"]]
              .groupby("date", as_index=False)
              .agg({c: "mean" for c in num_cols})
        )
        # Herstel afgeleide datumvelden (maand/season) voor downstream groupby's
        g["month"] = g["date"].dt.month
        def _season(m):
            return (
                "winter" if m in [12, 1, 2]
                else "lente" if m in [3, 4, 5]
                else "zomer" if m in [6, 7, 8]
                else "herfst"
            )
        g["season"] = g["month"].apply(_season)
        g["station"] = "Alle stations"
        g["station_key"] = "all"
        g["period"] = ", ".join(sel_periods)
        df_all = g
    return df_all, mode

# -----------------------------------------------------------------------------
# Sidebar (navigation ONLY)
# -----------------------------------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a page",
    [
        "Overzicht",
        "Temperatuur Trends",
        "Neerslag & Zon",
        "Windtrends & Topdagen",
        "Correlaties",
        "Voorspellingsmodel"
    ]
)

# -----------------------------------------------------------------------------
# KPI-tegels (op basis van alle data die aanwezig is)
# -----------------------------------------------------------------------------
# Gebruik alle bestanden en reken gemiddelde KPI's over alle stations (zonder UI-calls)
found_kpi = discover_files()
all_periods_kpi = sorted({p for _, p, _ in found_kpi})
all_station_keys_kpi = list(STATIONS_META.keys())
df_kpi = build_dataset(tuple(all_periods_kpi), tuple(all_station_keys_kpi))

if df_kpi.empty:
    avg_temp = total_rain = total_sun = None
else:
    avg_temp = df_kpi.get("TG_C").mean().round(1) if "TG_C" in df_kpi else None
    total_rain = df_kpi.get("RH_mm").sum().round(1) if "RH_mm" in df_kpi else None
    total_sun = df_kpi.get("SQ_h").sum().round(1) if "SQ_h" in df_kpi else None

kpi1, kpi2, kpi3 = st.columns(3)
if avg_temp is not None:
    kpi1.metric("🌡️ Gemiddelde Temp (°C)", avg_temp)
if total_rain is not None:
    kpi2.metric("🌧️ Totale Neerslag (mm)", total_rain)
if total_sun is not None:
    kpi3.metric("☀️ Totale Zonuren", total_sun)

# -----------------------------------------------------------------------------
# PAGE 1: Overzicht
# -----------------------------------------------------------------------------
if page == "Overzicht":
    st.header("🗺️ Interactieve kaart • Temperatuur, Neerslag & Zonuren")
    st.caption("Kies jaarperiodes en variabelen. De kaart toont waarden per KNMI-station. Stations zonder data worden grijs getoond (incl. IJmuiden).")

    found = discover_files()
    if not found:
        st.warning("Geen JSON-data gevonden.")
        st.stop()

    # Filters
    all_periods = sorted({p for _, p, _ in found})
    sel_periods = st.multiselect("📅 Kies jaarperiodes:", all_periods, default=all_periods, key="ov_periods")

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

    # Data samenvoegen
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

    if sel_month != "Alle":
        month_idx = month_names.index(sel_month)
        df_all = df_all[df_all["date"].dt.month == month_idx]

    for c in ["TG_C", "RH_mm", "SQ_h"]:
        if c in df_all.columns:
            df_all[c] = pd.to_numeric(df_all[c], errors="coerce")

    agg_df = (
        df_all.groupby(["station_key", "station"], as_index=False)
        .agg({map_var: agg_func, "TG_C": "mean", "RH_mm": "sum", "SQ_h": "sum"})
    )

    agg_df["lat"] = agg_df["station_key"].map(lambda k: STATIONS_META[k]["lat"])
    agg_df["lon"] = agg_df["station_key"].map(lambda k: STATIONS_META[k]["lon"])

    agg_df = agg_df.replace([np.inf, -np.inf], np.nan)

    # Maak een volledige stationslijst (zodat IJmuiden altijd zichtbaar is)
    stations_full = pd.DataFrame([
        {"station_key": k, "station": v["name"], "lat": v["lat"], "lon": v["lon"]}
        for k, v in STATIONS_META.items()
    ])
    agg_df = stations_full.merge(agg_df, on=["station_key", "station", "lat", "lon"], how="left")

    # Splits valid/missing per kaartvariabele
    valid = agg_df.dropna(subset=[map_var]).copy()
    missing = agg_df[agg_df[map_var].isna()].copy()

    color_scale = "RdYlBu_r" if map_var == "TG_C" else ("Blues" if map_var == "RH_mm" else "YlOrBr")
    map_title = {"TG_C": "Temperatuur (°C)", "RH_mm": "Neerslag (mm)", "SQ_h": "Zonuren (h)"}[map_var]

    if valid.empty and missing.empty:
        st.info("Geen geldige waarden om op de kaart te tonen voor de gekozen filters.")
    else:
        size_kwargs = {}
        if not valid.empty and (valid[map_var] >= 0).all():
            size_kwargs = {"size": map_var, "size_max": 28}

        fig_map = px.scatter_mapbox(
            valid if not valid.empty else agg_df,
            lat="lat",
            lon="lon",
            color=(map_var if not valid.empty else None),
            hover_name="station",
            hover_data={"lat": False, "lon": False, "TG_C": True, "RH_mm": True, "SQ_h": True},
            color_continuous_scale=(color_scale if not valid.empty else None),
            zoom=6,
            height=520,
            **({} if valid.empty else size_kwargs)
        )
        # Grijze markers voor stations zonder waarde (incl. IJmuiden)
        if not missing.empty:
            fig_map.add_trace(
                go.Scattermapbox(
                    lat=missing["lat"],
                    lon=missing["lon"],
                    mode="markers",
                    marker=dict(size=14, color="#A0A0A0"),
                    name="Geen data",
                    text=missing["station"],
                    hoverinfo="text",
                )
            )
        fig_map.update_layout(
            mapbox_style="open-street-map",
            margin=dict(l=0, r=0, t=10, b=0),
            coloraxis_colorbar=dict(title=map_title),
        )
        st.plotly_chart(fig_map, use_container_width=True)

# -----------------------------------------------------------------------------
# PAGE 2: Temperatuur Trends
# -----------------------------------------------------------------------------
elif page == "Temperatuur Trends":
    st.header("🌡️ Temperatuur Trends")
    df, mode = selection_controls(key_prefix="temp")
    if df.empty or "date" not in df.columns:
        st.info("Geen data beschikbaar voor de gekozen filters.")
        st.stop()

    # Lijnplot min/gem/max (per station of geaggregeerd)
    use_cols = [c for c in ["TN_C", "TG_C", "TX_C"] if c in df.columns]
    if use_cols:
        label_map = {"TN_C": "Min temp", "TG_C": "Gem temp", "TX_C": "Max temp"}
        if "station" not in df.columns:
            df["station"] = "Alle stations"
        temp = df[["date", "station"] + use_cols].melt(["date", "station"], var_name="type", value_name="temp_C")
        temp["type"] = temp["type"].replace(label_map)

        facet_args = {}
        if mode == "Vergelijk locaties":
            facet_args = {"facet_row": "station"}

        fig = px.line(
            temp, x="date", y="temp_C", color="type", **facet_args,
            labels={"temp_C": "Temperatuur (°C)", "date": "Datum", "type": "Type"},
            title="Dagelijkse temperatuur (min, gem, max)"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Boxplot temperatuur per maand
    dft = df.copy()
    dft["month_name"] = dft["date"].dt.month_name()
    fig_box = px.box(
        dft, x="month_name", y="TG_C", color=("station" if mode == "Vergelijk locaties" else None),
        category_orders={"month_name": [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ]},
        title="📦 Verdeling van gemiddelde temperatuur per maand",
        labels={"month_name": "Maand", "TG_C": "Gemiddelde
