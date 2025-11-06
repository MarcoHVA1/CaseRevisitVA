import json
from pathlib import Path
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# =========================================================
# App-config
# =========================================================
st.set_page_config(page_title="Weer Dashboard NL", layout="wide")

# Stations die we willen uitsluiten
EXCLUDED_STATIONS = {"ijmuiden"}

# =========================================================
# Basis helpers
# =========================================================
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    records = raw.get("data", raw) if isinstance(raw, dict) else raw
    df = pd.json_normalize(records)

    # Datum
    if "date" in df.columns:
        try:
            df["date"] = pd.to_datetime(df["date"])
        except Exception:
            df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d", errors="coerce")

    # KNMI schalen
    def scale(src, dst, div=10):
        if src in df.columns:
            df[dst] = pd.to_numeric(df[src], errors="coerce") / div

    scale("TN", "TN_C")
    scale("TG", "TG_C")
    scale("TX", "TX_C")
    scale("RH", "RH_mm")
    scale("SQ", "SQ_h")

    if "FG" in df.columns:
        df["FG_ms"] = pd.to_numeric(df["FG"], errors="coerce") / 10.0
    if "DDVEC" in df.columns:
        df["DDVEC"] = pd.to_numeric(df["DDVEC"], errors="coerce")

    if "date" in df.columns:
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month

        def season(m):
            if m in [12, 1, 2]:
                return "winter"
            if m in [3, 4, 5]:
                return "lente"
            if m in [6, 7, 8]:
                return "zomer"
            return "herfst"

        df["season"] = df["month"].apply(season)

    return df


STATIONS_META = {
    "amsterdam":  {"name": "Amsterdam",  "lat": 52.3676, "lon": 4.9041},
    "de_bilt":    {"name": "De Bilt",    "lat": 52.1010, "lon": 5.1790},
    "eelde":      {"name": "Eelde",      "lat": 53.1250, "lon": 6.5833},
    "eindhoven":  {"name": "Eindhoven",  "lat": 51.4500, "lon": 5.3740},
    "maastricht": {"name": "Maastricht", "lat": 50.8510, "lon": 5.6910},
    "twente":     {"name": "Twente",     "lat": 52.2700, "lon": 6.9000},
    "vlissingen": {"name": "Vlissingen", "lat": 51.4420, "lon": 3.5730},
}

@st.cache_data
def discover_files():
    pat = re.compile(
        r"^(amsterdam|de_bilt|eelde|eindhoven|ijmuiden|maastricht|twente|vlissingen)_(\d{4}_\d{4})\.json$",
        re.I
    )
    out = []
    for f in sorted(Path(".").glob("*.json")):
        m = pat.match(f.name)
        if m:
            skey = m.group(1).lower()
            if skey in EXCLUDED_STATIONS:
                continue
            out.append((skey, m.group(2), str(f)))
    return out


@st.cache_data
def build_dataset(selected_periods: tuple, selected_stations: tuple) -> pd.DataFrame:
    frames = []
    for skey, period, path in discover_files():
        if selected_periods and period not in selected_periods:
            continue
        if selected_stations and skey not in selected_stations:
            continue
        dfp = load_data(path)
        if dfp.empty:
            continue
        dfp["station_key"] = skey
        dfp["station"] = STATIONS_META[skey]["name"]
        dfp["period"] = period
        for c in ["TN_C", "TG_C", "TX_C", "RH_mm", "SQ_h", "FG_ms", "DDVEC"]:
            if c in dfp.columns:
                dfp[c] = pd.to_numeric(dfp[c], errors="coerce")
        frames.append(dfp)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def selection_controls(key_prefix: str = ""):
    found = discover_files()
    all_periods = sorted({p for _, p, _ in found})
    all_station_keys = [k for k in STATIONS_META.keys() if k not in EXCLUDED_STATIONS]

    st.subheader("⚙️ Filters")
    c1, c2 = st.columns([2, 3])
    with c1:
        sel_periods = st.multiselect(
            "📅 Jaarperiodes",
            options=all_periods,
            default=all_periods,
            key=f"{key_prefix}_periods",
        )
    with c2:
        mode = st.selectbox(
            "📍 Locatiekeuze",
            ["Alle locaties (geaggregeerd)", "Enkele locatie", "Vergelijk locaties"],
            index=0,
            key=f"{key_prefix}_mode",
        )

    if mode == "Enkele locatie":
        station = st.selectbox(
            "Station",
            options=all_station_keys,
            format_func=lambda k: STATIONS_META[k]["name"],
            key=f"{key_prefix}_single",
        )
        stations = [station]
    elif mode == "Vergelijk locaties":
        stations = st.multiselect(
            "Stations",
            options=all_station_keys,
            default=["amsterdam", "de_bilt"],
            format_func=lambda k: STATIONS_META[k]["name"],
            key=f"{key_prefix}_multi",
        )
        if not stations:
            st.warning("Kies minimaal één station.")
    else:
        stations = all_station_keys

    df_all = build_dataset(tuple(sel_periods), tuple(stations))
    return df_all, mode, tuple(sel_periods), tuple(stations)


# =========================================================
# Windroos component
# =========================================================
def render_windrose(raw_df: pd.DataFrame, *, title="🧭 Windroos", facet_per_station=False):
    st.subheader(title)
    if raw_df.empty:
        st.info("Geen data voor de windroos.")
        return
    if not {"DDVEC", "FG_ms"}.issubset(raw_df.columns):
        st.warning("Windroos niet mogelijk: kolommen ontbreken.")
        return

    w = raw_df[["station", "DDVEC", "FG_ms"]].dropna()
    if w.empty:
        st.info("Geen geldige winddata om te tonen.")
        return

    w["DDVEC"] = pd.to_numeric(w["DDVEC"], errors="coerce") % 360
    w["FG_ms"] = pd.to_numeric(w["FG_ms"], errors="coerce")

    dir_bin = st.selectbox("Richtingsbin (°)", [10, 15, 20, 30, 45], index=3)
    bins_text = st.text_input("Snelheidsklassen m/s (komma-gescheiden)", value="0,2,4,6,8,10,12,20")
    try:
        speed_bins = sorted({float(x.strip()) for x in bins_text.split(",") if x.strip()})
        if len(speed_bins) < 2:
            raise ValueError
    except Exception:
        speed_bins = [0, 2, 4, 6, 8, 10, 12, 20]
        st.warning("Kon de snelheidsklassen niet parsen; standaard gebruikt.")
    normalize = st.selectbox("Normalisatie", ["% van totaal", "% per richting", "Aantal (ruw)"], index=0)

    n_bins = int(360 / dir_bin)
    w["dir_bin_idx"] = (np.floor(w["DDVEC"] / dir_bin).astype(int)) % n_bins
    dir_labels = [f"{k*dir_bin}–{(k+1)*dir_bin}°" for k in range(n_bins)]
    w["dir_bin"] = w["dir_bin_idx"].map(lambda k: dir_labels[k])

    speed_labels = [f"{speed_bins[i]}–{speed_bins[i+1]} m/s" for i in range(len(speed_bins) - 1)]
    w["speed_bin"] = pd.cut(w["FG_ms"], bins=speed_bins, labels=speed_labels, include_lowest=True, right=False)

    agg = (
        w.groupby(["station", "dir_bin_idx", "dir_bin", "speed_bin"], as_index=False)
         .size()
         .rename(columns={"size": "count"})
    )
    if agg.empty:
        st.info("Geen data binnen de gekozen bins.")
        return

    if normalize == "% van totaal":
        total = agg.groupby("station")["count"].transform("sum")
        agg["value"] = 100 * agg["count"] / total
        r_title, tick_suffix = "Frequentie", "%"
    elif normalize == "% per richting":
        dir_tot = agg.groupby(["station", "dir_bin_idx"])["count"].transform("sum")
        agg["value"] = 100 * agg["count"] / dir_tot
        r_title, tick_suffix = "Aandeel binnen richting", "%"
    else:
        agg["value"] = agg["count"]
        r_title, tick_suffix = "Aantal", ""

    facets = {"facet_row": "station"} if facet_per_station and agg["station"].nunique() > 1 else {}
    fig = px.bar_polar(
        agg,
        r="value",
        theta="dir_bin",
        color="speed_bin",
        barmode="stack",
        **facets
    )
    fig.update_layout(
        polar=dict(
            angularaxis=dict(direction="clockwise", rotation=90),
            radialaxis=dict(title=r_title, ticksuffix=tick_suffix),
        ),
        legend_title_text="Snelheid (m/s)"
    )
    st.plotly_chart(fig, use_container_width=True)


# =========================================================
# Sidebar
# =========================================================
st.sidebar.title("Navigatie")
page = st.sidebar.radio(
    "Ga naar",
    ["Overzicht", "Temperatuur Trends", "Neerslag & Zon", "Windtrends & Topdagen", "Correlaties", "Voorspellingsmodel"],
)

# =========================================================
# Pagina’s
# =========================================================
if page == "Overzicht":
    st.header("🗺️ Overzicht — Kaart met temperatuur, neerslag en zonuren")
    df, mode, sel_periods, sel_stations = selection_controls("ov")
    if df.empty:
        st.warning("Geen data.")
        st.stop()
    map_var = st.selectbox("Variabele", ["TG_C", "RH_mm", "SQ_h"],
                           format_func=lambda k: {"TG_C": "🌡️ Temperatuur (°C)", "RH_mm": "🌧️ Neerslag (mm)", "SQ_h": "☀️ Zonuren (uur)"}[k])
    agg_df = df.groupby(["station_key", "station"], as_index=False).agg({map_var: "mean"})
    agg_df["lat"] = agg_df["station_key"].map(lambda k: STATIONS_META[k]["lat"])
    agg_df["lon"] = agg_df["station_key"].map(lambda k: STATIONS_META[k]["lon"])
    fig_map = px.scatter_mapbox(agg_df, lat="lat", lon="lon", color=map_var,
                                hover_name="station", color_continuous_scale="RdYlBu_r", zoom=6)
    fig_map.update_layout(mapbox_style="open-street-map", margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_map, use_container_width=True)

elif page == "Temperatuur Trends":
    st.header("🌡️ Temperatuur Trends — dag • maand • seizoen")
    df, mode, _, _ = selection_controls("temp")
    if df.empty:
        st.info("Geen data.")
        st.stop()

    # Dag
    use_cols = [c for c in ["TN_C", "TG_C", "TX_C"] if c in df.columns]
    if use_cols:
        m = df.melt(["date", "station"], value_vars=use_cols, var_name="type", value_name="temp_C")
        label_map = {"TN_C": "Min", "TG_C": "Gem", "TX_C": "Max"}
        m["type"] = m["type"].replace(label_map)
        fig = px.line(m, x="date", y="temp_C", color="type", title="Dagelijkse temperatuur",
                      labels={"temp_C": "Temperatuur (°C)", "date": "Datum", "type": "Reeks"})
        st.plotly_chart(fig, use_container_width=True)

    # Maand
    df["month_name"] = df["date"].dt.month_name()
    fig_box = px.box(df, x="month_name", y="TG_C", title="Gemiddelde temperatuur per maand",
                     labels={"month_name": "Maand", "TG_C": "Temperatuur (°C)"})
    st.plotly_chart(fig_box, use_container_width=True)

    # Seizoen
    season_temp = df.groupby("season")["TG_C"].mean().reset_index()
    fig_season = px.bar(season_temp, x="season", y="TG_C", title="Gemiddelde temperatuur per seizoen",
                        labels={"season": "Seizoen", "TG_C": "Gemiddelde temperatuur (°C)"})
    st.plotly_chart(fig_season, use_container_width=True)

elif page == "Neerslag & Zon":
    st.header("☔ Neerslag & Zon — relaties")
    df, mode, _, _ = selection_controls("rain")
    if df.empty:
        st.info("Geen data.")
        st.stop()
    if "RH_mm" in df.columns and "SQ_h" in df.columns:
        bins = [0, 1, 5, 10, 50]
        df["rain_cat"] = pd.cut(df["RH_mm"], bins=bins)
        fig = px.box(df, x="rain_cat", y="SQ_h", title="Zonuren per regenhoeveelheid",
                     labels={"rain_cat": "Neerslagcategorie (mm)", "SQ_h": "Zonuren (uur)"})
        st.plotly_chart(fig, use_container_width=True)

elif page == "Windtrends & Topdagen":
    st.header("💨 Windtrends & Topdagen — windroos en verdelingen")
    df, mode, sel_periods, sel_stations = selection_controls("wind")
    if df.empty:
        st.info("Geen data.")
        st.stop()
    raw = build_dataset(sel_periods, sel_stations)
    render_windrose(raw, title="🧭 Windroos", facet_per_station=(mode == "Vergelijk locaties"))

elif page == "Correlaties":
    st.header("🔗 Correlaties — begrijpelijke namen")
    df, mode, _, _ = selection_controls("corr")
    if df.empty:
        st.info("Geen data.")
        st.stop()
    nice = {
        "TN_C": "Minimum temperatuur (°C)",
        "TG_C": "Gemiddelde temperatuur (°C)",
        "TX_C": "Maximum temperatuur (°C)",
        "RH_mm": "Neerslag (mm)",
        "SQ_h": "Zonuren (uur)",
        "FG_ms": "Windsnelheid (m/s)",
    }
    vars_use = [c for c in nice if c in df.columns]
    corr = df[vars_use].corr().round(2)
    corr.index = [nice[c] for c in corr.index]
    corr.columns = [nice[c] for c in corr.columns]
    fig = px.imshow(corr, text_auto=True, aspect="auto", labels=dict(color="Correlatie"))
    st.plotly_chart(fig, use_container_width=True)
