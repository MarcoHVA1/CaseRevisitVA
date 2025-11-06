import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

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
        g = (df_all[keep_cols + ["station"]]
                .groupby("date", as_index=False)
                .agg({c: "mean" for c in num_cols}))
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
    # Fallback: hanteer lege KPIs
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
# PAGE 1: Overzicht (ongewijzigde logica maar altijd alle JSON's + periodekeuze)
# -----------------------------------------------------------------------------
if page == "Overzicht":
    st.header("🗺️ Interactieve kaart • Temperatuur, Neerslag & Zonuren")
    st.caption("Kies jaarperiodes en variabelen. De kaart toont waarden per KNMI-station; onderaan zie je de correlatie tussen twee variabelen.")

    # 1) JSON-bestanden detecteren
    found = discover_files()
    if not found:
        st.warning("Geen JSON-data gevonden.")
        st.stop()

    # 2) Filters (jaarperiodes bovenaan de pagina)
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

    # Numeriek maken
    for c in ["TG_C", "RH_mm", "SQ_h"]:
        if c in df_all.columns:
            df_all[c] = pd.to_numeric(df_all[c], errors="coerce")

    # Aggregatie per station
    agg_df = (
        df_all.groupby(["station_key", "station"], as_index=False)
        .agg({map_var: agg_func, "TG_C": "mean", "RH_mm": "sum", "SQ_h": "sum"})
    )

    # Coördinaten
    agg_df["lat"] = agg_df["station_key"].map(lambda k: STATIONS_META[k]["lat"])
    agg_df["lon"] = agg_df["station_key"].map(lambda k: STATIONS_META[k]["lon"])

    agg_df = agg_df.replace([np.inf, -np.inf], np.nan)
    # Toon waarschuwing als een station geen waarden heeft voor de gekozen variabele
    missing_stations = []
    for sk in STATIONS_META.keys():
        if sk not in set(agg_df["station_key"]):
            missing_stations.append(STATIONS_META[sk]["name"])
    agg_df = agg_df.dropna(subset=["lat", "lon", map_var])

    if missing_stations:
        st.caption("⚠️ Geen waarde voor gekozen variabele bij: " + ", ".join(missing_stations))

    if agg_df.empty:
        st.info("Geen geldige waarden om op de kaart te tonen voor de gekozen filters.")
    else:
        color_scale = "RdYlBu_r" if map_var == "TG_C" else ("Blues" if map_var == "RH_mm" else "YlOrBr")
        map_title = {"TG_C": "Temperatuur (°C)", "RH_mm": "Neerslag (mm)", "SQ_h": "Zonuren (h)"}[map_var]

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

# -----------------------------------------------------------------------------
# PAGE 2: Temperatuur Trends (+ heatmap toegevoegd + locatie/vergelijk/alle + periode)
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
        labels={"month_name": "Maand", "TG_C": "Gemiddelde temperatuur (°C)"}
    )
    fig_box.update_traces(line_width=2)
    st.plotly_chart(fig_box, use_container_width=True)

    # Gemiddelde temperatuur per seizoen
    group_cols = ["season"]
    if mode == "Vergelijk locaties":
        group_cols.insert(0, "station")
    season_temp = df.groupby(group_cols)["TG_C"].mean().reset_index()
    if mode != "Vergelijk locaties":
        season_temp.rename(columns={"TG_C": "Gem_TG_C"}, inplace=True)
        fig_season = px.bar(
            season_temp, x="season", y="Gem_TG_C", color="season",
            title="🌦️ Gemiddelde temperatuur per seizoen",
            labels={"season": "Seizoen", "Gem_TG_C": "Gemiddelde Temp (°C)"}
        )
    else:
        season_temp.rename(columns={"TG_C": "Gem_TG_C"}, inplace=True)
        fig_season = px.bar(
            season_temp, x="season", y="Gem_TG_C", color="station", barmode="group",
            title="🌦️ Gemiddelde temperatuur per seizoen (per station)",
            labels={"season": "Seizoen", "Gem_TG_C": "Gemiddelde Temp (°C)", "station": "Station"}
        )
    st.plotly_chart(fig_season, use_container_width=True)

    # 📅 Kalender-heatmap (verplaatst vanuit Windtrends & Topdagen)
    st.subheader("📅 Kalender-heatmap: gemiddelde temperatuur per dag")
    # Voor vergelijken tonen we tabs per station
    if mode == "Vergelijk locaties":
        tabs = st.tabs(sorted(df["station"].unique()))
        for tab, st_name in zip(tabs, sorted(df["station"].unique())):
            with tab:
                d = df[df["station"] == st_name].copy()
                d["day"] = d["date"].dt.day
                d["month"] = d["date"].dt.month
                pivot = d.pivot_table(index="month", columns="day", values="TG_C", aggfunc="mean")
                month_names_map = {
                    1: "Januari", 2: "Februari", 3: "Maart", 4: "April",
                    5: "Mei", 6: "Juni", 7: "Juli", 8: "Augustus",
                    9: "September", 10: "Oktober", 11: "November", 12: "December"
                }
                pivot.index = pivot.index.map(month_names_map)
                fig_heatmap = px.imshow(
                    pivot, color_continuous_scale="RdBu_r", origin="upper", aspect="auto",
                    labels=dict(color="Temperatuur (°C)", x="Dag van de maand", y="Maand")
                )
                fig_heatmap.update_xaxes(title="Dag van de maand", tickmode="linear")
                fig_heatmap.update_yaxes(title="Maand", tickmode="array",
                                         tickvals=list(pivot.index), ticktext=list(pivot.index))
                st.plotly_chart(fig_heatmap, use_container_width=True)
    else:
        d = df.copy()
        d["day"] = d["date"].dt.day
        d["month"] = d["date"].dt.month
        pivot = d.pivot_table(index="month", columns="day", values="TG_C", aggfunc="mean")
        month_names_map = {
            1: "Januari", 2: "Februari", 3: "Maart", 4: "April",
            5: "Mei", 6: "Juni", 7: "Juli", 8: "Augustus",
            9: "September", 10: "Oktober", 11: "November", 12: "December"
        }
        pivot.index = pivot.index.map(month_names_map)
        fig_heatmap = px.imshow(
            pivot, color_continuous_scale="RdBu_r", origin="upper", aspect="auto",
            labels=dict(color="Temperatuur (°C)", x="Dag van de maand", y="Maand")
        )
        fig_heatmap.update_xaxes(title="Dag van de maand", tickmode="linear")
        fig_heatmap.update_yaxes(title="Maand", tickmode="array",
                                 tickvals=list(pivot.index), ticktext=list(pivot.index))
        st.plotly_chart(fig_heatmap, use_container_width=True)

# -----------------------------------------------------------------------------
# PAGE 3: Neerslag & Zon (met locatie/vergelijk/alle + periode)
# -----------------------------------------------------------------------------
elif page == "Neerslag & Zon":
    st.header("☔ Neerslag vs. Zon")
    df, mode = selection_controls(key_prefix="rain_sun")
    if df.empty:
        st.info("Geen data beschikbaar voor de gekozen filters.")
        st.stop()

    if "RH_mm" in df.columns and "SQ_h" in df.columns:
        bins = [0, 1, 5, 10, 50]
        labels = ["0 mm", "0–5 mm", "5–10 mm", "10+ mm"]
        df["rain_cat"] = pd.cut(df["RH_mm"], bins=bins, labels=labels, include_lowest=True)
        ordered_cats = ["0 mm", "0–5 mm", "5–10 mm", "10+ mm"]
        df["rain_cat"] = pd.Categorical(df["rain_cat"], categories=ordered_cats, ordered=True)

        fig_box = px.box(
            df, x="rain_cat", y="SQ_h",
            color=("station" if mode == "Vergelijk locaties" else "rain_cat"),
            category_orders={"rain_cat": ordered_cats},
            title="📦 Verdeling zonuren per neerslagcategorie",
            labels={"SQ_h": "Zonuren", "rain_cat": "Neerslagcategorie"},
            points="all"
        )
        st.plotly_chart(fig_box, use_container_width=True)

        # Gemiddelde temperatuur bij toenemende regenval
        rain_bins = pd.cut(
            df["RH_mm"], bins=[0, 1, 5, 10, 20, 50], include_lowest=True,
            labels=["0–1 mm", "1–5 mm", "5–10 mm", "10–20 mm", "20+ mm"]
        )
        avg_temp_rain = df.groupby([rain_bins] + (["station"] if mode == "Vergelijk locaties" else []) )["TG_C"].mean().reset_index()
        avg_temp_rain.rename(columns={"RH_mm": "Neerslag"}, inplace=True)

        fig_temp_rain = px.bar(
            avg_temp_rain, x="RH_mm", y="TG_C",
            color=("station" if mode == "Vergelijk locaties" else None),
            barmode=("group" if mode == "Vergelijk locaties" else "relative"),
            title="🌧️ Gemiddelde temperatuur bij toenemende regenval",
            labels={"RH_mm": "Neerslagcategorie (mm per dag)", "TG_C": "Gemiddelde temperatuur (°C)", "station": "Station"},
            text_auto=".1f",
            color_continuous_scale="RdYlBu_r"
        )
        fig_temp_rain.update_layout(showlegend=True)
        st.plotly_chart(fig_temp_rain, use_container_width=True)

# -----------------------------------------------------------------------------
# PAGE 4: Windtrends & Topdagen (met locatie/vergelijk/alle + periode)
# -----------------------------------------------------------------------------
elif page == "Windtrends & Topdagen":
    st.header("📊 Windtrends & Topdagen")
    df, mode = selection_controls(key_prefix="wind")
    if df.empty:
        st.info("Geen data beschikbaar voor de gekozen filters.")
        st.stop()

    # 🧭 Interactieve windroos
    if "FG_ms" in df.columns and "DDVEC" in df.columns:
        st.subheader("🧭 Interactieve windroos")
        w = df[["station", "DDVEC", "FG_ms"]].dropna().copy()
        w["DDVEC"] = pd.to_numeric(w["DDVEC"], errors="coerce") % 360
        w["FG_ms"] = pd.to_numeric(w["FG_ms"], errors="coerce")
        w = w.dropna()
        if w.empty:
            st.info("Geen geldige winddata om te tonen.")
        else:
            colA, colB, colC = st.columns(3)
            with colA:
                dir_bin = st.selectbox("Richtingsbin (°)", [10, 15, 20, 30, 45], index=3)
            with colB:
                bins_text = st.text_input("Snelheidsklassen m/s (komma-gescheiden)", value="0,2,4,6,8,10,12,20")
                try:
                    speed_bins = sorted({float(x.strip()) for x in bins_text.split(",") if x.strip() != ""})
                    if len(speed_bins) < 2:
                        raise ValueError
                except Exception:
                    speed_bins = [0, 2, 4, 6, 8, 10, 12, 20]
                    st.warning("Kon de snelheidsklassen niet parsen; standaard gebruikt.")
            with colC:
                normalize = st.selectbox("Normalisatie", ["% van totaal", "% per richting", "Aantal (ruw)"], index=0)

            n_bins = int(360 / dir_bin)
            sector_idx = ((w["DDVEC"] // dir_bin).astype(int)) % n_bins
            w["dir_bin_idx"] = sector_idx
            dir_labels = [f"{k*dir_bin}–{(k+1)*dir_bin}°" for k in range(n_bins)]
            w["dir_bin"] = w["dir_bin_idx"].map(lambda k: dir_labels[k])

            speed_labels = [f"{speed_bins[i]}–{speed_bins[i+1]} m/s" for i in range(len(speed_bins)-1)]
            w["speed_bin"] = pd.cut(w["FG_ms"], bins=speed_bins, labels=speed_labels, include_lowest=True, right=False)

            agg = (
                w.dropna(subset=["dir_bin", "speed_bin"]) \
                 .groupby(["station", "dir_bin_idx", "dir_bin", "speed_bin"], as_index=False) \
                 .size() \
                 .rename(columns={"size": "count"})
            )

            if agg.empty:
                st.info("Geen data binnen de gekozen bins.")
            else:
                if normalize == "% van totaal":
                    total = agg.groupby("station")["count"].transform("sum")
                    agg["value"] = np.where(total > 0, 100.0 * agg["count"] / total, 0.0)
                    r_title, tick_suffix = "Frequentie (%)", "%"
                elif normalize == "% per richting":
                    dir_tot = agg.groupby(["station", "dir_bin_idx"])["count"].transform("sum")
                    agg["value"] = np.where(dir_tot > 0, 100.0 * agg["count"] / dir_tot, 0.0)
                    r_title, tick_suffix = "Aandeel binnen richting (%)", "%"
                else:
                    agg["value"] = agg["count"]
                    r_title, tick_suffix = "Aantal", ""

                agg = agg.sort_values(["station", "dir_bin_idx"]).reset_index(drop=True)

                fig_windrose = px.bar_polar(
                    agg,
                    r="value",
                    theta="dir_bin",
                    color="speed_bin",
                    barmode="stack",
                    facet_row=("station" if mode == "Vergelijk locaties" else None),
                    hover_data={"count": True, "value": True, "dir_bin_idx": False}
                )
                fig_windrose.update_layout(
                    polar=dict(
                        angularaxis=dict(direction="clockwise", rotation=90, categoryorder="array", categoryarray=dir_labels),
                        radialaxis=dict(title=r_title, ticksuffix=tick_suffix)
                    ),
                    margin=dict(l=0, r=0, t=40, b=0),
                    legend_title_text="Snelheid (m/s)"
                )
                st.plotly_chart(fig_windrose, use_container_width=True)

    # Boxplot windsnelheid per seizoen
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
        fig_box = px.box(
            df,
            x="season_box",
            y="FG_ms",
            color=("station" if mode == "Vergelijk locaties" else "season_box"),
            category_orders={"season_box": season_order},
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

# -----------------------------------------------------------------------------
# PAGE 5: Correlaties (NIEUW)
# -----------------------------------------------------------------------------
elif page == "Correlaties":
    st.header("🔗 Correlaties tussen variabelen")
    df, mode = selection_controls(key_prefix="corr")
    if df.empty:
        st.info("Geen data beschikbaar voor de gekozen filters.")
        st.stop()

    vars_use = [c for c in ["TN_C", "TG_C", "TX_C", "RH_mm", "SQ_h", "FG_ms"] if c in df.columns]
    if not vars_use:
        st.info("Geen geschikte variabelen gevonden voor correlatie.")
        st.stop()

    # Correlatiematrix (Pearson)
    st.subheader("📐 Correlatiematrix (Pearson)")
    if mode == "Vergelijk locaties":
        tabs = st.tabs(sorted(df["station"].unique()))
        for tab, st_name in zip(tabs, sorted(df["station"].unique())):
            with tab:
                cdf = df[df["station"] == st_name][vars_use]
                corr = cdf.corr().round(2)
                fig = px.imshow(corr, text_auto=True, aspect="auto", origin="upper")
                st.plotly_chart(fig, use_container_width=True)
    else:
        corr = df[vars_use].corr().round(2)
        fig = px.imshow(corr, text_auto=True, aspect="auto", origin="upper")
        st.plotly_chart(fig, use_container_width=True)

    # Scatter-matrix
    st.subheader("📊 Scatter matrix")
    sm = px.scatter_matrix(df, dimensions=vars_use, color=("station" if mode == "Vergelijk locaties" else None), height=700)
    st.plotly_chart(sm, use_container_width=True)

# -----------------------------------------------------------------------------
# PAGE 6: Voorspellingsmodel (inclusief IJmuiden)
# -----------------------------------------------------------------------------
elif page == "Voorspellingsmodel":
    st.header("🧠 Voorspellingsmodel — voorspelde temperatuur in Nederland")
    st.caption("Pas **maand**, **dag**, **neerslag** en **windsnelheid** aan. Het model voorspelt de **temperatuur (°C)** per station op basis van historische patronen. Alle stations incl. IJmuiden worden automatisch meegenomen.")

    # === Alle JSON-bestanden automatisch inlezen ===
    found = discover_files()
    if not found:
        st.warning("Geen JSON-data gevonden (bijv. 'ijmuiden_2023_2024.json').")
        st.stop()

    # Periodekeuze bovenaan
    all_periods = sorted({p for _, p, _ in found})
    sel_periods = st.multiselect("📅 Kies jaarperiodes:", all_periods, default=all_periods, key="model_periods")

    # === Alle stations samenvoegen ===
    frames = []
    for station_key, period, path_str in found:
        if period not in sel_periods:
            continue
        dfp = load_data(path_str)
        if "date" not in dfp.columns:
            continue
        for c in ["TG_C", "RH_mm", "FG_ms"]:
            if c not in dfp.columns:
                dfp[c] = np.nan
            dfp[c] = pd.to_numeric(dfp[c], errors="coerce")
        dfp["station_key"] = station_key
        dfp["station"] = STATIONS_META[station_key]["name"]
        frames.append(dfp)

    df_all = pd.concat(frames, ignore_index=True)
    df_all = df_all.dropna(subset=["TG_C"]).copy()

    # === Datumfeatures (dag van jaar → sin/cos) ===
    df_all["doy"] = df_all["date"].dt.dayofyear
    df_all["doy"] = df_all["doy"].fillna(df_all["doy"].median())
    df_all["doy_sin"] = np.sin(2 * np.pi * df_all["doy"] / 366.0)
    df_all["doy_cos"] = np.cos(2 * np.pi * df_all["doy"] / 366.0)

    # === Eenvoudig lineair model per station ===
    def fit_linear(X, y):
        # Kies dynamisch beschikbare features (fallback: alleen seizoenscomponenten)
        candidate_features = ["RH_mm", "FG_ms", "doy_sin", "doy_cos"]
        features = [c for c in candidate_features if c in X.columns and not X[c].isna().all()]
        # Zorg dat seizoensfeatures er altijd in zitten
        for c in ["doy_sin", "doy_cos"]:
            if c not in features and c in X.columns:
                features.append(c)
        if not features or len(features) < 2:  # minimaal sin & cos
            return None
        X_ = np.column_stack([np.ones(len(X))] + [X[c].values for c in features])
        mask = ~np.isnan(X_).any(axis=1) & ~np.isnan(y.values)
        Xc, yc = X_[mask], y.values[mask]
        if len(yc) < 5:
            return None
        beta, *_ = np.linalg.lstsq(Xc, yc, rcond=None)
        yhat = Xc @ beta
        resid = yc - yhat
        rmse = float(np.sqrt(np.mean(resid**2)))
        ss_res = float(np.sum(resid**2))
        ss_tot = float(np.sum((yc - np.mean(yc))**2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        return {"beta": beta, "rmse": rmse, "r2": r2, "n": len(yc), "features": features}

    models = {}
    for sk, g in df_all.groupby("station_key"):
        m = fit_linear(g[[c for c in ["RH_mm", "FG_ms", "doy_sin", "doy_cos"] if c in g.columns]], g["TG_C"])
        if m is not None:
            models[sk] = m

    if not models:
        st.info("Onvoldoende data om een stationmodel te trainen.")
        st.stop()

    # === Gebruikersinvoer ===
    import calendar
    st.subheader("⚙️ Stel je omstandigheden in")
    col1, col2, col3 = st.columns(3)
    with col1:
        maand = st.slider("📆 Maand", 1, 12, 7)
        max_dag = calendar.monthrange(2024, maand)[1]
        dag = st.slider("📅 Dag", 1, int(max_dag), min(15, max_dag))
    with col2:
        pred_rain = st.slider("🌧️ Neerslag (mm/dag)", 0.0, 50.0, 0.0, 0.5)
    with col3:
        pred_wind = st.slider("💨 Windsnelheid (m/s)", 0.0, 15.0, 3.0, 0.5)

    # === Valideer dag en maand ===
    try:
        selected_date = pd.Timestamp(year=2024, month=int(maand), day=int(dag))
        doy = selected_date.dayofyear
        doy_sin = float(np.sin(2 * np.pi * doy / 366.0))
        doy_cos = float(np.cos(2 * np.pi * doy / 366.0))
    except ValueError:
        st.error(f"❌ Ongeldige datum: {dag} / {maand}. Controleer of deze dag in de gekozen maand voorkomt.")
        st.stop()

    # === Voorspellen per station ===
    rows = []
    for sk, m in models.items():
        beta = m["beta"]
        feats = ["const"] + m.get("features", ["RH_mm", "FG_ms", "doy_sin", "doy_cos"])  # volgorde overeenkomstig fit
        # Bouw feature vector in dezelfde volgorde
        feat_values = []
        for f in feats[1:]:
            if f == "RH_mm":
                feat_values.append(pred_rain)
            elif f == "FG_ms":
                feat_values.append(pred_wind)
            elif f == "doy_sin":
                feat_values.append(doy_sin)
            elif f == "doy_cos":
                feat_values.append(doy_cos)
            else:
                feat_values.append(0.0)
        xvec = np.array([1.0] + feat_values)
        pred_temp = float(xvec @ beta)
        rows.append({
            "station_key": sk,
            "station": STATIONS_META.get(sk, {}).get("name", sk),
            "lat": STATIONS_META.get(sk, {}).get("lat", np.nan),
            "lon": STATIONS_META.get(sk, {}).get("lon", np.nan),
            "pred_TG_C": float(pred_temp),
            "r2": m["r2"],
            "rmse": m["rmse"],
            "n": m["n"]
        })

    pred_df = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)

    # === Kaart (ZWART-WIT) met voorspelde temperatuur ===
    st.subheader("🗺️ Voorspelde temperatuur per station (°C)")
    plot_df = pred_df.dropna(subset=["lat", "lon", "pred_TG_C"]).copy()
    if plot_df.empty:
        st.info("Geen geldige stations om te tonen.")
    else:
        if plot_df["pred_TG_C"].max() > plot_df["pred_TG_C"].min():
            plot_df["size"] = (plot_df["pred_TG_C"] - plot_df["pred_TG_C"].min()) / (
                plot_df["pred_TG_C"].max() - plot_df["pred_TG_C"].min()
            )
        else:
            plot_df["size"] = 0.5
        plot_df["size"] = (plot_df["size"] * 25) + 6

        TEMP_SCALE_MIN = -5.0
        TEMP_SCALE_MAX = 30.0

        fig = px.scatter_mapbox(
            plot_df,
            lat="lat",
            lon="lon",
            color="pred_TG_C",
            size="size",
            color_continuous_scale="RdYlBu_r",
            range_color=[TEMP_SCALE_MIN, TEMP_SCALE_MAX],
            zoom=6,
            hover_name="station",
            hover_data={
                "pred_TG_C": True,
                "r2": True,
                "rmse": True,
                "n": True,
                "lat": False,
                "lon": False,
                "size": False
            },
            height=520
        )
        fig.update_layout(
            mapbox_style="carto-darkmatter",
            margin=dict(l=0, r=0, t=10, b=0),
            coloraxis_colorbar=dict(
                title="Voorspelde temperatuur",
                ticksuffix="°C",
                tickmode="auto"
            )
        )
        st.plotly_chart(fig, use_container_width=True)

        # === Tabel: voorspelde temperatuur (°C) ===
        st.subheader("📄 Tabel: voorspelde temperatuur (°C)")
        temp_tbl = (
            pred_df[["station", "pred_TG_C"]]
            .rename(columns={"station": "Station", "pred_TG_C": "Voorspelde Temp (°C)"})
            .assign(**{"Voorspelde Temp (°C)": lambda d: d["Voorspelde Temp (°C)"].round(1)})
            .sort_values("Voorspelde Temp (°C)", ascending=False)
            .reset_index(drop=True)
        )
        st.dataframe(temp_tbl, use_container_width=True)

        # === Samenvatting over alle stations ===
        st.subheader("📈 Gemiddelde modelprestatie in Nederland")
        avg_temp = pred_df["pred_TG_C"].mean()
        avg_r2 = pred_df["r2"].mean()
        avg_rmse = pred_df["rmse"].mean()
        summary_df = pd.DataFrame([{
            "Gem. voorspelde temperatuur (°C)": round(avg_temp, 1),
            "Gem. R² (verklaarde variantie)": round(avg_r2, 2),
            "Gem. standaardafwijking (°C)": round(avg_rmse, 1)
        }])
        st.table(summary_df)
