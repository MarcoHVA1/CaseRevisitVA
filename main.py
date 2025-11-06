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


# ---------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------
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

    # Geschaalde velden (KNMI tienden)
    def make_scaled(src, dest, divisor=10):
        if src in df.columns:
            df[dest] = pd.to_numeric(df[src], errors="coerce") / divisor

    make_scaled("TG", "TG_C")
    make_scaled("TN", "TN_C")
    make_scaled("TX", "TX_C")
    make_scaled("RH", "RH_mm")
    make_scaled("SQ", "SQ_h")

    # Wind (FG in tienden m/s) + richting
    if "FG" in df.columns:
        df["FG_ms"] = pd.to_numeric(df["FG"], errors="coerce") / 10.0
    if "DDVEC" in df.columns:
        df["DDVEC"] = pd.to_numeric(df["DDVEC"], errors="coerce")

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
    "vlissingen": {"name": "Vlissingen", "lat": 51.4420, "lon": 3.5730},
}


@st.cache_data
def discover_files():
    """
    Zoek alle JSON-bestanden per station/jaartal (case-insensitive),
    o.a. Ijmuiden_2021_2022.json, Ijmuiden_2022_2023.json, Ijmuiden_2023_2024.json
    """
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
    return found


@st.cache_data
def build_dataset(selected_periods: tuple, selected_stations: tuple):
    """Bouw één DataFrame vanuit de gekozen jaarperiodes + stations."""
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

        for c in ["TG_C", "TN_C", "TX_C", "RH_mm", "SQ_h", "FG_ms", "DDVEC"]:
            if c in dfp.columns:
                dfp[c] = pd.to_numeric(dfp[c], errors="coerce")

        dfp["station_key"] = station_key
        dfp["station"] = STATIONS_META[station_key]["name"]
        dfp["period"] = period
        frames.append(dfp)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def selection_controls(key_prefix: str = ""):
    """UI: jaarperiodes + locatiemodus per pagina."""
    found = discover_files()
    all_periods = sorted({p for _, p, _ in found})
    all_station_keys = list(STATIONS_META.keys())

    st.subheader("⚙️ Filters")
    col1, col2 = st.columns([2, 3])
    with col1:
        sel_periods = st.multiselect(
            "📅 Jaarperiodes",
            options=all_periods,
            default=all_periods,
            key=f"{key_prefix}_periods"
        )
    with col2:
        mode = st.selectbox(
            "📍 Locatiekeuze",
            ["Alle locaties (geaggregeerd)", "Enkele locatie", "Vergelijk locaties"],
            index=0,
            key=f"{key_prefix}_mode"
        )

    if mode == "Enkele locatie":
        station = st.selectbox(
            "Station",
            options=all_station_keys,
            format_func=lambda k: STATIONS_META[k]["name"],
            key=f"{key_prefix}_single"
        )
        stations = [station]
    elif mode == "Vergelijk locaties":
        stations = st.multiselect(
            "Stations",
            options=all_station_keys,
            default=["amsterdam", "ijmuiden"],  # IJmuiden standaard mee
            format_func=lambda k: STATIONS_META[k]["name"],
            key=f"{key_prefix}_multi"
        )
        if not stations:
            st.warning("Kies minimaal één station.")
    else:
        stations = all_station_keys  # alle stations (incl. IJmuiden)

    df_all = build_dataset(tuple(sel_periods), tuple(stations))

    # Aggregaatmodus: middelen per datum over alle gekozen stations
    if mode == "Alle locaties (geaggregeerd)" and not df_all.empty:
        num_cols = [c for c in ["TN_C", "TG_C", "TX_C", "RH_mm", "SQ_h", "FG_ms"] if c in df_all.columns]
        keep_cols = ["date"] + num_cols
        g = (
            df_all[keep_cols + ["station"]]
            .groupby("date", as_index=False)
            .agg({c: "mean" for c in num_cols})
        )
        # Herstel datumfeatures voor downstream groupby's
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


# ---------------------------------------------------------------------
# Windroos component (netjes ingesprongen, 4 spaties)
# ---------------------------------------------------------------------
def render_windrose(df: pd.DataFrame, *, titel="🧭 Windroos", vergelijk_per_station=False):
    """
    df: DataFrame met kolommen DDVEC (graden), FG_ms (m/s) en optioneel 'station'.
    """
    st.subheader(titel)

    if "DDVEC" not in df.columns or "FG_ms" not in df.columns:
        st.warning("Benodigd: kolommen 'DDVEC' (richting in °) en 'FG_ms' (m/s).")
        return

    w = df.copy()
    w["DDVEC"] = pd.to_numeric(w["DDVEC"], errors="coerce") % 360
    w["FG_ms"] = pd.to_numeric(w["FG_ms"], errors="coerce")
    if "station" not in w.columns:
        w["station"] = "Alle stations"
    w = w.dropna(subset=["DDVEC", "FG_ms"])

    if w.empty:
        st.info("Geen geldige winddata om te tonen.")
        return

    # UI-controls
    c1, c2, c3 = st.columns(3)
    with c1:
        dir_bin = st.selectbox("Richtingsbin (°)", [10, 15, 20, 30, 45], index=3)
    with c2:
        bins_text = st.text_input("Snelheidsklassen m/s (komma-gescheiden)", value="0,2,4,6,8,10,12,20")
        try:
            speed_bins = sorted({float(x.strip()) for x in bins_text.split(",") if x.strip() != ""})
            if len(speed_bins) < 2:
                raise ValueError
        except Exception:
            speed_bins = [0, 2, 4, 6, 8, 10, 12, 20]
            st.warning("Kon de snelheidsklassen niet parsen; standaard gebruikt.")
    with c3:
        normalize = st.selectbox("Normalisatie", ["% van totaal", "% per richting", "Aantal (ruw)"], index=0)

    # Binning richting & snelheid
    n_bins = int(360 / dir_bin)
    sector_idx = (np.floor(w["DDVEC"] / dir_bin).astype(int)) % n_bins
    w["dir_bin_idx"] = sector_idx
    dir_labels = [f"{k*dir_bin}–{(k+1)*dir_bin}°" for k in range(n_bins)]
    w["dir_bin"] = w["dir_bin_idx"].map(lambda k: dir_labels[k])

    speed_labels = [f"{speed_bins[i]}–{speed_bins[i+1]} m/s" for i in range(len(speed_bins) - 1)]
    w["speed_bin"] = pd.cut(w["FG_ms"], bins=speed_bins, labels=speed_labels, include_lowest=True, right=False)

    agg = (
        w.dropna(subset=["dir_bin", "speed_bin"])
         .groupby(["station", "dir_bin_idx", "dir_bin", "speed_bin"], as_index=False)
         .size()
         .rename(columns={"size": "count"})
    )
    if agg.empty:
        st.info("Geen data binnen de gekozen bins.")
        return

    # Normalisatie
    if normalize == "% van totaal":
        total = agg.groupby("station")["count"].transform("sum")
        agg["value"] = np.where(total > 0, 100.0 * agg["count"] / total, 0.0)
        r_title, tick_suffix = "Frequentie", "%"
    elif normalize == "% per richting":
        dir_tot = agg.groupby(["station", "dir_bin_idx"])["count"].transform("sum")
        agg["value"] = np.where(dir_tot > 0, 100.0 * agg["count"] / dir_tot, 0.0)
        r_title, tick_suffix = "Aandeel binnen richting", "%"
    else:
        agg["value"] = agg["count"]
        r_title, tick_suffix = "Aantal", ""

    agg = agg.sort_values(["station", "dir_bin_idx"]).reset_index(drop=True)

    # Plot
    facets = {"facet_row": "station"} if vergelijk_per_station and agg["station"].nunique() > 1 else {}
    fig = px.bar_polar(
        agg,
        r="value",
        theta="dir_bin",
        color="speed_bin",
        barmode="stack",
        hover_data={"count": True, "value": True, "dir_bin_idx": False},
        **facets
    )
    fig.update_layout(
        polar=dict(
            angularaxis=dict(
                direction="clockwise",
                rotation=90,
                categoryorder="array",
                categoryarray=dir_labels
            ),
            radialaxis=dict(title=r_title, ticksuffix=tick_suffix)
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        legend_title_text="Snelheid (m/s)",
        title="🧭 Windroos"
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------
# Sidebar (navigation ONLY)
# ---------------------------------------------------------------------
st.sidebar.title("Navigatie")
page = st.sidebar.radio(
    "Ga naar",
    [
        "Overzicht",
        "Temperatuur Trends",
        "Neerslag & Zon",
        "Windtrends & Topdagen",
        "Correlaties",
        "Voorspellingsmodel",
    ],
)


# ---------------------------------------------------------------------
# KPI's over alle data
# ---------------------------------------------------------------------
_found_kpi = discover_files()
_all_periods_kpi = sorted({p for _, p, _ in _found_kpi})
_all_station_keys_kpi = list(STATIONS_META.keys())
df_kpi = build_dataset(tuple(_all_periods_kpi), tuple(_all_station_keys_kpi))

kpi1, kpi2, kpi3 = st.columns(3)
if not df_kpi.empty:
    kpi1.metric("🌡️ Gemiddelde Temp (°C)", round(df_kpi["TG_C"].mean(), 1) if "TG_C" in df_kpi else "—")
    kpi2.metric("🌧️ Totale Neerslag (mm)", round(df_kpi["RH_mm"].sum(), 1) if "RH_mm" in df_kpi else "—")
    kpi3.metric("☀️ Totale Zonuren", round(df_kpi["SQ_h"].sum(), 1) if "SQ_h" in df_kpi else "—")
else:
    kpi1.metric("🌡️ Gemiddelde Temp (°C)", "—")
    kpi2.metric("🌧️ Totale Neerslag (mm)", "—")
    kpi3.metric("☀️ Totale Zonuren", "—")


# ---------------------------------------------------------------------
# PAGE 1: Overzicht
# ---------------------------------------------------------------------
if page == "Overzicht":
    st.header("🗺️ Overzicht — Kaart met temperatuur, neerslag en zonuren")
    st.caption("Kies jaarperiodes en variabelen. Stations zonder waarde worden grijs getoond (incl. IJmuiden).")

    found = discover_files()
    if not found:
        st.warning("Geen JSON-data gevonden.")
        st.stop()

    all_periods = sorted({p for _, p, _ in found})
    sel_periods = st.multiselect("📅 Jaarperiodes", all_periods, default=all_periods, key="ov_periods")

    month_names = [
        "Alle", "01 - Januari", "02 - Februari", "03 - Maart", "04 - April",
        "05 - Mei", "06 - Juni", "07 - Juli", "08 - Augustus",
        "09 - September", "10 - Oktober", "11 - November", "12 - December",
    ]
    colA, colB, colC = st.columns([1, 1, 1])
    sel_month = colA.selectbox("📆 Maand", month_names)

    map_var = colB.selectbox(
        "🗺️ Variabele op kaart",
        ["TG_C", "RH_mm", "SQ_h"],
        format_func=lambda k: {"TG_C": "🌡️ Temperatuur (°C)", "RH_mm": "🌧️ Neerslag (mm)", "SQ_h": "☀️ Zonuren (h)"}[k],
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

    # Toon altijd alle stations (incl. IJmuiden)
    stations_full = pd.DataFrame([
        {"station_key": k, "station": v["name"], "lat": v["lat"], "lon": v["lon"]}
        for k, v in STATIONS_META.items()
    ])
    agg_df = stations_full.merge(agg_df, on=["station_key", "station", "lat", "lon"], how="left")

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
            **({} if valid.empty else size_kwargs),
        )
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


# ---------------------------------------------------------------------
# PAGE 2: Temperatuur Trends
# ---------------------------------------------------------------------
elif page == "Temperatuur Trends":
    st.header("🌡️ Temperatuur Trends — tijdreeksen, verdelingen en heatmap")
    df, mode = selection_controls(key_prefix="temp")
    if df.empty or "date" not in df.columns:
        st.info("Geen data beschikbaar voor de gekozen filters.")
        st.stop()

    if "season" not in df.columns:
        df["month"] = df["date"].dt.month

        def _season(m):
            return (
                "winter" if m in [12, 1, 2]
                else "lente" if m in [3, 4, 5]
                else "zomer" if m in [6, 7, 8]
                else "herfst"
            )

        df["season"] = df["month"].apply(_season)

    # Lijnplot min/gem/max
    use_cols = [c for c in ["TN_C", "TG_C", "TX_C"] if c in df.columns]
    if use_cols:
        label_map = {"TN_C": "Min", "TG_C": "Gem", "TX_C": "Max"}
        if "station" not in df.columns:
            df["station"] = "Alle stations"
        temp = df[["date", "station"] + use_cols].melt(["date", "station"], var_name="type", value_name="temp_C")
        temp["type"] = temp["type"].replace(label_map)

        facet_args = {}
        if mode == "Vergelijk locaties":
            facet_args = {"facet_row": "station"}

        fig = px.line(
            temp, x="date", y="temp_C", color="type", **facet_args,
            labels={"temp_C": "Temperatuur (°C)", "date": "Datum", "type": "Reeks"},
            title="Dagelijkse temperatuur (min/gem/max)"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Boxplot temperatuur per maand
    dft = df.copy()
    dft["month_name"] = dft["date"].dt.month_name()
    fig_box = px.box(
        dft, x="month_name", y="TG_C",
        color=("station" if mode == "Vergelijk locaties" else None),
        category_orders={"month_name": [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ]},
        title="Verdeling van gemiddelde temperatuur per maand",
        labels={"month_name": "Maand", "TG_C": "Gemiddelde temperatuur (°C)"},
    )
    fig_box.update_traces(line_width=2)
    st.plotly_chart(fig_box, use_container_width=True)

    # Gemiddelde temperatuur per seizoen
    group_cols = ["season"]
    if mode == "Vergelijk locaties":
        group_cols.insert(0, "station")
    season_temp = df.groupby(group_cols)["TG_C"].mean().reset_index()
    season_temp.rename(columns={"TG_C": "Gem_TG_C"}, inplace=True)

    if mode != "Vergelijk locaties":
        fig_season = px.bar(
            season_temp, x="season", y="Gem_TG_C", color="season",
            title="Gemiddelde temperatuur per seizoen",
            labels={"season": "Seizoen", "Gem_TG_C": "Gemiddelde temperatuur (°C)"},
        )
    else:
        fig_season = px.bar(
            season_temp, x="season", y="Gem_TG_C", color="station", barmode="group",
            title="Gemiddelde temperatuur per seizoen (per station)",
            labels={"season": "Seizoen", "Gem_TG_C": "Gemiddelde temperatuur (°C)", "station": "Station"},
        )
    st.plotly_chart(fig_season, use_container_width=True)

    # Kalender-heatmap TG_C per dag
    st.subheader("📅 Kalender-heatmap — gemiddelde temperatuur per dag")

    def heatmap_from(df_in: pd.DataFrame):
        d = df_in.copy()
        d["day"] = d["date"].dt.day
        d["month"] = d["date"].dt.month
        pivot = d.pivot_table(index="month", columns="day", values="TG_C", aggfunc="mean")

        month_names_map = {
            1: "Januari", 2: "Februari", 3: "Maart", 4: "April",
            5: "Mei", 6: "Juni", 7: "Juli", 8: "Augustus",
            9: "September", 10: "Oktober", 11: "November", 12: "December",
        }
        pivot.index = pivot.index.map(month_names_map)

        fig_heatmap = px.imshow(
            pivot,
            color_continuous_scale="RdBu_r",
            origin="upper",
            aspect="auto",
            labels=dict(color="Temperatuur (°C)", x="Dag van de maand", y="Maand"),
        )
        fig_heatmap.update_xaxes(title="Dag van de maand", tickmode="linear")
        fig_heatmap.update_yaxes(
            title="Maand",
            tickmode="array",
            tickvals=list(pivot.index),
            ticktext=list(pivot.index),
        )
        return fig_heatmap

    if mode == "Vergelijk locaties":
        tabs = st.tabs(sorted(df["station"].unique()))
        for tab, st_name in zip(tabs, sorted(df["station"].unique())):
            with tab:
                st.plotly_chart(heatmap_from(df[df["station"] == st_name]), use_container_width=True)
    else:
        st.plotly_chart(heatmap_from(df), use_container_width=True)


# ---------------------------------------------------------------------
# PAGE 3: Neerslag & Zon
# ---------------------------------------------------------------------
elif page == "Neerslag & Zon":
    st.header("☔ Neerslag & Zon — relaties en verdelingen")
    df, mode = selection_controls(key_prefix="rain_sun")
    if df.empty:
        st.info("Geen data beschikbaar voor de gekozen filters.")
        st.stop()

    if "RH_mm" in df.columns and "SQ_h" in df.columns:
        bins = [0, 1, 5, 10, 50]
        labels = ["0 mm", "0–5 mm", "5–10 mm", "10+ mm"]
        df["rain_cat"] = pd.cut(df["RH_mm"], bins=bins, labels=labels, include_lowest=True)
        df["rain_cat"] = pd.Categorical(df["rain_cat"], categories=labels, ordered=True)

        fig_box = px.box(
            df, x="rain_cat", y="SQ_h",
            color=("station" if mode == "Vergelijk locaties" else "rain_cat"),
            category_orders={"rain_cat": labels},
            title="Zonuren per neerslagcategorie",
            labels={"SQ_h": "Zonuren (uur)", "rain_cat": "Neerslagcategorie"},
            points="all",
        )
        st.plotly_chart(fig_box, use_container_width=True)

        rain_bins = pd.cut(
            df["RH_mm"], bins=[0, 1, 5, 10, 20, 50], include_lowest=True,
            labels=["0–1 mm", "1–5 mm", "5–10 mm", "10–20 mm", "20+ mm"],
        )
        rain_bins.name = "RH_mm"
        avg_temp_rain = df.groupby([rain_bins] + (["station"] if mode == "Vergelijk locaties" else []))["TG_C"].mean().reset_index()

        fig_temp_rain = px.bar(
            avg_temp_rain, x="RH_mm", y="TG_C",
            color=("station" if mode == "Vergelijk locaties" else None),
            barmode=("group" if mode == "Vergelijk locaties" else "relative"),
            title="Gemiddelde temperatuur bij toenemende regenval",
            labels={"RH_mm": "Neerslagcategorie (mm/dag)", "TG_C": "Gem. temperatuur (°C)", "station": "Station"},
            text_auto=".1f",
        )
        st.plotly_chart(fig_temp_rain, use_container_width=True)


# ---------------------------------------------------------------------
# PAGE 4: Windtrends & Topdagen (incl. windroos)
# ---------------------------------------------------------------------
elif page == "Windtrends & Topdagen":
    st.header("💨 Windtrends & Topdagen — windroos en verdelingen")
    df, mode = selection_controls(key_prefix="wind")
    if df.empty:
        st.info("Geen data beschikbaar voor de gekozen filters.")
        st.stop()

    # Windroos
    if "FG_ms" in df.columns and "DDVEC" in df.columns:
        # In aggregaatmodus de ruwe rijen herladen in dezelfde periode
        w_source = df
        if mode == "Alle locaties (geaggregeerd)":
            found = discover_files()
            all_periods = sorted({p for _, p, _ in found})
            all_stations = list(STATIONS_META.keys())
            raw = build_dataset(tuple(all_periods), tuple(all_stations))
            if not df.empty and "date" in df.columns:
                dmin, dmax = df["date"].min(), df["date"].max()
                raw = raw[(raw["date"] >= dmin) & (raw["date"] <= dmax)]
            w_source = raw

        vergelijk_flag = (mode == "Vergelijk locaties")
        render_windrose(w_source, titel="🧭 Windroos", vergelijk_per_station=vergelijk_flag)
    else:
        st.info("Windroos niet mogelijk: kolommen 'DDVEC' en/of 'FG_ms' ontbreken in de selectie.")

    # Boxplot windsnelheid per seizoen
    if "FG_ms" in df.columns and "date" in df.columns:
        st.subheader("📦 Verdeling windsnelheid per seizoen")
        def get_season_name(date):
            m = date.month
            if m in [3, 4, 5]:
                return "Lente"
            elif m in [6, 7, 8]:
                return "Zomer"
            elif m in [9, 10, 11]:
                return "Herfst"
            else:
                return "Winter"

        df["season_box"] = df["date"].apply(get_season_name)
        season_order = ["Lente", "Zomer", "Herfst", "Winter"]

        fig_box_w = px.box(
            df,
            x="season_box",
            y="FG_ms",
            color=("station" if mode == "Vergelijk locaties" else "season_box"),
            category_orders={"season_box": season_order},
            points="all",
            title="Windsnelheid per seizoen",
            labels={"season_box": "Seizoen", "FG_ms": "Windsnelheid (m/s)"},
        )
        fig_box_w.update_traces(line_width=3)
        fig_box_w.update_layout(title_x=0.5, boxmode="group")
        st.plotly_chart(fig_box_w, use_container_width=True)


# ---------------------------------------------------------------------
# PAGE 5: Correlaties
# ---------------------------------------------------------------------
elif page == "Correlaties":
    st.header("🔗 Correlaties — verbanden tussen temperatuur, neerslag, zon en wind")
    df, mode = selection_controls(key_prefix="corr")
    if df.empty:
        st.info("Geen data beschikbaar voor de gekozen filters.")
        st.stop()

    vars_use = [c for c in ["TN_C", "TG_C", "TX_C", "RH_mm", "SQ_h", "FG_ms"] if c in df.columns]
    if not vars_use:
        st.info("Geen geschikte variabelen gevonden voor correlatie.")
        st.stop()

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

    st.subheader("📊 Scattermatrix")
    sm = px.scatter_matrix(df, dimensions=vars_use, color=("station" if mode == "Vergelijk locaties" else None), height=700)
    st.plotly_chart(sm, use_container_width=True)


# ---------------------------------------------------------------------
# PAGE 6: Voorspellingsmodel
# ---------------------------------------------------------------------
elif page == "Voorspellingsmodel":
    st.header("🧠 Voorspellingsmodel — verwachte temperatuur per station")
    st.caption("Pas maand/dag/neerslag/windsnelheid aan. Alle stations (incl. IJmuiden) worden getoond; zonder voorspelling als grijze marker.")

    found = discover_files()
    if not found:
        st.warning("Geen JSON-data gevonden (bijv. 'Ijmuiden_2023_2024.json').")
        st.stop()

    all_periods = sorted({p for _, p, _ in found})
    sel_periods = st.multiselect("📅 Jaarperiodes", all_periods, default=all_periods, key="model_periods")

    # Alle stations samenvoegen
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

    # Datumfeatures
    df_all["doy"] = df_all["date"].dt.dayofyear
    df_all["doy"] = df_all["doy"].fillna(df_all["doy"].median())
    df_all["doy_sin"] = np.sin(2 * np.pi * df_all["doy"] / 366.0)
    df_all["doy_cos"] = np.cos(2 * np.pi * df_all["doy"] / 366.0)

    # Eenvoudig lineair model per station (met dynamische features)
    def fit_linear(X, y):
        candidate_features = ["RH_mm", "FG_ms", "doy_sin", "doy_cos"]
        features = [c for c in candidate_features if c in X.columns and not X[c].isna().all()]
        for c in ["doy_sin", "doy_cos"]:
            if c not in features and c in X.columns:
                features.append(c)
        if len(features) < 2:
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

    # Gebruikersinvoer
    import calendar
    st.subheader("⚙️ Stel de omstandigheden in")
    col1, col2, col3 = st.columns(3)
    with col1:
        maand = st.slider("📆 Maand", 1, 12, 7)
        max_dag = calendar.monthrange(2024, maand)[1]
        dag = st.slider("📅 Dag", 1, int(max_dag), min(15, max_dag))
    with col2:
        pred_rain = st.slider("🌧️ Neerslag (mm/dag)", 0.0, 50.0, 0.0, 0.5)
    with col3:
        pred_wind = st.slider("💨 Windsnelheid (m/s)", 0.0, 15.0, 3.0, 0.5)

    # Valideer dag en maand
    try:
        selected_date = pd.Timestamp(year=2024, month=int(maand), day=int(dag))
        doy = selected_date.dayofyear
        doy_sin = float(np.sin(2 * np.pi * doy / 366.0))
        doy_cos = float(np.cos(2 * np.pi * doy / 366.0))
    except ValueError:
        st.error(f"❌ Ongeldige datum: {dag} / {maand}. Controleer of deze dag in de gekozen maand voorkomt.")
        st.stop()

    # Voorspellen per station
    rows = []
    for sk, m in models.items():
        beta = m["beta"]
        feats = ["const"] + m.get("features", ["RH_mm", "FG_ms", "doy_sin", "doy_cos"])
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
            "n": m["n"],
        })

    pred_df = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)

    # Kaart — ALTIJD alle stations tonen (incl. IJmuiden)
    st.subheader("🗺️ Voorspelde temperatuur per station (°C)")
    stations_full = pd.DataFrame([
        {"station_key": k, "station": v["name"], "lat": v["lat"], "lon": v["lon"]}
        for k, v in STATIONS_META.items()
    ])
    plot_df = stations_full.merge(pred_df, on=["station_key", "station", "lat", "lon"], how="left")

    valid = plot_df.dropna(subset=["pred_TG_C"]).copy()
    missing = plot_df[plot_df["pred_TG_C"].isna()].copy()

    if valid.empty and missing.empty:
        st.info("Geen geldige stations om te tonen.")
    else:
        if not valid.empty:
            if valid["pred_TG_C"].max() > valid["pred_TG_C"].min():
                valid["size"] = (valid["pred_TG_C"] - valid["pred_TG_C"].min()) / (
                    valid["pred_TG_C"].max() - valid["pred_TG_C"].min()
                )
            else:
                valid["size"] = 0.5
            valid["size"] = (valid["size"] * 25) + 6

        TEMP_SCALE_MIN = -5.0
        TEMP_SCALE_MAX = 30.0

        fig = px.scatter_mapbox(
            valid if not valid.empty else plot_df,
            lat="lat",
            lon="lon",
            color=("pred_TG_C" if not valid.empty else None),
            size=("size" if not valid.empty else None),
            color_continuous_scale=("RdYlBu_r" if not valid.empty else None),
            range_color=([TEMP_SCALE_MIN, TEMP_SCALE_MAX] if not valid.empty else None),
            zoom=6,
            hover_name="station",
            hover_data={
                "pred_TG_C": True,
                "r2": True,
                "rmse": True,
                "n": True,
                "lat": False,
                "lon": False,
                "size": False,
            },
            height=520,
        )
        if not missing.empty:
            fig.add_trace(
                go.Scattermapbox(
                    lat=missing["lat"],
                    lon=missing["lon"],
                    mode="markers",
                    marker=dict(size=14, color="#A0A0A0"),
                    name="Geen voorspelling",
                    text=missing["station"],
                    hoverinfo="text",
                )
            )
        fig.update_layout(
            mapbox_style="carto-darkmatter",
            margin=dict(l=0, r=0, t=10, b=0),
            coloraxis_colorbar=dict(
                title="Voorspelde temperatuur",
                ticksuffix="°C",
                tickmode="auto",
            ),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Tabel voorspelde temperatuur
    st.subheader("📄 Tabel — voorspelde temperatuur (°C)")
    temp_tbl = (
        pred_df[["station", "pred_TG_C"]]
        .rename(columns={"station": "Station", "pred_TG_C": "Voorspelde temperatuur (°C)"})
        .assign(**{"Voorspelde temperatuur (°C)": lambda d: d["Voorspelde temperatuur (°C)"].round(1)})
        .sort_values("Voorspelde temperatuur (°C)", ascending=False)
        .reset_index(drop=True)
    )
    st.dataframe(temp_tbl, use_container_width=True)

    # Samenvatting modelprestatie
    st.subheader("📈 Gemiddelde modelprestatie")
    avg_temp_pred = pred_df["pred_TG_C"].mean()
    avg_r2 = pred_df["r2"].mean()
    avg_rmse = pred_df["rmse"].mean()
    summary_df = pd.DataFrame([{
        "Gem. voorspelde temperatuur (°C)": round(avg_temp_pred, 1),
        "Gem. R² (verklaarde variantie)": round(avg_r2, 2),
        "Gem. standaardafwijking (°C)": round(avg_rmse, 1),
    }])
    st.table(summary_df)
