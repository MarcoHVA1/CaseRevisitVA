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
        df["week"] = df["date"].dt.isocalendar().week

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

# Stations 
EXCLUDED_STATIONS = {"ijmuiden"}

STATIONS_META = {
    "amsterdam":  {"name": "Amsterdam",  "lat": 52.3676, "lon": 4.9041},
    "de_bilt":    {"name": "De Bilt",    "lat": 52.1010, "lon": 5.1790},
    "eelde":      {"name": "Eelde",      "lat": 53.1250, "lon": 6.5833},
    "eindhoven":  {"name": "Eindhoven",  "lat": 51.4500, "lon": 5.3740},
    "maastricht": {"name": "Maastricht", "lat": 50.8510, "lon": 5.6910},
    "twente":     {"name": "Twente",     "lat": 52.2700, "lon": 6.9000},
    "vlissingen": {"name": "Vlissingen", "lat": 51.4420, "lon": 3.5730},
}
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
        r"^(amsterdam|de_bilt|eelde|eindhoven|maastricht|twente|vlissingen)_(\d{4}_\d{4})\.json$",
        re.I
    )
    out = []
    for f in sorted(Path(".").glob("*.json")):
        m = pat.match(f.name)
        if m:
            out.append((m.group(1).lower(), m.group(2), str(f)))
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
        # zorg dat numeriek is
        for c in ["TN_C", "TG_C", "TX_C", "RH_mm", "SQ_h", "FG_ms", "DDVEC"]:
            if c in dfp.columns:
                dfp[c] = pd.to_numeric(dfp[c], errors="coerce")
        frames.append(dfp)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def selection_controls(key_prefix: str = ""):
    """Geeft (df_for_charts, mode, selected_periods, selected_stations) terug."""
    found = discover_files()
    all_periods = sorted({p for _, p, _ in found})
    all_station_keys = list(STATIONS_META.keys())

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

    # Aggregaatmodus: gemiddelden per datum over gekozen stations
    if mode == "Alle locaties (geaggregeerd)" and not df_all.empty:
        num_cols = [c for c in ["TN_C", "TG_C", "TX_C", "RH_mm", "SQ_h", "FG_ms"] if c in df_all.columns]
        keep_cols = ["date"] + num_cols
        g = (
            df_all[keep_cols + ["station"]]
            .groupby("date", as_index=False)
            .agg({c: "mean" for c in num_cols})
        )
        g["month"] = g["date"].dt.month

        def _season(m):
            if m in [12, 1, 2]:
                return "winter"
            if m in [3, 4, 5]:
                return "lente"
            if m in [6, 7, 8]:
                return "zomer"
            return "herfst"

        g["season"] = g["month"].apply(_season)
        g["station"] = "Alle stations"
        g["station_key"] = "all"
        g["period"] = ", ".join(sel_periods)
        df_all = g

    return df_all, mode, tuple(sel_periods), tuple(stations)


# =========================================================
# Windroos component
# =========================================================
def render_windrose(raw_df: pd.DataFrame, *, title="🧭 Windroos", facet_per_station=False):
    st.subheader(title)

    if raw_df.empty:
        st.info("Geen data voor de windroos.")
        return

    need = {"DDVEC", "FG_ms"}
    if not need.issubset(raw_df.columns):
        st.warning("Windroos niet mogelijk: kolommen DDVEC/FG ontbreken in de bronbestanden.")
        return

    w = raw_df[["station", "DDVEC", "FG_ms"]].copy().dropna()
    if w.empty:
        st.info("Geen geldige winddata om te tonen.")
        return

    w["DDVEC"] = pd.to_numeric(w["DDVEC"], errors="coerce") % 360
    w["FG_ms"] = pd.to_numeric(w["FG_ms"], errors="coerce")
    w = w.dropna()

    # Stations samenvoegen als we niet per station facetteren (voorkomt dubbel stapelen)
    if not facet_per_station:
        w["station"] = "Alle stations"

    c1, c2 = st.columns(2)
    with c1:
        dir_bin = st.selectbox("Richtingsbin (°)", [10, 15, 20, 30, 45], index=3)
    with c2:
        bins_text = st.text_input("Snelheidsklassen m/s (komma-gescheiden)", value="0,2,4,6,8,10,12,20")
        try:
            speed_bins = sorted({float(x.strip()) for x in bins_text.split(",") if x.strip()})
            if len(speed_bins) < 2:
                raise ValueError
        except Exception:
            speed_bins = [0, 2, 4, 6, 8, 10, 12, 20]
            st.warning("Kon de snelheidsklassen niet parsen; standaard gebruikt.")

    n_bins = int(360 / dir_bin)
    w["dir_bin_idx"] = (np.floor(w["DDVEC"] / dir_bin).astype(int)) % n_bins
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

    # === Normalisatie: totaal = 100% (niet per richting)
    total = agg.groupby("station")["count"].transform("sum")
    agg["value"] = np.where(total > 0, 100.0 * agg["count"] / total, 0.0)
    r_title, tick_suffix = "Aandeel (totaal = 100%)", "%"

    facets = {"facet_row": "station"} if facet_per_station and agg["station"].nunique() > 1 else {}
    fig = px.bar_polar(
        agg.sort_values(["station", "dir_bin_idx"]),
        r="value",
        theta="dir_bin",
        color="speed_bin",
        barmode="stack",
        hover_data={"count": True, "value": True, "dir_bin_idx": False},
        **facets
    )
    fig.update_layout(
        polar=dict(
            angularaxis=dict(direction="clockwise", rotation=90, categoryorder="array", categoryarray=dir_labels),
            radialaxis=dict(title=r_title, ticksuffix=tick_suffix),
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        legend_title_text="Snelheid (m/s)",
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
# Pagina's
# =========================================================
if page == "Overzicht":
    st.header("🗺️ Overzicht — Kaart met temperatuur, neerslag en zonuren")

    found = discover_files()
    if not found:
        st.warning("Geen JSON-data gevonden.")
        st.stop()

    all_periods = sorted({p for _, p, _ in found})
    sel_periods = st.multiselect("📅 Jaarperiodes", all_periods, default=all_periods, key="ov_periods")

    month_names = ["Alle"] + [f"{i:02d} - {m}" for i, m in enumerate(
        ["Januari","Februari","Maart","April","Mei","Juni","Juli","Augustus","September","Oktober","November","December"], start=1)]
    cA, cB, cC = st.columns(3)
    sel_month = cA.selectbox("📆 Maand", month_names)

    map_var = cB.selectbox(
        "🗺️ Variabele",
        ["TG_C", "RH_mm", "SQ_h"],
        format_func=lambda k: {"TG_C": "🌡️ Temperatuur (°C)", "RH_mm": "🌧️ Neerslag (mm)", "SQ_h": "☀️ Zonuren (uur)"}[k],
    )
    agg_choice = cC.radio("Aggregatie", ["Gemiddelde", "Som"], horizontal=True)
    agg_func = "mean" if agg_choice == "Gemiddelde" else "sum"

    frames = []
    for skey, period, path in found:
        if period not in sel_periods:
            continue
        dfp = load_data(path)
        if dfp.empty or "date" not in dfp.columns:
            continue
        dfp["station_key"] = skey
        dfp["station"] = STATIONS_META[skey]["name"]
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

    stations_full = pd.DataFrame([{"station_key": k, "station": v["name"], "lat": v["lat"], "lon": v["lon"]} for k, v in STATIONS_META.items()])
    agg_df = stations_full.merge(agg_df, on=["station_key", "station", "lat", "lon"], how="left")

    valid = agg_df.dropna(subset=[map_var]).copy()
    missing = agg_df[agg_df[map_var].isna()].copy()

    color_scale = "RdYlBu_r" if map_var == "TG_C" else ("Blues" if map_var == "RH_mm" else "YlOrBr")
    title_map = {"TG_C": "Temperatuur (°C)", "RH_mm": "Neerslag (mm)", "SQ_h": "Zonuren (uur)"}

    if valid.empty and missing.empty:
        st.info("Geen geldige waarden om op de kaart te tonen.")
    else:
        size_kwargs = {"size": map_var, "size_max": 28} if not valid.empty and (valid[map_var] >= 0).all() else {}
        fig_map = px.scatter_mapbox(
            valid if not valid.empty else agg_df,
            lat="lat", lon="lon",
            color=(map_var if not valid.empty else None),
            hover_name="station",
            hover_data={"lat": False, "lon": False, "TG_C": True, "RH_mm": True, "SQ_h": True},
            color_continuous_scale=(color_scale if not valid.empty else None),
            zoom=6, height=520,
            **({} if valid.empty else size_kwargs),
        )
        if not missing.empty:
            fig_map.add_trace(
                go.Scattermapbox(
                    lat=missing["lat"], lon=missing["lon"],
                    mode="markers", marker=dict(size=14, color="#A0A0A0"),
                    name="Geen data", text=missing["station"], hoverinfo="text",
                )
            )
        fig_map.update_layout(
            mapbox_style="open-street-map",
            margin=dict(l=0, r=0, t=10, b=0),
            coloraxis_colorbar=dict(title=title_map[map_var]),
        )
        st.plotly_chart(fig_map, use_container_width=True)

elif page == "Temperatuur Trends":
    st.header("🌡️ Temperatuur Trends — dag • maand • seizoen")
    df, mode, sel_periods, sel_stations = selection_controls(key_prefix="temp")
    if df.empty or "date" not in df.columns:
        st.info("Geen data beschikbaar voor de gekozen filters.")
        st.stop()

    # DAG
    use_cols = [c for c in ["TN_C", "TG_C", "TX_C"] if c in df.columns]
    if use_cols:
        label_map = {"TN_C": "Min", "TG_C": "Gem", "TX_C": "Max"}
        if "station" not in df.columns:
            df["station"] = "Alle stations"
        m = df[["date", "station"] + use_cols].melt(["date", "station"], var_name="type", value_name="temp_C")
        m["type"] = m["type"].replace(label_map)
        facet = {"facet_row": "station"} if mode == "Vergelijk locaties" else {}
        fig = px.line(m, x="date", y="temp_C", color="type", title="Dagelijkse temperatuur (min/gem/max)",
                      labels={"temp_C": "Temperatuur (°C)", "date": "Datum", "type": "Reeks"}, **facet)
        st.plotly_chart(fig, use_container_width=True)

    # MAAND
    dft = df.copy()
    dft["month_name"] = dft["date"].dt.month_name()
    fig_box = px.box(
        dft, x="month_name", y="TG_C",
        color=("station" if mode == "Vergelijk locaties" else None),
        category_orders={"month_name": ["January","February","March","April","May","June","July","August","September","October","November","December"]},
        title="Verdeling van gemiddelde temperatuur per maand",
        labels={"month_name": "Maand", "TG_C": "Gemiddelde temperatuur (°C)"}
    )
    fig_box.update_traces(line_width=2)
    st.plotly_chart(fig_box, use_container_width=True)

    # SEIZOEN
    df["season"] = df["season"].fillna("onbekend") if "season" in df.columns else "onbekend"
    group_cols = ["season"] if mode != "Vergelijk locaties" else ["station", "season"]
    season_temp = df.groupby(group_cols)["TG_C"].mean().reset_index().rename(columns={"TG_C": "Gem_TG_C"})
    if mode != "Vergelijk locaties":
        fig_season = px.bar(season_temp, x="season", y="Gem_TG_C", color="season",
                            title="Gemiddelde temperatuur per seizoen",
                            labels={"season": "Seizoen", "Gem_TG_C": "Gemiddelde temperatuur (°C)"})
    else:
        fig_season = px.bar(season_temp, x="season", y="Gem_TG_C", color="station", barmode="group",
                            title="Gemiddelde temperatuur per seizoen (per station)",
                            labels={"season": "Seizoen", "Gem_TG_C": "Gemiddelde temperatuur (°C)", "station": "Station"})
    st.plotly_chart(fig_season, use_container_width=True)

elif page == "Neerslag & Zon":
    st.header("☔ Neerslag & Zon — relaties en verdelingen")
    df, mode, _, _ = selection_controls(key_prefix="rain_sun")
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

        rain_bins = pd.cut(df["RH_mm"], bins=[0,1,5,10,20,50], include_lowest=True,
                           labels=["0–1 mm","1–5 mm","5–10 mm","10–20 mm","20+ mm"])
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

elif page == "Windtrends & Topdagen":
    st.header("💨 Windtrends & Topdagen — windroos en verdelingen")
    df_for_charts, mode, sel_periods, sel_stations = selection_controls(key_prefix="wind")
    if df_for_charts.empty:
        st.info("Geen data beschikbaar voor de gekozen filters.")
        st.stop()

    # === BELANGRIJK: windroos altijd op RUWE rijen opbouwen
    raw = build_dataset(sel_periods, sel_stations)
    if not raw.empty and "date" in raw.columns and "date" in df_for_charts.columns:
        # optioneel: filter ruwe rijen op exacte datumbereik dat in charts zichtbaar is
        dmin, dmax = df_for_charts["date"].min(), df_for_charts["date"].max()
        raw = raw[(raw["date"] >= dmin) & (raw["date"] <= dmax)]

    render_windrose(
        raw_df=raw,
        title="🧭 Windroos",
        facet_per_station=(mode == "Vergelijk locaties")
    )

    # Boxplot windsnelheid per seizoen
    if "FG_ms" in df_for_charts.columns and "date" in df_for_charts.columns:
        st.subheader("📦 Verdeling windsnelheid per seizoen")
        def season_name(d):
            m = d.month
            if m in [3, 4, 5]: return "Lente"
            if m in [6, 7, 8]: return "Zomer"
            if m in [9,10,11]: return "Herfst"
            return "Winter"
        t = df_for_charts.copy()
        t["season_box"] = t["date"].apply(season_name)
        fig_wb = px.box(
            t, x="season_box", y="FG_ms",
            color=("station" if mode == "Vergelijk locaties" else "season_box"),
            category_orders={"season_box": ["Lente","Zomer","Herfst","Winter"]},
            points="all",
            title="Windsnelheid per seizoen",
            labels={"season_box": "Seizoen", "FG_ms": "Windsnelheid (m/s)"}
        )
        fig_wb.update_traces(line_width=3)
        st.plotly_chart(fig_wb, use_container_width=True)

elif page == "Correlaties":
    st.header("🔗 Correlaties ")
    df, mode, _, _ = selection_controls(key_prefix="corr")
    if df.empty:
        st.info("Geen data beschikbaar voor de gekozen filters.")
        st.stop()

    # Variabele-naam mapping naar mensentaal
    nice = {
        "TN_C": "Minimum temp (°C)",
        "TG_C": "Gem. temp (°C)",
        "TX_C": "Maximum temp (°C)",
        "RH_mm": "Neerslag (mm)",
        "SQ_h": "Zonuren (uur)",
        "FG_ms": "Windsnelheid (m/s)",
    }
    vars_use = [c for c in nice if c in df.columns]
    if not vars_use:
        st.info("Geen geschikte variabelen gevonden voor correlatie.")
        st.stop()

    # Correlatiematrix
    st.subheader("📐 Correlatiematrix")
    if mode == "Vergelijk locaties":
        tabs = st.tabs(sorted(df["station"].unique()))
        for tab, st_name in zip(tabs, sorted(df["station"].unique())):
            with tab:
                sub = df[df["station"] == st_name][vars_use].copy()
                corr = sub.corr().round(2)
                corr.index = [nice[c] for c in corr.index]
                corr.columns = [nice[c] for c in corr.columns]
                fig = px.imshow(corr, text_auto=True, aspect="auto", origin="upper",
                                labels=dict(color="Correlatie"))
                st.plotly_chart(fig, use_container_width=True)
    else:
        corr = df[vars_use].corr().round(2)
        corr.index = [nice[c] for c in corr.index]
        corr.columns = [nice[c] for c in corr.columns]
        fig = px.imshow(corr, text_auto=True, aspect="auto", origin="upper",
                        labels=dict(color="Correlatie"))
        st.plotly_chart(fig, use_container_width=True)

elif page == "Voorspellingsmodel":
    st.header("🧠 Voorspellingsmodel — verwachte temperatuur per station")

    found = discover_files()
    if not found:
        st.warning("Geen JSON-data gevonden.")
        st.stop()

    all_periods = sorted({p for _, p, _ in found})
    sel_periods = st.multiselect("📅 Jaarperiodes", all_periods, default=all_periods, key="model_periods")

    frames = []
    for skey, period, path in found:
        if period not in sel_periods:
            continue
        dfp = load_data(path)
        if dfp.empty or "date" not in dfp.columns:
            continue
        for c in ["TG_C", "RH_mm", "FG_ms", "SQ_h"]:
            if c not in dfp.columns:
                dfp[c] = np.nan
            dfp[c] = pd.to_numeric(dfp[c], errors="coerce")
        dfp["station_key"] = skey
        dfp["station"] = STATIONS_META[skey]["name"]
        frames.append(dfp)
    if not frames:
        st.info("Geen data beschikbaar.")
        st.stop()

    df_all = pd.concat(frames, ignore_index=True).dropna(subset=["TG_C"]).copy()
    df_all["doy"] = df_all["date"].dt.dayofyear
    df_all["doy_sin"] = np.sin(2 * np.pi * df_all["doy"] / 366.0)
    df_all["doy_cos"] = np.cos(2 * np.pi * df_all["doy"] / 366.0)

    def fit_linear(X, y):
        feats = [c for c in ["RH_mm", "FG_ms", "SQ_h", "doy_sin", "doy_cos"] if c in X.columns]
        if "doy_sin" not in feats: feats.append("doy_sin")
        if "doy_cos" not in feats: feats.append("doy_cos")
        X_ = np.column_stack([np.ones(len(X))] + [X[c].values for c in feats])
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
        return {"beta": beta, "rmse": rmse, "r2": r2, "n": len(yc), "features": ["const"] + feats}

    models = {}
    for sk, g in df_all.groupby("station_key"):
        m = fit_linear(g[["RH_mm", "FG_ms", "SQ_h", "doy_sin", "doy_cos"]], g["TG_C"])
        if m:
            models[sk] = m

    if not models:
        st.info("Onvoldoende data om modellen te trainen.")
        st.stop()

    import calendar
    st.subheader("⚙️ Stel de omstandigheden in")

    # Maand, regen, wind, zon
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        maand = st.slider("📆 Maand", 1, 12, 7)
        max_dag = calendar.monthrange(2024, maand)[1]
        dag = st.slider("📅 Dag", 1, int(max_dag), min(15, max_dag))

    # Realistische standaardwaardes afhankelijk van maand
    if maand in [12, 1, 2]:
        default_sun = 2.0   # winter
    elif maand in [3, 4, 5]:
        default_sun = 5.0   # lente
    elif maand in [6, 7, 8]:
        default_sun = 8.0   # zomer
    else:
        default_sun = 4.0   # herfst

    with c2:
        pred_rain = st.slider("🌧️ Neerslag (mm/dag)", 0.0, 50.0, 2.0, 0.5)
    with c3:
        pred_wind = st.slider("💨 Windsnelheid (m/s)", 0.0, 15.0, 3.0, 0.5)
    with c4:
        pred_sun = st.slider("☀️ Zonuren (uur/dag)", 0.0, 12.0, default_sun, 0.25)

    try:
        selected_date = pd.Timestamp(year=2024, month=int(maand), day=int(dag))
        doy = selected_date.dayofyear
        sin, cos = float(np.sin(2 * np.pi * doy / 366.0)), float(np.cos(2 * np.pi * doy / 366.0))
    except ValueError:
        st.error("❌ Ongeldige datum.")
        st.stop()

    rows = []
    for sk, m in models.items():
        vals = [1.0]
        for f in m["features"][1:]:
            if f == "RH_mm":
                vals.append(pred_rain)
            elif f == "FG_ms":
                vals.append(pred_wind)
            elif f == "SQ_h":
                vals.append(pred_sun)
            elif f == "doy_sin":
                vals.append(sin)
            elif f == "doy_cos":
                vals.append(cos)
            else:
                vals.append(0.0)
        pred = float(np.array(vals) @ m["beta"])
        rows.append({
            "station_key": sk,
            "station": STATIONS_META[sk]["name"],
            "lat": STATIONS_META[sk]["lat"],
            "lon": STATIONS_META[sk]["lon"],
            "pred_TG_C": pred,
            "r2": m["r2"],
            "rmse": m["rmse"],
            "n": m["n"]
        })
    pred_df = pd.DataFrame(rows)

    # Kaart
    st.subheader("🗺️ Voorspelde temperatuur per station (°C)")
    all_points = pd.DataFrame([
        {"station_key": k, "station": v["name"], "lat": v["lat"], "lon": v["lon"]}
        for k, v in STATIONS_META.items() if k not in EXCLUDED_STATIONS
    ])
    plot_df = all_points.merge(pred_df, on=["station_key", "station", "lat", "lon"], how="left")

    valid = plot_df.dropna(subset=["pred_TG_C"]).copy()
    missing = plot_df[plot_df["pred_TG_C"].isna()].copy()
    if not valid.empty:
        rng = valid["pred_TG_C"].max() - valid["pred_TG_C"].min()
        valid["size"] = 0.5 if rng == 0 else (valid["pred_TG_C"] - valid["pred_TG_C"].min()) / rng
        valid["size"] = (valid["size"] * 25) + 6

    fig = px.scatter_mapbox(
        valid if not valid.empty else plot_df,
        lat="lat", lon="lon",
        color=("pred_TG_C" if not valid.empty else None),
        size=("size" if not valid.empty else None),
        color_continuous_scale=("RdYlBu_r" if not valid.empty else None),
        range_color=([-5.0, 30.0] if not valid.empty else None),
        zoom=6, height=520,
        hover_name="station",
        hover_data={"pred_TG_C": True, "r2": True, "rmse": True, "n": True,
                    "lat": False, "lon": False, "size": False},
    )
    if not missing.empty:
        fig.add_trace(go.Scattermapbox(
            lat=missing["lat"], lon=missing["lon"], mode="markers",
            marker=dict(size=14, color="#A0A0A0"), name="Geen voorspelling",
            text=missing["station"], hoverinfo="text"
        ))
    fig.update_layout(
        mapbox_style="carto-darkmatter",
        margin=dict(l=0, r=0, t=10, b=0),
        coloraxis_colorbar=dict(title="Voorspelde temperatuur", ticksuffix="°C")
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📄 Tabel — voorspelde temperatuur (°C)")
    st.dataframe(
        pred_df[["station", "pred_TG_C"]]
        .rename(columns={"station": "Station", "pred_TG_C": "Voorspelde temperatuur (°C)"})
        .assign(**{"Voorspelde temperatuur (°C)": lambda d: d["Voorspelde temperatuur (°C)"].round(1)})
        .sort_values("Voorspelde temperatuur (°C)", ascending=False)
        .reset_index(drop=True),
        use_container_width=True
    )
