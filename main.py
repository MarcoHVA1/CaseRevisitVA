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

     # 2. Windroos — Interactief (Plotly, robuuste binning)
    if "FG_ms" in df.columns and "DDVEC" in df.columns:
        st.subheader("🧭 Interactieve windroos")

        w = df[["DDVEC", "FG_ms"]].dropna().copy()
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
                w.dropna(subset=["dir_bin", "speed_bin"])
                 .groupby(["dir_bin_idx", "dir_bin", "speed_bin"], as_index=False)
                 .size()
                 .rename(columns={"size": "count"})
            )

            if agg.empty:
                st.info("Geen data binnen de gekozen bins.")
            else:
                if normalize == "% van totaal":
                    total = agg["count"].sum()
                    agg["value"] = np.where(total > 0, 100.0 * agg["count"] / total, 0.0)
                    r_title, tick_suffix = "Frequentie (%)", "%"
                elif normalize == "% per richting":
                    dir_tot = agg.groupby("dir_bin_idx")["count"].transform("sum")
                    agg["value"] = np.where(dir_tot > 0, 100.0 * agg["count"] / dir_tot, 0.0)
                    r_title, tick_suffix = "Aandeel binnen richting (%)", "%"
                else:
                    agg["value"] = agg["count"]
                    r_title, tick_suffix = "Aantal", ""

                agg = agg.sort_values("dir_bin_idx")

                fig_windrose = px.bar_polar(
                    agg,
                    r="value",
                    theta="dir_bin",
                    color="speed_bin",
                    barmode="stack",
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

# == Pagina 5 == 
elif page == "Voorspellingsmodel":
    st.header("🧠 Voorspellingsmodel — voorspelde temperatuur in Nederland")
    st.caption("Pas **maand**, **dag**, **neerslag** en **windsnelheid** aan. Het model voorspelt de **temperatuur (°C)** per station op basis van historische patronen.")

    # === Alle JSON-bestanden automatisch inlezen ===
    files = sorted(Path(".").glob("*.json"))
    pat = re.compile(r"^(amsterdam|de_bilt|eelde|eindhoven|ijmuiden|maastricht|twente|vlissingen)_(\d{4}_\d{4})\.json$", re.I)
    found = []
    for f in files:
        m = pat.match(f.name)
        if m:
            station_key, period = m.group(1).lower(), m.group(2)
            found.append((station_key, period, str(f)))

    if not found:
        st.warning("Geen JSON-data gevonden (bijv. 'amsterdam_2023_2024.json').")
        st.stop()

    # === Alle stations samenvoegen ===
    frames = []
    for station_key, _, path_str in found:
        dfp = load_data(path_str)
        if "date" not in dfp.columns:
            continue
        # Benodigde kolommen afdwingen + numeriek maken
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
        X_ = np.column_stack([np.ones(len(X))] + [X[c].values for c in ["RH_mm", "FG_ms", "doy_sin", "doy_cos"]])
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
        return {"beta": beta, "rmse": rmse, "r2": r2, "n": len(yc)}

    models = {}
    for sk, g in df_all.groupby("station_key"):
        m = fit_linear(g[["RH_mm", "FG_ms", "doy_sin", "doy_cos"]], g["TG_C"])
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
        b0, b_rain, b_wind, b_sin, b_cos = m["beta"]
        pred_temp = b0 + b_rain * pred_rain + b_wind * pred_wind + b_sin * doy_sin + b_cos * doy_cos
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
            plot_df["size"] = (plot_df["pred_TG_C"] - plot_df["pred_TG_C"].min()) / (plot_df["pred_TG_C"].max() - plot_df["pred_TG_C"].min())
        else:
            plot_df["size"] = 0.5
        plot_df["size"] = (plot_df["size"] * 25) + 6

        fig = px.scatter_mapbox(
            plot_df,
            lat="lat",
            lon="lon",
            color="pred_TG_C",
            size="size",
            color_continuous_scale="RdYlBu_r",
            zoom=6,
            hover_name="station",
            hover_data={"pred_TG_C": True, "r2": True, "rmse": True, "n": True, "lat": False, "lon": False, "size": False},
            height=520
        )
        fig.update_layout(
            mapbox_style="carto-darkmatter",
            margin=dict(l=0, r=0, t=10, b=0),
            coloraxis_colorbar=dict(title="Voorspelde temperatuur (°C)")
        )
        st.plotly_chart(fig, use_container_width=True)

    # === Tabel: voorspelde temperatuur (°C), afgerond op 1 decimaal ===
    st.subheader("📄 Tabel: voorspelde temperatuur (°C)")
    temp_tbl = (
        pred_df[["station", "pred_TG_C"]]
        .rename(columns={
            "station": "Station",
            "pred_TG_C": "Voorspelde Temp (°C)"
        })
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

