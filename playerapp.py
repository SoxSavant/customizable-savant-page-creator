import os
import time
import unicodedata
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from io import BytesIO, StringIO
from datetime import date
from pathlib import Path
import requests
from pybaseball import batting_stats, fielding_stats, bwar_bat, playerid_lookup
from pybaseball.statcast_fielding import statcast_outs_above_average

plt.rcdefaults()

st.set_page_config(page_title="Custom Savant Page App", layout="wide")

title_col, meta_col = st.columns([3, 1])
with title_col:
    st.title("Custom Savant Page App")
with meta_col:
    st.markdown(
        """
        <div style="text-align: right; font-size: 1rem; padding-top: 0.6rem;">
            Built by <a href="https://twitter.com/Sox_Savant" target="_blank">@Sox_Savant</a>
            <span style="color: #aaa;">(v 1.2)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

TRUTHY_STRINGS = {"true", "1", "yes", "y", "t"}
STAT_DISPLAY_NAMES = {"WAR": "fWAR", "HardHit%": "Hard Hit%"}

STAT_PRESETS = {
    "Statcast": [
        "WAR",
        "bWAR",
        "Off",
        "BsR",
        "Def",
        "xwOBA",
        "xBA",
        "xSLG",
        "EV",
        "Barrel%",
        "HardHit%",
        "O-Swing%",
        "Whiff%",
        "K%",
        "BB%",
    ],
    "Fielding": [
        "DRS",
        "FRV",
        "OAA",
        "ARM",
    ],
    "Standard": [
        "bWAR",
        "WAR",
        "PA",
        "AVG",
        "OBP",
        "SLG",
        "OPS",
        "H",
        "2B",
        "3B",
        "HR",
        "XBH",
        "RBI",
        "SB",
        "R",
        "K%",
        "BB%",
    ],
    "Miscellaneous": [
        "K-BB%",
        "O-Swing%",
        "Z-Swing%",
        "Swing%",
        "Contact%",
        "WPA",
        "Clutch",
        "Pull%",
        "Cent%",
        "Oppo%",
        "GB%",
        "FB%",
        "LD%",
    ],
}

STAT_ALLOWLIST = [
    "Off", "Def", "BsR", "WAR", "bWAR", "Barrel%", "HardHit%", "EV", "MaxEV",
    "wRC+", "wOBA", "xwOBA", "xBA", "xSLG", "OPS", "SLG", "OBP", "AVG", "ISO",
    "BABIP", "G","PA", "AB", "R", "RBI", "HR", "XBH", "H", "2B", "3B", "SB", "BB", "IBB", "SO",
    "K%", "BB%", "K-BB%", "O-Swing%", "Z-Swing%", "Swing%", "Contact%", "WPA", "Clutch",
    "Whiff%", "Pull%", "Cent%", "Oppo%", "GB%", "FB%", "LD%", "LA",
    "DRS", "FRV", "OAA", "ARM", "RANGE", "TZ", "UZR", "FRM",
]

FIELDING_COLS = ["DRS", "TZ", "UZR", "FRM", "FRV", "OAA", "ARM", "RANGE"]
STATCAST_FIELDING_START_YEAR = 2016
LOCAL_BWAR_FILE = Path(__file__).with_name("warhitters2025.txt")


def normalize_name_key(val: str) -> str:
    return normalize_statcast_name(val)


def local_bwar_signature() -> float:
    try:
        return LOCAL_BWAR_FILE.stat().st_mtime
    except FileNotFoundError:
        return 0.0


def normalize_statcast_name(raw: str) -> str:
    if not raw or not isinstance(raw, str):
        return ""
    cleaned = raw.replace("\xa0", " ").strip()
    if "," in cleaned:
        last, first = cleaned.split(",", 1)
        full = f"{first.strip()} {last.strip()}"
    else:
        full = cleaned
    try:
        full = unicodedata.normalize("NFKD", full).encode("ascii", "ignore").decode()
    except Exception:
        pass
    return " ".join(full.split())



def reorder_savant_name(raw: str) -> str:
    if not raw or not isinstance(raw, str):
        return ""
    raw = raw.replace("\xa0", " ").strip()
    if "," in raw:
        last, first = raw.split(",", 1)
        return f"{first.strip()} {last.strip()}".strip()
    return raw


def fetch_csv(url: str) -> pd.DataFrame:
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
    except Exception:
        return pd.DataFrame()
    try:
        data = StringIO(resp.content.decode("utf-8"))
        df = pd.read_csv(data)
    except Exception:
        return pd.DataFrame()
    return df


def load_savant_frv_year(year: int) -> pd.DataFrame:
    url = (
        "https://baseballsavant.mlb.com/leaderboard/fielding-run-value?"
        f"gameType=Regular&seasonStart={year}&seasonEnd={year}"
        "&type=fielder&position=&minInnings=0&minResults=1&csv=true"
    )
    df = fetch_csv(url)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.rename(
        columns={
            "name": "NameRaw",
            "total_runs": "FRV",
            "arm_runs": "ARM",
            "range_runs": "RANGE",
        }
    )
    df["Name"] = df["NameRaw"].apply(reorder_savant_name)
    df["NameKey"] = df["Name"].apply(normalize_statcast_name)
    for metric in ["FRV", "ARM", "RANGE"]:
        df[metric] = pd.to_numeric(df.get(metric), errors="coerce")
    return df[["NameKey", "Name", "FRV", "ARM", "RANGE"]]


def load_savant_oaa_year(year: int) -> pd.DataFrame:
    try:
        df = statcast_outs_above_average(year, "all")
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    name_col = None
    for col in ["player_name", "last_name, first_name", "name"]:
        if col in df.columns:
            name_col = col
            break
    if not name_col:
        return pd.DataFrame()
    if name_col == "last_name, first_name":
        df["Name"] = df[name_col].apply(reorder_savant_name)
    else:
        df["Name"] = df[name_col].astype(str).str.strip()
    df["NameKey"] = df["Name"].apply(normalize_statcast_name)
    oaa_col = None
    for col in ["outs_above_average", "oaa"]:
        if col in df.columns:
            oaa_col = col
            break
    if not oaa_col:
        return pd.DataFrame()
    df["OAA"] = pd.to_numeric(df[oaa_col], errors="coerce")
    return df[["NameKey", "Name", "OAA"]]


@st.cache_data(show_spinner=False, ttl=3600)
def load_local_bwar_data():
    path = LOCAL_BWAR_FILE
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    pitcher_col = df.get("pitcher")
    if pitcher_col is not None:
        pitcher_mask = (
            pitcher_col.astype(str)
            .str.strip()
            .str.upper()
            .isin({"Y", "1", "TRUE"})
        )
        df = df[~pitcher_mask]
    df["Name"] = df.get("name_common", df.get("Name", "")).astype(str).str.strip()
    df["NameKey"] = df["Name"].apply(normalize_statcast_name)
    df["year_ID"] = pd.to_numeric(df.get("year_ID"), errors="coerce")
    df["WAR"] = pd.to_numeric(df.get("WAR"), errors="coerce")
    df = df.dropna(subset=["NameKey", "year_ID", "WAR"])
    return df[["NameKey", "Name", "year_ID", "WAR"]]


@st.cache_data(show_spinner=False, ttl=3600)
def load_bwar_dataset(local_sig: float) -> pd.DataFrame:
    _ = local_sig
    frames: list[pd.DataFrame] = []
    try:
        data = bwar_bat(return_all=True)
    except Exception:
        data = None
    if data is not None and not data.empty:
        data = data.copy()
        data["year_ID"] = pd.to_numeric(data.get("year_ID"), errors="coerce")
        data["WAR"] = pd.to_numeric(data.get("WAR"), errors="coerce")
        if "pitcher" in data.columns:
            data = data[pd.to_numeric(data["pitcher"], errors="coerce").fillna(1) == 0]
        data["Name"] = data["name_common"].astype(str).str.strip()
        data["NameKey"] = data["Name"].apply(normalize_statcast_name)
        frames.append(data[["NameKey", "Name", "year_ID", "WAR"]])

    local = load_local_bwar_data()
    if local is not None and not local.empty:
        frames.append(local)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["NameKey", "year_ID", "WAR"])
    combined = combined.sort_values(["NameKey", "year_ID"])
    combined = combined.drop_duplicates(subset=["NameKey", "year_ID"], keep="last")
    combined = combined.rename(columns={"WAR": "bWAR"})
    return combined.reset_index(drop=True)


@st.cache_data(show_spinner=False, ttl=900)
def load_bwar_for_year(year: int) -> pd.DataFrame:
    data = load_bwar_dataset(local_bwar_signature())
    if data is None or data.empty:
        return pd.DataFrame()
    df_year = data[pd.to_numeric(data["year_ID"], errors="coerce") == year].copy()
    if df_year.empty:
        return pd.DataFrame()
    agg = df_year.groupby("NameKey", as_index=False).agg({
        "bWAR": lambda s: s.sum(min_count=1),
        "Name": "first",
    })
    return agg


@st.cache_data(show_spinner=False, ttl=900)
def load_bwar_span(
    start_year: int,
    end_year: int,
    target_names: tuple[str, ...] | None = None,
    target_bbref: str | None = None,
    target_mlbam: int | None = None,
) -> pd.DataFrame:
    data = load_bwar_dataset(local_bwar_signature())
    if data is None or data.empty:
        return pd.DataFrame()
    mask = data["year_ID"].between(start_year, end_year)
    pool = data[mask]
    if pool.empty:
        return pd.DataFrame()

    def match_by_names() -> pd.DataFrame:
        if not target_names:
            return pd.DataFrame()
        keys = {normalize_statcast_name(name) for name in target_names if name}
        if not keys:
            return pd.DataFrame()
        return pool[pool["NameKey"].isin(keys)]

    def match_by_bbref() -> pd.DataFrame:
        if not target_bbref:
            return pd.DataFrame()
        slug = str(target_bbref).strip().lower()
        if not slug:
            return pd.DataFrame()
        if "player_ID" not in pool.columns:
            return pd.DataFrame()
        return pool[pool["player_ID"].astype(str).str.lower() == slug]

    def match_by_mlbam() -> pd.DataFrame:
        if target_mlbam is None:
            return pd.DataFrame()
        try:
            mlbam_val = int(target_mlbam)
        except Exception:
            return pd.DataFrame()
        if "mlb_ID" not in pool.columns:
            return pd.DataFrame()
        return pool[pool["mlb_ID"] == mlbam_val]

    df = match_by_names()
    if df.empty:
        alt = match_by_bbref()
        if not alt.empty:
            df = alt
    if df.empty:
        alt = match_by_mlbam()
        if not alt.empty:
            df = alt
    if df.empty:
        return pd.DataFrame()
    agg = df.groupby("NameKey", as_index=False).agg({
        "Name": "first",
        "bWAR": lambda s: s.sum(min_count=1),
    })
    return agg





@st.cache_data(show_spinner=False, ttl=900)
def load_player_fielding_profile(fg_id: int, start_year: int, end_year: int) -> dict[str, float]:
    try:
        df = fielding_stats(start_year, end_year, qual=0, split_seasons=False, players=str(fg_id))
    except Exception:
        df = None
    if df is None or df.empty:
        frames = []
        for year in range(start_year, end_year + 1):
            try:
                yearly = fielding_stats(year, year, qual=0, split_seasons=False, players=str(fg_id))
            except Exception:
                yearly = None
            if yearly is not None and not yearly.empty:
                frames.append(yearly)
        if not frames:
            return {}
        df = pd.concat(frames, ignore_index=True)
    result: dict[str, float] = {}
    for key in ["DRS", "TZ", "UZR", "FRM"]:
        if key in df.columns:
            series = pd.to_numeric(df[key], errors="coerce")
            if not series.isna().all():
                result[key] = series.sum(skipna=True)
    return result





@st.cache_data(show_spinner=False, ttl=900)
def load_player_fielding_profile_span(fg_id: int, start_year: int, end_year: int) -> dict[str, float]:
    try:
        df = fielding_stats(start_year, end_year, qual=0, split_seasons=False, players=str(fg_id))
    except Exception:
        df = None
    if df is None or df.empty:
        frames = []
        for year in range(start_year, end_year + 1):
            try:
                yearly = fielding_stats(year, year, qual=0, split_seasons=False, players=str(fg_id))
            except Exception:
                yearly = None
            if yearly is not None and not yearly.empty:
                frames.append(yearly)
        if not frames:
            return {}
        df = pd.concat(frames, ignore_index=True)
    result: dict[str, float] = {}
    for key in ["DRS", "TZ", "UZR", "FRM"]:
        if key in df.columns:
            series = pd.to_numeric(df[key], errors="coerce")
            if not series.isna().all():
                result[key] = series.sum(skipna=True)
    return result


@st.cache_data(show_spinner=False, ttl=900)
def load_statcast_fielding_span(
    start_year: int,
    end_year: int,
    target_names: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    start = min(start_year, end_year)
    end = max(start_year, end_year)
    start = max(start, STATCAST_FIELDING_START_YEAR)
    if start > end:
        return pd.DataFrame()
    name_filter = None
    if target_names:
        name_filter = {normalize_statcast_name(name) for name in target_names if name}
    frames: list[pd.DataFrame] = []
    for year in range(start, end + 1):
        frv = load_savant_frv_year(year)
        oaa = load_savant_oaa_year(year)
        if (frv is None or frv.empty) and (oaa is None or oaa.empty):
            continue
        if frv is None or frv.empty:
            yearly = oaa.copy()
        elif oaa is None or oaa.empty:
            yearly = frv.copy()
        else:
            yearly = pd.merge(
                frv,
                oaa,
                on=["NameKey", "Name"],
                how="outer",
            )
        for metric in ["FRV", "OAA", "ARM", "RANGE"]:
            if metric not in yearly.columns:
                yearly[metric] = np.nan
            else:
                yearly[metric] = pd.to_numeric(yearly[metric], errors="coerce")
        if name_filter is not None:
            yearly = yearly[yearly["NameKey"].isin(name_filter)]
            if yearly.empty:
                continue
        frames.append(yearly)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)

    def pick_name(series: pd.Series) -> str:
        for val in series:
            if isinstance(val, str) and val.strip():
                return val
        return ""

    agg = combined.groupby("NameKey", as_index=False).agg({
        "Name": pick_name,
        "FRV": lambda s: s.sum(min_count=1),
        "OAA": lambda s: s.sum(min_count=1),
        "ARM": lambda s: s.sum(min_count=1),
        "RANGE": lambda s: s.sum(min_count=1),
    })
    return agg


@st.cache_data(show_spinner=False, ttl=900)
def load_fielding_year(year: int) -> pd.DataFrame:
    try:
        df = fielding_stats(year, year, qual=0, split_seasons=False)
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["NameKey"] = df["Name"].astype(str).apply(normalize_statcast_name)
    for col in FIELDING_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan
    agg = df.groupby(["NameKey"], as_index=False)[FIELDING_COLS].sum(min_count=1)

    frv = load_savant_frv_year(year)
    if frv is not None and not frv.empty:
        agg = agg.merge(frv[["NameKey", "FRV", "ARM", "RANGE"]], on="NameKey", how="left")
    oaa = load_savant_oaa_year(year)
    if oaa is not None and not oaa.empty:
        agg = agg.merge(oaa[["NameKey", "OAA"]], on="NameKey", how="left")

    for col in FIELDING_COLS:
        if col not in agg.columns:
            agg[col] = np.nan
    return agg


# ...existing code...
@st.cache_data(show_spinner=True, ttl=900)
def load_batting(y: int) -> pd.DataFrame:
    base = batting_stats(y, y, qual=0)
    if base is None or base.empty:
        return pd.DataFrame()
    df = base.copy()

    # remove any accidental duplicate column names (prevents merge suffix errors)
    df = df.loc[:, ~df.columns.duplicated()]

    df["NameKey"] = df["Name"].astype(str).apply(normalize_statcast_name)

    bwar_df = load_bwar_for_year(y)
    if bwar_df is not None and not bwar_df.empty:
        df = df.merge(bwar_df[["NameKey", "bWAR"]], on="NameKey", how="left")
    field_df = load_fielding_year(y)
    if field_df is not None and not field_df.empty:
        df = df.merge(field_df, on="NameKey", how="left")

    # Merge statcast fielding metrics (FRV/OAA/ARM/RANGE) safely.
    statcast_df = load_statcast_fielding_span(y, y)
    if statcast_df is not None and not statcast_df.empty:
        metrics = ["FRV", "OAA", "ARM", "RANGE"]
        available = [m for m in metrics if m in statcast_df.columns]
        cols = ["NameKey"] + available
        if len(available) > 0:
            sc = statcast_df[cols].copy()

            # Rename any statcast metric columns that already exist in df to avoid merge conflicts.
            rename_map = {m: f"sc_{m}" for m in available if m in df.columns}
            if rename_map:
                sc = sc.rename(columns=rename_map)

            merge_cols = ["NameKey"] + [c for c in sc.columns if c != "NameKey"]
            df = df.merge(sc[merge_cols], on="NameKey", how="left")

            # For metrics that were merged as sc_<metric>, fill original metric from statcast and drop temp cols.
            for m in available:
                sc_col = f"sc_{m}"
                if sc_col in df.columns:
                    if m in df.columns:
                        df[m] = df[m].fillna(df[sc_col])
                    else:
                        df[m] = df[sc_col]
                    df.drop(columns=[sc_col], inplace=True)

    for col in ["bWAR"] + FIELDING_COLS:
        if col not in df.columns:
            df[col] = np.nan
    return df
# ...existing code...

# --------------------- Controls ---------------------
left_col, right_col = st.columns([1.2, 1.5])

with left_col:
    controls_container = st.container()
    stat_builder_container = st.container()

# Controls (top of left column)
with controls_container:
    today = date.today()
    current_year = today.year
    default_year = current_year if today.month >= 3 else current_year - 1
    year = st.number_input("Season", value=default_year, step=1, format="%d")
    player_mode = st.selectbox("Player Input", ["Name", "FanGraphs ID"], key="player_mode")
    default_player = st.session_state.get("player_select", "Mookie Betts")
    if player_mode == "Name":
        player_input = st.text_input("Player Name", value=default_player, key="player_select")
    else:
        player_input = st.text_input("Player FanGraphs ID", value=st.session_state.get("player_fg_id", ""), key="player_fg_id")
# --------------------- Data ---------------------
df = load_batting(year).copy()

if df is None or df.empty:
    fallback_df = pd.DataFrame()
    if year == current_year:
        fallback_year = year - 1
        fallback_df = load_batting(fallback_year).copy()
        if fallback_df is not None and not fallback_df.empty:
            st.info(f"No data returned for {year}. Showing {fallback_year} instead.")
            df = fallback_df
            year = fallback_year

if df is None or df.empty:
    st.error("No data returned from pybaseball.")
    st.stop()

df["PA"] = pd.to_numeric(df["PA"], errors="coerce")
df["Team"] = df["Team"].astype(str).str.upper()
df["Name"] = df["Name"].astype(str).str.replace(".", "", regex=False).str.strip()
if "Contact%" in df.columns:
    contact = pd.to_numeric(df["Contact%"], errors="coerce")
    needs_percent_scale = contact.dropna().abs().le(1).mean() > 0.9 if contact.notna().any() else False
    if needs_percent_scale:
        contact = contact * 100
    df["Contact%"] = contact
    df["Whiff%"] = 100 - contact
else:
    df["Whiff%"] = np.nan

# League for percentile distribution (Savant uses 340+ PA)
PCT_PA = 126 if year == 2020 else 340
league_for_pct = df[df["PA"] >= PCT_PA].copy()
if league_for_pct.empty:
    st.error(f"No league hitters ≥ {PCT_PA} PA in {year}.")
    st.stop()

player_row = None
if player_mode == "Name":
    name_input = player_input.strip()
    if not name_input:
        st.error("Enter a player name.")
        st.stop()
    matches = df[df["Name"].str.lower() == name_input.lower()]
    if matches.empty:
        st.error("Player not found for that name.")
        st.stop()
    player_row = matches.sort_values("PA", ascending=False).head(1).squeeze()
else:
    try:
        fg_id = int(str(player_input).strip())
    except Exception:
        st.error("Enter a valid FanGraphs ID.")
        st.stop()
    id_col = pd.to_numeric(df.get("IDfg"), errors="coerce")
    matches = df[id_col == fg_id]
    if matches.empty:
        st.error("Player not found for that FanGraphs ID.")
        st.stop()
    player_row = matches.sort_values("PA", ascending=False).head(1).squeeze()

if player_row is None or player_row.empty or pd.isna(player_row.get("PA", np.nan)):
    st.error("Selected player has no valid data.")
    st.stop()
player_name = str(player_row.get("Name", "")).strip()
player_name_key = normalize_statcast_name(player_name)

fg_id_val = pd.to_numeric(player_row.get("IDfg"), errors="coerce")
if pd.notna(fg_id_val):
    fielding_profile = load_player_fielding_profile(int(fg_id_val), year, year)
    for key, val in fielding_profile.items():
        player_row[key] = val

statcast_field = load_statcast_fielding_span(year, year, target_names=(player_name_key,))
if statcast_field is not None and not statcast_field.empty:
    stat_row = statcast_field.iloc[0]
    for metric in ["FRV", "OAA", "ARM", "RANGE"]:
        player_row[metric] = pd.to_numeric(stat_row.get(metric), errors="coerce")
for metric in ["FRV", "OAA", "ARM", "RANGE"]:
    if metric not in player_row.index:
        player_row[metric] = np.nan
for metric in FIELDING_COLS:
    if metric not in player_row.index:
        player_row[metric] = np.nan
    else:
        player_row[metric] = pd.to_numeric(player_row[metric], errors="coerce")

# Collect teams the player appeared for this season (exclude TOT aggregate)
player_teams_raw = (
    df[df["Name"] == player_name]["Team"]
    .dropna()
    .astype(str)
    .str.upper()
    .unique()
    .tolist()
)
placeholders = {"TOT", "- - -", "---", "--", ""}
player_teams = [t for t in player_teams_raw if t not in placeholders]
# If no clean teams, fall back to TOT (if present) or the raw team value
if not player_teams:
    if "- - -" in player_teams_raw:
        player_teams = ["2+ Tms"]
    elif "TOT" in player_teams_raw:
        player_teams = ["TOT"]
    else:
        raw_team = str(player_row.get("Team", "N/A")).upper()
        player_teams = [raw_team] if raw_team not in placeholders else ["N/A"]

if len(player_teams) > 1:
    player_team_display = f"{len(player_teams)} Tms"
else:
    player_team_display = player_teams[0]

# --------------------- Stat builder setup ---------------------
numeric_stats = [
    col for col in df.columns
    if pd.api.types.is_numeric_dtype(df[col])
]
stat_exclusions = {"Season"}
numeric_stats = [col for col in numeric_stats if col not in stat_exclusions]

preferred_stats = [stat for stat in STAT_ALLOWLIST if stat in numeric_stats]
other_stats = [stat for stat in numeric_stats if stat not in preferred_stats]
stat_options = preferred_stats + other_stats
allowed_add_stats = preferred_stats if preferred_stats else stat_options.copy()

if not stat_options:
    st.error("No numeric stats available to display.")
    st.stop()

default_preset_name = "Statcast"
stat_preset_key = "stat_preset_select"
preset_options = list(STAT_PRESETS.keys())
stat_state_key = "stat_config"
manual_stat_update_key = "stat_config_manual_update"
add_select_key = "add_stat_select"
remove_select_key = "remove_stat_select"
add_reset_key = "reset_add_select"
remove_reset_key = "reset_remove_select"
stat_version_key = "stat_config_version"

# --- Callbacks ---
def bump_stat_config_version():
    st.session_state[stat_version_key] = st.session_state.get(stat_version_key, 0) + 1


def add_stat_callback(stat_key: str, select_key: str, reset_key: str, sentinel: str):
    choice = st.session_state.get(select_key)
    if not choice or choice == sentinel:
        return
    current_preset_for_base = st.session_state.get(stat_preset_key, default_preset_name)
    preset_base_candidates = [stat for stat in STAT_PRESETS[current_preset_for_base] if stat in stat_options]
    if not preset_base_candidates and stat_options:
        preset_base_candidates = [stat_options[0]]
    preset_base_config = [{"Stat": stat, "Show": True} for stat in preset_base_candidates]

    config = st.session_state.get(stat_key, preset_base_config)
    config = normalize_stat_rows(config, preset_base_config)
    if not any(row["Stat"] == choice for row in config):
        config.append({"Stat": choice, "Show": True})
    st.session_state[stat_key] = config
    bump_stat_config_version()
    st.session_state[manual_stat_update_key] = True
    st.session_state[reset_key] = True


def remove_stat_callback(stat_key: str, select_key: str, reset_key: str, sentinel: str):
    choice = st.session_state.get(select_key)
    if not choice or choice == sentinel:
        return
    current_preset_for_base = st.session_state.get(stat_preset_key, default_preset_name)
    preset_base_candidates = [stat for stat in STAT_PRESETS[current_preset_for_base] if stat in stat_options]
    if not preset_base_candidates and stat_options:
        preset_base_candidates = [stat_options[0]]
    preset_base_config = [{"Stat": stat, "Show": True} for stat in preset_base_candidates]

    config = st.session_state.get(stat_key, preset_base_config)
    config = normalize_stat_rows(config, preset_base_config)
    new_config = [row for row in config if row.get("Stat") != choice]
    st.session_state[stat_key] = new_config or [row.copy() for row in preset_base_config]
    bump_stat_config_version()
    st.session_state[manual_stat_update_key] = True
    st.session_state[reset_key] = True


def stat_preset_callback(preset_key: str, stat_key: str, available_stats: list[str]):
    preset_name = st.session_state.get(preset_key, default_preset_name)
    preset_stats = STAT_PRESETS.get(preset_name, [])
    filtered_stats = [stat for stat in preset_stats if stat in available_stats]
    if not filtered_stats and available_stats:
        filtered_stats = [available_stats[0]]
    if not filtered_stats:
        return
    st.session_state[stat_key] = [{"Stat": stat, "Show": True} for stat in filtered_stats]
    bump_stat_config_version()
    st.session_state[manual_stat_update_key] = True
    st.session_state[add_reset_key] = True
    st.session_state[remove_reset_key] = True


def normalize_stat_rows(rows, fallback):
    cleaned = []
    seen_stats = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        stat_name = row.get("Stat")
        if (
            not stat_name
            or stat_name not in stat_options
            or stat_name in seen_stats
        ):
            continue
        show_val = row.get("Show", True)
        if pd.isna(show_val):
            show_bool = True
        elif isinstance(show_val, str):
            show_bool = show_val.strip().lower() in TRUTHY_STRINGS
        else:
            show_bool = bool(show_val)
        cleaned.append({"Stat": stat_name, "Show": show_bool})
        seen_stats.add(stat_name)
    if not cleaned:
        cleaned = [row.copy() for row in fallback]
    return cleaned
# --- End Callbacks ---


def move_stat_row(delta: int, index: int, fallback):
    """Move a stat row up/down and persist."""
    rows = normalize_stat_rows(st.session_state.get(stat_state_key, fallback), fallback)
    target = index + delta
    if 0 <= target < len(rows):
        rows[index], rows[target] = rows[target], rows[index]
        st.session_state[stat_state_key] = rows
        bump_stat_config_version()
        st.session_state[manual_stat_update_key] = True


def toggle_stat_show(index: int, state_key: str, fallback):
    """Toggle the Show flag for a row and persist."""
    rows = normalize_stat_rows(st.session_state.get(stat_state_key, fallback), fallback)
    if 0 <= index < len(rows):
        rows[index]["Show"] = bool(st.session_state.get(state_key, True))
        st.session_state[stat_state_key] = rows
        bump_stat_config_version()
        st.session_state[manual_stat_update_key] = True

# Initialize state once
if stat_state_key not in st.session_state:
    st.session_state[stat_preset_key] = default_preset_name
    current_preset_for_base = st.session_state[stat_preset_key]
    preset_base_candidates = [stat for stat in STAT_PRESETS[current_preset_for_base] if stat in stat_options]
    if not preset_base_candidates and stat_options:
        preset_base_candidates = [stat_options[0]]
    preset_base_config = [{"Stat": stat, "Show": True} for stat in preset_base_candidates]
    st.session_state[stat_state_key] = preset_base_config
    st.session_state[stat_version_key] = 0
elif stat_version_key not in st.session_state:
    st.session_state[stat_version_key] = 0

current_preset_for_base = st.session_state.get(stat_preset_key, default_preset_name)
preset_base_candidates = [stat for stat in STAT_PRESETS[current_preset_for_base] if stat in stat_options]
if not preset_base_candidates and stat_options:
    preset_base_candidates = [stat_options[0]]
preset_base_config = [{"Stat": stat, "Show": True} for stat in preset_base_candidates]

current_stat_config = st.session_state.get(stat_state_key, preset_base_config)
current_stat_config = normalize_stat_rows(current_stat_config, preset_base_config)

with stat_builder_container:
    prior_preset = st.session_state.get(stat_preset_key, default_preset_name)
    preset_index = preset_options.index(prior_preset) if prior_preset in preset_options else 0
    st.selectbox(
        "Stat Preset",
        preset_options,
        index=preset_index,
        key=stat_preset_key,
        on_change=stat_preset_callback,
        args=(stat_preset_key, stat_state_key, stat_options),
    )

    st.markdown("### Customize stats")
    st.markdown(
        "<div style='margin-bottom: -0.25rem; color: inherit; font-size: 0.9rem;'>"
        "Use the drop downs to add or remove stats and the arrows to reorder."
        "</div>",
        unsafe_allow_html=True,
    )

    stats_in_config = [row.get("Stat") for row in current_stat_config if row.get("Stat")]
    available_pool = allowed_add_stats if allowed_add_stats else stat_options
    available_stats = [stat for stat in available_pool if stat not in stats_in_config]

    add_col, remove_col = st.columns(2)
    sentinel_add = "Select stat to add"
    sentinel_remove = "Select stat to remove"

    add_options = [sentinel_add] + available_stats
    remove_options = [sentinel_remove] + stats_in_config

    if st.session_state.get(add_select_key) not in add_options:
        st.session_state[add_select_key] = sentinel_add
    if st.session_state.pop(add_reset_key, False):
        st.session_state[add_select_key] = sentinel_add

    if st.session_state.get(remove_select_key) not in remove_options:
        st.session_state[remove_select_key] = sentinel_remove
    if st.session_state.pop(remove_reset_key, False):
        st.session_state[remove_select_key] = sentinel_remove

    with add_col:
        st.selectbox(
            "Add stat",
            add_options,
            label_visibility="hidden",
            key=add_select_key,
            on_change=add_stat_callback,
            args=(stat_state_key, add_select_key, add_reset_key, sentinel_add),
            format_func=lambda s: STAT_DISPLAY_NAMES.get(s, s) if s != sentinel_add else s,
        )

    with remove_col:
        st.selectbox(
            "Remove stat",
            remove_options,
            label_visibility="hidden",
            key=remove_select_key,
            on_change=remove_stat_callback,
            args=(stat_state_key, remove_select_key, remove_reset_key, sentinel_remove),
            format_func=lambda s: STAT_DISPLAY_NAMES.get(s, s) if s != sentinel_remove else s,
        )

    current_stat_config = normalize_stat_rows(st.session_state.get(stat_state_key, preset_base_config), preset_base_config)

    st.markdown("#### Order & visibility")

    st.markdown('<div class="stat-table">', unsafe_allow_html=True)
    st.markdown('<div class="table-header">', unsafe_allow_html=True)
    header_cols = st.columns([0.25, 0.25, 0.25, 0.25])
    header_cols[0].markdown("**Up**")
    header_cols[1].markdown("**Down**")
    header_cols[2].markdown("**Stat**")
    header_cols[3].markdown("**Show**")
    st.markdown('</div>', unsafe_allow_html=True)

    for idx, row in enumerate(current_stat_config):
        st.markdown('<div class="table-row">', unsafe_allow_html=True)
        up_col, down_col, stat_col, show_col = st.columns([0.25, 0.25, 0.25, 0.25])
        with up_col:
            st.button(
                "▲",
                key=f"stat_up_{idx}",
                disabled=idx == 0,
                on_click=move_stat_row,
                args=(-1, idx, preset_base_config),
            )
        with down_col:
            st.button(
                "▼",
                key=f"stat_down_{idx}",
                disabled=idx == len(current_stat_config) - 1,
                on_click=move_stat_row,
                args=(1, idx, preset_base_config),
            )
        with stat_col:
            stat_name = row.get("Stat", "")
            display_name = STAT_DISPLAY_NAMES.get(stat_name, stat_name)
            st.write(display_name)
        with show_col:
            checkbox_key = f"stat_show_{idx}"
            st.checkbox(
                "",
                value=bool(row.get("Show", True)),
                key=checkbox_key,
                label_visibility="collapsed",
                on_change=toggle_stat_show,
                args=(idx, checkbox_key, preset_base_config),
            )
        st.markdown('</div>', unsafe_allow_html=True)

    cleaned_config = normalize_stat_rows(st.session_state.get(stat_state_key, current_stat_config), preset_base_config)
    st.session_state[stat_state_key] = cleaned_config
    st.markdown('</div>', unsafe_allow_html=True)

stats_order = [row["Stat"] for row in st.session_state[stat_state_key] if row.get("Show", True)]
if not stats_order:
    st.info("Add at least one stat and mark it as shown to build the chart.")
    st.stop()

# --------------------- Formatting ---------------------
def format_stat(stat: str, val) -> str:
    if pd.isna(val):
        return ""

    upper_stat = stat.upper()
    if upper_stat == "FRV":
        return f"{float(val):.0f}"
    if upper_stat in {"WAR", "BWAR", "FWAR", "EV", "AVG EXIT VELO", "OFF", "DEF", "BSR"}:
        v = float(val)
        if abs(v - round(v)) < 1e-9:
            return f"{int(round(v))}.0"
        return f"{v:.1f}"

    if upper_stat in {"WPA", "CLUTCH"}:
        return f"{float(val):.2f}"

    if upper_stat in {"AVG", "OBP", "SLG", "OPS", "WOBA", "XWOBA", "XBA", "XSLG", "BABIP", "ISO"}:
        return f"{float(val):.3f}".lstrip("0")

    if (
        "Barrel" in stat or "Hard" in stat or "K%" in stat
        or "Swing" in stat or "Whiff" in stat or "%" in stat
    ):
        v = float(val)
        if v <= 1:
            v *= 100
        return f"{v:.1f}%"

    v = float(val)
    return f"{v:.0f}" if abs(v - round(v)) < 1e-6 else f"{v:.1f}"


# --------------------- Build rows ---------------------
leaders = []

label_map = {
    **STAT_DISPLAY_NAMES,
    "EV": "Avg Exit Velo",
}
lower_better = {"K%", "O-Swing%", "Whiff%", "GB%"}

for stat in stats_order:
    if stat not in df.columns:
        continue

    player_val = player_row.get(stat, np.nan)
    if pd.isna(player_val):
        continue

    league_vals = league_for_pct[stat].dropna()
    if league_vals.empty:
        continue

    pct = (league_vals <= player_val).mean() * 100.0
    if stat in lower_better:
        pct = 100 - pct
    pct = float(np.clip(pct, 0, 100))

    label = label_map.get(stat, stat)

    leaders.append({
        "Stat": label,
        "Leader": player_name,
        "Value": float(player_val),
        "Pct": pct
    })

lead_df = pd.DataFrame(leaders)

if lead_df.empty:
    st.warning("No stats available to display.")
    st.stop()

lead_df["Display"] = lead_df.apply(lambda r: format_stat(r["Stat"], r["Value"]), axis=1)

with right_col:
    cmap = LinearSegmentedColormap.from_list(
        "savant",
        [
            (0, "#335AA1"),
            (0.5, "#E8E8E8"),
            (1, "#D92229"),
        ],
    )

    fig_height = 1.7 + len(lead_df) * 0.40
    fig, ax = plt.subplots(figsize=(7.5, fig_height))

    top_pad = 0.12 if len(lead_df) > 6 else 0.14 if len(lead_df) > 4 else 0.4
    ax_height = 0.85 - top_pad
    ax.set_position([0.08, top_pad, 0.8, ax_height])

    fig.text(
        0.5, 0.885,
        f"{year} {player_name}" + (f" | {player_team_display}" if player_team_display else ""),
        ha="center", va="center",
        fontsize=22, fontweight="bold"
    )

    fig.text(
        0.2, 0.08,
        "By: Sox_Savant",
        ha="center", va="center",
        fontsize=13, color="#555"
    )
    fig.text(
        0.75, 0.08,
        "Data: FanGraphs",
        ha="center", va="center",
        fontsize=13, color="#555"
    )

    y = np.arange(len(lead_df))

    TRACK_H = 0.82
    BAR_H = 0.82
    LEFT_OFFSET = 3
    BAR_LENGTH = 45 
    VALUE_X = LEFT_OFFSET + BAR_LENGTH + 12
    BUBBLE_SIZE = 700

    ax.barh(
        y,
        BAR_LENGTH,
        left=LEFT_OFFSET,
        height=TRACK_H,
        color="#F1F1F1",
        edgecolor="none",
    )

    for i, row in lead_df.iterrows():
        pct = row["Pct"]
        color = cmap(pct / 100)

        bar_width = pct / 100 * BAR_LENGTH
        ax.barh(
            i,
            bar_width,
            left=LEFT_OFFSET,
            height=BAR_H,
            color=color,
            edgecolor="none",
        )

        bubble_x = LEFT_OFFSET + bar_width

        ax.scatter(bubble_x, i, s=BUBBLE_SIZE, color=color,
                   edgecolors="white", linewidth=2.4, zorder=3)

        ax.text(
            bubble_x, i + 0.04, f"{int(round(pct))}",
            ha="center", va="center",
            fontsize=11, fontweight="bold", color="white"
        )

        ax.text(VALUE_X - 9, i, row["Display"],
                ha="left", va="center", fontsize=12, color="#111")

        ax.text(0, i, row["Stat"],
                ha="right", va="center", fontsize=13,)

    # Add translucent percentile guide lines at 10th, 50th, 90th
    for pos in (0.1, 0.5, 0.9):
        guide_x = LEFT_OFFSET + BAR_LENGTH * pos
        ax.vlines(
            guide_x,
            -0.5,
            len(lead_df) - 0.5,
            colors="white",
            linewidth=1.2,
            alpha=0.25,
            zorder=2.6,
        )

    ax.set_xlim(-10, VALUE_X)
    ax.set_ylim(-0.5, len(lead_df) - 0.5)
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")

    # Render to both PNG (for on-page display) and PDF (for download) to reduce
    # chances of blank renders on some hosted envs.
    png_buffer = BytesIO()
    fig.savefig(png_buffer, format="png", bbox_inches="tight", pad_inches=.25, dpi=220)
    png_buffer.seek(0)

    pdf_buffer = BytesIO()
    fig.savefig(pdf_buffer, format="pdf", bbox_inches="tight", pad_inches=.25)
    pdf_buffer.seek(0)

    st.image(png_buffer, use_container_width=True)
    download_name = f"{player_name.replace(' ', '_')}_{year}_savant.pdf"
    st.download_button(
        "Download as PDF",
        data=pdf_buffer,
        file_name=download_name,
        mime="application/pdf",
    )
    plt.close(fig)

    st.caption("Percentiles based on players with at least 2.1 PA per team game, or 340 PA over a full season")
