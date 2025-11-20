import os
import time
import unicodedata
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from io import BytesIO, StringIO
from datetime import date
from pathlib import Path
import re
import requests
os.environ.setdefault("AGGRID_RELEASE", "True")
from pybaseball import batting_stats, fielding_stats, bwar_bat
from pybaseball.statcast_fielding import statcast_outs_above_average
from st_aggrid import (
    AgGrid,
    GridOptionsBuilder,
    GridUpdateMode,
    DataReturnMode,
    JsCode,
)


def safe_aggrid(df, **kwargs):
    """
    Retries AG Grid loading up to 3 times to avoid Streamlit component
    handshake failures in production environments like Cloud Run.
    """
    class _DFProxy(pd.DataFrame):
        @property
        def _constructor(self):
            return _DFProxy
        def __bool__(self):
            return True

    grid_opts = kwargs.get("gridOptions", {})
    if "rowData" in grid_opts and grid_opts["rowData"] is not None:
        grid_opts["rowData"] = None
    kwargs["gridOptions"] = grid_opts

    data_arg = _DFProxy(df) if isinstance(df, pd.DataFrame) else df
    for attempt in range(3):
        try:
            return AgGrid(data=data_arg, **kwargs)
        except Exception:
            if attempt == 2:
                raise  # rethrow after last attempt
            time.sleep(0.3)



GRID_THEME = "balham"
GRID_CUSTOM_CSS = {
    ".ag-root-wrapper": {"border": "1px solid #2d2d2d"},
    ".ag-root": {"background-color": "#1b1b1d"},
    ".ag-header": {"background-color": "#2c2c2c", "color": "#dcdcdc"},
    ".ag-header-row": {"background-color": "#2c2c2c", "color": "#dcdcdc"},
    ".ag-row": {"color": "#e0e0e0"},
    ".ag-row-odd": {"background-color": "#1f1f1f"},
    ".ag-row-even": {"background-color": "#242424"},
    ".ag-center-cols-viewport": {"background-color": "#1b1b1d"},
    ".ag-body-viewport": {"background-color": "#1b1b1d"},
    ".ag-center-cols-container": {"background-color": "#1b1b1d"},
    ".ag-body-horizontal-scroll-viewport": {"background-color": "#1b1b1d"},
    ".ag-body-vertical-scroll-viewport": {"background-color": "#1b1b1d"},
    ".ag-rich-select-popup": {"background-color": "#1b1b1d", "color": "#f0f0f0"},
    ".ag-rich-select-list": {"background-color": "#1b1b1d"},
    ".ag-virtual-list-viewport": {"background-color": "#1b1b1d"},
    ".ag-list-item": {"background-color": "#1b1b1d", "color": "#f0f0f0"},
    ".ag-list-item.ag-active-item": {"background-color": "#2c2c2c", "color": "#f0f0f0"},
    ".ag-rich-select-value": {"color": "#f0f0f0"},
}

plt.rcdefaults()

st.set_page_config(page_title="Custom Savant Page App", layout="wide")

st.markdown(
    """
    <style>
        [data-testid="stToolbar"] {visibility: hidden;}
        [data-testid="stDecoration"] {display: none;}
        [data-testid="stStatusWidget"] {display: none;}
        .viewerBadge_link__qRi_k {display: none;}
        /* Hide slider end labels */
        [data-testid="stTickBarMin"],
        [data-testid="stTickBarMax"] {display: none;}
        /* Keep row selector column hidden without touching data columns */
        div.ag-header-cell[col-id="ag-RowSelector"],
        div.ag-pinned-left-cols-container [col-id="ag-RowSelector"],
        div.ag-center-cols-container [col-id="ag-RowSelector"] {
            display: none !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

title_col, meta_col = st.columns([3, 1])
with title_col:
    st.title("Custom Savant Page App")
with meta_col:
    st.markdown(
        """
        <div style="text-align: right; font-size: 1rem; padding-top: 0.6rem;">
            Built by <a href="https://twitter.com/Sox_Savant" target="_blank">@Sox_Savant</a>
            <span style="color: #aaa;">(v 1.0)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

TRUTHY_STRINGS = {"true", "1", "yes", "y", "t"}

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
        "TZ",
        "UZR",
        "FRM",
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

FIELDING_COLS = ["DRS", "TZ", "UZR", "FRM"]
STATCAST_FIELDING_START_YEAR = 2016
LOCAL_BWAR_FILE = Path(__file__).with_name("warhitters2025.txt")


def normalize_name_key(val: str) -> str:
    if val is None:
        return ""
    txt = str(val).strip()
    try:
        txt = unicodedata.normalize("NFKD", txt).encode("ascii", "ignore").decode()
    except Exception:
        pass
    return " ".join(txt.split()).lower()


def local_bwar_signature() -> float:
    try:
        return LOCAL_BWAR_FILE.stat().st_mtime
    except FileNotFoundError:
        return 0.0


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
    df["NameKey"] = df["Name"].apply(normalize_name_key)
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
        data["NameKey"] = data["Name"].apply(normalize_name_key)
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
def load_fielding_year(year: int) -> pd.DataFrame:
    try:
        df = fielding_stats(year, year, qual=0, split_seasons=False)
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["NameKey"] = df["Name"].astype(str).apply(normalize_name_key)
    for col in FIELDING_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan
    agg = df.groupby(["NameKey"], as_index=False)[FIELDING_COLS].sum(min_count=1)
    return agg


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
    df["NameKey"] = df["Name"].apply(normalize_name_key)
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
    df["NameKey"] = df["Name"].apply(normalize_name_key)
    oaa_col = None
    for col in ["outs_above_average", "oaa"]:
        if col in df.columns:
            oaa_col = col
            break
    if not oaa_col:
        return pd.DataFrame()
    df["OAA"] = pd.to_numeric(df[oaa_col], errors="coerce")
    return df[["NameKey", "Name", "OAA"]]


@st.cache_data(show_spinner=False, ttl=900)
def load_statcast_fielding_year(year: int) -> pd.DataFrame:
    if year < STATCAST_FIELDING_START_YEAR:
        return pd.DataFrame()
    frv = load_savant_frv_year(year)
    oaa = load_savant_oaa_year(year)
    if (frv is None or frv.empty) and (oaa is None or oaa.empty):
        return pd.DataFrame()
    if frv is None or frv.empty:
        combined = oaa.copy()
    elif oaa is None or oaa.empty:
        combined = frv.copy()
    else:
        combined = pd.merge(
            frv,
            oaa,
            on=["NameKey", "Name"],
            how="outer",
        )
    for metric in ["FRV", "OAA", "ARM", "RANGE"]:
        if metric not in combined.columns:
            combined[metric] = np.nan
        else:
            combined[metric] = pd.to_numeric(combined[metric], errors="coerce")
    agg = combined.groupby("NameKey", as_index=False).agg({
        "Name": "first",
        "FRV": lambda s: s.sum(min_count=1),
        "OAA": lambda s: s.sum(min_count=1),
        "ARM": lambda s: s.sum(min_count=1),
        "RANGE": lambda s: s.sum(min_count=1),
    })
    return agg


@st.cache_data(show_spinner=True, ttl=900)
def load_batting(y: int) -> pd.DataFrame:
    base = batting_stats(y, y, qual=0)
    if base is None or base.empty:
        return pd.DataFrame()
    df = base.copy()
    df["NameKey"] = df["Name"].astype(str).apply(normalize_name_key)

    bwar_df = load_bwar_for_year(y)
    if bwar_df is not None and not bwar_df.empty:
        df = df.merge(bwar_df[["NameKey", "bWAR"]], on="NameKey", how="left")
    field_df = load_fielding_year(y)
    if field_df is not None and not field_df.empty:
        df = df.merge(field_df, on="NameKey", how="left")
    statcast_df = load_statcast_fielding_year(y)
    if statcast_df is not None and not statcast_df.empty:
        df = df.merge(statcast_df[["NameKey", "FRV", "OAA", "ARM", "RANGE"]], on="NameKey", how="left")
    for col in ["bWAR"] + FIELDING_COLS:
        if col not in df.columns:
            df[col] = np.nan
    for col in ["FRV", "OAA", "ARM", "RANGE"]:
        if col not in df.columns:
            df[col] = np.nan
    return df


# --------------------- Controls ---------------------
left_col, right_col = st.columns([1.2, 1.5])

with left_col:
    controls_container = st.container()
    stat_builder_container = st.container()

with controls_container:
    year = st.slider("Season", 1900, date.today().year, date.today().year)

    # Player input controls (name or FanGraphs ID)
    player_mode = st.selectbox(
        "Player Input",
        ["Name", "FanGraphs ID"],
        key="player_input_mode",
    )
    if player_mode == "Name":
        player_name_input = st.text_input(
            "Player Name",
            st.session_state.get("player_name_input", "Mookie Betts"),
            key="player_name_input",
        )
        player_id_input = st.session_state.get("player_id_input", "")
    else:
        player_id_input = st.text_input(
            "Player FanGraphs ID",
            st.session_state.get("player_id_input", ""),
            key="player_id_input",
        )
        player_name_input = st.session_state.get("player_name_input", "")
# --------------------- Data ---------------------
df = load_batting(year).copy()

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

def resolve_player_row(df: pd.DataFrame, mode: str, name_input: str, fg_id_input: str) -> pd.Series | None:
    name = name_input.strip()
    fg_raw = str(fg_id_input).strip()
    if mode == "FanGraphs ID":
        if not fg_raw:
            st.warning("Enter a FanGraphs ID or switch to Name input.")
            return None
        try:
            fg_id = int(fg_raw)
        except Exception:
            st.error("FanGraphs ID must be an integer.")
            return None
        # Try matching in the loaded dataframe first
        if "IDfg" in df.columns:
            match = df[pd.to_numeric(df["IDfg"], errors="coerce") == fg_id]
            if not match.empty:
                return match.sort_values("PA", ascending=False).head(1).squeeze()
        # Fallback: fetch directly by FG ID to be resilient
        try:
            fetched = batting_stats(year, year, qual=0, split_seasons=False, players=str(fg_id))
        except Exception:
            fetched = None
        if fetched is not None and not fetched.empty:
            fetched = fetched.copy()
            fetched["Name"] = fetched["Name"].astype(str).str.replace(".", "", regex=False).str.strip()
            fetched["NameKey"] = fetched["Name"].astype(str).apply(normalize_name_key)
            # Merge bWAR/fielding by NameKey
            bwar_df = load_bwar_for_year(year)
            field_df = load_fielding_year(year)
            if bwar_df is not None and not bwar_df.empty:
                fetched = fetched.merge(bwar_df[["NameKey", "bWAR"]], on="NameKey", how="left")
            if field_df is not None and not field_df.empty:
                fetched = fetched.merge(field_df, on="NameKey", how="left")
            for col in ["bWAR"] + FIELDING_COLS:
                if col not in fetched.columns:
                    fetched[col] = np.nan
            return fetched.head(1).squeeze()
        st.error(f"Could not find data for FG ID {fg_id}.")
        return None

    # Name mode
    if not name:
        st.warning("Enter a player name or switch to FanGraphs ID input.")
        return None
    target_key = normalize_name_key(name)
    if "NameKey" in df.columns:
        match = df[df["NameKey"] == target_key]
        if not match.empty:
            return match.sort_values("PA", ascending=False).head(1).squeeze()
    # Fallback: exact name match
    match = df[df["Name"].str.casefold() == name.casefold()]
    if not match.empty:
        return match.sort_values("PA", ascending=False).head(1).squeeze()
    st.error(f"Could not find data for {name}.")
    return None


player_row = resolve_player_row(df, player_mode, player_name_input, player_id_input)
if player_row is None or player_row.empty or pd.isna(player_row.get("PA", np.nan)):
    st.stop()

player_name = str(player_row.get("Name", "")).strip()
if not player_name:
    player_name = player_name_input if player_mode == "Name" else f"FG#{player_id_input or 'N/A'}"
player_name_key = str(player_row.get("NameKey", normalize_name_key(player_name)))

# Collect teams the player appeared for this season (exclude TOT aggregate)
if "NameKey" in df.columns:
    player_teams_raw = (
        df[df["NameKey"] == player_name_key]["Team"]
        .dropna()
        .astype(str)
        .str.upper()
        .unique()
        .tolist()
    )
else:
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

# --- AG Grid Checkbox Renderer JS ---
show_checkbox_renderer = JsCode(
    """
    class ShowCheckboxRenderer {
        init(params) {
            this.params = params;
            this.eGui = document.createElement('div');
            this.eGui.style.display = 'flex';
            this.eGui.style.justifyContent = 'center';
            this.eGui.style.alignItems = 'center';
            this.eGui.style.height = '100%';
            this.eGui.style.width = '100%';
            this.checkbox = document.createElement('input');
            this.checkbox.type = 'checkbox';
            this.checkbox.checked = Boolean(params.value);
            this.checkbox.addEventListener('change', () => {
                params.node.setDataValue(params.column.colId, this.checkbox.checked);
            });
            this.eGui.appendChild(this.checkbox);
        }
        getGui() {
            return this.eGui;
        }
        refresh(params) {
            this.checkbox.checked = Boolean(params.value);
            return true;
        }
    }
    """
)
# --- End AG Grid Checkbox Renderer JS ---

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
        "Drag to reorder. Use the drop downs to add or remove stats."
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
        )

    with remove_col:
        st.selectbox(
            "Remove stat",
            remove_options,
            label_visibility="hidden",
            key=remove_select_key,
            on_change=remove_stat_callback,
            args=(stat_state_key, remove_select_key, remove_reset_key, sentinel_remove),
        )

    current_stat_config = normalize_stat_rows(st.session_state.get(stat_state_key, preset_base_config), preset_base_config)

    stat_config_df = pd.DataFrame(current_stat_config)
    if stat_config_df.empty:
        stat_config_df = pd.DataFrame(preset_base_config)
    if "Show" not in stat_config_df.columns:
        stat_config_df["Show"] = True
    if "Stat" not in stat_config_df.columns:
        stat_config_df["Stat"] = preset_base_config[0]["Stat"]
    stat_config_df = stat_config_df[["Show", "Stat"]].copy()
    stat_config_df["Show"] = stat_config_df["Show"].apply(
        lambda val: True
        if pd.isna(val)
        else val.strip().lower() in TRUTHY_STRINGS
        if isinstance(val, str)
        else bool(val)
    )
    stat_config_df.insert(0, "Drag", ["↕"] * len(stat_config_df))

    gb = GridOptionsBuilder.from_dataframe(stat_config_df)
    gb.configure_default_column(
        editable=True,
        filter=False,
        sortable=False,
        resizable=True,
    )
    gb.configure_selection(selection_mode="disabled", use_checkbox=False)
    gb.configure_grid_options(
        rowDragManaged=True,
        rowDragMultiRow=True,
        rowDragEntireRow=True,
        animateRows=True,
        suppressMovableColumns=True,
        suppressRowClickSelection=True,
        singleClickEdit=True,
        stopEditingWhenCellsLoseFocus=True,
    )
    gb.configure_column(
        "Drag",
        header_name="",
        rowDrag=True,
        editable=False,
        width=70,
        suppressMenu=True,
        suppressSizeToFit=True,
    )
    gb.configure_column(
        "Show",
        header_name="Show",
        cellRenderer=show_checkbox_renderer,
        editable=False,
        width=100,
    )
    gb.configure_column(
        "Stat",
        header_name="Stat",
        editable=True,
        cellEditor="agSelectCellEditor",
        cellEditorParams={"values": allowed_add_stats or stat_options},
        flex=1,
    )

    grid_options = gb.build()
    grid_options["rowData"] = None

    grid_height = min(480, 90 + len(stat_config_df) * 44)
    grid_key = f"stat_grid_{st.session_state.get(stat_version_key, 0)}"
    time.sleep(0.1)
    grid_response = safe_aggrid(
        stat_config_df,
        gridOptions=grid_options,
        height=grid_height,
        width="100%",
        theme=GRID_THEME,
        custom_css=GRID_CUSTOM_CSS,
        data_return_mode=DataReturnMode.AS_INPUT,
        reload_data=True,
        fit_columns_on_grid_load=True,
        allow_unsafe_jscode=True,
        enable_enterprise_modules=False,
        key=grid_key,
        update_on=["selectionChanged"],
    )

grid_df = None
if grid_response and grid_response.data is not None:
    if isinstance(grid_response.data, pd.DataFrame):
        grid_df = grid_response.data
    else:
        grid_df = pd.DataFrame(grid_response.data)
    if "Drag" in grid_df.columns:
        grid_df = grid_df.drop(columns=["Drag"])
    if "Stat" in grid_df.columns:
        grid_df = grid_df[grid_df["Stat"].astype(str).str.strip().ne("")]
    grid_records = grid_df.to_dict("records")
else:
    grid_records = []

manual_override = st.session_state.pop(manual_stat_update_key, False)
current_config_records = [{k: v for k, v in row.items() if k in ["Stat", "Show"]} for row in current_stat_config]
is_config_identical = (
    grid_records is not None and
    len(grid_records) == len(current_config_records) and
    all(a["Stat"] == b["Stat"] and a["Show"] == b["Show"] for a, b in zip(grid_records, current_config_records))
)

if manual_override:
    cleaned_config = current_stat_config.copy()
elif grid_records and not is_config_identical:
    cleaned_config = normalize_stat_rows(grid_records, preset_base_config)
else:
    cleaned_config = current_stat_config.copy()

st.session_state[stat_state_key] = cleaned_config

stats_order = [row["Stat"] for row in cleaned_config if row.get("Show", True)]
if not stats_order:
    st.info("Add at least one stat and mark it as shown to build the chart.")
    st.stop()

# --------------------- Formatting ---------------------
def format_stat(stat: str, val) -> str:
    if pd.isna(val):
        return ""

    upper_stat = stat.upper()
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
    "HardHit%": "Hard Hit%",
    "WAR": "fWAR",
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
        f"{year} {player_name}" + (f" ({player_team_display})" if player_team_display else ""),
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

    pdf_buffer = BytesIO()
    fig.savefig(pdf_buffer, format="pdf", bbox_inches="tight", pad_inches=.25)
    pdf_buffer.seek(0)

    st.pyplot(fig, use_container_width=True, clear_figure=True)
    download_name = f"{player_name.replace(' ', '_')}_{year}_savant.pdf"
    st.download_button(
        "Download as PDF",
        data=pdf_buffer,
        file_name=download_name,
        mime="application/pdf",
    )
    st.caption("If dragging doesn't update in table, drag it again.")
    st.caption("Find a player's Fangraphs ID in their Fangraphs profile URL")
    st.caption("TZ records ended in 2001, DRS started in 2002")
