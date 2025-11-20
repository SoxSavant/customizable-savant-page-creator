import os
import time
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import html
import io
import unicodedata
import re
import json
os.environ.setdefault("AGGRID_RELEASE", "True")
from datetime import date
from pathlib import Path
from pybaseball import batting_stats, fielding_stats, playerid_lookup, bwar_bat
from pybaseball.statcast_fielding import statcast_outs_above_average
from st_aggrid import (
    AgGrid,
    GridOptionsBuilder,
    GridUpdateMode,
    DataReturnMode,
    JsCode,
)
import requests


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

    grid_opts = kwargs.get("gridOptions")
    if isinstance(grid_opts, dict) and "rowData" in grid_opts:
        clean_grid_opts = dict(grid_opts)
        clean_grid_opts.pop("rowData", None)
        kwargs = {**kwargs, "gridOptions": clean_grid_opts}

    data_arg = _DFProxy(df) if isinstance(df, pd.DataFrame) else df
    for attempt in range(3):
        try:
            return AgGrid(data_arg, **kwargs)
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

st.set_page_config(page_title="Player Comparison App", layout="wide")

st.markdown(
    """
    <style>
        [data-testid="stToolbar"] {visibility: hidden;}
        [data-testid="stDecoration"] {display: none;}
        [data-testid="stStatusWidget"] {display: none;}
        .viewerBadge_link__qRi_k {display: none;}
        /* Keep row selector column hidden without touching data columns */
        div.ag-header-cell[col-id="ag-RowSelector"],
        div.ag-pinned-left-cols-container [col-id="ag-RowSelector"],
        div.ag-center-cols-container [col-id="ag-RowSelector"] {
            display: none !important;
        }
        /* Comparison card styling */
        .compare-card {
            background: #ffffff;
            border: 1px solid #d0d0d0;
            border-radius: 10px;
            padding: 1.25rem 1.5rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.12);
            color: #111111;
        }
        .compare-card h3 {
            margin: 0;
            text-align: center;
            color: #7b0d0d;
            font-weight: 800;
        }
        .compare-card .headshot-row {
            display: flex;
            align-items: center;
            justify-content: space-around;
            gap: 16rem;
            margin-bottom: .2rem;
        }
        .compare-card .headshot-col {
            flex: 0 0 220px;
            text-align: center;
            padding-top: .1rem;
        }
        .compare-card .headshot-col img {
            border: 1px solid #d0d0d0;
            background: #f2f2f2;
            border-radius: 4px;
            padding: 4px;
            width: 230px;
            max-height: 230px;
            height: auto;
            object-fit: contain;
        }
        .compare-card .player-name {
            font-size: 1.35rem;
            font-weight: 800;
            margin: .2rem 0 0 0;
        }
        .compare-card .player-meta {
            color: #555;
            margin: 0 0 0.3rem 0;
            font-size: 1.3rem;
        }
        .compare-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
            table-layout: fixed;
            line-height: 1.5;
        }
        .compare-table th, .compare-table td {
            border: 1px solid #d0d0d0;
            padding: 3px 6px;
            text-align: center;
            background: #ffffff;
            color: #111111;
            width: 33.333%;
        }
        .compare-table th {
            background: #f1f1f1;
            font-weight: 800;
            color: #7b0d0d;
            font-size: 15px;
            line-height: 1.2;
        }
        .compare-table .overall-row th {
            background: #f1f1f1;
            color: #7b0d0d;
            font-weight: 800;
            font-size: 15px;
            padding: 5px 0 3px 0;
            border-top: 1px solid #d0d0d0;
            border-bottom: 1px solid #d0d0d0;
            border-left: 1px solid #d0d0d0;
            border-right: 1px solid #d0d0d0;
        }
        .compare-table .stat-col {
            font-weight: 700;
            background: #fafafa;
            color: #111;
        }
        .compare-table .best {
            background: #E5F1E4;
            font-weight: 800;
            color: #111111;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

title_col, meta_col = st.columns([3, 1])
with title_col:
    st.title("Custom Player Comparison")
with meta_col:
    st.markdown(
        """
        <div style="text-align: right; font-size: 1rem; padding-top: 0.6rem;">
            Built by <a href="https://twitter.com/Sox_Savant" target="_blank">@Sox_Savant</a>
        </div>
        """,
        unsafe_allow_html=True,
    )


TRUTHY_STRINGS = {"true", "1", "yes", "y", "t"}

STAT_PRESETS = {
    "Stathead": [
        "bWAR",
        "WAR",
        "G",
        "PA",
        "H",
        "HR",
        "RBI",
        "SB",
        "AVG",
        "OBP",
        "SLG",
        "OPS",
        "wRC+",
    ],
    "Statcast": [
        "WAR",
        "Off",
        "BsR",
        "Def",
        "OAA",
        "FRV",
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
    "Standard": [
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
        "DRS",
        "FRV",
    ],
    "Fielding": [
        "DRS",
        "FRV",
        "OAA",
        "ARM",
    ],
    "Miscellaneous": [
        "K-BB%",
        "O-Swing%",
        "Z-Swing%",
        "Swing%",
        "Contact%",
        "DRS",
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
    "Off", "Def", "BsR", "WAR", "Barrel%", "HardHit%", "EV", "MaxEV",
    "wRC+", "wOBA", "xwOBA", "xBA", "xSLG", "OPS", "SLG", "OBP", "AVG", "ISO",
    "BABIP", "G", "PA", "AB", "R", "RBI", "HR", "XBH", "H", "2B", "3B", "SB", "BB", "IBB", "SO",
    "K%", "BB%", "K-BB%", "O-Swing%", "Z-Swing%", "Swing%", "Contact%", "WPA", "Clutch",
    "Whiff%", "Pull%", "Cent%", "Oppo%", "GB%", "FB%", "LD%", "LA",
    "FRV", "OAA", "ARM", "RANGE", "DRS", "TZ", "FRM", "UZR", "bWAR",
]
STATCAST_FIELDING_START_YEAR = 2016
FIELDING_STATS = ["DRS", "TZ", "UZR", "FRM", "FRV", "OAA", "ARM", "RANGE", "bWAR"]
STAT_DISPLAY_NAMES = {
    "WAR": "fWAR",
}


def display_stat_name(stat) -> str:
    if stat is None:
        return ""
    text = str(stat)
    return STAT_DISPLAY_NAMES.get(text, text)
SUM_STATS = {
    "G", "PA", "AB", "R", "H", "1B", "2B", "3B", "HR", "RBI", "SB", "CS",
    "BB", "IBB", "SO", "HBP", "SF", "SH", "XBH", "TB",
    "WAR", "Off", "Def", "BsR", "ISO", "GDP", "wRAA", "wRC",
    "TZ",
}
RATE_STATS = {
    "AVG", "OBP", "SLG", "OPS", "wOBA", "xwOBA", "xBA", "xSLG", "BABIP",
    "K%", "BB%", "K-BB%", "O-Swing%", "Z-Swing%", "Swing%", "Contact%", "Whiff%",
    "Barrel%", "HardHit%", "Pull%", "Cent%", "Oppo%", "GB%", "FB%", "LD%",
    "LA", "EV", "MaxEV", "CSW%", "BB/K",
}

HEADSHOT_BASES = [
    # Standard silo path (real photos when they exist)
    "https://img.mlbstatic.com/mlb-photos/image/upload/w_240,q_auto:best,f_auto/people/{mlbam}/headshot/silo/current",
    # Generic fallback path with slash
    "https://img.mlbstatic.com/mlb-photos/image/upload/w_213,d_people:generic:headshot:silo:current.png,q_auto:best,f_auto/v1/people/{mlbam}/headshot/67/current",
    # Alternate path provided (kept last to avoid overriding real photos)
    "https://img.mlbstatic.com/mlb-photos/image/upload/w_213,d_people:generic:headshot:silo:current.png,q_auto:best,f_auto/v1/people/{mlbam}headshot/67/current",
]
HEADSHOT_BREF_BASES = [
    "https://content-static.baseball-reference.com/req/202406/images/headshots/{folder}/{bref_id}.jpg",
    "https://content-static.baseball-reference.com/req/202310/images/headshots/{folder}/{bref_id}.jpg",
    "https://www.baseball-reference.com/req/202108020/images/headshots/{folder}/{bref_id}.jpg",
]
HEADSHOT_CHECK_TIMEOUT = 1.0
HEADSHOT_USER_AGENT = "headshot-fetcher/1.0"
HEADSHOT_PLACEHOLDER = (
    "data:image/svg+xml;base64,"
    "PHN2ZyB3aWR0aD0nMjQwJyBoZWlnaHQ9JzI0MCcgdmlld0JveD0nMCAwIDI0MCAyNDAnIHhtbG5zPSdodHRwOi8v"
    "d3d3LnczLm9yZy8yMDAwL3N2Zyc+CjxyZWN0IHdpZHRoPScyNDAnIGhlaWdodD0nMjQwJyBmaWxsPScjZWVmJy8+"
    "CjxjaXJjbGUgY3g9JzEyMCcgY3k9Jzk1JyByPSc1NScgZmlsbD0nI2RkZScvPgo8Y2lyY2xlIGN4PScxMjAnIGN5"
    "PSc4NScgcj0nNDInIGZpbGw9JyNmZmYnIHN0cm9rZT0nI2NjYycvPgo8cGF0aCBkPSdNMTIwIDE1MGMtMzAgMC01"
    "NSAyNS01NSA1NXMzNSAxNS41IDU1IDE1LjUgNTUtMTUuNSA1NS0xNS41LTM1LTU1LTU1LTU1eicgZmlsbD0nI2Nj"
    "YycvPgo8L3N2Zz4="
)
LOCAL_BWAR_FILE = Path(__file__).with_name("warhitters2025.txt")


def local_bwar_signature() -> float:
    try:
        return LOCAL_BWAR_FILE.stat().st_mtime
    except FileNotFoundError:
        return 0.0


@st.cache_data(show_spinner=False, ttl=900)
def load_year(y: int) -> pd.DataFrame:
    """Cached single-year fetch."""
    return batting_stats(y, y, qual=0, split_seasons=False)


def compute_team_display(team_values: list[str]) -> str:
    placeholders = {"TOT", "- - -", "---", "--", "", "N/A"}
    teams = [str(t).upper() for t in team_values if str(t).strip()]
    valid = [t for t in teams if t not in placeholders]
    if not valid:
        if any(t == "TOT" for t in teams):
            return "TOT"
        if any(t == "- - -" for t in teams):
            return "2+ Tms"
        return teams[0] if teams else "N/A"
    unique_valid = sorted(set(valid))
    if len(unique_valid) == 1:
        return unique_valid[0]
    return f"{len(unique_valid)} Tms"


def aggregate_player_group(grp: pd.DataFrame, name: str | None = None) -> dict:
    result: dict[str, object] = {}
    if name is None and "Name" in grp.columns:
        val = grp["Name"].dropna()
        if not val.empty:
            name = str(val.iloc[0])
    if name:
        result["Name"] = name
    if "IDfg" in grp.columns:
        ids = grp["IDfg"].dropna()
        if not ids.empty:
            result["IDfg"] = ids.iloc[0]
    teams = grp["Team"].dropna().astype(str).tolist() if "Team" in grp.columns else []
    display = compute_team_display(teams)
    result["Team"] = display
    result["TeamDisplay"] = display
    numeric_cols = [
        col for col in grp.columns
        if pd.api.types.is_numeric_dtype(grp[col]) and col not in {"IDfg"}
    ]
    if "PA" in grp.columns:
        pa_weight = pd.to_numeric(grp["PA"], errors="coerce").fillna(0)
    else:
        pa_weight = pd.Series(np.zeros(len(grp)), index=grp.index, dtype=float)
    pa_total = pa_weight.sum()
    for col in numeric_cols:
        series = pd.to_numeric(grp[col], errors="coerce")
        if series.isna().all():
            continue
        if col in SUM_STATS:
            result[col] = series.sum(skipna=True)
        elif col in RATE_STATS and pa_total > 0:
            result[col] = (series * pa_weight).sum(skipna=True) / pa_total
        else:
            result[col] = series.mean(skipna=True)
    return result


@st.cache_data(show_spinner=False, ttl=900)
def load_batting(start_year: int, end_year: int) -> pd.DataFrame:
    """Load aggregated batting stats for a single year or a span of years."""
    start = min(start_year, end_year)
    end = max(start_year, end_year)
    try:
        df = batting_stats(start, end, qual=0, split_seasons=False)
        if df is not None and not df.empty:
            return df
    except Exception:
        # Fall back silently to per-year fetches
        pass

    chunk_size = 10
    frames = []
    failed_years = []
    for chunk_start in range(start, end + 1, chunk_size):
        chunk_end = min(chunk_start + chunk_size - 1, end)
        try:
            chunk_df = batting_stats(
                chunk_start,
                chunk_end,
                qual=0,
                split_seasons=False,
            )
        except Exception:
            chunk_df = None
        if chunk_df is not None and not chunk_df.empty:
            frames.append(chunk_df)
            continue
        for year in range(chunk_start, chunk_end + 1):
            try:
                yearly = load_year(year)
                if yearly is not None and not yearly.empty:
                    frames.append(yearly)
                else:
                    failed_years.append(year)
            except Exception:
                failed_years.append(year)
    if frames:
        combined = pd.concat(frames, ignore_index=True)
        grouped_rows = []
        for name, grp in combined.groupby("Name"):
            row = aggregate_player_group(grp, name)
            grouped_rows.append(row)
        aggregated = pd.DataFrame(grouped_rows)
        if failed_years:
            st.info(f"Loaded partial data; skipped years: {', '.join(map(str, failed_years))}")
        return aggregated

    st.error(f"Could not load batting data for {start}-{end}. Please try another span.")
    return pd.DataFrame()


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
        data = io.StringIO(resp.content.decode("utf-8"))
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


def load_local_bwar_data() -> pd.DataFrame:
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
    player_series = df["player_ID"] if "player_ID" in df.columns else pd.Series("", index=df.index)
    df["player_ID"] = player_series.astype(str).str.strip().str.lower()
    mlb_series = df["mlb_ID"] if "mlb_ID" in df.columns else pd.Series(np.nan, index=df.index)
    df["mlb_ID"] = pd.to_numeric(mlb_series, errors="coerce")
    df = df.dropna(subset=["NameKey", "year_ID", "WAR"])
    return df[["NameKey", "Name", "year_ID", "WAR", "player_ID", "mlb_ID"]]


@st.cache_data(show_spinner=False, ttl=3600)
def load_bwar_dataset(local_sig: float) -> pd.DataFrame:
    _ = local_sig  # cache key
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
        player_series = data["player_ID"] if "player_ID" in data.columns else pd.Series("", index=data.index)
        data["player_ID"] = player_series.astype(str).str.strip().str.lower()
        mlb_series = data["mlb_ID"] if "mlb_ID" in data.columns else pd.Series(np.nan, index=data.index)
        data["mlb_ID"] = pd.to_numeric(mlb_series, errors="coerce")
        frames.append(data[["NameKey", "Name", "year_ID", "WAR", "player_ID", "mlb_ID"]])

    local = load_local_bwar_data()
    if local is not None and not local.empty:
        frames.append(local)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["NameKey", "year_ID", "WAR"])
    combined = combined.sort_values(["NameKey", "year_ID"])
    combined = combined.drop_duplicates(subset=["NameKey", "year_ID"], keep="last")
    return combined.reset_index(drop=True)


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

    def clean_key(val: str) -> str:
        try:
            val = unicodedata.normalize("NFKD", val).encode("ascii", "ignore").decode()
        except Exception:
            pass
        return "".join(ch for ch in str(val) if ch.isalnum()).lower()

    def match_by_names() -> pd.DataFrame:
        if not target_names:
            return pd.DataFrame()
        keys = {normalize_statcast_name(name) for name in target_names if name}
        if not keys:
            return pd.DataFrame()
        return pool[pool["NameKey"].isin(keys)]

    def match_by_clean_name() -> pd.DataFrame:
        if not target_names:
            return pd.DataFrame()
        targets = {clean_key(name) for name in target_names if name}
        if not targets:
            return pd.DataFrame()
        pool_local = pool.copy()
        pool_local["__clean"] = pool_local["Name"].astype(str).apply(clean_key)
        return pool_local[pool_local["__clean"].isin(targets)]

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
        alt = match_by_clean_name()
        if not alt.empty:
            df = alt
    if df.empty:
        return pd.DataFrame()
    agg = df.groupby("NameKey", as_index=False).agg({
        "Name": "first",
        "WAR": lambda s: s.sum(min_count=1),
    })
    agg = agg.rename(columns={"WAR": "bWAR"})
    return agg


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


def normalize_display_team(team_value: str) -> str:
    return compute_team_display([team_value]) if team_value else "N/A"


@st.cache_data(show_spinner=False, ttl=900)
def load_player_batting_profile(fg_id: int, start_year: int, end_year: int) -> pd.Series | None:
    try:
        df = batting_stats(start_year, end_year, qual=0, split_seasons=False, players=str(fg_id))
    except Exception:
        df = None
    if df is not None and not df.empty:
        row = df.iloc[0].copy()
        row["TeamDisplay"] = normalize_display_team(str(row.get("Team", "")).strip())
        row["Name"] = str(row.get("Name", "")).strip()
        return row
    frames = []
    for year in range(start_year, end_year + 1):
        try:
            yearly = batting_stats(year, year, qual=0, split_seasons=False, players=str(fg_id))
        except Exception:
            yearly = None
        if yearly is not None and not yearly.empty:
            frames.append(yearly)
    if not frames:
        return None
    combined = pd.concat(frames, ignore_index=True)
    aggregated = aggregate_player_group(combined)
    return pd.Series(aggregated)


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


def build_player_profile(fg_id: int, start_year: int, end_year: int) -> pd.Series | None:
    batting = load_player_batting_profile(fg_id, start_year, end_year)
    if batting is None:
        return None
    fielding = load_player_fielding_profile(fg_id, start_year, end_year)
    for key, val in fielding.items():
        batting[key] = val
    name_value = str(batting.get("Name", "")).strip()
    mlbam_id = None
    bbref_id = None
    if name_value:
        mlbam_id, bbref_id = lookup_mlbam_id(name_value, return_bbref=True)
    name_key = normalize_statcast_name(str(batting.get("Name", "")))
    if name_key:
        statcast = load_statcast_fielding_span(start_year, end_year, target_names=(name_key,))
        if statcast is not None and not statcast.empty:
            match = statcast[statcast["NameKey"] == name_key]
            if not match.empty:
                for metric in ["FRV", "OAA", "ARM", "RANGE"]:
                    value = pd.to_numeric(match[metric].iloc[0], errors="coerce") if metric in match.columns else np.nan
                    batting[metric] = value
    name_targets = (name_key,) if name_key else None
    bwar = load_bwar_span(
        start_year,
        end_year,
        target_names=name_targets,
        target_bbref=bbref_id,
        target_mlbam=mlbam_id,
    )
    if bwar is not None and not bwar.empty:
        match_bwar = bwar
        if name_key:
            match = bwar[bwar["NameKey"] == name_key]
            if not match.empty:
                match_bwar = match
        batting["bWAR"] = pd.to_numeric(match_bwar["bWAR"].iloc[0], errors="coerce")
    for metric in FIELDING_STATS:
        if metric not in batting.index:
            batting[metric] = np.nan
        else:
            batting[metric] = pd.to_numeric(batting[metric], errors="coerce")
    return batting


@st.cache_data(show_spinner=False, ttl=3600)
def lookup_fg_id_by_name(full_name: str) -> int | None:
    if not full_name or not isinstance(full_name, str):
        return None
    tokens = full_name.strip().split()
    suffixes = {"jr", "jr.", "sr", "sr.", "ii", "iii", "iv", "v"}
    while tokens and tokens[-1].lower() in suffixes:
        tokens.pop()
    if len(tokens) < 2:
        return None
    last = tokens[-1]
    first = " ".join(tokens[:-1])

    def normalize_token(val: str) -> str:
        val = val.replace(".", "").strip()
        try:
            return unicodedata.normalize("NFKD", val).encode("ascii", "ignore").decode()
        except Exception:
            return val

    variants = [
        (last, first),
        (normalize_token(last), normalize_token(first)),
        (normalize_token(last).lower(), normalize_token(first).lower()),
        (last.lower(), first.lower()),
    ]
    seen: set[tuple[str, str]] = set()
    for last_variant, first_variant in variants:
        key = (last_variant, first_variant)
        if not last_variant or not first_variant or key in seen:
            continue
        seen.add(key)
        try:
            lookup = playerid_lookup(last_variant, first_variant)
        except Exception:
            continue
        if lookup is None or lookup.empty or "key_fangraphs" not in lookup.columns:
            continue
        val = lookup["key_fangraphs"].dropna()
        if val.empty:
            continue
        try:
            fg_id = int(val.iloc[0])
        except Exception:
            continue
        if fg_id > 0:
            return fg_id
    return None


def resolve_player_fg_id(name: str, pool_df: pd.DataFrame | None = None) -> int | None:
    if pool_df is not None and not pool_df.empty and "IDfg" in pool_df.columns:
        ids = pool_df.loc[pool_df["Name"] == name, "IDfg"].dropna().astype(int)
        if not ids.empty:
            return int(ids.iloc[0])
    return lookup_fg_id_by_name(name)


@st.cache_data(show_spinner=False)
def lookup_mlbam_id(full_name: str, return_bbref: bool = False):
    """Best-effort MLBAM lookup using pybaseball's playerid_lookup. Optionally returns bbref id."""
    if not full_name or not full_name.strip():
        return (None, None) if return_bbref else None
    suffixes = {"jr", "jr.", "sr", "sr.", "ii", "iii", "iv", "v"}

    def normalize_token(tok: str) -> str:
        if not tok:
            return ""
        tok = tok.replace(".", "").strip()
        try:
            return unicodedata.normalize("NFKD", tok).encode("ascii", "ignore").decode()
        except Exception:
            return tok

    def clean_full(val: str) -> str:
        try:
            val = unicodedata.normalize("NFKD", val).encode("ascii", "ignore").decode()
        except Exception:
            pass
        return "".join(ch for ch in val if ch.isalnum()).lower()

    def strip_suffix(tokens: list[str]) -> list[str]:
        toks = tokens.copy()
        while toks and toks[-1].lower() in suffixes:
            toks.pop()
        return toks

    parts = full_name.split()
    base_tokens = strip_suffix(parts)
    if len(base_tokens) < 2:
        return (None, None) if return_bbref else None

    first_raw = base_tokens[0]
    last_raw = " ".join(base_tokens[1:])
    target_clean = clean_full(first_raw + last_raw)

    def initial_forms(token: str) -> list[str]:
        forms = []
        if not token:
            return forms
        stripped = token.replace(".", "")
        if stripped and stripped.isupper() and 1 <= len(stripped) <= 4:
            dotted = ".".join(list(stripped)) + "."
            spaced = " ".join(list(stripped))
            forms.extend([dotted, spaced, stripped, stripped + "."])
        return forms

    first_forms = initial_forms(first_raw)
    variants = [
        (last_raw, first_raw),  # raw as-is (keeps accents/dots)
        (normalize_token(last_raw), normalize_token(first_raw)),
        (normalize_token(last_raw).lower(), normalize_token(first_raw).lower()),
        (last_raw.replace(".", ""), first_raw.replace(".", "")),  # no dots
    ]
    for form in first_forms:
        variants.append((last_raw, form))
        variants.append((normalize_token(last_raw), normalize_token(form)))

    first_hit_mlbam = None
    first_hit_bbref = None
    best_match_mlbam = None
    best_match_bbref = None

    def consider_row(row):
        nonlocal first_hit_mlbam, first_hit_bbref, best_match_mlbam, best_match_bbref
        combo = clean_full(str(row.get("name_first", "")) + str(row.get("name_last", "")))
        mlbam_val = row.get("key_mlbam")
        bbref_val = row.get("key_bbref")
        if combo == target_clean:
            if pd.notna(mlbam_val):
                try:
                    best_match_mlbam = int(mlbam_val)
                except Exception:
                    pass
            if pd.notna(bbref_val):
                try:
                    best_match_bbref = str(bbref_val)
                except Exception:
                    pass
        if first_hit_mlbam is None and pd.notna(mlbam_val):
            try:
                first_hit_mlbam = int(mlbam_val)
            except Exception:
                pass
        if first_hit_bbref is None and pd.notna(bbref_val):
            try:
                first_hit_bbref = str(bbref_val)
            except Exception:
                pass
    for last, first in variants:
        try:
            lookup_df = playerid_lookup(last, first)
        except Exception:
            continue
        if lookup_df is None or lookup_df.empty:
            continue
        for _, row in lookup_df.iterrows():
            consider_row(row)

    # Fallback: search by last name only, then match cleaned full name
    try:
        lookup_df = playerid_lookup(last_raw, None)
    except Exception:
        lookup_df = None
    if lookup_df is not None and not lookup_df.empty:
        for _, row in lookup_df.iterrows():
            consider_row(row)

    mlbam_result = best_match_mlbam if best_match_mlbam is not None else first_hit_mlbam
    bbref_result = best_match_bbref if best_match_bbref is not None else first_hit_bbref

    if return_bbref:
        return mlbam_result, bbref_result
    return mlbam_result


@st.cache_data(show_spinner=False, ttl=21600)
def build_mlb_headshot(mlbam: int | str | None) -> str | None:
    """Try MLB headshot URLs in order; return the first that responds (200)."""
    if mlbam is None:
        return None
    mlbam_val = str(mlbam).strip()
    if not mlbam_val:
        return None
    headers = {"User-Agent": HEADSHOT_USER_AGENT}
    fallback_url = None
    for base in HEADSHOT_BASES:
        try:
            url = base.format(mlbam=mlbam_val)
            if fallback_url is None:
                fallback_url = url
        except Exception:
            continue
        try:
            resp = requests.head(url, headers=headers, timeout=HEADSHOT_CHECK_TIMEOUT, allow_redirects=True)
            status = resp.status_code
            if status == 200:
                return url
            # Some endpoints reject HEAD; try a lightweight GET
            if status in (403, 404, 405):
                resp_get = requests.get(url, headers=headers, timeout=HEADSHOT_CHECK_TIMEOUT, stream=True)
                if resp_get.status_code == 200:
                    return url
        except Exception:
            continue
    return fallback_url


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_bbref_headshot(bref_id: str | None) -> str | None:
    if not bref_id:
        return None
    slug = str(bref_id).strip().lower()
    if not slug:
        return None
    first_letter = slug[0]
    url = f"https://www.baseball-reference.com/players/{first_letter}/{slug}.shtml"
    headers = {"User-Agent": HEADSHOT_USER_AGENT}
    try:
        resp = requests.get(url, headers=headers, timeout=HEADSHOT_CHECK_TIMEOUT)
    except Exception:
        return None
    if resp.status_code != 200 or not resp.text:
        return None
    html_text = resp.text
    urls = []
    for pattern in [
        r'https?://[^"\']*headshots[^"\']*\.(?:jpg|png)',
        r'//[^"\']*headshots[^"\']*\.(?:jpg|png)',
    ]:
        urls.extend(re.findall(pattern, html_text, flags=re.IGNORECASE))
    for raw in urls:
        if not raw:
            continue
        candidate = raw if raw.startswith("http") else f"https:{raw}"
        return candidate
    return None


def get_headshot_url(name: str, df: pd.DataFrame) -> str | None:
    id_cols = ["mlbamid", "mlbam_id", "mlbam", "MLBID", "MLBAMID", "key_mlbam"]
    fg_cols = ["playerid", "IDfg", "fg_id", "FGID"]
    bref_cols = ["key_bbref", "bbref_id", "BBREFID", "bref_id", "BREFID"]

    def build_bref_headshot(bref_id: str | None) -> str | None:
        if not bref_id:
            return None
        raw_slug = str(bref_id).strip()
        if not raw_slug:
            return None
        slug_variants = {raw_slug.lower(), raw_slug.upper()}
        for slug in slug_variants:
            folder_variants = {slug[0].lower(), slug[0].upper()} if slug else set()
            for folder in folder_variants:
                for base in HEADSHOT_BREF_BASES:
                    try:
                        return base.format(folder=folder, bref_id=slug)
                    except Exception:
                        continue
        return None

    def resolve_bref_headshot(bref_id: str | None) -> str | None:
        direct = build_bref_headshot(bref_id)
        if direct:
            return direct
        return fetch_bbref_headshot(bref_id)

    for col in id_cols:
        if col in df.columns:
            vals = df.loc[df["Name"] == name, col].dropna()
            if not vals.empty:
                try:
                    mlbam = int(vals.iloc[0])
                    headshot = build_mlb_headshot(mlbam)
                    if headshot:
                        return headshot
                except Exception:
                    pass

    # If we have a FanGraphs id, try to resolve MLBAM/BBRef via reverse lookup
    for col in fg_cols:
        if col in df.columns:
            vals = df.loc[df["Name"] == name, col].dropna()
            if not vals.empty:
                try:
                    fg_id = int(vals.iloc[0])
                except Exception:
                    fg_id = None
                if fg_id:
                    try:
                        from pybaseball import playerid_reverse_lookup
                        rev = playerid_reverse_lookup([fg_id], key_type="fangraphs")
                        if rev is not None and not rev.empty:
                            mlbam = rev.iloc[0].get("key_mlbam")
                            if pd.notna(mlbam):
                                headshot = build_mlb_headshot(int(mlbam))
                                if headshot:
                                    return headshot
                            bbref = rev.iloc[0].get("key_bbref")
                            if pd.notna(bbref):
                                bref_url = resolve_bref_headshot(str(bbref))
                                if bref_url:
                                    return bref_url
                    except Exception:
                        pass

    def clean_name(val: str) -> str:
        if not val:
            return ""
        try:
            val = unicodedata.normalize("NFKD", val).encode("ascii", "ignore").decode()
        except Exception:
            pass
        return "".join(ch for ch in val if ch.isalnum() or ch.isspace()).strip().lower()

    def heuristic_bbref_slug(full_name: str) -> list[str]:
        """Best-effort guesses for bbref slug when lookup fails (last5 + first2 + 2-digit index)."""
        cleaned = clean_name(full_name)
        if not cleaned:
            return []
        parts = cleaned.split()
        if len(parts) < 2:
            return []
        first = parts[0]
        last = parts[-1]
        if not first or not last:
            return []
        base_slug = f"{last[:5]}{first[:2]}"
        if len(base_slug) < 6:
            return []
        slugs = []
        for i in range(1, 16):  # try 01-15 to account for name collisions
            slugs.append(f"{base_slug}{i:02d}")
        return slugs

    target_clean = clean_name(name)

    candidate_cols = [
        "mlbamid", "MLBID", "mlbam_id", "mlbam", "key_mlbam", "MLBAMID", "playerid"
    ]
    for col in candidate_cols:
        if col in df.columns:
            vals = df.loc[df["Name"] == name, col].dropna()
            if not vals.empty:
                try:
                    mlbam = int(vals.iloc[0])
                    headshot = build_mlb_headshot(mlbam)
                    if headshot:
                        return headshot
                except Exception:
                    pass

    for col in bref_cols:
        if col in df.columns:
            vals = df.loc[df["Name"] == name, col].dropna()
            if not vals.empty:
                bref_url = resolve_bref_headshot(str(vals.iloc[0]))
                if bref_url:
                    return bref_url

    # Try matching by cleaned name in the df
    if "Name" in df.columns:
        df_clean = df.copy()
        df_clean["__clean_name"] = df_clean["Name"].astype(str).apply(clean_name)
        matches = df_clean[df_clean["__clean_name"] == target_clean]
        if not matches.empty:
            for col in candidate_cols:
                if col in matches.columns:
                    vals = matches[col].dropna()
                    if not vals.empty:
                        try:
                            mlbam = int(vals.iloc[0])
                            headshot = build_mlb_headshot(mlbam)
                            if headshot:
                                return headshot
                        except Exception:
                            pass
            for col in bref_cols:
                if col in matches.columns:
                    vals = matches[col].dropna()
                    if not vals.empty:
                        bref_url = resolve_bref_headshot(str(vals.iloc[0]))
                        if bref_url:
                            return bref_url

    mlbam_fallback, bbref_fallback = lookup_mlbam_id(name, return_bbref=True)
    if mlbam_fallback:
        headshot = build_mlb_headshot(mlbam_fallback)
        if headshot:
            return headshot
    if bbref_fallback:
        bref_url = resolve_bref_headshot(bbref_fallback)
        if bref_url:
            return bref_url
    for slug in heuristic_bbref_slug(name):
        bref_url = resolve_bref_headshot(slug)
        if bref_url:
            return bref_url
    return HEADSHOT_PLACEHOLDER


# --------------------- Layout containers ---------------------
left_col, right_col = st.columns([1.2, 1.55])

with left_col:
    controls_container = st.container()
    stat_builder_container = st.container()

# --------------------- Controls ---------------------
current_year = date.today().year
years_desc = list(range(current_year, 1870, -1))
single_a_key = "comp_single_year_a"
single_b_key = "comp_single_year_b"
single_year_a_key = "comp_year_a_single"
single_year_b_key = "comp_year_b_single"
with controls_container:
    year_col = st.columns(2)
    with year_col[0]:
        single_a = st.checkbox(
            "Single season (Player A)",
            value=st.session_state.get(single_a_key, True),
            key=single_a_key,
        )
        if single_a:
            year_a_single = st.selectbox(
                "Season (Player A)",
                years_desc,
                index=0,
                key=single_year_a_key,
            )
            year_a_start = year_a_single
            year_a_end = year_a_single
        else:
            year_a_start = st.selectbox(
                "Season Start (Player A)",
                years_desc,
                index=0,
                key="comp_year_a_start",
            )
            year_a_end = st.selectbox(
                "Season End (Player A)",
                years_desc,
                index=0,
                key="comp_year_a_end",
            )
    with year_col[1]:
        single_b = st.checkbox(
            "Single season (Player B)",
            value=st.session_state.get(single_b_key, True),
            key=single_b_key,
        )
        if single_b:
            year_b_single = st.selectbox(
                "Season (Player B)",
                years_desc,
                index=0,
                key=single_year_b_key,
            )
            year_b_start = year_b_single
            year_b_end = year_b_single
        else:
            year_b_start = st.selectbox(
                "Season Start (Player B)",
                years_desc,
                index=0,
                key="comp_year_b_start",
            )
            year_b_end = st.selectbox(
                "Season End (Player B)",
                years_desc,
                index=0,
                key="comp_year_b_end",
            )

range_a = (min(year_a_start, year_a_end), max(year_a_start, year_a_end))
range_b = (min(year_b_start, year_b_end), max(year_b_start, year_b_end))


DEFAULT_PLAYER_A = "Mookie Betts"
DEFAULT_PLAYER_B = "Aaron Judge"
if "comp_player_a" not in st.session_state:
    st.session_state["comp_player_a"] = DEFAULT_PLAYER_A
if "comp_player_b" not in st.session_state:
    st.session_state["comp_player_b"] = DEFAULT_PLAYER_B
if "comp_player_a_id" not in st.session_state:
    st.session_state["comp_player_a_id"] = ""
if "comp_player_b_id" not in st.session_state:
    st.session_state["comp_player_b_id"] = ""

with controls_container:
    sel_a_col, sel_b_col = st.columns(2)
    with sel_a_col:
        player_a_mode = st.selectbox(
            "Player A Input",
            ["Name", "FanGraphs ID"],
            key="comp_player_a_mode",
        )
        if player_a_mode == "Name":
            player_a_name_input = st.text_input("Player A", key="comp_player_a")
            player_a_id_input = st.session_state.get("comp_player_a_id", "")
        else:
            player_a_id_input = st.text_input("Player A FanGraphs ID", key="comp_player_a_id")
            player_a_name_input = st.session_state.get("comp_player_a", "")
    with sel_b_col:
        player_b_mode = st.selectbox(
            "Player B Input",
            ["Name", "FanGraphs ID"],
            key="comp_player_b_mode",
        )
        if player_b_mode == "Name":
            player_b_name_input = st.text_input("Player B", key="comp_player_b")
            player_b_id_input = st.session_state.get("comp_player_b_id", "")
        else:
            player_b_id_input = st.text_input("Player B FanGraphs ID", key="comp_player_b_id")
            player_b_name_input = st.session_state.get("comp_player_b", "")

player_a_name = player_a_name_input.strip()
player_b_name = player_b_name_input.strip()
player_a_id_raw = str(player_a_id_input).strip()
player_b_id_raw = str(player_b_id_input).strip()

if player_a_mode == "Name":
    if not player_a_name:
        st.warning("Enter a name for Player A or switch to FanGraphs ID input.")
        st.stop()
    player_a_fg_id = resolve_player_fg_id(player_a_name)
else:
    if not player_a_id_raw:
        st.warning("Enter a FanGraphs ID for Player A or switch to name input.")
        st.stop()
    try:
        player_a_fg_id = int(player_a_id_raw)
    except Exception:
        player_a_fg_id = None

if player_b_mode == "Name":
    if not player_b_name:
        st.warning("Enter a name for Player B or switch to FanGraphs ID input.")
        st.stop()
    player_b_fg_id = resolve_player_fg_id(player_b_name)
else:
    if not player_b_id_raw:
        st.warning("Enter a FanGraphs ID for Player B or switch to name input.")
        st.stop()
    try:
        player_b_fg_id = int(player_b_id_raw)
    except Exception:
        player_b_fg_id = None

if not player_a_fg_id or player_a_fg_id <= 0:
    if player_a_mode == "Name":
        st.error(f"Could not resolve FanGraphs ID for {player_a_name}. Check the spelling or use the ID input.")
    else:
        st.error("Player A FanGraphs ID must be a positive integer.")
    st.stop()
if not player_b_fg_id or player_b_fg_id <= 0:
    if player_b_mode == "Name":
        st.error(f"Could not resolve FanGraphs ID for {player_b_name}. Check the spelling or use the ID input.")
    else:
        st.error("Player B FanGraphs ID must be a positive integer.")
    st.stop()

player_a_row = build_player_profile(player_a_fg_id, *range_a)
player_b_row = build_player_profile(player_b_fg_id, *range_b)
if player_a_row is None or player_b_row is None:
    st.error("Could not load data for one of the selected players.")
    st.stop()

player_a_display_name = str(player_a_row.get("Name", "")).strip()
if not player_a_display_name:
    player_a_display_name = player_a_name if player_a_mode == "Name" else f"FG#{player_a_fg_id}"
player_b_display_name = str(player_b_row.get("Name", "")).strip()
if not player_b_display_name:
    player_b_display_name = player_b_name if player_b_mode == "Name" else f"FG#{player_b_fg_id}"

df_a = pd.DataFrame([player_a_row])
df_b = pd.DataFrame([player_b_row])
for metric in FIELDING_STATS:
    if metric in df_a.columns:
        df_a[metric] = pd.to_numeric(df_a[metric], errors="coerce")
    if metric in df_b.columns:
        df_b[metric] = pd.to_numeric(df_b[metric], errors="coerce")
player_a_team = player_a_row.get("TeamDisplay", normalize_display_team(player_a_row.get("Team", "")))
player_b_team = player_b_row.get("TeamDisplay", normalize_display_team(player_b_row.get("Team", "")))
year_a_label = f"{range_a[0]}" if range_a[0] == range_a[1] else f"{range_a[0]}-{range_a[1]}"
year_b_label = f"{range_b[0]}" if range_b[0] == range_b[1] else f"{range_b[0]}-{range_b[1]}"
player_a_col_label = player_a_display_name
player_b_col_label = player_b_display_name
if player_a_col_label == player_b_col_label:
    if year_a_label != year_b_label:
        player_a_col_label = f"{player_a_display_name} ({year_a_label})"
        player_b_col_label = f"{player_b_display_name} ({year_b_label})"
    else:
        player_a_col_label = f"{player_a_display_name} (Player A)"
        player_b_col_label = f"{player_b_display_name} (Player B)"

# --------------------- Stat builder setup ---------------------
stat_exclusions = {"Season"}
numeric_a = [col for col in df_a.columns if pd.api.types.is_numeric_dtype(df_a[col])]
numeric_b = [col for col in df_b.columns if pd.api.types.is_numeric_dtype(df_b[col])]
numeric_stats = [col for col in numeric_a if col in numeric_b and col not in stat_exclusions]

preferred_stats = [stat for stat in STAT_ALLOWLIST if stat in numeric_stats]
other_stats = [stat for stat in numeric_stats if stat not in preferred_stats]
stat_options = preferred_stats + other_stats
allowed_add_stats = preferred_stats if preferred_stats else stat_options.copy()

if not stat_options:
    st.error("No numeric stats available to display.")
    st.stop()

default_preset_name = "Stathead"
stat_preset_key = "comp_stat_preset_select"
preset_options = list(STAT_PRESETS.keys())
stat_state_key = "comp_stat_config"
manual_stat_update_key = "comp_stat_config_manual_update"
add_select_key = "comp_add_stat_select"
remove_select_key = "comp_remove_stat_select"
add_reset_key = "comp_reset_add_select"
remove_reset_key = "comp_reset_remove_select"
stat_version_key = "comp_stat_config_version"

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
            format_func=display_stat_name,
            key=add_select_key,
            on_change=add_stat_callback,
            args=(stat_state_key, add_select_key, add_reset_key, sentinel_add),
        )

    with remove_col:
        st.selectbox(
            "Remove stat",
            remove_options,
            label_visibility="hidden",
            format_func=display_stat_name,
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

    display_map_json = json.dumps(STAT_DISPLAY_NAMES)
    stat_value_formatter = JsCode(
        f"""
        function(params) {{
            const map = {display_map_json};
            const value = params.value;
            if (value === undefined || value === null) {{
                return '';
            }}
            return map[value] || value;
        }}
        """
    )

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
        valueFormatter=stat_value_formatter,
    )

    grid_options = gb.build()
    grid_options.pop("rowData", None)

    grid_height = min(480, 90 + len(stat_config_df) * 44)
    grid_key = f"comp_stat_grid_{st.session_state.get(stat_version_key, 0)}"
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
        update_mode=GridUpdateMode.GRID_CHANGED,
        allow_unsafe_jscode=True,
        enable_enterprise_modules=False,
        key=grid_key,
        update_on=["rowDragEnd", "cellValueChanged"],
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
    st.info("Add at least one stat and mark it as shown to build the comparison.")
    st.stop()


# --------------------- Formatting ---------------------
def format_stat(stat: str, val) -> str:
    if pd.isna(val):
        return ""

    upper_stat = stat.upper()
    if upper_stat == "FRV":
        return f"{int(round(float(val)))}"

    if upper_stat == "ARM":
        return f"{int(round(float(val)))}"

    if upper_stat in {"WAR", "BWAR", "FWAR", "EV", "AVG EXIT VELO", "OFF", "DEF", "BSR"}:
        v = float(val)
        if abs(v - round(v)) < 1e-9:
            return f"{int(round(v))}.0"
        return f"{v:.1f}"

    if upper_stat in {"WPA", "CLUTCH"}:
        return f"{float(val):.2f}"

    if upper_stat in {"AVG", "OBP", "SLG", "OPS", "WOBA", "XWOBA", "XBA", "XSLG", "BABIP", "ISO"}:
        return f"{float(val):.3f}".lstrip("0")

    if upper_stat in {"WRC+", "OPS+"}:
        return f"{int(round(float(val)))}"

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


# --------------------- Comparison table ---------------------
label_map = {
    "HardHit%": "Hard Hit%",
    "WAR": "fWAR",
    "EV": "Avg Exit Velo",
    "ARM": "Arm Value",
}
lower_better = {"K%", "O-Swing%", "Whiff%", "GB%"}

comparison_rows = []
winner_map: dict[str, str | None] = {}
for stat in stats_order:
    if stat not in df_a.columns or stat not in df_b.columns:
        continue
    a_val = player_a_row.get(stat, np.nan)
    b_val = player_b_row.get(stat, np.nan)
    raw_label = label_map.get(stat, stat)

    if pd.isna(a_val) and pd.isna(b_val):
        winner = None
    elif pd.isna(a_val):
        winner = player_b_col_label
    elif pd.isna(b_val):
        winner = player_a_col_label
    else:
        better = a_val < b_val if stat in lower_better else a_val > b_val
        if a_val == b_val:
            winner = "Tie"
        else:
            winner = player_a_col_label if better else player_b_col_label

    comparison_rows.append({
        "Stat": raw_label,
        player_a_col_label: format_stat(stat, a_val),
        player_b_col_label: format_stat(stat, b_val),
    })
    winner_map[raw_label] = winner

table_df = pd.DataFrame(comparison_rows)

headshot_a = get_headshot_url(player_a_display_name, df_a)
headshot_b = get_headshot_url(player_b_display_name, df_b)
esc = html.escape

with right_col:
    if table_df.empty:
        st.warning("No stats available to compare.")
    else:
        # Build headshot HTML safely before creating the rows list
        img_a = f'<img src="{esc(headshot_a)}" width="200" />' if headshot_a else ''
        img_b = f'<img src="{esc(headshot_b)}" width="200" />' if headshot_b else ''

        rows = [
            "<div class=\"compare-card\">",
            "  <div class=\"headshot-row\">",
            "    <div class=\"headshot-col\">",
            f"      <div class=\"player-meta\">{esc(str(year_a_label))} | {esc(str(player_a_team))}</div>",
            f"      {img_a}",
            f"      <div class=\"player-name\">{esc(player_a_display_name)}</div>",
            "    </div>",
            "    <div class=\"headshot-col\">",
            f"      <div class=\"player-meta\">{esc(str(year_b_label))} | {esc(str(player_b_team))}</div>",
            f"      {img_b}",
            f"      <div class=\"player-name\">{esc(player_b_display_name)}</div>",
            "    </div>",
            "  </div>",
            "  <table class=\"compare-table\">",
            "    <thead>",
            "      <tr class=\"overall-row\">",
            "        <th colspan=\"3\">Overall Stats</th>",
            "      </tr>",
            "    </thead>",
            "    <tbody>",
        ]
        for _, row in table_df.iterrows():
            raw_label = str(row["Stat"])
            stat_label = esc(raw_label)
            best = winner_map.get(raw_label)
            a_class = "best" if best in {player_a_col_label, "Tie"} else ""
            b_class = "best" if best in {player_b_col_label, "Tie"} else ""
            a_val = esc(str(row[player_a_col_label]))
            b_val = esc(str(row[player_b_col_label]))
            rows.extend([
                "      <tr>",
                f"        <td class=\"{a_class}\">{a_val}</td>",
                f"        <td class=\"stat-col\">{stat_label}</td>",
                f"        <td class=\"{b_class}\">{b_val}</td>",
                "      </tr>",
            ])
        rows.extend([
            "    </tbody>",
            "  </table>",
            "  <div style=\"display:flex; justify-content:space-between; margin-top:0.35rem; color:#555; font-size:0.9rem;\">",
            "    <div>By: Sox_Savant</div>",
            "    <div>Data: FanGraphs</div>",
            "  </div>",
            "</div>",
        ])
        rows_html = "\n".join(rows)
        st.markdown(rows_html, unsafe_allow_html=True)
        st.caption("PDF Export wasn't working, so screenshot to save.")
        st.caption("If dragging doesn't update in table, drag it again.")
        st.caption("Find a player's Fangraphs ID in their Fangraphs profile URL")
        st.caption("TZ records ended in 2001, DRS started in 2002")
        st.caption("Rookies with accents, initials, etc. may not return a headshot")
       
