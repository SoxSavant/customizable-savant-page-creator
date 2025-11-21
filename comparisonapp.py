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
from datetime import date
from pathlib import Path
from pybaseball import batting_stats, fielding_stats, playerid_lookup, bwar_bat
from pybaseball.statcast_fielding import statcast_outs_above_average
import requests
from bs4 import BeautifulSoup

st.set_page_config(page_title="Player Comparison App", layout="wide")

st.markdown(
    """
    <style>
        :root {
            --stat-col-width: 120px;
            --headshot-col-width: 220px;
            --headshot-img-width: 200px;
            --player-name-size: 1.35rem;
            --player-meta-size: 1.3rem;
        }
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
            max-width: 100%;
            margin: 0 auto;
        }

        .compare-card .headshot-row {
            display: grid;
            grid-auto-flow: column;
            grid-auto-columns: 1fr;
            grid-template-columns: var(--stat-col-width) 1fr 1fr; /* default, overridden inline for more players */
            align-items: center;
            justify-items: center;
            width: 100%;
            max-width: 100%;
            overflow: hidden;
            margin-bottom: .2rem;
            gap: 0;
        }
        .compare-card .headshot-spacer {
            width: var(--stat-col-width);
        }

        .compare-card .headshot-col {
            flex: 1 1 auto;
            width: auto;
            max-width: var(--headshot-col-width);
            min-width: 0;
            text-align: center;
            padding-top: .1rem;
        }
        .compare-card .headshot-col img { /*headshot size, background color, border */
            border: 1px solid #d0d0d0;
            background: #f2f2f2;
            border-radius: 4px;
            padding: 4px;
            width: 100%;
            max-width: var(--headshot-img-width);
            max-height: var(--headshot-img-width);
            height: auto;
            object-fit: contain;
        }
        .compare-card .player-name {
            font-size: var(--player-name-size);
            font-weight: 800;
            line-height: 1.2;
            margin: .2rem 0 0 0;
        }
        .compare-card .player-meta {
            color: #555;
            margin: 0 0 0.3rem 0;
            font-size: var(--player-meta-size);
        }
        .compare-table { /* sets the line height, width, font size */
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
            table-layout: fixed;
            line-height: 1.5;
        }
        .compare-table td {
            width: auto;
        }
        .compare-table th, .compare-table td { /* centers text, white background for rows */
            border: 1px solid #d0d0d0;
            padding: 3px 3px;
            text-align: center;
            background: #ffffff;
            color: #111111;
        }
        .compare-table th { /* Overall Stats row */
            background: #f1f1f1;
            font-weight: 800;
            color: #7b0d0d;
            font-size: 15px;
            line-height: 1.2;
        }
        .compare-table .overall-row th { /* More styling for Overall Stats row */
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
        .compare-table .stat-col { /* stats rows */
            font-weight: 700;
            background: #fafafa;
            color: #111;
            width: var(--stat-col-width);
        }
        .compare-table col.col-stat {
            width: var(--stat-col-width);
        }
        .compare-table col.col-player {
            width: auto;
        }
        .compare-table .best { /* highlights the winner in green */
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
    st.title("Custom Hitter Comparison")
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
    "Blank – Create your own": [
        "WAR",
    ],
}

STAT_ALLOWLIST = [
    "Off", "Def", "BsR", "WAR", "Barrel%", "HardHit%", "EV", "MaxEV",
    "wRC+", "wOBA", "xwOBA", "xBA", "xSLG", "OPS", "SLG", "OBP", "AVG", "ISO",
    "BABIP", "G", "PA", "AB", "R", "RBI", "HR", "XBH", "H", "2B", "3B", "SB", "BB", "IBB", "SO",
    "K%", "BB%", "K-BB%", "O-Swing%", "Z-Swing%", "Swing%", "Contact%", "WPA", "Clutch",
    "Whiff%", "Pull%", "Cent%", "Oppo%", "GB%", "FB%", "LD%", "LA",
    "FRV", "OAA", "ARM", "RANGE", "DRS", "TZ", "FRM", "UZR", "bWAR", "Age",
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


def compute_team_display(teams: list[str]) -> str:
    """
    Convert a list of team codes into a display string.
    """
    if not teams:
        return "N/A"
    if len(teams) == 1:
        return teams[0]       # Includes the collapsed "OAK/ATH"
    return f"{len(teams)} Teams"


def collapse_athletics(teams: list[str]) -> list[str]:
    """
    Collapse OAK + ATH into a single franchise for counting purposes
    but preserve all other teams.
    """
    has_oak = "OAK" in teams
    has_ath = "ATH" in teams

    # If both appear, collapse them into one entry
    if has_oak and has_ath:
        new_list = [t for t in teams if t not in {"OAK", "ATH"}]
        new_list.append("OAK/ATH")
        return sorted(new_list)

    # If only one of OAK or ATH appears → keep as-is
    return teams



VALID_TEAMS = {
    "ARI","ATL","BAL","BOS","CHC","CIN","CLE","COL","CHW","DET",
    "HOU","KCR","LAA","LAD","MIA","MIL","MIN","NYM","NYY",
    "OAK","ATH","PHI","PIT","SDP","SEA","SFG","STL","TBR",
    "TEX","TOR","WSN"
}


def normalize_team_code(team: str, year: int) -> str:
    """
    Normalize Athletics team codes depending on the year.
    Before 2025 → OAK
    2025+ → ATH
    """
    if not team:
        return team

    team = team.upper().strip()

    if team in {"", "-", "--", "---", "- - -", "TOT"}:
        return None

    # Athletics → year-aware
    if year < 2025:
        if team in {"ATH", "OAK"}:
            return "OAK"
    else:
        if team in {"ATH", "OAK"}:
            return "ATH"

    return team


def get_player_teams_fangraphs(fg_id: int, start_year: int, end_year: int) -> list[str]:
    """
    Scrape Fangraphs batting tables and extract MLB team codes.
    Handles:
    - multiple teams in a year
    - ATH/OAK transition
    - split-season rows
    """

    url = f"https://www.fangraphs.com/players/x/{fg_id}/stats?season=all"
    r = requests.get(url, timeout=10)
    soup = BeautifulSoup(r.text, "html.parser")

    found = []

    # Scan all tables (MLB + Minors + College)
    for table in soup.find_all("table"):
        for row in table.find_all("tr"):
            cols = [c.get_text(strip=True) for c in row.find_all("td")]
            if len(cols) < 2:
                continue

            # Year check
            if not (cols[0].isdigit() and len(cols[0]) == 4):
                continue

            year = int(cols[0])
            team_raw = cols[1].strip()
            # Ensure this is a Major League row
            league = cols[2] if len(cols) > 2 else ""
            if league != "MLB":
                continue

            if team_raw not in VALID_TEAMS:
                continue

            if start_year <= year <= end_year:
                team_norm = normalize_team_code(team_raw, year)
                if team_norm:
                    found.append(team_norm)

    # Unique + alphabetical
    unique = sorted(set(found))

    # 👇 Collapse OAK + ATH → OAK/ATH
    collapsed = collapse_athletics(unique)

    return collapsed

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
    result["Team"] = "None"
    result["TeamDisplay"] = "None"
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
        if col == "Age":
            age_min = series.min(skipna=True)
            age_max = series.max(skipna=True)
            if pd.isna(age_min) or pd.isna(age_max):
                continue
            if abs(age_min - age_max) < 0.01:
                result[col] = float(age_min)
            else:
                result[col] = f"{int(round(age_min))}-{int(round(age_max))}"
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

    # Single year: return directly
    if start == end:
        try:
            df = batting_stats(start, end, qual=0, split_seasons=False)
            if df is not None and not df.empty:
                return df
        except Exception:
            pass
        fallback = load_year(start)
        return fallback if fallback is not None else pd.DataFrame()

    # Multi-year: always build from per-year frames so Age spans are accurate
    frames = []
    failed_years = []
    for year in range(start, end + 1):
        try:
            yearly = batting_stats(year, year, qual=0, split_seasons=False)
        except Exception:
            yearly = None
        if yearly is None or yearly.empty:
            try:
                yearly = load_year(year)
            except Exception:
                yearly = None
        if yearly is not None and not yearly.empty:
            frames.append(yearly)
        else:
            failed_years.append(year)
    if frames:
        combined = pd.concat(frames, ignore_index=True)

        age_span_map: dict[str, str | float] = {}
        if "Name" in combined.columns and "Age" in combined.columns:
            for name, grp in combined.groupby("Name"):
                series = pd.to_numeric(grp["Age"], errors="coerce")
                if series.isna().all():
                    continue
                age_min = series.min(skipna=True)
                age_max = series.max(skipna=True)
                if pd.isna(age_min) or pd.isna(age_max):
                    continue
                if abs(age_min - age_max) < 0.01:
                    age_span_map[name] = float(age_min)
                else:
                    age_span_map[name] = f"{int(round(age_min))}-{int(round(age_max))}"

        grouped_rows = []
        for name, grp in combined.groupby("Name"):
            row = aggregate_player_group(grp, name)
            if age_span_map and name in age_span_map:
                row["Age"] = age_span_map[name]
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
    # Drop pitcher rows only when they have no plate appearances (true pitchers)
    pitcher_col = df.get("pitcher")
    pa_col = pd.to_numeric(df.get("PA"), errors="coerce") if "PA" in df.columns else pd.Series(np.nan, index=df.index)
    if pitcher_col is not None:
        pitcher_mask = (
            pitcher_col.astype(str)
            .str.strip()
            .str.upper()
            .isin({"Y", "1", "TRUE"})
        )
        no_pa_mask = pa_col.isna() | (pa_col <= 0)
        drop_mask = pitcher_mask & no_pa_mask
        df = df[~drop_mask]
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
            pitcher_mask = pd.to_numeric(data["pitcher"], errors="coerce").fillna(0) != 0
            pa_series = pd.to_numeric(data.get("PA"), errors="coerce") if "PA" in data.columns else pd.Series(np.nan, index=data.index)
            no_pa_mask = pa_series.isna() | (pa_series <= 0)
            drop_mask = pitcher_mask & no_pa_mask
            data = data[~drop_mask]
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
    """
    Load the player's batting profile for either a single season or a multi-year span.
    Also attaches corrected Team / TeamDisplay based on direct Fangraphs HTML scraping.
    """

    # -----------------------------
    # SINGLE YEAR
    # -----------------------------
    if start_year == end_year:
        try:
            df = batting_stats(start_year, end_year, qual=0, split_seasons=False, players=str(fg_id))
        except Exception:
            df = None

        if df is not None and not df.empty:
            row = df.iloc[0].copy()

            # Pull real teams from HTML
            team_values = get_player_teams_fangraphs(fg_id, start_year, end_year)
            if team_values:
                team_display = compute_team_display(team_values)
                row["Team"] = team_display
                row["TeamDisplay"] = team_display
            else:
                # fallback: use pybaseball Team column
                row["TeamDisplay"] = normalize_display_team(str(row.get("Team", "")).strip())

            row["Name"] = str(row.get("Name", "")).strip()
            return row

        # if single year but no data, fall through to multi-year logic
        # (rare, but harmless)

    # -----------------------------
    # MULTI-YEAR SPAN
    # -----------------------------
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

    # Pull actual teams from HTML scraper
    team_values = get_player_teams_fangraphs(fg_id, start_year, end_year)
    if team_values:
        team_display = compute_team_display(team_values)
        aggregated["Team"] = team_display
        aggregated["TeamDisplay"] = team_display

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


def build_player_profile(
    fg_id: int,
    start_year: int,
    end_year: int,
    mlbam_override: int | None = None,
) -> pd.Series | None:
    batting = load_player_batting_profile(fg_id, start_year, end_year)
    if batting is None:
        return None
    fielding = load_player_fielding_profile(fg_id, start_year, end_year)
    for key, val in fielding.items():
        batting[key] = val
    name_value = str(batting.get("Name", "")).strip()
    mlbam_id = mlbam_override
    bbref_id = None
    if name_value:
        lookup_mlbam, lookup_bbref = lookup_mlbam_id(name_value, return_bbref=True)
        if mlbam_id is None:
            mlbam_id = lookup_mlbam
        bbref_id = lookup_bbref
    batting["mlbam_override"] = mlbam_id if mlbam_id is not None else np.nan
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
    id_cols = ["mlbam_override", "mlbamid", "mlbam_id", "mlbam", "MLBID", "MLBAMID", "key_mlbam"]
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
        "mlbam_override", "mlbamid", "MLBID", "mlbam_id", "mlbam", "key_mlbam", "MLBAMID", "playerid"
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
player_mode_options = ["2 players", "3 players", "4 players"]
player_mode = st.radio(
    "Players to compare",
    player_mode_options,
    index=0,
    horizontal=True,
    help="Switch between 2, 3, or 4 player comparison modes.",
)
player_count = int(player_mode.split()[0])
column_weights_map = {
    "2 players": [1, 1],
    "3 players": [1, 1.5],
    "4 players": [1, 2],
}
column_weights = column_weights_map.get(player_mode, [1, 1])

left_col, right_col = st.columns(column_weights)

with left_col:
    controls_container = st.container()
    stat_builder_container = st.container()

# --------------------- Controls ---------------------
current_year = date.today().year
years_desc = list(range(current_year, 1870, -1))
MAX_PLAYERS = 4
default_names = ["Mookie Betts", "Aaron Judge", "", ""]
# If mode increases, default new players to single-season
prev_count = st.session_state.get("comp_prev_player_count", 2)
if player_count > prev_count:
    for idx in range(prev_count, player_count):
        single_key = f"comp_single_year_{idx}"
        st.session_state[single_key] = True
st.session_state["comp_prev_player_count"] = player_count
# Initialize state for all player slots up-front to avoid missing keys when switching modes.
for idx in range(MAX_PLAYERS):
    name_key = f"comp_player_{idx}"
    id_key = f"comp_player_{idx}_id"
    mode_key = f"comp_player_{idx}_mode"
    mlbam_key = f"comp_player_{idx}_mlbam"
    mlbam_enabled_key = f"comp_player_{idx}_mlbam_enabled"
    single_key = f"comp_single_year_{idx}"
    year_single_key = f"comp_year_{idx}_single"
    year_start_key = f"comp_year_{idx}_start"
    year_end_key = f"comp_year_{idx}_end"
    if name_key not in st.session_state:
        st.session_state[name_key] = default_names[idx] if idx < len(default_names) else ""
    if id_key not in st.session_state:
        st.session_state[id_key] = ""
    if mode_key not in st.session_state:
        st.session_state[mode_key] = "Name"
    if mlbam_key not in st.session_state:
        st.session_state[mlbam_key] = ""
    if mlbam_enabled_key not in st.session_state:
        st.session_state[mlbam_enabled_key] = False
    if single_key not in st.session_state:
        st.session_state[single_key] = True
    if year_single_key not in st.session_state:
        st.session_state[year_single_key] = years_desc[0]
    if year_start_key not in st.session_state:
        st.session_state[year_start_key] = years_desc[0]
    if year_end_key not in st.session_state:
        st.session_state[year_end_key] = years_desc[0]


def parse_mlbam_override(raw: str, enabled: bool) -> int | None:
    if not enabled:
        return None
    raw_val = str(raw).strip()
    if not raw_val:
        return None
    try:
        return int(raw_val)
    except Exception:
        return None


with controls_container:
    year_cols = st.columns(player_count)
    year_ranges: list[tuple[int, int]] = []
    for idx in range(player_count):
        label = chr(ord("A") + idx)
        single_key = f"comp_single_year_{idx}"
        year_single_key = f"comp_year_{idx}_single"
        year_start_key = f"comp_year_{idx}_start"
        year_end_key = f"comp_year_{idx}_end"
        with year_cols[idx]:
            single = st.checkbox(
                f"Single season (Player {label})",
                key=single_key,
            )
            if single:
                year_single = st.selectbox(
                    f"Season (Player {label})",
                    years_desc,
                    index=0,
                    key=year_single_key,
                )
                year_start = year_single
                year_end = year_single
            else:
                year_start = st.selectbox(
                    f"Season Start (Player {label})",
                    years_desc,
                    index=0,
                    key=year_start_key,
                )
                year_end = st.selectbox(
                    f"Season End (Player {label})",
                    years_desc,
                    index=0,
                    key=year_end_key,
                )
        year_ranges.append((min(year_start, year_end), max(year_start, year_end)))

    # Ensure visible player slots have a default name before rendering inputs
    for idx in range(player_count):
        name_key = f"comp_player_{idx}"
        if not st.session_state.get(name_key) and default_names[idx]:
            st.session_state[name_key] = default_names[idx]

    input_cols = st.columns(player_count)
    player_inputs = []
    for idx in range(player_count):
        label = chr(ord("A") + idx)
        name_key = f"comp_player_{idx}"
        id_key = f"comp_player_{idx}_id"
        mode_key = f"comp_player_{idx}_mode"
        with input_cols[idx]:
            mode_val = st.selectbox(
                f"Player {label} Input",
                ["Name", "FanGraphs ID"],
                key=mode_key,
            )
            if mode_val == "Name":
                name_input = st.text_input(f"Player {label}", key=name_key)
                id_input = st.session_state.get(id_key, "")
            else:
                id_input = st.text_input(f"Player {label} FanGraphs ID", key=id_key)
                name_input = st.session_state.get(name_key, "")
        player_inputs.append({
            "mode": mode_val,
            "name_input": name_input.strip(),
            "id_input": str(id_input).strip(),
            "years": year_ranges[idx],
            "mlbam_override": parse_mlbam_override(
                st.session_state.get(f"comp_player_{idx}_mlbam", ""),
                st.session_state.get(f"comp_player_{idx}_mlbam_enabled", False),
            ),
        })

players_data = []
for idx, cfg in enumerate(player_inputs):
    label = chr(ord("A") + idx)
    years = cfg["years"]
    if cfg["mode"] == "Name":
        if not cfg["name_input"]:
            st.warning(f"Enter a name for Player {label} or switch to FanGraphs ID input.")
            st.stop()
        fg_id = resolve_player_fg_id(cfg["name_input"])
    else:
        if not cfg["id_input"]:
            st.warning(f"Enter a FanGraphs ID for Player {label} or switch to name input.")
            st.stop()
        try:
            fg_id = int(cfg["id_input"])
        except Exception:
            fg_id = None

    if not fg_id or fg_id <= 0:
        if cfg["mode"] == "Name":
            st.error(f"Could not find data for {cfg['name_input'] or f'Player {label}'}. Check the spelling or use the ID input.")
        else:
            st.error(f"Player {label} FanGraphs ID must be a positive integer.")
        st.stop()

    player_row = build_player_profile(fg_id, *years, mlbam_override=cfg["mlbam_override"])
    if player_row is None:
        st.error(f"Could not load data for Player {label}.")
        st.stop()

    display_name = str(player_row.get("Name", "")).strip()
    if not display_name:
        display_name = cfg["name_input"] if cfg["mode"] == "Name" else f"FG#{fg_id}"

    df = pd.DataFrame([player_row])
    for metric in FIELDING_STATS:
        if metric in df.columns:
            df[metric] = pd.to_numeric(df[metric], errors="coerce")
    if cfg["mlbam_override"] is not None:
        df["mlbam_override"] = cfg["mlbam_override"]

    team_display = player_row.get("TeamDisplay", normalize_display_team(player_row.get("Team", "")))
    year_label = f"{years[0]}" if years[0] == years[1] else f"{years[0]}-{years[1]}"

    players_data.append({
        "fg_id": fg_id,
        "display_name": display_name,
        "input_name": cfg["name_input"],
        "mode": cfg["mode"],
        "team": team_display,
        "year_label": year_label,
        "df": df,
        "row": player_row,
        "mlbam_override": cfg["mlbam_override"],
        "label_char": label,
    })

# Ensure column labels are unique
seen_labels = set()
for idx, pdata in enumerate(players_data):
    base = pdata["display_name"]
    label = base
    if label in seen_labels and pdata["year_label"]:
        label = f"{base} ({pdata['year_label']})"
    if label in seen_labels:
        label = f"{base} (Player {pdata['label_char']})"
    seen_labels.add(label)
    pdata["col_label"] = label

dfs = [p["df"] for p in players_data]

# --------------------- Stat builder setup ---------------------
stat_exclusions = {"Season"}
numeric_sets = []
for df in dfs:
    numeric_sets.append({col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])})
if not numeric_sets:
    st.error("No numeric stats available to display.")
    st.stop()

if len(numeric_sets) == 1:
    numeric_stats = list(numeric_sets[0] - stat_exclusions)
else:
    numeric_stats = list(set.intersection(*numeric_sets) - stat_exclusions)
# Ensure Age is available even if represented as a string span.
if all("Age" in df.columns for df in dfs) and "Age" not in numeric_stats:
    numeric_stats.append("Age")

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
        "Use the drop downs to add or remove stats and the arrows to reorder them."
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

    st.markdown('<div class="stat-table">', unsafe_allow_html=True)
    st.markdown('<div class="table-header">', unsafe_allow_html=True)
    header_cols = st.columns([0.25, 0.25, .25, 1])
    header_cols[0].markdown("**Up**")
    header_cols[1].markdown("**Down**")
    header_cols[2].markdown("**Stat**")
    header_cols[3].markdown("**Show**")
    st.markdown('</div>', unsafe_allow_html=True)

    for idx, row in enumerate(current_stat_config):
        st.markdown('<div class="table-row">', unsafe_allow_html=True)
        up_col, down_col, stat_col, show_col = st.columns([0.25, 0.25, .25, 1])
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

stats_order = [row["Stat"] for row in st.session_state[stat_state_key] if row.get("Show", True)]
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

    if upper_stat == "AGE":
        if isinstance(val, str):
            return val
        v = float(val)
        return f"{int(round(v))}" if abs(v - round(v)) < 1e-9 else f"{v:.1f}"

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
winner_map: dict[str, set[str]] = {}
col_order = [p["col_label"] for p in players_data]
for stat in stats_order:
    # Skip stats not common to all players
    if any(stat not in pdata["df"].columns for pdata in players_data):
        continue

    raw_label = label_map.get(stat, stat)
    values = []
    numeric_vals = []
    has_non_numeric = False
    for pdata in players_data:
        val = pdata["row"].get(stat, np.nan)
        values.append(val)
        if pd.isna(val):
            numeric_vals.append(np.nan)
            continue
        try:
            numeric_vals.append(float(val))
        except Exception:
            has_non_numeric = True
            numeric_vals.append(np.nan)

    winners = set()
    numeric_candidates = [v for v in numeric_vals if not pd.isna(v)]
    if numeric_candidates and not has_non_numeric and stat.upper() != "AGE":
        best_val = min(numeric_candidates) if stat in lower_better else max(numeric_candidates)
        winners = {
            col_order[idx]
            for idx, v in enumerate(numeric_vals)
            if not pd.isna(v) and abs(v - best_val) < 1e-9
        }

    row_dict = {"Stat": raw_label}
    for idx, pdata in enumerate(players_data):
        row_dict[pdata["col_label"]] = format_stat(stat, values[idx])
    comparison_rows.append(row_dict)
    winner_map[raw_label] = winners

table_df = pd.DataFrame(comparison_rows, columns=["Stat"] + col_order)

for pdata in players_data:
    pdata["headshot"] = get_headshot_url(pdata["display_name"], pdata["df"])
esc = html.escape
# Even column widths for all modes
if player_count == 2:
    stat_col_width = "calc(100% / 3)"
    player_col_width = "calc(100% / 3)"
    grid_template = "1fr 1fr 1fr"
else:
    shared_width = f"calc(100% / {player_count + 1})"
    stat_col_width = shared_width
    player_col_width = shared_width
    grid_template = " ".join(["1fr"] * (player_count + 1))
if player_count == 2:
    headshot_width = 200
    headshot_col_width = 220
    player_name_size = "1.35rem"
    player_meta_size = "1.3rem"
else:
    headshot_width = f"clamp(110px, calc(80vw / {player_count + 1}), 140px)"
    headshot_col_width = f"clamp(125px, calc(84vw / {player_count + 1}), 160px)"
    # Smaller text for 3–4 player layouts
    player_name_size = ".9rem"
    player_meta_size = ".95rem"
name_style_attr = f' style="font-size:{player_name_size}; line-height:1.1;"' if player_count > 2 else ""

with right_col:
    if table_df.empty:
        st.warning("No stats available to compare.")
    else:
        # Build headshot + table HTML
        rows = [
            f"<div class=\"compare-card\" style=\"--stat-col-width: {stat_col_width}; --headshot-col-width: {headshot_col_width}px; --headshot-img-width: {headshot_width}px; --player-name-size: {player_name_size}; --player-meta-size: {player_meta_size};\">",
            f"  <div class=\"headshot-row\" style=\"grid-template-columns: {grid_template};\">",
        ]
        if player_count == 2:
            # left player
            pdata = players_data[0]
            img_html = f'<img src="{esc(pdata["headshot"])}" width="{headshot_width}" />' if pdata["headshot"] else ""
            rows.extend([
                '    <div class="headshot-col">',
                f"      <div class=\"player-meta\">{esc(str(pdata['year_label']))} | {esc(str(pdata['team']))}</div>",
                f"      {img_html}",
                f"      <div class=\"player-name\"{name_style_attr}>{esc(pdata['display_name'])}</div>",
                "    </div>",
            ])
            rows.append("    <div class=\"headshot-spacer\"></div>")
            # right player
            pdata = players_data[1]
            img_html = f'<img src="{esc(pdata["headshot"])}" width="{headshot_width}" />' if pdata["headshot"] else ""
            rows.extend([
                '    <div class="headshot-col">',
                f"      <div class=\"player-meta\">{esc(str(pdata['year_label']))} | {esc(str(pdata['team']))}</div>",
                f"      {img_html}",
                f"      <div class=\"player-name\"{name_style_attr}>{esc(pdata['display_name'])}</div>",
                "    </div>",
            ])
        else:
            rows.append("    <div class=\"headshot-spacer\"></div>")
            for pdata in players_data:
                img_html = f'<img src="{esc(pdata["headshot"])}" width="{headshot_width}" />' if pdata["headshot"] else ""
                rows.extend([
                    '    <div class="headshot-col">',
                    f"      <div class=\"player-meta\">{esc(str(pdata['year_label']))} | {esc(str(pdata['team']))}</div>",
                    f"      {img_html}",
                    f"      <div class=\"player-name\"{name_style_attr}>{esc(pdata['display_name'])}</div>",
                    "    </div>",
                ])
        rows.extend([
            "  </div>",
            "  <table class=\"compare-table\">",
            "    <colgroup>",
        ])
        if player_count == 2:
            rows.append(f"      <col class=\"col-player\" style=\"width: {player_col_width};\" />")
            rows.append(f"      <col class=\"col-stat\" style=\"width: {stat_col_width};\" />")
            rows.append(f"      <col class=\"col-player\" style=\"width: {player_col_width};\" />")
            render_cols = [players_data[0]["col_label"], "__STAT__", players_data[1]["col_label"]]
        else:
            rows.append(f"      <col class=\"col-stat\" style=\"width: {stat_col_width};\" />")
            for _ in players_data:
                rows.append(f"      <col class=\"col-player\" style=\"width: {player_col_width};\" />")
            render_cols = ["__STAT__"] + [p["col_label"] for p in players_data]
        rows.extend([
            "    </colgroup>",
            "    <thead>",
            f"      <tr class=\"overall-row\">",
            f"        <th colspan=\"{player_count + 1}\">Overall Stats</th>",
            "      </tr>",
            "    </thead>",
            "    <tbody>",
        ])
        for row in comparison_rows:
            stat_label = esc(str(row["Stat"]))
            winners = winner_map.get(str(row["Stat"]), set())
            rows.append("      <tr>")
            for col_id in render_cols:
                if col_id == "__STAT__":
                    rows.append(f"        <td class=\"stat-col\">{stat_label}</td>")
                else:
                    val = esc(str(row.get(col_id, "")))
                    cell_class = "best" if col_id in winners else ""
                    rows.append(f"        <td class=\"{cell_class}\">{val}</td>")
            rows.append("      </tr>")
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
        with st.expander("Optional MLB ID overrides (use if bWAR/headshot is missing)", expanded=False):
            override_cols = st.columns(player_count)
            for idx, pdata in enumerate(players_data):
                with override_cols[idx]:
                    st.checkbox("Use MLB ID override", key=f"comp_player_{idx}_mlbam_enabled")
                    st.text_input(
                        f"Player {pdata['label_char']} MLB ID",
                        key=f"comp_player_{idx}_mlbam",
                        placeholder="e.g. 608070",
                        disabled=not st.session_state.get(f"comp_player_{idx}_mlbam_enabled", False),
                    )
        st.caption("Screenshot to save")
        st.caption("Find a player's Fangraphs/MLB ID in their Fangraphs/MLB profile URL")
        st.caption("TZ records ended in 2001, DRS started in 2002")
      
