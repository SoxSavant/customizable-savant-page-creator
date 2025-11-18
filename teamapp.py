import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from pathlib import Path
from io import BytesIO
from pybaseball import batting_stats
from datetime import date
from st_aggrid import (
    AgGrid,
    GridOptionsBuilder,
    GridUpdateMode,
    DataReturnMode,
    JsCode,
)

plt.rcdefaults()  # ensure default fonts/styles each run

st.set_page_config(page_title="Custom Team Savant Page App", layout="wide")

# Hide Streamlit Cloud toolbar + profile badge on deployed app
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
    </style>
    """,
    unsafe_allow_html=True,
)
title_col, meta_col = st.columns([3, 1])
with title_col:
    st.title("Custom Team Savant Page App")
with meta_col:
    st.markdown(
        """
        <div style="text-align: right; font-size: 1rem; padding-top: 0.6rem;">
            Built by <a href="https://twitter.com/Sox_Savant" target="_blank">@Sox_Savant</a>
            <span style="color: #aaa;">(v 1.1)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

CSS_KEY = "stat_builder_css"
if not st.session_state.get(CSS_KEY):
    st.markdown(
        """
        <style>
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
    st.session_state[CSS_KEY] = True

# --------------------- Teams ---------------------
TEAMS = {
    "ARI": "Arizona Diamondbacks", "ATL": "Atlanta Braves",
    "BAL": "Baltimore Orioles",    "BOS": "Boston Red Sox",
    "CHC": "Chicago Cubs",         "CIN": "Cincinnati Reds",
    "CLE": "Cleveland Guardians",  "COL": "Colorado Rockies",
    "CHW": "Chicago White Sox",    "DET": "Detroit Tigers",
    "HOU": "Houston Astros",       "KCR":  "Kansas City Royals",
    "LAA": "Los Angeles Angels",   "LAD": "Los Angeles Dodgers",
    "MIA": "Miami Marlins",        "MIL": "Milwaukee Brewers",
    "MIN": "Minnesota Twins",      "NYM": "New York Mets",
    "NYY": "New York Yankees",     "OAK": "Oakland Athletics",
    "ATH": "Athletics",
    "PHI": "Philadelphia Phillies","PIT": "Pittsburgh Pirates",
    "SDP":  "San Diego Padres",     "SEA": "Seattle Mariners",
    "SFG":  "San Francisco Giants", "STL": "St. Louis Cardinals",
    "TBR":  "Tampa Bay Rays",       "TEX": "Texas Rangers",
    "TOR": "Toronto Blue Jays",    "WSN": "Washington Nationals"
}

TRUTHY_STRINGS = {"true", "1", "yes", "y", "t"}

STAT_PRESETS = {
    "Statcast": [
        "WAR",
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
    "Standard": [
        "WAR"
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
    "Off", "Def", "BsR", "WAR", "Barrel%", "HardHit%", "EV", "MaxEV",
    "wRC+", "wOBA", "xwOBA", "xBA", "xSLG", "OPS", "SLG", "OBP", "AVG", "ISO",
    "BABIP", "R", "RBI", "HR", "XBH", "H", "2B", "3B", "SB", "BB", "IBB", "SO",
    "K%", "BB%", "K-BB%", "O-Swing%", "Z-Swing%", "Swing%", "Contact%", "WPA", "Clutch",
    "Whiff%", "Pull%", "Cent%", "Oppo%", "GB%", "FB%", "LD%", "LA",
]

@st.cache_data(show_spinner=True)
def load_batting(y: int) -> pd.DataFrame:
    return batting_stats(y, y, qual=0)

def get_teams_for_year(season: int) -> dict[str, str]:
    """Return team mapping for provided season, handling Athletics rename."""
    show_athletics_key = "ATH" if season >= 2025 else "OAK"
    teams_for_year = {}
    for abbr, name in TEAMS.items():
        if abbr in {"OAK", "ATH"} and abbr != show_athletics_key:
            continue
        teams_for_year[abbr] = name
    return teams_for_year


def get_team_nickname(full_name: str) -> str:
    """Return nickname portion for logo lookup, handling multi-word cities."""
    multi_word_cities = {
        "Kansas City", "Los Angeles", "New York", "San Diego",
        "San Francisco", "St. Louis", "Tampa Bay"
    }
    for city in multi_word_cities:
        prefix = f"{city} "
        if full_name.startswith(prefix):
            return full_name[len(prefix):]
    return full_name.split(" ", 1)[-1]

# --------------------- Controls ---------------------
left_col, right_col = st.columns([1.2, 1.5])

with left_col:
    year = st.slider("Season", 1900, date.today().year, date.today().year)
    teams_for_year = get_teams_for_year(year)
    team_options = list(teams_for_year.keys())
    team_select_key = "team_abbr_select"
    preferred_team = st.session_state.get(
        team_select_key,
        "ARI" if "ARI" in team_options else team_options[0],
    )
    if preferred_team not in team_options:
        preferred_team = team_options[0]
    default_index = team_options.index(preferred_team)
    team_abbr = st.selectbox(
        "Team",
        team_options,
        index=default_index,
        key=team_select_key,
    )
    min_pa = st.number_input("Minimum PA", 0, 800, 502)

stat_builder_container = left_col.container()

# --------------------- Data ---------------------
team_full_name = TEAMS[team_abbr]
nickname = get_team_nickname(team_full_name)
logo_dir = Path(__file__).parent / "logos"
logo_path = logo_dir / f"{nickname}.png"
logo_img = None
if logo_path.exists():
    try:
        logo_img = mpimg.imread(logo_path)
    except Exception:
        logo_img = None

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

# Team leaders for selected PA
team_df = df[(df["Team"] == team_abbr) & (df["PA"] >= min_pa)].copy()
if team_df.empty:
    st.warning(f"No players on {team_abbr} with ≥ {min_pa} PA.")
    st.stop()

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
def add_stat_callback(stat_key: str, select_key: str, reset_key: str, sentinel: str):
    choice = st.session_state.get(select_key)
    if not choice or choice == sentinel:
        return
    # Recalculate base config to pass to normalize_stat_rows
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
    st.session_state[manual_stat_update_key] = True
    st.session_state[reset_key] = True

def remove_stat_callback(stat_key: str, select_key: str, reset_key: str, sentinel: str):
    choice = st.session_state.get(select_key)
    if not choice or choice == sentinel:
        return
    # Recalculate base config to pass to normalize_stat_rows
    current_preset_for_base = st.session_state.get(stat_preset_key, default_preset_name)
    preset_base_candidates = [stat for stat in STAT_PRESETS[current_preset_for_base] if stat in stat_options]
    if not preset_base_candidates and stat_options:
        preset_base_candidates = [stat_options[0]]
    preset_base_config = [{"Stat": stat, "Show": True} for stat in preset_base_candidates]

    config = st.session_state.get(stat_key, preset_base_config)
    config = normalize_stat_rows(config, preset_base_config)
    new_config = [row for row in config if row.get("Stat") != choice]
    st.session_state[stat_key] = new_config or [row.copy() for row in preset_base_config]
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
    st.session_state[manual_stat_update_key] = True
    st.session_state[add_reset_key] = True
    st.session_state[remove_reset_key] = True

def normalize_stat_rows(rows, fallback):
    """Clean incoming rows into a valid stat config list."""
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

# ====================================================================================
# FIX: Ensure stat_config persists across Team/PA changes by initializing it only once.
# ====================================================================================

# 1. Initialize State ONLY ONCE if not present (This is what prevents the reset)
if stat_state_key not in st.session_state:
    st.session_state[stat_preset_key] = default_preset_name
    
    # Calculate initial base config for the default preset
    current_preset_for_base = st.session_state[stat_preset_key]
    preset_base_candidates = [stat for stat in STAT_PRESETS[current_preset_for_base] if stat in stat_options]
    if not preset_base_candidates and stat_options:
        preset_base_candidates = [stat_options[0]]
    preset_base_config = [{"Stat": stat, "Show": True} for stat in preset_base_candidates]
    
    # Set the initial stat config
    st.session_state[stat_state_key] = preset_base_config

# 2. Update config variables for the current run based on session state
current_preset_for_base = st.session_state.get(stat_preset_key, default_preset_name)
preset_base_candidates = [stat for stat in STAT_PRESETS[current_preset_for_base] if stat in stat_options]
if not preset_base_candidates and stat_options:
    preset_base_candidates = [stat_options[0]]
preset_base_config = [{"Stat": stat, "Show": True} for stat in preset_base_candidates]

current_stat_config = st.session_state.get(stat_state_key, preset_base_config)
current_stat_config = normalize_stat_rows(current_stat_config, preset_base_config)

# ====================================================================================
# END FIX
# ====================================================================================

with stat_builder_container:
    # 3. New Preset Select Box (Simplified)
    # Using key and on_change prevents the config from being reset on Min PA/Team change.
    prior_preset = st.session_state.get(stat_preset_key, default_preset_name)
    preset_index = preset_options.index(prior_preset) if prior_preset in preset_options else 0
    st.selectbox(
        "Stat Preset",
        preset_options,
        index=preset_index,
        key=stat_preset_key, # Use the session state key directly
        on_change=stat_preset_callback,
        args=(stat_preset_key, stat_state_key, stat_options)
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
        suppressMenu=True,
    )
    gb.configure_column(
        "Stat",
        header_name="Stat",
        cellEditor="agSelectCellEditor",
        cellEditorParams={"values": allowed_add_stats or stat_options},
        wrapText=True,
        autoHeight=True,
        flex=1,
    )
    grid_height = min(480, 90 + len(stat_config_df) * 44)
    grid_response = AgGrid(
        stat_config_df,
        gridOptions=gb.build(),
        height=grid_height,
        theme="streamlit",
        data_return_mode=DataReturnMode.AS_INPUT,
        reload_data=False,
        fit_columns_on_grid_load=True,
        update_mode=GridUpdateMode.MODEL_CHANGED,
        allow_unsafe_jscode=True,
        enable_enterprise_modules=False,
        key="stat_builder_grid",
        update_on=["rowDragEnd", "cellValueChanged"], 
    )

grid_records = None
if grid_response and grid_response.data is not None:
    if isinstance(grid_response.data, pd.DataFrame):
        grid_df = grid_response.data.copy()
    else:
        grid_df = pd.DataFrame(grid_response.data)
    if "Drag" in grid_df.columns:
        grid_df = grid_df.drop(columns=["Drag"])
    if "Stat" in grid_df.columns:
        # Drop AG Grid placeholder rows that have no stat selected
        grid_df = grid_df[grid_df["Stat"].astype(str).str.strip().ne("")]
    grid_records = grid_df.to_dict("records")

# ... (code up to line 664, after grid_records is defined)

manual_override = st.session_state.pop(manual_stat_update_key, False)

# Convert current_stat_config to records for comparison
current_config_records = [{k: v for k, v in row.items() if k in ["Stat", "Show"]} for row in current_stat_config]

# Determine if the grid data is identical to the current state data
is_config_identical = (
    grid_records is not None and 
    len(grid_records) == len(current_config_records) and 
    all(a["Stat"] == b["Stat"] and a["Show"] == b["Show"] for a, b in zip(grid_records, current_config_records))
)

if manual_override:
    # 1. If add/remove was used, the state is already correct from the callback
    cleaned_config = current_stat_config.copy()
elif grid_records and not is_config_identical:
    # 2. If grid data exists AND it is DIFFERENT from the current state, 
    #    assume the user edited/dragged and update the state.
    cleaned_config = normalize_stat_rows(grid_records, preset_base_config)
else:
    # 3. Otherwise (Min PA/Team change), stick with the existing state order
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
    # append .0 to these stats if its a whole number
    if upper_stat in {"WAR", "FWAR", "EV", "AVG EXIT VELO", "OFF", "DEF", "BSR"}: 
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

# --------------------- Build leader rows ---------------------
leaders = []

label_map = {
    "HardHit%": "Hard Hit%",
    "WAR": "fWAR",
    "EV": "Avg Exit Velo",
}
lower_better = {"K%", "O-Swing%", "Whiff%", "GB%"}

for stat in stats_order:

    # Completely skip stats missing from dataset
    if stat not in df.columns:
        continue

    # TEAM values
    team_vals = team_df[[stat, "Name"]].dropna(subset=[stat])
    if team_vals.empty:
        continue  # no one on the team has a real value for this stat

    team_leader_row = team_vals.sort_values(
        stat,
        ascending=stat in lower_better
    ).iloc[0]
    leader_val = float(team_leader_row[stat])

    # LEAGUE percentile distribution (340+ PA)
    league_vals = league_for_pct[stat].dropna()
    if league_vals.empty:
        continue  # league does not have values for this stat

    # Compute percentile safely (higher = better)
    pct = (league_vals <= leader_val).mean() * 100.0
    if stat in lower_better:
        pct = 100 - pct
    pct = float(np.clip(pct, 0, 100))


    # Pretty output label
    label = label_map.get(stat, stat)

    leaders.append({
        "Stat": label,
        "Leader": team_leader_row["Name"],
        "Value": leader_val,
        "Pct": pct
    })


lead_df = pd.DataFrame(leaders)

if lead_df.empty:
    st.warning("No stats available to display.")
    st.stop()

lead_df["Display"] = lead_df.apply(lambda r: format_stat(r["Stat"], r["Value"]), axis=1)

with right_col:
    # --------------------- Plot ---------------------

    cmap = LinearSegmentedColormap.from_list(
        "savant",
        [
            (0, "#335AA1"),   # low end (blue)
            (0.5, "#E8E8E8"), # neutral
            (1, "#D92229"),   # high end (bright red)
        ],
    )

    # Increased height based on number of rows
    fig_height = 1.7 + len(lead_df) * 0.40
    fig, ax = plt.subplots(figsize=(7.5, fig_height))

    # Adjust white card space with more breathing room between subtitle and bars
    top_pad = 0.12 if len(lead_df) > 6 else 0.14 if len(lead_df) > 4 else 0.4
    ax_height = 0.82 - top_pad
    ax.set_position([0.08, top_pad, 0.8, ax_height])

    # Title
    fig.text(
        0.5, 0.885,
        f"{year} {team_full_name}",
        ha="center", va="center",
        fontsize=22, fontweight="bold"
    )

    fig.text(
        0.5, 0.84,
        f"(min {min_pa} PA)",
        ha="center", va="center",
        fontsize=13, color="#555"
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
    # if logo_img is not None:
    #     imagebox = OffsetImage(logo_img, zoom=0.08)
    #     ab = AnnotationBbox(
    #         imagebox,
    #         (0.88, 0.90),
    #         xycoords="figure fraction",
    #         frameon=False,
    #         pad=0,
    #         zorder=0.5
    #     )
    #     ax.add_artist(ab)

    y = np.arange(len(lead_df))

    TRACK_H = 0.82
    BAR_H = 0.82
    LEFT_OFFSET = 3
    BAR_LENGTH = 45
    VALUE_X = LEFT_OFFSET + BAR_LENGTH + 12
    LABEL_X = 0
    BUBBLE_SIZE = 700

    # Background track bars
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

        name = row["Leader"]
        bubble_x = LEFT_OFFSET + bar_width
        name_x = LEFT_OFFSET + bar_width / 2
        needs_shift = pct < len(str(name)) * 3.2
        if needs_shift:
            name_x = bubble_x + (VALUE_X -  bubble_x) * 0.2
            name_ha = "left"
        else:
            name_x = LEFT_OFFSET + bar_width / 2 - 1
            name_ha = "center"

        ax.text(
            name_x, i, name,
            ha=name_ha, va="center",
            fontsize=13, fontweight="bold", color="#111"
        )

        ax.scatter(bubble_x, i, s=BUBBLE_SIZE, color=color,
                   edgecolors="white", linewidth=2.4, zorder=3)

        ax.text(
            bubble_x, i + 0.04, f"{int(round(pct))}",
            ha="center", va="center",
            fontsize=11, fontweight="bold", color="white"
        )

        ax.text(VALUE_X - 9.5, i, row["Display"],
                ha="left", va="center", fontsize=12, color="#111")

        ax.text(LABEL_X, i, row["Stat"],
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
    fig.savefig(pdf_buffer, format="pdf", bbox_inches = "tight", pad_inches = .25)
    pdf_buffer.seek(0)

    st.pyplot(fig, use_container_width=True, clear_figure=True)
    download_name = f"{team_abbr}_{year}_stat_leaders.pdf"
    st.download_button(
        "Download as PDF",
        data=pdf_buffer,
        file_name=download_name,
        mime="application/pdf",
    )
