import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
from io import BytesIO
from pybaseball import batting_stats, playerid_lookup
from st_aggrid import (
    AgGrid,
    GridOptionsBuilder,
    GridUpdateMode,
    DataReturnMode,
    JsCode,
)

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
            gap: 14em;
            margin-bottom: 1.2rem;
        }
        .compare-card .headshot-col {
            flex: 0 0 220px;
            text-align: center;
        }
        .compare-card .headshot-col img {
            border: 1px solid #dcdcdc;
            border-radius: 4px;
        }
        .compare-card .player-name {
            font-size: 1.35rem;
            font-weight: 800;
            margin: 0.2rem 0 0 0;
        }
        .compare-card .player-meta {
            color: #555;
            margin: 0 0 0.3rem 0;
            font-size: 0.95rem;
        }
        .compare-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 15px;
        }
        .compare-table th, .compare-table td {
            border: 1px solid #d0d0d0;
            padding: 8px 10px;
            text-align: center;
            background: #ffffff;
            color: #111111;
        }
        .compare-table th {
            background: #f1f1f1;
            font-weight: 800;
            color: #7b0d0d;
        }
        .compare-table .overall-row th {
            background: #f1f1f1;
            color: #7b0d0d;
            font-weight: 800;
            font-size: 17px;
            padding: 10px 0 6px 0;
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

HEADSHOT_BASE = "https://img.mlbstatic.com/mlb-photos/image/upload/w_240,q_auto:best,f_auto/people/{mlbam}/headshot/silo/current"


@st.cache_data(show_spinner=True)
def load_batting(y: int) -> pd.DataFrame:
    return batting_stats(y, y, qual=0)


@st.cache_data(show_spinner=False)
def lookup_mlbam_id(full_name: str):
    """Best-effort MLBAM lookup using pybaseball's playerid_lookup."""
    if not full_name or " " not in full_name:
        return None
    parts = full_name.split(" ")
    first = parts[0]
    last = " ".join(parts[1:])
    try:
        lookup_df = playerid_lookup(last, first)
    except Exception:
        return None
    if lookup_df is None or lookup_df.empty:
        return None
    mlbam = lookup_df.iloc[0].get("key_mlbam")
    try:
        return int(mlbam) if pd.notna(mlbam) else None
    except Exception:
        return None


def get_headshot_url(name: str, df: pd.DataFrame) -> str | None:
    candidate_cols = [
        "mlbamid", "MLBID", "mlbam_id", "mlbam", "key_mlbam", "MLBAMID", "playerid"
    ]
    for col in candidate_cols:
        if col in df.columns:
            vals = df.loc[df["Name"] == name, col].dropna()
            if not vals.empty:
                try:
                    mlbam = int(vals.iloc[0])
                    return HEADSHOT_BASE.format(mlbam=mlbam)
                except Exception:
                    pass
    mlbam_fallback = lookup_mlbam_id(name)
    if mlbam_fallback:
        return HEADSHOT_BASE.format(mlbam=mlbam_fallback)
    return None


def get_player_row(df: pd.DataFrame, name: str) -> pd.Series | None:
    data = (
        df[df["Name"] == name]
        .sort_values("PA", ascending=False)
        .head(1)
    )
    return data.squeeze() if not data.empty else None


def get_team_display(df: pd.DataFrame, player_name: str) -> str:
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
    if not player_teams:
        if "- - -" in player_teams_raw:
            player_teams = ["2+ Tms"]
        elif "TOT" in player_teams_raw:
            player_teams = ["TOT"]
        else:
            player_teams = [str(df.loc[df["Name"] == player_name, "Team"].iloc[0]).upper()]
    if len(player_teams) > 1:
        return f"{len(player_teams)} Tms"
    return player_teams[0]


# --------------------- Layout containers ---------------------
left_col, right_col = st.columns([1.2, 1.55])

with left_col:
    controls_container = st.container()
    stat_builder_container = st.container()

# --------------------- Controls ---------------------
min_pa_key = "comp_min_pa_input"
min_pa_default = 200
with controls_container:
    year = st.slider("Season", 1900, date.today().year, date.today().year)
    min_pa = st.number_input(
        "Minimum PA (for player list)",
        0,
        800,
        st.session_state.get(min_pa_key, min_pa_default),
        key=min_pa_key,
    )

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

eligible_players = df[df["PA"] >= min_pa].copy()
if eligible_players.empty:
    eligible_players = df.copy()

player_lookup = (
    eligible_players.sort_values(["Name", "PA"], ascending=[True, False])
    .drop_duplicates(subset=["Name"], keep="first")
    .sort_values("Name")
)
player_options = player_lookup["Name"].tolist()
if not player_options:
    st.error("No players available to display.")
    st.stop()

default_a = st.session_state.get("comp_player_a", player_options[0])
if default_a not in player_options:
    default_a = player_options[0]
default_b = st.session_state.get("comp_player_b", player_options[1] if len(player_options) > 1 else player_options[0])
if default_b not in player_options:
    default_b = player_options[0]

with controls_container:
    sel_a_col, sel_b_col = st.columns(2)
    with sel_a_col:
        player_a = st.selectbox("Player A", player_options, index=player_options.index(default_a), key="comp_player_a")
    with sel_b_col:
        player_b = st.selectbox("Player B", player_options, index=player_options.index(default_b), key="comp_player_b")

if player_a == player_b:
    st.warning("Select two different players to compare.")
    st.stop()

player_a_row = get_player_row(df, player_a)
player_b_row = get_player_row(df, player_b)
if player_a_row is None or player_b_row is None:
    st.error("Could not load data for one of the selected players.")
    st.stop()

player_a_team = get_team_display(df, player_a)
player_b_team = get_team_display(df, player_b)

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
stat_preset_key = "comp_stat_preset_select"
preset_options = list(STAT_PRESETS.keys())
stat_state_key = "comp_stat_config"
manual_stat_update_key = "comp_stat_config_manual_update"
add_select_key = "comp_add_stat_select"
remove_select_key = "comp_remove_stat_select"
add_reset_key = "comp_reset_add_select"
remove_reset_key = "comp_reset_remove_select"

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
            "",
            add_options,
            label_visibility="hidden",
            key=add_select_key,
            on_change=add_stat_callback,
            args=(stat_state_key, add_select_key, add_reset_key, sentinel_add),
        )

    with remove_col:
        st.selectbox(
            "",
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
    )

    grid_options = gb.build()

    grid_height = min(480, 90 + len(stat_config_df) * 44)
    grid_response = AgGrid(
        stat_config_df,
        gridOptions=grid_options,
        height=grid_height,
        width="100%",
        theme="streamlit",
        data_return_mode=DataReturnMode.AS_INPUT,
        reload_data=True,
        fit_columns_on_grid_load=True,
        update_mode=GridUpdateMode.VALUE_CHANGED,
        allow_unsafe_jscode=True,
        enable_enterprise_modules=False,
        key="comp_stat_grid",
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


# --------------------- Comparison table ---------------------
label_map = {
    "HardHit%": "Hard Hit%",
    "WAR": "fWAR",
    "EV": "Avg Exit Velo",
}
lower_better = {"K%", "O-Swing%", "Whiff%", "GB%"}

comparison_rows = []
winner_map: dict[str, str | None] = {}
for stat in stats_order:
    if stat not in df.columns:
        continue
    a_val = player_a_row.get(stat, np.nan)
    b_val = player_b_row.get(stat, np.nan)
    if pd.isna(a_val) and pd.isna(b_val):
        continue

    label = label_map.get(stat, stat)

    if pd.isna(a_val):
        winner = player_b
    elif pd.isna(b_val):
        winner = player_a
    else:
        better = a_val < b_val if stat in lower_better else a_val > b_val
        if a_val == b_val:
            winner = None
        else:
            winner = player_a if better else player_b

    comparison_rows.append({
        "Stat": label,
        player_a: format_stat(stat, a_val),
        player_b: format_stat(stat, b_val),
    })
    winner_map[label] = winner

table_df = pd.DataFrame(comparison_rows)

headshot_a = get_headshot_url(player_a, df)
headshot_b = get_headshot_url(player_b, df)

with right_col:
    if table_df.empty:
        st.warning("No stats available to compare.")
    else:
        rows = [
            f"<div class=\"compare-card\">",
            "  <div class=\"headshot-row\">",
            "    <div class=\"headshot-col\">",
            f"      <div class=\"player-meta\">{year} • {player_a_team}</div>",
            f"      {f'<img src=\"{headshot_a}\" width=\"200\" />' if headshot_a else ''}",
            f"      <div class=\"player-name\">{player_a}</div>",
            "    </div>",
            "    <div class=\"headshot-col\">",
            f"      <div class=\"player-meta\">{year} • {player_b_team}</div>",
            f"      {f'<img src=\"{headshot_b}\" width=\"200\" />' if headshot_b else ''}",
            f"      <div class=\"player-name\">{player_b}</div>",
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
            stat_label = row["Stat"]
            best = winner_map.get(stat_label)
            a_class = "best" if best == player_a else ""
            b_class = "best" if best == player_b else ""
            rows.extend([
                "      <tr>",
                f"        <td class=\"{a_class}\">{row[player_a]}</td>",
                f"        <td class=\"stat-col\">{stat_label}</td>",
                f"        <td class=\"{b_class}\">{row[player_b]}</td>",
                "      </tr>",
            ])
        rows.extend([
            "    </tbody>",
            "  </table>",
            "</div>",
        ])
        st.markdown("\n".join(rows), unsafe_allow_html=True)

        # -------- Download as PDF --------
        pdf_buffer = BytesIO()
        col_labels = [player_a, "Stat", player_b]
        cell_text = []
        cell_colors = []
        for _, row in table_df.iterrows():
            stat_label = row["Stat"]
            best = winner_map.get(stat_label)
            a_val = row[player_a]
            b_val = row[player_b]
            cell_text.append([a_val, stat_label, b_val])
            a_color = "#E5F1E4" if best == player_a else "#ffffff"
            b_color = "#E5F1E4" if best == player_b else "#ffffff"
            cell_colors.append([a_color, "#fafafa", b_color])

        fig, ax = plt.subplots(figsize=(8.5, 0.55 * (len(table_df) + 2)))
        ax.axis("off")
        table = ax.table(
            cellText=cell_text,
            cellColours=cell_colors,
            colLabels=col_labels,
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.3)

        # Bold the header row
        for col_idx in range(len(col_labels)):
            cell = table[0, col_idx]
            cell.set_facecolor("#f1f1f1")
            cell.get_text().set_color("#7b0d0d")
            cell.get_text().set_fontweight("bold")

        # Bold stat names and winners
        for i in range(1, len(cell_text) + 1):
            stat_cell = table[i, 1]
            stat_cell.get_text().set_fontweight("bold")
            if cell_colors[i - 1][0] != "#ffffff":
                table[i, 0].get_text().set_fontweight("bold")
            if cell_colors[i - 1][2] != "#ffffff":
                table[i, 2].get_text().set_fontweight("bold")

        plt.tight_layout()
        fig.savefig(pdf_buffer, format="pdf", bbox_inches="tight")
        pdf_buffer.seek(0)

        st.download_button(
            "Download PDF",
            data=pdf_buffer,
            file_name=f"{player_a.replace(' ', '_')}_{player_b.replace(' ', '_')}_{year}_comparison.pdf",
            mime="application/pdf",
        )
