import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path
import numpy as np

# Attempt to import OpenAI
try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    OpenAI = None
    _OPENAI_AVAILABLE = False

# -----------------------------------------------------------------------------
# CONSTANTS & SETUP
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
LOGO_DIR = BASE_DIR / "assets"  # put ubc.png, ame.png, cov.png in ./assets

# Route IDs
ID_REF = "ref"
ID_MOF = "mof"

# Polymer fraction of Ref-Bead carried into U@Bead (for electricity allocation)
POLYMER_FRACTION_MOF = 0.87

# Default equipment parameters for electricity-intensive steps
# Masses are in grams; power in kW; time in hours.
EQUIPMENT_DEFAULTS = {
    "ref_micro": {
        "label": "Ref-Bead microfluidiser (PDChNF fibrillation)",
        "route_id": ID_REF,
        "power_kw": 1.5,
        "time_h": 0.33,
        "batch_mass_g": 0.4,    # dry bead mass per batch
        "alloc_mass_g": 0.4,    # allocation mass / effective capacity
    },
    "ref_mix": {
        "label": "Ref-Bead hotplate mixing (50 °C)",
        "route_id": ID_REF,
        "power_kw": 0.625,
        "time_h": 6.0,
        "batch_mass_g": 0.4,
        "alloc_mass_g": 0.4,
    },
    "ref_cross": {
        "label": "Ref-Bead hotplate crosslinking (50 °C)",
        "route_id": ID_REF,
        "power_kw": 0.625,
        "time_h": 6.0,
        "batch_mass_g": 0.4,
        "alloc_mass_g": 0.4,
    },
    "ref_freeze": {
        "label": "Ref-Bead freeze-drying",
        "route_id": ID_REF,
        "power_kw": 1.84,
        "time_h": 16.0,
        "batch_mass_g": 0.4,
        "alloc_mass_g": 0.4,
    },
    "mof_stir_zr": {
        "label": "UiO stirring (Zr step)",
        "route_id": ID_MOF,
        "power_kw": 0.625,
        "time_h": 12.0,
        "batch_mass_g": 0.6,
        "alloc_mass_g": 0.6,
    },
    "mof_stir_linker": {
        "label": "UiO stirring (linker step)",
        "route_id": ID_MOF,
        "power_kw": 0.625,
        "time_h": 12.0,
        "batch_mass_g": 0.6,
        "alloc_mass_g": 0.6,
    },
    "mof_freeze": {
        "label": "Second freeze-drying (U@Bead)",
        "route_id": ID_MOF,
        "power_kw": 1.84,
        "time_h": 16.0,
        "batch_mass_g": 0.6,
        "alloc_mass_g": 0.6,
    },
}

st.set_page_config(page_title="LCA Explorer: Interactive & AI-Enhanced", layout="wide")


# -----------------------------------------------------------------------------
# AI HELPER FUNCTION
# -----------------------------------------------------------------------------
def get_ai_insight(context_data, user_question):
    """
    Sends calculation results + user question to OpenAI for analysis.
    """
    if not _OPENAI_AVAILABLE:
        return "Error: OpenAI library not installed."

    api_key = st.secrets.get("openai_api_key2")
    if not api_key:
        return "Error: API Key 'openai_api_key2' not found in secrets."

    client = OpenAI(api_key=api_key)

    # Prepare a summary of the current results to give the AI context
    context_str = "Current LCA Results:\n"
    for res in context_data:
        context_str += (
            f"- {res['name']}: "
            f"Total GWP={res['Total GWP']:.2e}, "
            f"Elec GWP={res['Electricity GWP']:.2e}, "
            f"Elec%={res['Electricity %']:.1f}%\n"
        )

    system_prompt = f"""
You are an expert in Life Cycle Assessment (LCA) for materials science.
Use the provided data to answer the user's question.

Context Data:
{context_str}

Guidelines:
- Be concise and scientific.
- Comment on why certain impacts are high (for example electricity at lab scale).
- If the user asks about optimisation, suggest practical process improvements, especially around electricity demand, batch size and grid mix.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_question},
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"AI Error: {e}"


# -----------------------------------------------------------------------------
# DATA LOADING & SESSION STATE
# -----------------------------------------------------------------------------
@st.cache_data
def load_default_data():
    """Load default CSV files from disk."""

    def read_safe(path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(path, encoding="utf-8")
        except Exception:
            return pd.read_csv(path, encoding="latin1")

    return (
        read_safe(DATA_DIR / "emission_factors.csv"),
        read_safe(DATA_DIR / "lca_routes.csv"),
        read_safe(DATA_DIR / "performance.csv"),
        read_safe(DATA_DIR / "literature.csv"),
    )


def reset_data():
    """Resets session state to default CSV values and equipment settings."""
    ef, routes, perf, lit = load_default_data()
    st.session_state["ef_df"] = ef
    st.session_state["routes_df"] = routes
    st.session_state["perf_df"] = perf
    st.session_state["lit_df"] = lit

    # Reset custom grids
    st.session_state["custom_grids"] = {
        "QC Hydro": 0.002,
        "Canada Avg": 0.1197,
        "UK Grid": 0.225,
        "EU Avg": 0.25,
        "US Avg": 0.38,
        "China Grid": 0.58,
    }

    # Reset equipment parameters
    st.session_state["equipment_params"] = {k: dict(v) for k, v in EQUIPMENT_DEFAULTS.items()}

    # Clear widget states related to equipment controls
    for key in list(st.session_state.keys()):
        if key.startswith("equip_"):
            del st.session_state[key]


# Initialise Session State
if "ef_df" not in st.session_state:
    reset_data()

if "equipment_params" not in st.session_state:
    st.session_state["equipment_params"] = {k: dict(v) for k, v in EQUIPMENT_DEFAULTS.items()}

# Shortcuts (not used directly in main, but kept for completeness)
EF_DF = st.session_state["ef_df"]
ROUTES_DF = st.session_state["routes_df"]
PERF_DF = st.session_state["perf_df"]
LIT_DF = st.session_state["lit_df"]


# -----------------------------------------------------------------------------
# HEADER LOGOS
# -----------------------------------------------------------------------------
def render_header_logos():
    cov_path = DATA_DIR  / "cov.png"
    ubc_path = DATA_DIR / "ubc.png"

    col_left, col_spacer, col_right = st.columns([1, 2, 1])
    with col_left:
        if cov_path.exists():
            st.image(str(cov_path))
    with col_right:
        if ubc_path.exists():
            st.image(str(ubc_path))


# -----------------------------------------------------------------------------
# EQUIPMENT-BASED ELECTRICITY CALCULATIONS
# -----------------------------------------------------------------------------
def compute_electricity_from_equipment(equipment_params):
    """
    Compute electricity intensity (kWh/kg bead) for each route and step
    from equipment power, time and allocation mass.
    """
    step_rows = []
    ref_total = 0.0
    mof_specific_total = 0.0

    # Ref-Bead steps
    for step_id, cfg in equipment_params.items():
        if cfg.get("route_id") == ID_REF:
            mass_g = cfg.get("alloc_mass_g", cfg.get("batch_mass_g", 0.0))
            mass_kg = max(mass_g, 1e-9) / 1000.0
            power_kw = cfg.get("power_kw", 0.0)
            time_h = cfg.get("time_h", 0.0)
            kwh_per_kg = power_kw * time_h / mass_kg
            ref_total += kwh_per_kg
            step_rows.append(
                {
                    "route_id": ID_REF,
                    "Bead": "Ref-Bead (Polymer)",
                    "Step": cfg.get("label", step_id),
                    "kWh_per_kg": kwh_per_kg,
                }
            )

    # Carry-over of Ref-Bead electricity into U@Bead (87 wt% polymer support)
    carry_over = POLYMER_FRACTION_MOF * ref_total
    step_rows.append(
        {
            "route_id": ID_MOF,
            "Bead": "U@Bead (MOF-Functionalised)",
            "Step": "Carry-over (Ref support)",
            "kWh_per_kg": carry_over,
        }
    )

    # MOF-specific steps
    for step_id, cfg in equipment_params.items():
        if cfg.get("route_id") == ID_MOF:
            mass_g = cfg.get("alloc_mass_g", cfg.get("batch_mass_g", 0.0))
            mass_kg = max(mass_g, 1e-9) / 1000.0
            power_kw = cfg.get("power_kw", 0.0)
            time_h = cfg.get("time_h", 0.0)
            kwh_per_kg = power_kw * time_h / mass_kg
            mof_specific_total += kwh_per_kg
            step_rows.append(
                {
                    "route_id": ID_MOF,
                    "Bead": "U@Bead (MOF-Functionalised)",
                    "Step": cfg.get("label", step_id),
                    "kWh_per_kg": kwh_per_kg,
                }
            )

    route_totals = {
        ID_REF: ref_total,
        ID_MOF: carry_over + mof_specific_total,
    }

    return route_totals, pd.DataFrame(step_rows)


def sync_equipment_params_from_widgets():
    """
    Sync equipment parameter values from widget state into session_state['equipment_params'].
    """
    if "equipment_params" not in st.session_state:
        st.session_state["equipment_params"] = {k: dict(v) for k, v in EQUIPMENT_DEFAULTS.items()}

    equipment_params = st.session_state["equipment_params"]

    for step_id, cfg in equipment_params.items():
        for field in ("power_kw", "time_h", "batch_mass_g", "alloc_mass_g"):
            widget_key = f"equip_{step_id}_{field}"
            if widget_key in st.session_state:
                val = st.session_state[widget_key]
                try:
                    cfg[field] = float(val)
                except (TypeError, ValueError):
                    pass

    st.session_state["equipment_params"] = equipment_params


# -----------------------------------------------------------------------------
# CALCULATION ENGINE
# -----------------------------------------------------------------------------
def calculate_impacts(
    route_id,
    ef_df,
    routes_df,
    electricity_override_map=None, # New parameter for optional override
    efficiency_factor: float = 1.0,
    recycling_rate: float = 0.0,
    yield_rate: float = 100.0,
    transport_pct: float = 0.0,
):
    """Calculates GWP based on session state and efficiency modifiers."""
    route_data = routes_df[routes_df["route_id"] == route_id].copy()
    if route_data.empty:
        return None, None

    yield_multiplier = 1.0 / (yield_rate / 100.0)

    # Electricity Logic: Check if we have an override (for modified scaling), else use CSV/Default
    if electricity_override_map and route_id in electricity_override_map:
        base_elec_kwh = electricity_override_map[route_id]
    else:
        base_elec_kwh = float(route_data.iloc[0]["electricity_kwh_per_fu"])
    
    elec_kwh = base_elec_kwh * efficiency_factor * yield_multiplier

    elec_source = route_data.iloc[0]["electricity_source"]
    ef_elec_row = ef_df[ef_df["reagent_name"] == elec_source]
    ef_elec = float(ef_elec_row["GWP_kgCO2_per_kg"].iloc[0]) if not ef_elec_row.empty else 0.0
    gwp_elec = elec_kwh * ef_elec

    # Reagents
    contributions = [
        {
            "Component": "Electricity",
            "Category": "Electricity",
            "Mass (kg)": 0.0,
            "GWP": gwp_elec,
        }
    ]

    total_reagent_gwp = 0.0

    for _, row in route_data.iterrows():
        reagent = row["reagent_name"]
        base_mass = float(row["mass_kg_per_fu"])

        # Yield impact
        mass_needed = base_mass * yield_multiplier

        # Recycling impact
        is_solvent = reagent in ["Ethanol", "Formic acid (88%)", "Acetic acid"]
        if is_solvent:
            effective_mass = mass_needed * (1 - (recycling_rate / 100.0))
        else:
            effective_mass = mass_needed

        ef_row = ef_df[ef_df["reagent_name"] == reagent]
        ef_val = float(ef_row["GWP_kgCO2_per_kg"].iloc[0]) if not ef_row.empty else 0.0

        gwp_val = effective_mass * ef_val
        total_reagent_gwp += gwp_val

        if reagent in ["Chitosan", "PDChNF"]:
            cat = "Polymers"
        elif reagent in ["Zirconium tetrachloride", "2-Aminoterephthalic acid", "Formic acid (88%)", "Ethanol"]:
            cat = "MOF Reagents"
        else:
            cat = "Solvents/Other"

        contributions.append(
            {
                "Component": reagent,
                "Category": cat,
                "Mass (kg)": effective_mass,
                "GWP": gwp_val,
            }
        )

    # Transport
    raw_total_gwp = gwp_elec + total_reagent_gwp
    transport_gwp = raw_total_gwp * (transport_pct / 100.0)

    if transport_gwp > 0:
        contributions.append(
            {
                "Component": "Transport",
                "Category": "Logistics",
                "Mass (kg)": 0.0,
                "GWP": transport_gwp,
            }
        )

    final_total_gwp = raw_total_gwp + transport_gwp

    results = {
        "id": route_id,
        "name": route_data.iloc[0]["route_name"],
        "Total GWP": final_total_gwp,
        "Electricity GWP": gwp_elec,
        "Non-Electric GWP": total_reagent_gwp + transport_gwp,
        "Electricity kWh": elec_kwh,
        "Electricity EF Used": ef_elec,
    }

    return results, pd.DataFrame(contributions)


# -----------------------------------------------------------------------------
# PLOTTING HELPERS
# -----------------------------------------------------------------------------
def plot_sankey_diagram(results_list, route_id=None, title_prefix=""):
    """Build a simple Sankey diagram for a given route (Ref or MOF)."""
    if not results_list:
        return go.Figure()

    if route_id is not None:
        target_res = next((r for r in results_list if r["id"] == route_id), None)
        if target_res is None:
            target_res = results_list[0]
    else:
        target_res = results_list[0]

    elec_gwp = target_res["Electricity GWP"]
    chem_gwp = target_res["Non-Electric GWP"]
    total_gwp = target_res["Total GWP"]

    labels = ["Electricity Source", "Chemical Supply", "Lab Synthesis", "Total GWP"]
    colors = ["#FFD700", "#90EE90", "#87CEFA", "#FF6347"]

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=labels,
                    color=colors,
                ),
                link=dict(
                    source=[0, 1, 2],
                    target=[2, 2, 3],
                    value=[elec_gwp, chem_gwp, total_gwp],
                    color=[
                        "rgba(255, 215, 0, 0.4)",
                        "rgba(144, 238, 144, 0.4)",
                        "rgba(135, 206, 250, 0.4)",
                    ],
                ),
            )
        ]
    )

    fig.update_layout(
        title_text=f"{title_prefix}{target_res['name']}",
        font_size=10,
        height=400,
    )
    return fig


def get_electricity_step_data(equip_params) -> pd.DataFrame:
    """
    Electricity per process step, per kg bead, calculated from provided equipment settings.
    """
    _, step_df = compute_electricity_from_equipment(equip_params)
    return step_df


def create_system_boundary_figure() -> go.Figure:
    """
    Simple system boundary diagram for the gate to gate LCA scope.
    """
    fig = go.Figure()

    # Main system boundary box
    fig.add_shape(
        type="rect",
        x0=0.25,
        y0=0.2,
        x1=0.75,
        y1=0.8,
        line=dict(color="black", width=2),
        fillcolor="rgba(144, 238, 144, 0.1)",
    )

    # Upstream box (excluded)
    fig.add_shape(
        type="rect",
        x0=0.02,
        y0=0.35,
        x1=0.20,
        y1=0.65,
        line=dict(color="grey", width=1),
        fillcolor="rgba(200, 200, 200, 0.1)",
    )
    fig.add_annotation(
        x=0.11,
        y=0.5,
        text="Upstream:\nFisheries,\ncrab processing,\nchitin purification\n(excluded)",
        showarrow=False,
        font=dict(size=10),
    )

    # Downstream box (excluded)
    fig.add_shape(
        type="rect",
        x0=0.80,
        y0=0.35,
        x1=0.98,
        y1=0.65,
        line=dict(color="grey", width=1),
        fillcolor="rgba(200, 200, 200, 0.1)",
    )
    fig.add_annotation(
        x=0.89,
        y=0.5,
        text="Use phase,\nregeneration,\nend of life\n(excluded)",
        showarrow=False,
        font=dict(size=10),
    )

    # Main system annotation
    fig.add_annotation(
        x=0.5,
        y=0.6,
        text=(
            "System boundary (included):\n"
            "Lab scale bead synthesis\n"
            "PDChNF–chitosan bead (Ref-Bead)\n"
            "+ UiO-66-NH₂ growth and NaOH step\n"
            "(U@Bead-2step-aUiO)"
        ),
        showarrow=False,
        font=dict(size=11),
    )

    # Arrows
    fig.add_annotation(x=0.22, y=0.5, ax=0.25, ay=0.5, showarrow=True, arrowhead=2)
    fig.add_annotation(x=0.75, y=0.5, ax=0.78, ay=0.5, showarrow=True, arrowhead=2)

    fig.update_xaxes(visible=False, range=[0, 1])
    fig.update_yaxes(visible=False, range=[0, 1])

    fig.update_layout(
        title="System boundary for the screening LCA (gate to gate, lab scale)",
        height=350,
        margin=dict(l=20, r=20, t=70, b=20),
    )

    return fig


# -----------------------------------------------------------------------------
# MAIN APP
# -----------------------------------------------------------------------------
def main():
    # Logos on top of everything
    render_header_logos()

    st.title("Interactive LCA Explorer: Ref-Bead vs U@Bead")
    st.markdown(
        """
**Dashboard:** Adjust scenarios on the left. Use the tabs below to explore results, sensitivity and AI supported interpretation.
"""
    )

    # -------------------------------------------------------------------------
    # SIDEBAR: USER INPUTS
    # -------------------------------------------------------------------------
    with st.sidebar:
        st.header("Control Panel")

        if st.button("Reset to paper defaults"):
            reset_data()
            st.success("Data reset to defaults from the LCA section.")
            st.rerun()

        st.divider()

        # --- QUICK ADJUST ---
        st.subheader("1. Scenario parameters")

        # Grid Intensity
        current_grid_ef = float(
            st.session_state["ef_df"]
            .loc[
                st.session_state["ef_df"]["reagent_name"] == "Electricity (Canada)",
                "GWP_kgCO2_per_kg",
            ]
            .iloc[0]
        )

        new_grid_ef = st.slider(
            "Grid carbon intensity (kg CO₂ per kWh)",
            min_value=0.0,
            max_value=1.0,
            value=current_grid_ef,
            step=0.01,
            help="Controls the cleanliness of the power source used in the LCA.",
        )

        if new_grid_ef != current_grid_ef:
            st.session_state["ef_df"].loc[
                st.session_state["ef_df"]["reagent_name"] == "Electricity (Canada)",
                "GWP_kgCO2_per_kg",
            ] = new_grid_ef

        # Efficiency
        st.write("**Process efficiency**")
        eff_factor = st.slider(
            "Efficiency multiplier",
            min_value=0.1,
            max_value=1.0,
            value=1.0,
            help="1.0 = lab scale baseline. Lower values approximate improved industrial efficiency.",
        )

        # Recycling
        st.write("**Solvent recovery**")
        recycle_rate = st.slider(
            "Recycling rate (%)",
            min_value=0,
            max_value=95,
            value=0,
            help="Percentage of ethanol and formic acid recovered and reused.",
        )

        # Yield
        st.write("**Global yield**")
        yield_rate = st.slider(
            "Synthesis yield (%)",
            min_value=10,
            max_value=100,
            value=100,
            help="Lower yield increases the required inputs per kg of bead.",
        )

        # Transport
        st.write("**Transport overhead**")
        transport_overhead = st.slider(
            "Add transport (%)",
            min_value=0,
            max_value=50,
            value=0,
            help="Adds a fixed percentage to account for logistics emissions.",
        )

        st.divider()

        # --- TABLES ---
        st.subheader("2. Input tables")
        with st.expander("Edit detailed inputs"):
            st.caption("Emission factors")
            st.session_state["ef_df"] = st.data_editor(
                st.session_state["ef_df"], key="ed_ef", num_rows="dynamic"
            )

            st.caption("Bead recipes and electricity")
            st.session_state["routes_df"] = st.data_editor(
                st.session_state["routes_df"], key="ed_routes", num_rows="dynamic"
            )

            st.caption("Performance data")
            st.session_state["perf_df"] = st.data_editor(
                st.session_state["perf_df"], key="ed_perf", num_rows="dynamic"
            )

        # Refresh shortcuts after any edits
        ef_df = st.session_state["ef_df"]
        routes_df = st.session_state["routes_df"]
        perf_df = st.session_state["perf_df"]
        lit_df = st.session_state["lit_df"]

    # Sync equipment parameters from widget state
    sync_equipment_params_from_widgets()
    equipment_params_modified = st.session_state["equipment_params"]
    
    # CALCULATE ELECTRICITY FOR BOTH SCENARIOS
    # 1. Baseline (Default equipment params)
    route_elec_baseline, _ = compute_electricity_from_equipment(EQUIPMENT_DEFAULTS)
    
    # 2. Modified (User edited params)
    route_elec_modified, _ = compute_electricity_from_equipment(equipment_params_modified)

    # -------------------------------------------------------------------------
    # CALCULATIONS - RUN TWICE
    # -------------------------------------------------------------------------
    unique_routes = routes_df["route_id"].unique()

    # --- Run Baseline ---
    results_list_base = []
    dfs_list_base = []
    for rid in unique_routes:
        res, df = calculate_impacts(
            rid, ef_df, routes_df, 
            electricity_override_map=route_elec_baseline, # use baseline elec
            efficiency_factor=eff_factor,
            recycling_rate=recycle_rate,
            yield_rate=yield_rate,
            transport_pct=transport_overhead,
        )
        if res:
            res["Electricity %"] = (res["Electricity GWP"] / res["Total GWP"]) * 100.0 if res["Total GWP"] > 0 else 0.0
            results_list_base.append(res)
            dfs_list_base.append(df)

    # --- Run Modified ---
    results_list_mod = []
    dfs_list_mod = []
    for rid in unique_routes:
        res, df = calculate_impacts(
            rid, ef_df, routes_df, 
            electricity_override_map=route_elec_modified, # use modified elec
            efficiency_factor=eff_factor,
            recycling_rate=recycle_rate,
            yield_rate=yield_rate,
            transport_pct=transport_overhead,
        )
        if res:
            res["Electricity %"] = (res["Electricity GWP"] / res["Total GWP"]) * 100.0 if res["Total GWP"] > 0 else 0.0
            results_list_mod.append(res)
            dfs_list_mod.append(df)

    if not results_list_base:
        st.warning("No routes found. Check the input tables in the sidebar.")
        return

    # Helper to make summary DF
    def make_summary_df(res_list):
        perf_map = {row["route_id"]: float(row["capacity_mg_g"]) for _, row in perf_df.iterrows()}
        rows = []
        for r in res_list:
            cap = perf_map.get(r["id"], 0.001)
            rows.append({
                "Bead": r["name"],
                "Total GWP": r["Total GWP"],
                "Non-Electric GWP": r["Non-Electric GWP"],
                "GWP per g Cu": r["Total GWP"] / cap,
                "Electricity %": (r["Electricity GWP"] / r["Total GWP"]) * 100.0 if r["Total GWP"] > 0 else 0.0,
            })
        return pd.DataFrame(rows)

    sum_df_base = make_summary_df(results_list_base)
    sum_df_mod = make_summary_df(results_list_mod)

    # -------------------------------------------------------------------------
    # TABS
    # -------------------------------------------------------------------------
    tab_scale, tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Scaling (equipment)",
            "LCA results",
            "Sensitivity and scaling",
            "Inventory and flows",
            "Literature comparison",
            "AI insights",
        ]
    )

    # --- TAB: EQUIPMENT SCALING ---
    with tab_scale:
        st.header("Equipment utilisation and electricity scaling")
        st.markdown(
            """
The electricity intensities used in the LCA are derived from the power and runtime of each
unit operation, divided by an allocation mass (effective batch capacity).
Adjust the **Modified Scenario** values below. The graphs in other tabs will show the comparison.
"""
        )

        eq = st.session_state["equipment_params"]

        st.subheader("Ref-Bead (polymer) steps")
        for step_id in ["ref_micro", "ref_mix", "ref_cross", "ref_freeze"]:
            cfg = eq.get(step_id)
            if cfg is None: continue
            st.markdown(f"**{cfg['label']}**")
            col1, col2, col3, col4 = st.columns(4)
            with col1: st.number_input("Power (kW)", min_value=0.0, value=float(cfg["power_kw"]), step=0.01, key=f"equip_{step_id}_power_kw")
            with col2: st.number_input("Time per batch (h)", min_value=0.0, value=float(cfg["time_h"]), step=0.25, key=f"equip_{step_id}_time_h")
            with col3: st.number_input("Bead mass per batch (g)", min_value=0.0001, value=float(cfg["batch_mass_g"]), step=0.01, key=f"equip_{step_id}_batch_mass_g")
            with col4: st.number_input("Allocation mass (g)", min_value=0.0001, value=float(cfg["alloc_mass_g"]), step=0.01, key=f"equip_{step_id}_alloc_mass_g")

        st.subheader("U@Bead (MOF-functionalised) specific steps")
        for step_id in ["mof_stir_zr", "mof_stir_linker", "mof_freeze"]:
            cfg = eq.get(step_id)
            if cfg is None: continue
            st.markdown(f"**{cfg['label']}**")
            col1, col2, col3, col4 = st.columns(4)
            with col1: st.number_input("Power (kW)", min_value=0.0, value=float(cfg["power_kw"]), step=0.01, key=f"equip_{step_id}_power_kw")
            with col2: st.number_input("Time per batch (h)", min_value=0.0, value=float(cfg["time_h"]), step=0.25, key=f"equip_{step_id}_time_h")
            with col3: st.number_input("Bead mass per batch (g)", min_value=0.0001, value=float(cfg["batch_mass_g"]), step=0.01, key=f"equip_{step_id}_batch_mass_g")
            with col4: st.number_input("Allocation mass (g)", min_value=0.0001, value=float(cfg["alloc_mass_g"]), step=0.01, key=f"equip_{step_id}_alloc_mass_g")

        # Electricity summary table (Baseline vs Modified)
        st.markdown("### Electricity intensity comparison (kWh/kg)")
        comp_data = {
            "Bead": ["Ref-Bead (Polymer)", "U@Bead (MOF)"],
            "Baseline (Default)": [route_elec_baseline.get(ID_REF, 0), route_elec_baseline.get(ID_MOF, 0)],
            "Modified (Scaled)": [route_elec_modified.get(ID_REF, 0), route_elec_modified.get(ID_MOF, 0)],
        }
        st.dataframe(pd.DataFrame(comp_data).style.format({"Baseline (Default)": "{:.2e}", "Modified (Scaled)": "{:.2e}"}))

    # --- TAB 1: RESULTS ---
    with tab1:
        st.header("LCA results: Baseline vs. Modified")

        st.markdown("#### Summary Data")
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Baseline (Paper Defaults)")
            st.dataframe(sum_df_base.style.format({"Total GWP": "{:.2e}", "Non-Electric GWP": "{:.2f}", "GWP per g Cu": "{:.2f}", "Electricity %": "{:.1f}%"}))
        with c2:
            st.caption("Modified Scenario (Scaled Equipment)")
            st.dataframe(sum_df_mod.style.format({"Total GWP": "{:.2e}", "Non-Electric GWP": "{:.2f}", "GWP per g Cu": "{:.2f}", "Electricity %": "{:.1f}%"}))

        # Charts Side by Side
        col1, col2 = st.columns(2)
        
        # LEFT: BASELINE
        with col1:
            st.subheader("Baseline: Total GWP (log)")
            fig_log_base = px.bar(sum_df_base, x="Bead", y="Total GWP", color="Bead", log_y=True, title="Baseline: Total GWP (FU1)", text_auto=".2s")
            st.plotly_chart(fig_log_base, use_container_width=True)

            st.subheader("Baseline: Elec vs Chemicals")
            stack_data_base = []
            for r in results_list_base:
                stack_data_base.append({"Bead": r["name"], "Source": "Electricity", "GWP": r["Electricity GWP"]})
                stack_data_base.append({"Bead": r["name"], "Source": "Chemicals", "GWP": r["Non-Electric GWP"]})
            fig_stack_base = px.bar(pd.DataFrame(stack_data_base), x="Bead", y="GWP", color="Source", title="Baseline Contribution", text_auto=".2s")
            st.plotly_chart(fig_stack_base, use_container_width=True)

        # RIGHT: MODIFIED
        with col2:
            st.subheader("Modified: Total GWP (log)")
            fig_log_mod = px.bar(sum_df_mod, x="Bead", y="Total GWP", color="Bead", log_y=True, title="Modified: Total GWP (FU1)", text_auto=".2s")
            st.plotly_chart(fig_log_mod, use_container_width=True)

            st.subheader("Modified: Elec vs Chemicals")
            stack_data_mod = []
            for r in results_list_mod:
                stack_data_mod.append({"Bead": r["name"], "Source": "Electricity", "GWP": r["Electricity GWP"]})
                stack_data_mod.append({"Bead": r["name"], "Source": "Chemicals", "GWP": r["Non-Electric GWP"]})
            fig_stack_mod = px.bar(pd.DataFrame(stack_data_mod), x="Bead", y="GWP", color="Source", title="Modified Contribution", text_auto=".2s")
            st.plotly_chart(fig_stack_mod, use_container_width=True)

        st.divider()
        st.subheader("Performance Normalised (FU2)")
        c1, c2 = st.columns(2)
        with c1:
            fig_fu2_base = px.bar(sum_df_base, x="Bead", y="GWP per g Cu", color="Bead", title="Baseline: GWP per g Cu", text_auto=".2f")
            st.plotly_chart(fig_fu2_base, use_container_width=True)
        with c2:
            fig_fu2_mod = px.bar(sum_df_mod, x="Bead", y="GWP per g Cu", color="Bead", title="Modified: GWP per g Cu", text_auto=".2f")
            st.plotly_chart(fig_fu2_mod, use_container_width=True)

    # --- TAB 2: SENSITIVITY ---
    with tab2:
        st.header("Sensitivity and scaling (Applied to Modified Scenario)")
        # Note: Sensitivity usually applies logic on top of the current state.
        # For simplicity, we keep the original sensitivity logic but apply it to the modified state context mostly.
        
        # 1. Grid intensity sensitivity
        st.subheader("1. Grid intensity sensitivity")
        col_s1, col_s2 = st.columns([1, 2])
        with col_s1:
            st.write("Add custom grid point:")
            new_grid_name = st.text_input("Grid name", "My local grid")
            new_grid_val = st.number_input("Intensity", min_value=0.0, max_value=1.0, value=0.45)
            if st.button("Add grid to chart"):
                st.session_state["custom_grids"][new_grid_name] = new_grid_val

        sens_rows = []
        for g_name, g_val in st.session_state["custom_grids"].items():
            temp_ef = ef_df.copy()
            temp_ef.loc[temp_ef["reagent_name"].str.contains("Electricity"), "GWP_kgCO2_per_kg"] = g_val
            # We calculate sensitivity using the Modified Equipment Params
            for rid in unique_routes:
                res, _ = calculate_impacts(
                    rid, temp_ef, routes_df,
                    electricity_override_map=route_elec_modified,
                    efficiency_factor=eff_factor,
                    recycling_rate=recycle_rate,
                    yield_rate=yield_rate,
                    transport_pct=transport_overhead,
                )
                sens_rows.append({"Grid": g_name, "Grid Value": g_val, "Bead": res["name"], "Total GWP": res["Total GWP"]})

        df_sens = pd.DataFrame(sens_rows).sort_values("Grid Value")
        fig_sens = px.line(df_sens, x="Grid", y="Total GWP", color="Bead", markers=True, title="Total GWP vs Grid (using Modified Equipment)", hover_data=["Grid Value"])
        st.plotly_chart(fig_sens, use_container_width=True)

        st.divider()

        # 3. Electricity demand per process step
        st.subheader("3. Electricity demand per process step comparison")
        c1, c2 = st.columns(2)
        with c1:
            step_df_base = get_electricity_step_data(EQUIPMENT_DEFAULTS)
            fig_steps_base = px.bar(step_df_base, x="Step", y="kWh_per_kg", color="Bead", barmode="group", log_y=True, title="Baseline: Elec Demand", text_auto=".2s")
            fig_steps_base.update_layout(xaxis_tickangle=-30)
            st.plotly_chart(fig_steps_base, use_container_width=True)
        with c2:
            step_df_mod = get_electricity_step_data(equipment_params_modified)
            fig_steps_mod = px.bar(step_df_mod, x="Step", y="kWh_per_kg", color="Bead", barmode="group", log_y=True, title="Modified: Elec Demand", text_auto=".2s")
            fig_steps_mod.update_layout(xaxis_tickangle=-30)
            st.plotly_chart(fig_steps_mod, use_container_width=True)

    # --- TAB 3: INVENTORY ---
    with tab3:
        st.header("Inventory and impact breakdown")

        # Helper to combine DFs
        def get_combined_breakdown(dfs, res_list):
            comb = []
            for i, d in enumerate(dfs):
                d = d.copy()
                d["Bead"] = res_list[i]["name"]
                comb.append(d)
            return pd.concat(comb) if comb else pd.DataFrame()

        df_all_base = get_combined_breakdown(dfs_list_base, results_list_base)
        df_all_mod = get_combined_breakdown(dfs_list_mod, results_list_mod)

        if not df_all_base.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Baseline Breakdown")
                fig_br_base = px.bar(df_all_base, x="Bead", y="GWP", color="Component", title="Baseline: Total Breakdown (log)", barmode="group", log_y=True)
                st.plotly_chart(fig_br_base, use_container_width=True)
                
                st.markdown("**Baseline Sankey (Ref)**")
                st.plotly_chart(plot_sankey_diagram(results_list_base, route_id=ID_REF, title_prefix="Baseline: "), use_container_width=True)
                st.markdown("**Baseline Sankey (MOF)**")
                st.plotly_chart(plot_sankey_diagram(results_list_base, route_id=ID_MOF, title_prefix="Baseline: "), use_container_width=True)

            with col2:
                st.subheader("Modified Breakdown")
                fig_br_mod = px.bar(df_all_mod, x="Bead", y="GWP", color="Component", title="Modified: Total Breakdown (log)", barmode="group", log_y=True)
                st.plotly_chart(fig_br_mod, use_container_width=True)

                st.markdown("**Modified Sankey (Ref)**")
                st.plotly_chart(plot_sankey_diagram(results_list_mod, route_id=ID_REF, title_prefix="Modified: "), use_container_width=True)
                st.markdown("**Modified Sankey (MOF)**")
                st.plotly_chart(plot_sankey_diagram(results_list_mod, route_id=ID_MOF, title_prefix="Modified: "), use_container_width=True)

    # --- TAB 4: LITERATURE ---
    with tab4:
        st.header("Literature comparison")
        # We generally compare the "Modified" result to literature to see if improvement makes it competitive
        current_data = []
        for r in results_list_mod:
            current_data.append({
                "Material": f"{r['name']} (Modified Scen.)",
                "GWP_kgCO2_per_kg": r["Total GWP"],
                "Source": "This work (Modified)",
                "Type": "This work",
            })
        # Optionally add baseline to literature chart too?
        for r in results_list_base:
            current_data.append({
                "Material": f"{r['name']} (Baseline)",
                "GWP_kgCO2_per_kg": r["Total GWP"],
                "Source": "This work (Baseline)",
                "Type": "This work",
            })

        lit_combined = pd.concat([lit_df, pd.DataFrame(current_data)])
        fig_lit = px.bar(
            lit_combined,
            x="Material",
            y="GWP_kgCO2_per_kg",
            color="Source",
            log_y=True,
            title="GWP comparison: Baseline vs Modified vs Literature",
            text="Source",
        )
        fig_lit.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_lit, use_container_width=True)

    # --- TAB 5: AI INSIGHTS ---
    with tab5:
        st.header("AI insights (Modified Scenario)")
        st.caption("The AI analyzes the *Modified* scenario results.")
        
        # Focus on modified results for AI
        route_name_map = {r["id"]: r["name"] for r in results_list_mod}
        focus_options = ["All routes"]
        for rid in unique_routes:
            if rid in route_name_map:
                focus_options.append(route_name_map[rid])

        focus_choice = st.radio("Route focus", options=focus_options, index=0)
        
        if focus_choice == "All routes":
            ai_context_results = results_list_mod
        else:
            chosen_id = next((rid for rid, name in route_name_map.items() if name == focus_choice), None)
            ai_context_results = [r for r in results_list_mod if r["id"] == chosen_id] if chosen_id else results_list_mod

        st.write("Sample questions:")
        col_q1, col_q2 = st.columns(2)
        sample_questions = [
            "How does the modified scaling affect the GWP?",
            "Compare Ref-Bead and U@Bead results in this scenario.",
            "What is the biggest hotspot now?",
            "How can I reduce the carbon footprint further?",
        ]
        
        if "ai_custom_q" not in st.session_state: st.session_state["ai_custom_q"] = ""
        for i, q in enumerate(sample_questions):
            with (col_q1 if i % 2 == 0 else col_q2):
                if st.button(q, key=f"sample_q_{i}"): st.session_state["ai_custom_q"] = q
        
        user_q = st.text_area("Type your question:", key="ai_custom_q", height=140)
        if st.button("Analyse results"):
            if user_q.strip():
                with st.spinner("AI is analysing..."):
                    answer = get_ai_insight(ai_context_results, user_q)
                st.markdown("### AI analysis")
                st.info(answer)
            else:
                st.warning("Please enter a question.")

if __name__ == "__main__":
    main()
