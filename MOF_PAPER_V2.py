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
LOGO_DIR = BASE_DIR / "assets"

# Route IDs
ID_REF = "ref"
ID_MOF = "mof"

st.set_page_config(page_title="LCA Explorer: Interactive & AI-Enhanced", layout="wide")

# -----------------------------------------------------------------------------
# SCALING DEFAULTS (Derived from paper logic to match exact inventory)
# -----------------------------------------------------------------------------
# Ref-Bead Steps (Batch ~ 0.4g or 0.0004 kg)
# MOF-Bead Steps (Batch ~ 0.6g or 0.0006 kg)

DEFAULT_SCALING_PARAMS = {
    "ref_steps": [
        {"name": "Microfluidiser", "power_kw": 1.5, "time_h": 0.33, "batch_kg": 0.0004, "max_cap_kg": 4.0},
        {"name": "Hotplate mixing (50°C)", "power_kw": 0.625, "time_h": 6.0, "batch_kg": 0.0004, "max_cap_kg": 1.0},
        {"name": "Hotplate crosslinking (50°C)", "power_kw": 0.625, "time_h": 6.0, "batch_kg": 0.0004, "max_cap_kg": 1.0},
        {"name": "Freeze-drying (Step 1)", "power_kw": 1.84, "time_h": 16.0, "batch_kg": 0.0004, "max_cap_kg": 2.0},
    ],
    "mof_steps": [
        {"name": "UiO stirring (Zr step)", "power_kw": 0.625, "time_h": 12.0, "batch_kg": 0.0006, "max_cap_kg": 1.0},
        {"name": "UiO stirring (Linker step)", "power_kw": 0.625, "time_h": 12.0, "batch_kg": 0.0006, "max_cap_kg": 1.0},
        {"name": "Freeze-drying (Step 2)", "power_kw": 1.84, "time_h": 16.0, "batch_kg": 0.0006, "max_cap_kg": 2.0},
    ]
}

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
    """Resets session state to default CSV values and parameters."""
    ef, routes, perf, lit = load_default_data()
    st.session_state["ef_df"] = ef
    st.session_state["routes_df"] = routes
    st.session_state["perf_df"] = perf
    st.session_state["lit_df"] = lit

    # Reset Custom Grids
    st.session_state["custom_grids"] = {
        "QC Hydro": 0.002,
        "Canada Avg": 0.1197,
        "UK Grid": 0.225,
        "EU Avg": 0.25,
        "US Avg": 0.38,
        "China Grid": 0.58,
    }
    
    # Reset Scaling Params (Deep copy to ensure independence)
    import copy
    st.session_state["scaling_params"] = copy.deepcopy(DEFAULT_SCALING_PARAMS)

# Initialise Session State
if "ef_df" not in st.session_state:
    reset_data()

# Shortcuts
EF_DF = st.session_state["ef_df"]
ROUTES_DF = st.session_state["routes_df"]
PERF_DF = st.session_state["perf_df"]
LIT_DF = st.session_state["lit_df"]

# -----------------------------------------------------------------------------
# HEADER LOGOS
# -----------------------------------------------------------------------------
def render_header_logos():
    ubc_path = LOGO_DIR / "ubc.png"
    # ame_path = LOGO_DIR / "ame.png" 
    cov_path = LOGO_DIR / "cov.png"

    # Layout: COV on Left, Gap, UBC on Right
    col_left, col_mid, col_right = st.columns([1, 4, 1])

    with col_left:
        if cov_path.exists():
            st.image(str(cov_path), width=150)
        else:
            st.write("cov.png not found")

    with col_right:
        if ubc_path.exists():
            st.image(str(ubc_path), width=150)
        else:
             st.write("ubc.png not found")

# -----------------------------------------------------------------------------
# CALCULATION ENGINE
# -----------------------------------------------------------------------------
def calculate_electricity_demand(route_id, scaling_params):
    """
    Calculates kWh/kg based on the dynamic process steps defined in Tab 1.
    """
    
    # Helper to sum energy for a list of steps
    def sum_energy(steps_list):
        total_kwh_per_kg = 0.0
        for step in steps_list:
            # Energy (kWh) = Power (kW) * Time (h)
            # Intensity (kWh/kg) = Energy / Batch Mass (kg)
            batch_mass = step["batch_kg"] if step["batch_kg"] > 0 else 0.0001
            kwh_batch = step["power_kw"] * step["time_h"]
            total_kwh_per_kg += kwh_batch / batch_mass
        return total_kwh_per_kg

    ref_energy = sum_energy(scaling_params["ref_steps"])
    
    if route_id == ID_REF:
        return ref_energy
    elif route_id == ID_MOF:
        # MOF bead calculation from paper:
        # 1. Start with Ref bead structure (scaled by mass fraction ~0.87)
        # 2. Add MOF specific steps
        mof_specific_energy = sum_energy(scaling_params["mof_steps"])
        # Support mass fraction assumption from inventory (approx 13% MOF loading -> 87% Ref core)
        # Using 0.87 scaling factor for the support energy contribution
        total_mof_energy = (ref_energy * 0.87) + mof_specific_energy
        return total_mof_energy
    
    return 0.0

def calculate_impacts(
    route_id,
    ef_df,
    routes_df,
    scaling_params,  # NEW ARGUMENT
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

    # --- ELECTRICITY CALCULATION (DYNAMIC) ---
    # Instead of reading 'electricity_kwh_per_fu' from CSV, we calculate it
    raw_elec_kwh_per_kg = calculate_electricity_demand(route_id, scaling_params)
    
    # Apply efficiency factor and yield multiplier
    elec_kwh = raw_elec_kwh_per_kg * efficiency_factor * yield_multiplier

    elec_source = route_data.iloc[0]["electricity_source"]
    ef_elec_row = ef_df[ef_df["reagent_name"] == elec_source]
    ef_elec = float(ef_elec_row["GWP_kgCO2_per_kg"].iloc[0]) if not ef_elec_row.empty else 0.0

    gwp_elec = elec_kwh * ef_elec

    # --- REAGENTS ---
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
        
        # Yield Impact
        mass_needed = base_mass * yield_multiplier

        # Recycling Impact
        is_solvent = reagent in ["Ethanol", "Formic acid (88%)", "Acetic acid"]
        effective_mass = mass_needed * (1 - (recycling_rate / 100.0)) if is_solvent else mass_needed

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
def plot_sankey_diagram(results_list, route_id=None):
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
        title_text=f"Impact flow: {target_res['name']}",
        font_size=10,
        height=400,
    )
    return fig

def get_electricity_step_data(scaling_params) -> pd.DataFrame:
    """
    Generates bar chart data based on the CURRENT scaling parameters.
    """
    data = []
    
    # Ref Steps
    for step in scaling_params["ref_steps"]:
        # kWh/kg = (P * t) / m
        val = (step["power_kw"] * step["time_h"]) / step["batch_kg"]
        data.append({
            "Bead": "Ref-Bead (Polymer)", 
            "Step": step["name"], 
            "kWh_per_kg": val
        })

    # MOF Steps
    # Note: MOF bead includes Ref steps (scaled) + MOF steps
    # For this chart, we usually just show the unique steps contribution
    
    # 1. Add Ref Steps scaled down
    for step in scaling_params["ref_steps"]:
        val = ((step["power_kw"] * step["time_h"]) / step["batch_kg"]) * 0.87
        data.append({
            "Bead": "U@Bead (MOF-Functionalised)", 
            "Step": f"{step['name']} (Core)", 
            "kWh_per_kg": val
        })
        
    # 2. Add MOF specific steps
    for step in scaling_params["mof_steps"]:
        val = (step["power_kw"] * step["time_h"]) / step["batch_kg"]
        data.append({
            "Bead": "U@Bead (MOF-Functionalised)", 
            "Step": step["name"], 
            "kWh_per_kg": val
        })

    return pd.DataFrame(data)

def create_system_boundary_figure() -> go.Figure:
    """Simple system boundary diagram."""
    fig = go.Figure()
    
    # Main system boundary box
    fig.add_shape(
        type="rect", x0=0.25, y0=0.2, x1=0.75, y1=0.8,
        line=dict(color="black", width=2),
        fillcolor="rgba(144, 238, 144, 0.1)",
    )
    
    # Upstream box (excluded)
    fig.add_shape(
        type="rect", x0=0.02, y0=0.35, x1=0.20, y1=0.65,
        line=dict(color="grey", width=1),
        fillcolor="rgba(200, 200, 200, 0.1)",
    )
    fig.add_annotation(
        x=0.11, y=0.5, text="Upstream:\nFisheries,\ncrab processing,\nchitin purification\n(excluded)",
        showarrow=False, font=dict(size=10),
    )

    # Downstream box (excluded)
    fig.add_shape(
        type="rect", x0=0.80, y0=0.35, x1=0.98, y1=0.65,
        line=dict(color="grey", width=1),
        fillcolor="rgba(200, 200, 200, 0.1)",
    )
    fig.add_annotation(
        x=0.89, y=0.5, text="Use phase,\nregeneration,\nend of life\n(excluded)",
        showarrow=False, font=dict(size=10),
    )

    # Main system annotation
    fig.add_annotation(
        x=0.5, y=0.6,
        text=("System boundary (included):\nLab scale bead synthesis\nPDChNF–chitosan bead (Ref-Bead)\n+ UiO-66-NH₂ growth and NaOH step\n(U@Bead-2step-aUiO)"),
        showarrow=False, font=dict(size=11),
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
        **Dashboard:** Adjust scenarios on the left. Use the **Process Scaling** tab to adjust equipment capacity, and explore results in subsequent tabs.
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
            .loc[st.session_state["ef_df"]["reagent_name"] == "Electricity (Canada)", "GWP_kgCO2_per_kg"]
            .iloc[0]
        )
        new_grid_ef = st.slider(
            "Grid carbon intensity (kg CO₂ per kWh)",
            min_value=0.0, max_value=1.0, value=current_grid_ef, step=0.01,
            help="Controls the cleanliness of the power source used in the LCA.",
        )
        if new_grid_ef != current_grid_ef:
            st.session_state["ef_df"].loc[
                st.session_state["ef_df"]["reagent_name"] == "Electricity (Canada)", "GWP_kgCO2_per_kg"
            ] = new_grid_ef

        # Efficiency
        st.write("**Process efficiency**")
        eff_factor = st.slider(
            "Efficiency multiplier",
            min_value=0.1, max_value=1.0, value=1.0,
            help="1.0 = lab scale baseline. Lower values approximate improved industrial efficiency.",
        )
        
        # Recycling
        st.write("**Solvent recovery**")
        recycle_rate = st.slider(
            "Recycling rate (%)",
            min_value=0, max_value=95, value=0,
            help="Percentage of ethanol and formic acid recovered and reused.",
        )
        
        # Yield
        st.write("**Global yield**")
        yield_rate = st.slider(
            "Synthesis yield (%)",
            min_value=10, max_value=100, value=100,
            help="Lower yield increases the required inputs per kg of bead.",
        )
        
        # Transport
        st.write("**Transport overhead**")
        transport_overhead = st.slider(
            "Add transport (%)",
            min_value=0, max_value=50, value=0,
            help="Adds a fixed percentage to account for logistics emissions.",
        )
        
        st.divider()

        # --- TABLES ---
        st.subheader("2. Input tables")
        with st.expander("Edit detailed inputs"):
            st.caption("Emission factors")
            st.session_state["ef_df"] = st.data_editor(st.session_state["ef_df"], key="ed_ef", num_rows="dynamic")
            
            st.caption("Bead recipes (Material inputs only)")
            st.session_state["routes_df"] = st.data_editor(st.session_state["routes_df"], key="ed_routes", num_rows="dynamic")
            
            st.caption("Performance data")
            st.session_state["perf_df"] = st.data_editor(st.session_state["perf_df"], key="ed_perf", num_rows="dynamic")

        # Refresh shortcuts after any edits
        ef_df = st.session_state["ef_df"]
        routes_df = st.session_state["routes_df"]
        perf_df = st.session_state["perf_df"]
        lit_df = st.session_state["lit_df"]
        scaling_params = st.session_state["scaling_params"]

    # -------------------------------------------------------------------------
    # CALCULATIONS
    # -------------------------------------------------------------------------
    unique_routes = routes_df["route_id"].unique()
    results_list = []
    dfs_list = []

    for rid in unique_routes:
        res, df = calculate_impacts(
            rid,
            ef_df,
            routes_df,
            scaling_params=scaling_params,
            efficiency_factor=eff_factor,
            recycling_rate=recycle_rate,
            yield_rate=yield_rate,
            transport_pct=transport_overhead,
        )
        if res:
            # add Electricity % once so that AI tab can use it
            if res["Total GWP"] > 0:
                res["Electricity %"] = (res["Electricity GWP"] / res["Total GWP"]) * 100.0
            else:
                res["Electricity %"] = 0.0
            results_list.append(res)
            dfs_list.append(df)

    if not results_list:
        st.warning("No routes found. Check the input tables in the sidebar.")
        return

    perf_map = {row["route_id"]: float(row["capacity_mg_g"]) for _, row in perf_df.iterrows()}

    # Prepare summary data
    summary_rows = []
    for r in results_list:
        cap = perf_map.get(r["id"], 0.001)  # mg/g
        summary_rows.append({
            "Bead": r["name"],
            "Total GWP": r["Total GWP"],
            "Non-Electric GWP": r["Non-Electric GWP"],
            "GWP per g Cu": r["Total GWP"] / cap,
            "Electricity %": (r["Electricity GWP"] / r["Total GWP"]) * 100.0 if r["Total GWP"] > 0 else 0.0,
        })
    sum_df = pd.DataFrame(summary_rows)

    # -------------------------------------------------------------------------
    # TABS
    # -------------------------------------------------------------------------
    tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Process Scaling",
        "LCA Results",
        "Sensitivity",
        "Inventory",
        "Literature",
        "AI Insights"
    ])

    # --- TAB 0: PROCESS SCALING (NEW) ---
    with tab0:
        st.header("Process Scaling & Electricity Configuration")
        st.markdown(
            """
            Here you can modify the operational parameters for each synthesis step. 
            **Laboratory batches (approx 0.4 - 0.6 g)** make inefficient use of equipment (high power, long time, tiny output), 
            leading to high electricity intensity per kg.
            
            Adjust the **Batch Mass** or **Power** below to simulate scaling up (or down).
            """
        )
        
        st.subheader("1. Ref-Bead Process Steps (Polymer Core)")
        
        # Iterate over Ref Steps
        for i, step in enumerate(st.session_state["scaling_params"]["ref_steps"]):
            st.markdown(f"**{step['name']}**")
            c1, c2, c3, c4 = st.columns(4)
            
            with c1:
                new_p = st.number_input(
                    f"Power (kW)", 
                    value=float(step['power_kw']), 
                    step=0.1, 
                    key=f"ref_p_{i}"
                )
                st.session_state["scaling_params"]["ref_steps"][i]["power_kw"] = new_p

            with c2:
                new_t = st.number_input(
                    f"Time (h)", 
                    value=float(step['time_h']), 
                    step=0.5, 
                    key=f"ref_t_{i}"
                )
                st.session_state["scaling_params"]["ref_steps"][i]["time_h"] = new_t

            with c3:
                new_m = st.number_input(
                    f"Batch Mass (kg)", 
                    value=float(step['batch_kg']), 
                    format="%.5f",
                    step=0.0001, 
                    key=f"ref_m_{i}"
                )
                st.session_state["scaling_params"]["ref_steps"][i]["batch_kg"] = new_m

            with c4:
                new_cap = st.number_input(
                    f"Max Capacity (kg)", 
                    value=float(step['max_cap_kg']), 
                    step=1.0, 
                    key=f"ref_cap_{i}"
                )
                st.session_state["scaling_params"]["ref_steps"][i]["max_cap_kg"] = new_cap
            
            st.divider()

        st.subheader("2. MOF-Functionalisation Steps (Additional)")
        
        # Iterate over MOF Steps
        for i, step in enumerate(st.session_state["scaling_params"]["mof_steps"]):
            st.markdown(f"**{step['name']}**")
            c1, c2, c3, c4 = st.columns(4)
            
            with c1:
                new_p = st.number_input(
                    f"Power (kW)", 
                    value=float(step['power_kw']), 
                    step=0.1, 
                    key=f"mof_p_{i}"
                )
                st.session_state["scaling_params"]["mof_steps"][i]["power_kw"] = new_p

            with c2:
                new_t = st.number_input(
                    f"Time (h)", 
                    value=float(step['time_h']), 
                    step=0.5, 
                    key=f"mof_t_{i}"
                )
                st.session_state["scaling_params"]["mof_steps"][i]["time_h"] = new_t

            with c3:
                new_m = st.number_input(
                    f"Batch Mass (kg)", 
                    value=float(step['batch_kg']), 
                    format="%.5f",
                    step=0.0001, 
                    key=f"mof_m_{i}"
                )
                st.session_state["scaling_params"]["mof_steps"][i]["batch_kg"] = new_m

            with c4:
                new_cap = st.number_input(
                    f"Max Capacity (kg)", 
                    value=float(step['max_cap_kg']), 
                    step=1.0, 
                    key=f"mof_cap_{i}"
                )
                st.session_state["scaling_params"]["mof_steps"][i]["max_cap_kg"] = new_cap
            
            st.divider()


    # --- TAB 1: RESULTS ---
    with tab1:
        st.header("LCA Results")
        st.dataframe(
            sum_df.style.format({
                "Total GWP": "{:.2e}",
                "Non-Electric GWP": "{:.2f}",
                "GWP per g Cu": "{:.2f}",
                "Electricity %": "{:.1f}%",
            })
        )
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Total GWP per kg bead (log scale, FU1)")
            fig_log = px.bar(
                sum_df, x="Bead", y="Total GWP", color="Bead",
                log_y=True, title="Total GWP per kg bead (FU1)", text_auto=".2s",
            )
            st.plotly_chart(fig_log, use_container_width=True)

        with col2:
            st.subheader("Electricity versus chemicals")
            stack_data = []
            for r in results_list:
                stack_data.append({"Bead": r["name"], "Source": "Electricity", "GWP": r["Electricity GWP"]})
                stack_data.append({"Bead": r["name"], "Source": "Chemicals and transport", "GWP": r["Non-Electric GWP"]})
            
            fig_stack = px.bar(
                pd.DataFrame(stack_data), x="Bead", y="GWP", color="Source",
                title="Electricity versus non electricity contributions", text_auto=".2s",
            )
            st.plotly_chart(fig_stack, use_container_width=True)

        st.subheader("Performance normalised impacts (FU2)")
        fig_fu2 = px.bar(
            sum_df, x="Bead", y="GWP per g Cu", color="Bead",
            title="GWP per g Cu removed (FU2)", text_auto=".2f",
        )
        st.plotly_chart(fig_fu2, use_container_width=True)

        st.subheader("System boundary (schematic)")
        st.plotly_chart(create_system_boundary_figure(), use_container_width=True)

    # --- TAB 2: SENSITIVITY ---
    with tab2:
        st.header("Sensitivity and Scaling Analysis")

        # 1. Grid intensity sensitivity
        st.subheader("1. Grid intensity sensitivity")
        col_s1, col_s2 = st.columns([1, 2])
        with col_s1:
            st.write("Add custom grid point:")
            new_grid_name = st.text_input("Grid name", "My local grid")
            new_grid_val = st.number_input("Intensity (kg CO₂/kWh)", min_value=0.0, max_value=1.0, value=0.45)
            if st.button("Add grid to chart"):
                st.session_state["custom_grids"][new_grid_name] = new_grid_val
                st.success(f"Added grid: {new_grid_name}")

        sens_rows = []
        for g_name, g_val in st.session_state["custom_grids"].items():
            temp_ef = ef_df.copy()
            temp_ef.loc[temp_ef["reagent_name"].str.contains("Electricity"), "GWP_kgCO2_per_kg"] = g_val
            
            for rid in unique_routes:
                res, _ = calculate_impacts(
                    rid, temp_ef, routes_df, scaling_params,
                    efficiency_factor=eff_factor,
                    recycling_rate=recycle_rate,
                    yield_rate=yield_rate,
                    transport_pct=transport_overhead,
                )
                sens_rows.append({"Grid": g_name, "Grid Value": g_val, "Bead": res["name"], "Total GWP": res["Total GWP"]})
        
        df_sens = pd.DataFrame(sens_rows).sort_values("Grid Value")
        fig_sens = px.line(
            df_sens, x="Grid", y="Total GWP", color="Bead", markers=True,
            title="Total GWP versus grid carbon intensity", hover_data=["Grid Value"],
        )
        st.plotly_chart(fig_sens, use_container_width=True)

        st.divider()

        # 2. Batch scaling effect (Conceptual log-log)
        st.subheader("2. Batch scaling effect (Projected)")
        st.caption("This projection assumes electricity per kg scales inversely with batch size, starting from your current process settings.")
        
        LAB_BATCH_REF_KG = scaling_params["ref_steps"][0]["batch_kg"] 
        
        batch_sizes = np.logspace(np.log10(LAB_BATCH_REF_KG), np.log10(100.0), num=40)
        scale_rows = []
        
        for rid in unique_routes:
            base_res, _ = calculate_impacts(rid, ef_df, routes_df, scaling_params)
            if base_res is None: continue
            
            base_elec_intensity = base_res["Electricity kWh"]
            # Simplified scaler using the Ref Batch size as anchor for visual trend
            lab_batch = LAB_BATCH_REF_KG
            
            for b_size in batch_sizes:
                scale_factor = lab_batch / b_size
                new_elec_intensity = base_elec_intensity * scale_factor
                new_gwp = new_elec_intensity * base_res["Electricity EF Used"] + base_res["Non-Electric GWP"]
                scale_rows.append({"Batch size (kg)": b_size, "Bead": base_res["name"], "Estimated GWP": new_gwp})
                
        df_scale = pd.DataFrame(scale_rows)
        fig_scale = px.line(
            df_scale, x="Batch size (kg)", y="Estimated GWP", color="Bead",
            log_x=True, log_y=True, markers=True,
            title="Projected GWP versus batch size (log log)",
        )
        st.plotly_chart(fig_scale, use_container_width=True)

        st.divider()

        # 3. Electricity demand per process step
        st.subheader("3. Electricity demand per process step (Current Settings)")
        elec_step_df = get_electricity_step_data(scaling_params)
        
        fig_steps = px.bar(
            elec_step_df, x="Step", y="kWh_per_kg", color="Bead",
            barmode="group", log_y=True,
            title="Electricity demand by process step (kWh per kg bead)", text_auto=".2s",
        )
        fig_steps.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig_steps, use_container_width=True)

    # --- TAB 3: INVENTORY ---
    with tab3:
        st.header("Inventory and Impact Breakdown")
        
        all_contribs = []
        for i, df in enumerate(dfs_list):
            df = df.copy()
            df["Bead"] = results_list[i]["name"]
            all_contribs.append(df)
        
        df_all = pd.concat(all_contribs) if all_contribs else pd.DataFrame()
        
        if not df_all.empty:
            st.subheader("A. Chemical impacts (excluding electricity)")
            df_ne = df_all[df_all["Category"] != "Electricity"]
            fig_ne = px.bar(
                df_ne, x="Bead", y="GWP", color="Component",
                title="Chemical GWP (no electricity)", barmode="group",
            )
            st.plotly_chart(fig_ne, use_container_width=True)
            
            st.divider()

            st.subheader("B. Total breakdown (log scale)")
            fig_breakdown = px.bar(
                df_all, x="Bead", y="GWP", color="Component",
                title="Total GWP breakdown (log scale)", barmode="group", log_y=True,
            )
            st.plotly_chart(fig_breakdown, use_container_width=True)

            st.divider()

            st.subheader("C. Mass inventory per kg bead")
            fig_mass = px.bar(
                df_ne, x="Component", y="Mass (kg)", color="Component",
                facet_col="Bead", title="Mass input per kg product",
            )
            fig_mass.update_yaxes(matches=None, showticklabels=True)
            st.plotly_chart(fig_mass, use_container_width=True)
            
            st.subheader("D. Impact flow (Sankey diagrams)")
            st.markdown("**Ref-Bead (polymer only)**")
            st.plotly_chart(plot_sankey_diagram(results_list, route_id=ID_REF), use_container_width=True)
            st.markdown("**U@Bead (MOF functionalised)**")
            st.plotly_chart(plot_sankey_diagram(results_list, route_id=ID_MOF), use_container_width=True)

    # --- TAB 4: LITERATURE ---
    with tab4:
        st.header("Literature Comparison")
        current_data = []
        for r in results_list:
            current_data.append({
                "Material": f"{r['name']} (this work)",
                "GWP_kgCO2_per_kg": r["Total GWP"],
                "Source": "This work",
                "Type": "This work",
            })
        
        lit_combined = pd.concat([lit_df, pd.DataFrame(current_data)])
        
        fig_lit = px.bar(
            lit_combined, x="Material", y="GWP_kgCO2_per_kg", color="Source",
            log_y=True, title="GWP comparison with literature (log scale)", text="Source",
        )
        fig_lit.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_lit, use_container_width=True)

    # --- TAB 5: AI INSIGHTS ---
    with tab5:
        st.header("AI Insights")
        st.caption("Ask questions about the current results. You can focus on one bead or consider both together.")
        
        # Optional route focus
        route_name_map = {r["id"]: r["name"] for r in results_list}
        focus_options = ["All routes"]
        for rid in unique_routes:
            if rid in route_name_map:
                focus_options.append(route_name_map[rid])
        
        focus_choice = st.radio("Route focus (optional)", options=focus_options, index=0)
        
        if focus_choice == "All routes":
            ai_context_results = results_list
        else:
            chosen_id = None
            for rid, name in route_name_map.items():
                if name == focus_choice:
                    chosen_id = rid
                    break
            if chosen_id is None:
                ai_context_results = results_list
            else:
                ai_context_results = [r for r in results_list if r["id"] == chosen_id]
        
        st.write("Sample questions (click to populate the box):")
        col_q1, col_q2 = st.columns(2)
        sample_questions = [
            "Why is the GWP so high compared to literature?",
            "Compare Ref-Bead and U@Bead results.",
            "What is the biggest hotspot in this scenario?",
            "How can I reduce the carbon footprint of bead production?",
        ]
        
        if "ai_custom_q" not in st.session_state:
            st.session_state["ai_custom_q"] = ""
            
        for i, q in enumerate(sample_questions):
            col = col_q1 if i % 2 == 0 else col_q2
            with col:
                if st.button(q, key=f"sample_q_{i}"):
                    st.session_state["ai_custom_q"] = q
        
        user_q = st.text_area("Type your question about the LCA results:", key="ai_custom_q", height=140)
        
        if st.button("Analyse results"):
            if user_q.strip():
                with st.spinner("AI is analysing your data..."):
                    answer = get_ai_insight(ai_context_results, user_q)
                st.markdown("### AI Analysis")
                st.info(answer)
            else:
                st.warning("Please enter a question or click one of the sample prompts.")

if __name__ == "__main__":
    main()
