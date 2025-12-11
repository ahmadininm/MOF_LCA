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
# DEFAULT ENGINEERING PARAMETERS (LAB SCALE)
# -----------------------------------------------------------------------------
# Values derived from Table 3 of the paper (Lab Scale inputs)
DEFAULT_PARAMS = {
    # Ref-Bead Steps
    "ref_mf_power": 1.5,       # kW (Microfluidiser)
    "ref_mf_time": 0.33,       # h
    "ref_mf_batch": 0.0004,    # kg (0.4g) -> Throughput bottleneck
    
    "ref_mix_power": 0.625,    # kW (Hotplate mixing)
    "ref_mix_time": 6.0,       # h
    "ref_mix_batch": 0.0004,   # kg
    
    "ref_xl_power": 0.625,     # kW (Hotplate crosslinking)
    "ref_xl_time": 6.0,        # h
    "ref_xl_batch": 0.0004,    # kg
    
    "ref_fd_power": 1.84,      # kW (Freeze Dryer)
    "ref_fd_time": 16.0,       # h
    "ref_fd_batch": 0.0004,    # kg
    
    # MOF-Bead Steps
    "mof_zr_power": 0.625,     # kW (Stirrer Zr)
    "mof_zr_time": 12.0,       # h
    "mof_zr_batch": 0.0006,    # kg (0.6g)
    
    "mof_lnk_power": 0.625,    # kW (Stirrer Linker)
    "mof_lnk_time": 12.0,      # h
    "mof_lnk_batch": 0.0006,   # kg
    
    "mof_fd_power": 1.84,      # kW (Freeze Dryer 2)
    "mof_fd_time": 16.0,       # h
    "mof_fd_batch": 0.0006,    # kg
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
    - Comment on why certain impacts are high (e.g., electricity at lab scale).
    - If asked about optimization, suggest practical process improvements.
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
    """Resets session state to default CSV values."""
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
    # Reset Engineering Params
    st.session_state["eng_params"] = DEFAULT_PARAMS.copy()

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
    ame_path = LOGO_DIR / "ame.png"
    cov_path = LOGO_DIR / "cov.png"
    
    col_left, col_mid, col_right = st.columns([1, 1, 1])
    
    with col_left:
        # User requested cov.png on top left
        if cov_path.exists():
            st.image(str(cov_path), width=150)
            
    with col_mid:
        if ame_path.exists():
            st.image(str(ame_path), width=150)
            
    with col_right:
        # User requested ubc.png on top right
        if ubc_path.exists():
            st.image(str(ubc_path), width=150)

# -----------------------------------------------------------------------------
# CALCULATION ENGINE
# -----------------------------------------------------------------------------
def calculate_electricity_demand(params):
    """
    Calculates the electricity intensity (kWh/kg product) based on machine params.
    Returns dictionary with {ID_REF: val, ID_MOF: val}
    """
    # 1. Ref-Bead Calculation
    # Step 1: Microfluidiser
    e_mf = (params["ref_mf_power"] * params["ref_mf_time"]) / params["ref_mf_batch"]
    # Step 2: Mixing
    e_mix = (params["ref_mix_power"] * params["ref_mix_time"]) / params["ref_mix_batch"]
    # Step 3: Crosslinking
    e_xl = (params["ref_xl_power"] * params["ref_xl_time"]) / params["ref_xl_batch"]
    # Step 4: Freeze Drying
    e_fd = (params["ref_fd_power"] * params["ref_fd_time"]) / params["ref_fd_batch"]
    
    ref_total_kwh_kg = e_mf + e_mix + e_xl + e_fd
    
    # 2. MOF-Bead Calculation
    # MOF beads use Ref beads as a support. 
    # Mass balance: ~0.87 kg Ref bead per 1 kg MOF bead (13 wt% loading).
    ref_support_ratio = 0.87
    e_support = ref_total_kwh_kg * ref_support_ratio
    
    # Step 1: Zr Growth
    e_zr = (params["mof_zr_power"] * params["mof_zr_time"]) / params["mof_zr_batch"]
    # Step 2: Linker Growth
    e_lnk = (params["mof_lnk_power"] * params["mof_lnk_time"]) / params["mof_lnk_batch"]
    # Step 3: Final Freeze Drying
    e_fd2 = (params["mof_fd_power"] * params["mof_fd_time"]) / params["mof_fd_batch"]
    
    mof_total_kwh_kg = e_support + e_zr + e_lnk + e_fd2
    
    return {ID_REF: ref_total_kwh_kg, ID_MOF: mof_total_kwh_kg}

def calculate_impacts(
    route_id,
    ef_df,
    routes_df,
    eng_params,  # Passed from session state
    recycling_rate: float = 0.0,
    yield_rate: float = 100.0,
    transport_pct: float = 0.0,
):
    """Calculates GWP based on session state and engineering modifiers."""
    route_data = routes_df[routes_df["route_id"] == route_id].copy()
    if route_data.empty:
        return None, None
    
    yield_multiplier = 1.0 / (yield_rate / 100.0)
    
    # --- ELECTRICITY CALCULATION (Dynamic) ---
    elec_intensities = calculate_electricity_demand(eng_params)
    if route_id == ID_REF:
        base_elec_kwh = elec_intensities[ID_REF]
    elif route_id == ID_MOF:
        base_elec_kwh = elec_intensities[ID_MOF]
    else:
        # Fallback to CSV if unknown route
        base_elec_kwh = float(route_data.iloc[0]["electricity_kwh_per_fu"])
    
    # Apply yield to electricity? Usually batch size is fixed, but if yield is low, 
    # you produce less product for same energy. So yes, multiply by yield factor.
    elec_kwh = base_elec_kwh * yield_multiplier
    
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
        
    # --- TRANSPORT ---
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
    if not results_list:
        return go.Figure()
        
    target_res = next((r for r in results_list if r["id"] == route_id), results_list[0])
    
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
    fig.update_layout(title_text=f"Impact flow: {target_res['name']}", font_size=10, height=400)
    return fig

def create_system_boundary_figure() -> go.Figure:
    fig = go.Figure()
    # Main system boundary
    fig.add_shape(type="rect", x0=0.25, y0=0.2, x1=0.75, y1=0.8, line=dict(color="black", width=2), fillcolor="rgba(144, 238, 144, 0.1)")
    # Upstream
    fig.add_shape(type="rect", x0=0.02, y0=0.35, x1=0.20, y1=0.65, line=dict(color="grey", width=1), fillcolor="rgba(200, 200, 200, 0.1)")
    fig.add_annotation(x=0.11, y=0.5, text="Upstream:\nExcluded", showarrow=False, font=dict(size=10))
    # Downstream
    fig.add_shape(type="rect", x0=0.80, y0=0.35, x1=0.98, y1=0.65, line=dict(color="grey", width=1), fillcolor="rgba(200, 200, 200, 0.1)")
    fig.add_annotation(x=0.89, y=0.5, text="Downstream:\nExcluded", showarrow=False, font=dict(size=10))
    
    fig.add_annotation(x=0.5, y=0.6, text="System Boundary:\nLab scale synthesis", showarrow=False, font=dict(size=11))
    
    # Arrows
    fig.add_annotation(x=0.22, y=0.5, ax=0.25, ay=0.5, showarrow=True, arrowhead=2)
    fig.add_annotation(x=0.75, y=0.5, ax=0.78, ay=0.5, showarrow=True, arrowhead=2)
    
    fig.update_xaxes(visible=False, range=[0, 1])
    fig.update_yaxes(visible=False, range=[0, 1])
    fig.update_layout(height=200, margin=dict(l=10, r=10, t=30, b=10))
    return fig

# -----------------------------------------------------------------------------
# MAIN APP
# -----------------------------------------------------------------------------
def main():
    render_header_logos()
    
    st.title("Interactive LCA Explorer: Ref-Bead vs U@Bead")
    st.markdown("**Dashboard:** Use the new 'Scaling & Engineering' tab to adjust process capacities.")
    
    # -------------------------------------------------------------------------
    # SIDEBAR: GENERAL SETTINGS
    # -------------------------------------------------------------------------
    with st.sidebar:
        st.header("General Settings")
        if st.button("Reset to paper defaults"):
            reset_data()
            st.success("Data reset to defaults.")
            st.rerun()
        
        st.divider()
        st.subheader("Grid & Efficiency")
        
        # Grid Intensity
        current_grid_ef = float(st.session_state["ef_df"].loc[st.session_state["ef_df"]["reagent_name"] == "Electricity (Canada)", "GWP_kgCO2_per_kg"].iloc[0])
        new_grid_ef = st.slider("Grid Carbon Intensity (kg CO₂/kWh)", 0.0, 1.0, current_grid_ef, 0.01)
        if new_grid_ef != current_grid_ef:
            st.session_state["ef_df"].loc[st.session_state["ef_df"]["reagent_name"] == "Electricity (Canada)", "GWP_kgCO2_per_kg"] = new_grid_ef
            
        # Recycling
        st.write("**Solvent Recovery**")
        recycle_rate = st.slider("Recycling rate (%)", 0, 95, 0)
        
        # Yield
        st.write("**Global Yield**")
        yield_rate = st.slider("Synthesis yield (%)", 10, 100, 100)
        
        # Transport
        st.write("**Transport Overhead**")
        transport_overhead = st.slider("Add transport (%)", 0, 50, 0)

    # -------------------------------------------------------------------------
    # TABS
    # -------------------------------------------------------------------------
    tab1, tab_scale, tab2, tab3, tab4, tab5 = st.tabs([
        "LCA Results",
        "Scaling & Engineering", 
        "Sensitivity",
        "Inventory",
        "Literature",
        "AI Insights"
    ])

    # -------------------------------------------------------------------------
    # TAB: SCALING & ENGINEERING (NEW)
    # -------------------------------------------------------------------------
    with tab_scale:
        st.header("Process Scaling & Engineering Inputs")
        st.markdown(
            """
            This section allows you to define the **actual capacity** and **usage** of the lab equipment. 
            The high electricity footprint in the paper is due to using large equipment (kW scale) for very small batches (grams).
            **Adjust the 'Batch Mass' sliders to simulate scaling up to full capacity.**
            """
        )
        
        # We use columns to organize the inputs
        col_eng1, col_eng2 = st.columns(2)
        
        params = st.session_state["eng_params"]
        
        with col_eng1:
            st.subheader("1. Ref-Bead Equipment")
            st.caption("Base polymer bead production steps")
            
            with st.expander("Microfluidiser (Step 1)", expanded=True):
                params["ref_mf_power"] = st.number_input("Power (kW)", value=params["ref_mf_power"], key="mf_p")
                params["ref_mf_time"] = st.number_input("Time per run (h)", value=params["ref_mf_time"], key="mf_t")
                params["ref_mf_batch"] = st.number_input(
                    "Batch Mass (kg)", 
                    value=params["ref_mf_batch"], 
                    format="%.5f", 
                    step=0.0001,
                    help="Mass of beads processed per run. Lab scale = 0.0004 kg (0.4g). Increase this to scale up.", 
                    key="mf_b"
                )

            with st.expander("Hotplate Mixing & Crosslinking (Step 2 & 3)"):
                params["ref_mix_power"] = st.number_input("Heater Power (kW)", value=params["ref_mix_power"], key="mx_p")
                params["ref_mix_time"] = st.number_input("Mixing Time (h)", value=params["ref_mix_time"], key="mx_t")
                params["ref_xl_time"] = st.number_input("Crosslinking Time (h)", value=params["ref_xl_time"], key="xl_t")
                params["ref_mix_batch"] = st.number_input("Batch Mass (kg)", value=params["ref_mix_batch"], format="%.5f", key="mx_b")
                # Assume XL batch is same as Mix batch for simplicity in UI, but keep param separate in backend if needed
                params["ref_xl_batch"] = params["ref_mix_batch"]

            with st.expander("Freeze Dryer (Ref) (Step 4)"):
                params["ref_fd_power"] = st.number_input("FD Power (kW)", value=params["ref_fd_power"], key="fd_p")
                params["ref_fd_time"] = st.number_input("Drying Time (h)", value=params["ref_fd_time"], key="fd_t")
                params["ref_fd_batch"] = st.number_input("Batch Mass (kg)", value=params["ref_fd_batch"], format="%.5f", key="fd_b")

        with col_eng2:
            st.subheader("2. U@Bead (MOF) Equipment")
            st.caption("MOF growth on Ref-Bead support")
            
            with st.expander("Heated Stirring (Zr & Linker Steps)"):
                params["mof_zr_power"] = st.number_input("Stirrer Power (kW)", value=params["mof_zr_power"], key="mz_p")
                params["mof_zr_time"] = st.number_input("Zr Step Time (h)", value=params["mof_zr_time"], key="mz_t")
                params["mof_lnk_time"] = st.number_input("Linker Step Time (h)", value=params["mof_lnk_time"], key="ml_t")
                params["mof_zr_batch"] = st.number_input(
                    "Batch Mass (kg)", 
                    value=params["mof_zr_batch"], 
                    format="%.5f", 
                    help="Lab scale = 0.0006 kg (0.6g).",
                    key="mz_b"
                )
                params["mof_lnk_batch"] = params["mof_zr_batch"]
                params["mof_lnk_power"] = params["mof_zr_power"]

            with st.expander("Freeze Dryer (MOF) (Final Step)"):
                params["mof_fd_power"] = st.number_input("FD Power (kW)", value=params["mof_fd_power"], key="mfd_p")
                params["mof_fd_time"] = st.number_input("Drying Time (h)", value=params["mof_fd_time"], key="mfd_t")
                params["mof_fd_batch"] = st.number_input("Batch Mass (kg)", value=params["mof_fd_batch"], format="%.5f", key="mfd_b")

        # Save back to session
        st.session_state["eng_params"] = params
        
        # Display current calculated intensity
        current_elec = calculate_electricity_demand(params)
        st.metric("Calculated Electricity (Ref-Bead)", f"{current_elec[ID_REF]:.1e} kWh/kg")
        st.metric("Calculated Electricity (MOF-Bead)", f"{current_elec[ID_MOF]:.1e} kWh/kg")

    # -------------------------------------------------------------------------
    # CALCULATIONS (Happens after scaling updates)
    # -------------------------------------------------------------------------
    ef_df = st.session_state["ef_df"]
    routes_df = st.session_state["routes_df"]
    perf_df = st.session_state["perf_df"]
    lit_df = st.session_state["lit_df"]
    eng_params = st.session_state["eng_params"]

    unique_routes = routes_df["route_id"].unique()
    results_list = []
    dfs_list = []

    for rid in unique_routes:
        res, df = calculate_impacts(
            rid,
            ef_df,
            routes_df,
            eng_params,
            recycling_rate=recycle_rate,
            yield_rate=yield_rate,
            transport_pct=transport_overhead,
        )
        if res:
            if res["Total GWP"] > 0:
                res["Electricity %"] = (res["Electricity GWP"] / res["Total GWP"]) * 100.0
            else:
                res["Electricity %"] = 0.0
            results_list.append(res)
            dfs_list.append(df)

    # Summary DF
    perf_map = {row["route_id"]: float(row["capacity_mg_g"]) for _, row in perf_df.iterrows()}
    summary_rows = []
    for r in results_list:
        cap = perf_map.get(r["id"], 0.001)
        summary_rows.append({
            "Bead": r["name"],
            "Total GWP": r["Total GWP"],
            "Non-Electric GWP": r["Non-Electric GWP"],
            "GWP per g Cu": r["Total GWP"] / cap,
            "Electricity %": r["Electricity %"]
        })
    sum_df = pd.DataFrame(summary_rows)

    # -------------------------------------------------------------------------
    # TAB 1: RESULTS
    # -------------------------------------------------------------------------
    with tab1:
        st.header("LCA Results")
        st.dataframe(sum_df.style.format({
            "Total GWP": "{:.2e}",
            "Non-Electric GWP": "{:.2f}",
            "GWP per g Cu": "{:.2f}",
            "Electricity %": "{:.1f}%"
        }))
        
        col1, col2 = st.columns(2)
        with col1:
            fig_log = px.bar(sum_df, x="Bead", y="Total GWP", color="Bead", log_y=True, title="Total GWP per kg bead", text_auto=".2s")
            st.plotly_chart(fig_log, use_container_width=True)
            
        with col2:
            fig_fu2 = px.bar(sum_df, x="Bead", y="GWP per g Cu", color="Bead", title="GWP per g Cu removed", text_auto=".2f")
            st.plotly_chart(fig_fu2, use_container_width=True)
            
        st.subheader("Electricity vs Chemicals")
        stack_data = []
        for r in results_list:
            stack_data.append({"Bead": r["name"], "Source": "Electricity", "GWP": r["Electricity GWP"]})
            stack_data.append({"Bead": r["name"], "Source": "Chemicals/Transport", "GWP": r["Non-Electric GWP"]})
        fig_stack = px.bar(pd.DataFrame(stack_data), x="Bead", y="GWP", color="Source", title="Contribution Breakdown", text_auto=".2s")
        st.plotly_chart(fig_stack, use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 2: SENSITIVITY
    # -------------------------------------------------------------------------
    with tab2:
        st.header("Sensitivity Analysis")
        
        # Grid Intensity Plot
        st.subheader("Grid Intensity Sensitivity")
        grids = st.session_state["custom_grids"]
        sens_rows = []
        
        # We need to temporarily modify EF DF for the loop, but restore later or copy
        temp_ef = ef_df.copy()
        
        for g_name, g_val in grids.items():
            temp_ef.loc[temp_ef["reagent_name"].str.contains("Electricity"), "GWP_kgCO2_per_kg"] = g_val
            for rid in unique_routes:
                res, _ = calculate_impacts(
                    rid, temp_ef, routes_df, eng_params,
                    recycling_rate=recycle_rate, yield_rate=yield_rate, transport_pct=transport_overhead
                )
                sens_rows.append({
                    "Grid": g_name, "Grid Value": g_val, "Bead": res["name"], "Total GWP": res["Total GWP"]
                })
                
        df_sens = pd.DataFrame(sens_rows).sort_values("Grid Value")
        fig_sens = px.line(df_sens, x="Grid", y="Total GWP", color="Bead", markers=True, title="GWP vs Grid Intensity")
        st.plotly_chart(fig_sens, use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 3: INVENTORY
    # -------------------------------------------------------------------------
    with tab3:
        st.header("Inventory Breakdown")
        if dfs_list:
            all_contribs = []
            for i, df in enumerate(dfs_list):
                df = df.copy()
                df["Bead"] = results_list[i]["name"]
                all_contribs.append(df)
            df_all = pd.concat(all_contribs)
            
            fig_break = px.bar(df_all, x="Bead", y="GWP", color="Component", title="Detailed Breakdown (Log Scale)", log_y=True)
            st.plotly_chart(fig_break, use_container_width=True)
            
            st.subheader("Sankey Diagrams")
            col_sk1, col_sk2 = st.columns(2)
            with col_sk1:
                st.plotly_chart(plot_sankey_diagram(results_list, ID_REF), use_container_width=True)
            with col_sk2:
                st.plotly_chart(plot_sankey_diagram(results_list, ID_MOF), use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 4: LITERATURE
    # -------------------------------------------------------------------------
    with tab4:
        st.header("Literature Comparison")
        current_data = []
        for r in results_list:
            current_data.append({
                "Material": f"{r['name']} (This Work)",
                "GWP_kgCO2_per_kg": r["Total GWP"],
                "Source": "This Work",
                "Type": "This Work"
            })
        lit_combined = pd.concat([lit_df, pd.DataFrame(current_data)])
        fig_lit = px.bar(lit_combined, x="Material", y="GWP_kgCO2_per_kg", color="Source", log_y=True, title="Comparison with Literature")
        fig_lit.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_lit, use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 5: AI INSIGHTS
    # -------------------------------------------------------------------------
    with tab5:
        st.header("AI Insights")
        st.write("Ask AI to analyze the results based on your current scaling parameters.")
        
        user_q = st.text_area("Question:", "Why is the GWP so high compared to literature?")
        if st.button("Analyze"):
            with st.spinner("Consulting AI..."):
                ans = get_ai_insight(results_list, user_q)
                st.markdown("### Analysis")
                st.info(ans)

if __name__ == "__main__":
    main()
