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

# Route IDs
ID_REF = "ref"
ID_MOF = "mof"

st.set_page_config(page_title="LCA Explorer: Interactive & AI-Enhanced", layout="wide")

# -----------------------------------------------------------------------------
# AI HELPER FUNCTION
# -----------------------------------------------------------------------------
def get_ai_insight(context_data, user_question, current_params):
    """
    Sends calculation results + user question to OpenAI for analysis.
    """
    if not _OPENAI_AVAILABLE:
        return "Error: OpenAI library not installed."
    
    api_key = st.secrets.get("openai_api_key2")
    if not api_key:
        return "Error: API Key 'openai_api_key2' not found in secrets."

    client = OpenAI(api_key=api_key)
    
    # Prepare a summary of the current results
    context_str = "Current Simulation Results:\n"
    for res in context_data:
        context_str += f"- {res['name']}: Total GWP={res['Total GWP']:.2e} kg CO2e, Elec%={res['Electricity %']:.1f}%\n"
    
    # Add current slider settings to context
    settings_str = f"""
    Current User Settings:
    - Grid Intensity: {current_params['grid']} kg CO2/kWh
    - Synthesis Yield: {current_params['yield']}%
    - Drying Time: {current_params['dry_time']} hours
    - Solvent Usage: {current_params['solvent_vol'] * 100}% of baseline
    """

    system_prompt = f"""
    You are an expert in Life Cycle Assessment (LCA) for materials science (MOFs/biopolymers).
    Use the provided data to answer the user's question.
    
    Context Data:
    {context_str}
    
    {settings_str}
    
    Guidelines:
    - Be scientific but accessible.
    - If the user asks for predictions, base them on general industrial scaling principles (e.g., economies of scale).
    - Explain *why* certain impacts are high (e.g., freeze-drying duration).
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_question}
            ],
            temperature=0.3
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
    def read_safe(path):
        if not path.exists(): return pd.DataFrame()
        try: return pd.read_csv(path, encoding="utf-8")
        except: return pd.read_csv(path, encoding="latin1")

    return (
        read_safe(DATA_DIR / "emission_factors.csv"),
        read_safe(DATA_DIR / "lca_routes.csv"),
        read_safe(DATA_DIR / "performance.csv"),
        read_safe(DATA_DIR / "literature.csv")
    )

def reset_data():
    """Resets session state to default CSV values."""
    ef, routes, perf, lit = load_default_data()
    st.session_state["ef_df"] = ef
    st.session_state["routes_df"] = routes
    st.session_state["perf_df"] = perf
    st.session_state["lit_df"] = lit
    st.session_state["custom_grids"] = {
        "QC Hydro": 0.002, 
        "Canada Avg": 0.1197, 
        "UK Grid": 0.225, 
        "EU Avg": 0.25, 
        "US Avg": 0.38, 
        "China Grid": 0.58
    }

# Initialize Session State
if "ef_df" not in st.session_state:
    reset_data()

# -----------------------------------------------------------------------------
# SIDEBAR: USER INPUTS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Control Panel")
    
    if st.button("🔄 Reset to Paper Defaults", help="Restores values from the V11 manuscript."):
        reset_data()
        st.success("Data reset!")
        st.rerun()

    st.divider()
    
    # --- QUICK ADJUST ---
    st.subheader("1. Scenario Parameters")
    
    # Grid Intensity
    current_grid_ef = float(st.session_state["ef_df"].loc[
        st.session_state["ef_df"]["reagent_name"] == "Electricity (Canada)", 
        "GWP_kgCO2_per_kg"
    ].iloc[0])
    
    new_grid_ef = st.slider(
        "⚡ Grid Carbon Intensity (kg CO2/kWh)", 
        min_value=0.0, max_value=1.0, 
        value=current_grid_ef, 
        step=0.01,
        help="0.12=Canada, 0.38=US, 0.58=China. This determines the footprint of every kWh used."
    )
    if new_grid_ef != current_grid_ef:
        st.session_state["ef_df"].loc[
            st.session_state["ef_df"]["reagent_name"] == "Electricity (Canada)", 
            "GWP_kgCO2_per_kg"
        ] = new_grid_ef

    # Yield
    st.write("**🧪 Synthesis Yield**")
    yield_rate = st.slider(
        "Global Yield (%)",
        min_value=10, max_value=100, value=100,
        help="If yield is 50%, you need 2x the raw materials and energy to make the same amount of product."
    )

    # NEW: Drying Time (The major hotspot)
    st.write("**⏳ Freeze Drying Duration**")
    dry_time = st.slider(
        "Drying Time (Hours)",
        min_value=1, max_value=48, value=16,
        help="The paper assumes 16h drying. Reducing this directly lowers electricity consumption."
    )

    # NEW: Solvent Volume
    st.write("**💧 Solvent Usage**")
    solvent_vol_factor = st.slider(
        "Volume Multiplier (vs Baseline)",
        min_value=0.1, max_value=2.0, value=1.0, step=0.1,
        help="1.0 = Standard Recipe. 0.5 = Using half the solvent (Process Intensification)."
    )

    # Recycling
    st.write("**♻️ Solvent Recovery**")
    recycle_rate = st.slider(
        "Recycling Rate (%)", 
        min_value=0, max_value=95, value=0,
        help="Percentage of used Ethanol/Formic Acid captured and reused."
    )

    # Transport
    st.write("**🚛 Transport Overhead**")
    transport_overhead = st.slider(
        "Logistics Surcharge (%)",
        min_value=0, max_value=50, value=0,
        help="Adds % to GWP to account for shipping raw materials."
    )

    st.divider()

    # --- TABLES ---
    st.subheader("2. Input Tables")
    with st.expander("Edit Detailed Inputs"):
        st.caption("Emission Factors")
        st.session_state["ef_df"] = st.data_editor(st.session_state["ef_df"], key="ed_ef", num_rows="dynamic")
        st.caption("Recipes")
        st.session_state["routes_df"] = st.data_editor(st.session_state["routes_df"], key="ed_routes", num_rows="dynamic")
        st.caption("Performance")
        st.session_state["perf_df"] = st.data_editor(st.session_state["perf_df"], key="ed_perf", num_rows="dynamic")

# Shortcuts
EF_DF = st.session_state["ef_df"]
ROUTES_DF = st.session_state["routes_df"]
PERF_DF = st.session_state["perf_df"]
LIT_DF = st.session_state["lit_df"]

# Capture current params for AI
current_params = {
    "grid": new_grid_ef,
    "yield": yield_rate,
    "dry_time": dry_time,
    "solvent_vol": solvent_vol_factor
}

# -----------------------------------------------------------------------------
# CALCULATION ENGINE
# -----------------------------------------------------------------------------
def calculate_impacts(route_id, ef_df, routes_df, dry_time_h, solvent_factor, recycling_rate, yield_rate, transport_pct):
    """Calculates GWP dynamically based on all sliders."""
    route_data = routes_df[routes_df["route_id"] == route_id].copy()
    if route_data.empty: return None, None

    yield_multiplier = 1.0 / (yield_rate / 100.0)

    # --- Electricity Adjustment ---
    # The paper assumes ~93.6MWh/kg for Ref, dominated by 16h drying.
    # We model electricity as: Base * (NewTime / 16h) * YieldMultiplier
    # Note: This is a simplification; stirring doesn't scale with drying time, but drying is dominant (>80%).
    base_elec_kwh = float(route_data.iloc[0]["electricity_kwh_per_fu"])
    
    # Approx: Scale electricity linearly with drying time relative to baseline (16h)
    time_factor = dry_time_h / 16.0
    elec_kwh = base_elec_kwh * time_factor * yield_multiplier
    
    elec_source = route_data.iloc[0]["electricity_source"]
    ef_elec_row = ef_df[ef_df["reagent_name"] == elec_source]
    ef_elec = float(ef_elec_row["GWP_kgCO2_per_kg"].iloc[0]) if not ef_elec_row.empty else 0.0
    gwp_elec = elec_kwh * ef_elec

    # --- Reagents Adjustment ---
    contributions = [{"Component": "Electricity", "Category": "Electricity", "Mass (kg)": 0.0, "GWP": gwp_elec}]
    total_reagent_gwp = 0.0

    for _, row in route_data.iterrows():
        reagent = row["reagent_name"]
        base_mass = float(row["mass_kg_per_fu"])
        
        # 1. Yield Impact
        mass_needed = base_mass * yield_multiplier

        # 2. Solvent Volume Reduction (Process Intensification)
        is_solvent = reagent in ["Ethanol", "Formic acid (88%)", "Acetic acid"]
        if is_solvent:
            mass_needed = mass_needed * solvent_factor

        # 3. Recycling (End-of-pipe recovery)
        effective_mass = mass_needed * (1 - (recycling_rate/100)) if is_solvent else mass_needed
        
        ef_row = ef_df[ef_df["reagent_name"] == reagent]
        ef_val = float(ef_row["GWP_kgCO2_per_kg"].iloc[0]) if not ef_row.empty else 0.0
        
        gwp_val = effective_mass * ef_val
        total_reagent_gwp += gwp_val
        
        if reagent in ["Chitosan", "PDChNF"]: cat = "Polymers"
        elif reagent in ["Zirconium tetrachloride", "2-Aminoterephthalic acid", "Formic acid (88%)", "Ethanol"]: cat = "MOF Reagents"
        else: cat = "Solvents/Other"
            
        contributions.append({"Component": reagent, "Category": cat, "Mass (kg)": effective_mass, "GWP": gwp_val})

    # Transport
    raw_total_gwp = gwp_elec + total_reagent_gwp
    transport_gwp = raw_total_gwp * (transport_pct / 100.0)
    
    if transport_gwp > 0:
        contributions.append({"Component": "Transport", "Category": "Logistics", "Mass (kg)": 0.0, "GWP": transport_gwp})

    final_total_gwp = raw_total_gwp + transport_gwp
    
    results = {
        "id": route_id,
        "name": route_data.iloc[0]["route_name"],
        "Total GWP": final_total_gwp,
        "Electricity GWP": gwp_elec,
        "Non-Electric GWP": total_reagent_gwp + transport_gwp,
        "Electricity kWh": elec_kwh,
        "Electricity EF Used": ef_elec
    }
    return results, pd.DataFrame(contributions)

# -----------------------------------------------------------------------------
# VISUALIZATION
# -----------------------------------------------------------------------------
def plot_sankey_diagram(results_list):
    target_res = next((r for r in results_list if r["id"] == ID_MOF), results_list[0])
    elec_gwp = target_res["Electricity GWP"]
    chem_gwp = target_res["Non-Electric GWP"]
    total_gwp = target_res["Total GWP"]
    
    labels = ["Electricity Source", "Chemical Supply", "Lab Synthesis", "Total GWP"]
    colors = ["#FFD700", "#90EE90", "#87CEFA", "#FF6347"]
    
    fig = go.Figure(data=[go.Sankey(
        node = dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=labels, color=colors),
        link = dict(source=[0, 1, 2], target=[2, 2, 3], value=[elec_gwp, chem_gwp, total_gwp], 
                    color=["rgba(255, 215, 0, 0.4)", "rgba(144, 238, 144, 0.4)", "rgba(135, 206, 250, 0.4)"])
      )])
    fig.update_layout(title_text=f"Impact Flow: {target_res['name']}", font_size=10, height=400)
    return fig

# -----------------------------------------------------------------------------
# MAIN APP
# -----------------------------------------------------------------------------
def main():
    st.title("Interactive LCA Explorer: Ref-Bead vs U@Bead")
    st.markdown("Use the **Control Panel** to simulate Drying Time, Recycling, and Yield changes.")

    # --- CALCULATION ---
    unique_routes = ROUTES_DF["route_id"].unique()
    results_list = []
    dfs_list = []
    
    for rid in unique_routes:
        res, df = calculate_impacts(
            rid, EF_DF, ROUTES_DF, 
            dry_time, solvent_vol_factor, recycle_rate, yield_rate, transport_overhead
        )
        if res:
            results_list.append(res)
            dfs_list.append(df)
            
    if not results_list:
        st.warning("No routes. Check Inputs.")
        return

    perf_map = {row["route_id"]: float(row["capacity_mg_g"]) for _, row in PERF_DF.iterrows()}

    summary_rows = []
    for r in results_list:
        cap = perf_map.get(r["id"], 0.001)
        summary_rows.append({
            "Bead": r["name"],
            "Total GWP": r["Total GWP"],
            "Non-Electric GWP": r["Non-Electric GWP"],
            "GWP per g Cu": r["Total GWP"] / cap,
            "Electricity %": (r["Electricity GWP"] / r["Total GWP"]) * 100
        })
    sum_df = pd.DataFrame(summary_rows)

    # --- TABS ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Results", "📈 Sensitivity", "📦 Inventory", "📚 Literature", "🤖 AI Insights"
    ])

    # --- TAB 1: RESULTS ---
    with tab1:
        st.dataframe(sum_df.style.format({"Total GWP": "{:.2e}", "Non-Electric GWP": "{:.2f}", "GWP per g Cu": "{:.2f}", "Electricity %": "{:.1f}%"}))
        
        col1, col2 = st.columns(2)
        with col1:
            fig_log = px.bar(sum_df, x="Bead", y="Total GWP", color="Bead", 
                             log_y=True, title="Total GWP (Log Scale)", text_auto='.2s')
            st.plotly_chart(fig_log, use_container_width=True)
        with col2:
            fig_fu2 = px.bar(sum_df, x="Bead", y="GWP per g Cu", color="Bead", title="GWP per g Cu Removed", text_auto='.2f')
            st.plotly_chart(fig_fu2, use_container_width=True)

    # --- TAB 2: SENSITIVITY ---
    with tab2:
        st.header("Sensitivity Analysis")
        
        st.subheader("1. Grid Intensity")
        # Calc logic for chart
        sens_rows = []
        grids = st.session_state["custom_grids"]
        
        for g_name, g_val in grids.items():
            temp_ef = EF_DF.copy()
            temp_ef.loc[temp_ef["reagent_name"].str.contains("Electricity"), "GWP_kgCO2_per_kg"] = g_val
            for rid in unique_routes:
                res, _ = calculate_impacts(rid, temp_ef, ROUTES_DF, dry_time, solvent_vol_factor, recycle_rate, yield_rate, transport_overhead)
                sens_rows.append({"Grid": g_name, "Grid Val": g_val, "Bead": res["name"], "Total GWP": res["Total GWP"]})
        
        fig_sens = px.line(pd.DataFrame(sens_rows).sort_values("Grid Val"), x="Grid", y="Total GWP", color="Bead", markers=True, 
                           title="GWP vs Grid Intensity")
        st.plotly_chart(fig_sens, use_container_width=True)

        st.divider()

        st.subheader("2. Batch Scaling (Economies of Scale)")
        st.markdown("""
        **Transparency:** The scaling model assumes: $E_{total} \approx (E_{fixed\_overhead} / Mass_{batch}) + E_{variable}$  
        At small lab scales (0.5 g), the fixed overhead (drying pump, heating bath) dominates.
        """)
        
        ref_batch_kg = st.slider("Reference Batch Size (kg)", 0.0001, 1.0, 0.0005, format="%.4f",
                                 help="The small batch size used in the lab experiment (0.0005 kg = 0.5 g).")
        
        batch_sizes = np.logspace(-4, 1, 20) # 0.1g to 10kg
        scale_rows = []
        
        for rid in unique_routes:
            base_res, _ = calculate_impacts(rid, EF_DF, ROUTES_DF, dry_time, solvent_vol_factor, recycle_rate, yield_rate, 0)
            base_elec_per_kg = base_res["Electricity kWh"]
            
            for b_size in batch_sizes:
                # Scaling factor: If we make 10kg instead of 0.0005kg, the per-kg energy drops massively
                # Model: 10% is variable energy (heats reaction), 90% is fixed overhead (vacuum pump runs regardless)
                scale_ratio = ref_batch_kg / b_size
                new_elec = base_elec_per_kg * (0.1 + 0.9 * scale_ratio)
                
                # Apply current grid EF
                new_gwp = (new_elec * base_res["Electricity EF Used"]) + base_res["Non-Electric GWP"]
                scale_rows.append({"Batch Size (kg)": b_size, "Bead": base_res["name"], "Estimated GWP": new_gwp})
                
        fig_scale = px.line(pd.DataFrame(scale_rows), x="Batch Size (kg)", y="Estimated GWP", color="Bead", 
                            log_x=True, log_y=True, title="Projected GWP vs Batch Size")
        fig_scale.add_vrect(x0=1.0, x1=10.0, fillcolor="green", opacity=0.1, annotation_text="Pilot/Ind")
        st.plotly_chart(fig_scale, use_container_width=True)

    # --- TAB 3: INVENTORY ---
    with tab3:
        all_contribs = []
        for i, df in enumerate(dfs_list):
            df["Bead"] = results_list[i]["name"]
            all_contribs.append(df)
        df_all = pd.concat(all_contribs)

        col_i1, col_i2 = st.columns(2)
        with col_i1:
            st.markdown("#### Chemical Impacts (No Elec)")
            df_ne = df_all[df_all["Category"] != "Electricity"]
            fig_ne = px.bar(df_ne, x="Bead", y="GWP", color="Component", barmode="group")
            st.plotly_chart(fig_ne, use_container_width=True)
            
        with col_i2:
            st.markdown("#### Total Breakdown (Log Scale)")
            fig_tot = px.bar(df_all, x="Bead", y="GWP", color="Component", barmode="group", log_y=True)
            st.plotly_chart(fig_tot, use_container_width=True)
            
        st.markdown("#### Mass Inventory")
        fig_mass = px.bar(df_ne, x="Component", y="Mass (kg)", color="Component", facet_col="Bead")
        fig_mass.update_yaxes(matches=None, showticklabels=True) # FIX: Uncouple axes
        st.plotly_chart(fig_mass, use_container_width=True)
        
        st.markdown("#### Impact Flow")
        st.plotly_chart(plot_sankey_diagram(results_list), use_container_width=True)

    # --- TAB 4: LITERATURE ---
    with tab4:
        current_data = []
        for r in results_list:
            current_data.append({
                "Material": f"{r['name']} (This Work)",
                "GWP_kgCO2_per_kg": r["Total GWP"],
                "Source": "This Work", "Type": "This Work"
            })
        lit_combined = pd.concat([LIT_DF, pd.DataFrame(current_data)])
        
        fig_lit = px.bar(lit_combined, x="Material", y="GWP_kgCO2_per_kg", color="Source", 
                         log_y=True, title="Literature Benchmark (Log Scale)", text="Source")
        st.plotly_chart(fig_lit, use_container_width=True)

    # --- TAB 5: AI INSIGHTS ---
    with tab5:
        st.header("🤖 AI Insights & Predictions")
        st.markdown("Click a question or type your own to analyze the current simulation results.")
        
        col_buttons, col_chat = st.columns([1, 2])
        
        user_q = ""
        with col_buttons:
            st.caption("Quick Ask:")
            if st.button("🔥 Identify Hotspots"):
                user_q = "Identify the top 3 contributors to GWP in the current simulation."
            if st.button("🏭 Scaling Potential"):
                user_q = "How much would GWP likely decrease if this process is scaled to 1 ton/day?"
            if st.button("⚖️ Ref vs MOF Trade-off"):
                user_q = "Is the extra GWP of the MOF bead justified by its capacity?"
            if st.button("🔮 Future Prediction"):
                user_q = "Predict the GWP in 2030 assuming a 50% greener grid and 90% solvent recycling."

        with col_chat:
            custom_q = st.text_area("Custom Question:", value=user_q, height=100)
            if st.button("Analyze with AI"):
                if custom_q:
                    with st.spinner("Consulting AI..."):
                        answer = get_ai_insight(results_list, custom_q, current_params)
                        st.success("Analysis Complete")
                        st.markdown(answer)
                else:
                    st.warning("Please enter a question.")

if __name__ == "__main__":
    main()
