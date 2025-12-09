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
def get_ai_suggestion(query):
    """Uses OpenAI to find GWP values or grid intensities."""
    if not _OPENAI_AVAILABLE:
        return "Error: OpenAI library not installed. Please check requirements."
    
    api_key = st.secrets.get("openai_api_key2")
    if not api_key:
        return "Error: API Key 'openai_api_key2' not found in secrets."

    client = OpenAI(api_key=api_key)
    
    prompt = f"""
    You are an LCA expert. The user needs a specific value for their study.
    Provide a best-estimate number and a brief source/explanation.
    
    User Query: {query}
    
    Format:
    Value: [Numeric Value]
    Unit: [Unit, e.g., kg CO2/kg]
    Source/Note: [Brief explanation]
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
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
    st.session_state["reset_trigger"] = False

# Initialize Session State
if "ef_df" not in st.session_state:
    reset_data()

# -----------------------------------------------------------------------------
# SIDEBAR: USER INPUTS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Control Panel")
    
    if st.button("🔄 Reset to Paper Defaults"):
        reset_data()
        st.success("Data reset to manuscript values!")
        st.rerun()

    st.divider()
    
    # --- QUICK ADJUST (Sliders) ---
    st.subheader("1. Quick Adjustments")
    st.caption("Simulate scenarios beyond the paper's baseline.")
    
    # 1. Grid Intensity
    current_grid_ef = float(st.session_state["ef_df"].loc[
        st.session_state["ef_df"]["reagent_name"] == "Electricity (Canada)", 
        "GWP_kgCO2_per_kg"
    ].iloc[0])
    
    new_grid_ef = st.slider(
        "Grid Intensity (kg CO2/kWh)", 
        min_value=0.0, max_value=1.0, 
        value=current_grid_ef, 
        step=0.01,
        help="0.12=Canada, 0.38=US, 0.58=China"
    )
    
    if new_grid_ef != current_grid_ef:
        st.session_state["ef_df"].loc[
            st.session_state["ef_df"]["reagent_name"] == "Electricity (Canada)", 
            "GWP_kgCO2_per_kg"
        ] = new_grid_ef

    # 2. Process Efficiency (Scaling Factor)
    st.write("**Process Efficiency**")
    eff_factor = st.slider(
        "Efficiency Multiplier", 
        min_value=0.1, max_value=1.0, value=1.0, 
        help="1.0 = Lab Scale (Baseline). 0.5 = 50% less electricity (Pilot Scale)."
    )

    # 3. Solvent Recovery
    st.write("**Solvent Recovery**")
    recycle_rate = st.slider(
        "Recycling Rate (%)", 
        min_value=0, max_value=95, value=0,
        help="Reduces impact of Ethanol and Formic Acid."
    )

    st.divider()

    # --- ADVANCED EDIT (Tables) ---
    st.subheader("2. Detailed Tables")
    with st.expander("Edit Reagents & Factors"):
        st.write("**Emission Factors**")
        st.session_state["ef_df"] = st.data_editor(st.session_state["ef_df"], key="ed_ef", num_rows="dynamic")
        
        st.write("**Route Recipes**")
        st.session_state["routes_df"] = st.data_editor(st.session_state["routes_df"], key="ed_routes", num_rows="dynamic")
        
        st.write("**Performance Stats**")
        st.session_state["perf_df"] = st.data_editor(st.session_state["perf_df"], key="ed_perf", num_rows="dynamic")

    st.divider()
    st.subheader("3. AI Assistant")
    ai_query = st.text_input("Ask for data (e.g., 'GWP of acetone')")
    if st.button("Ask AI"):
        with st.spinner("Consulting AI..."):
            st.info(get_ai_suggestion(ai_query))

# Shortcuts
EF_DF = st.session_state["ef_df"]
ROUTES_DF = st.session_state["routes_df"]
PERF_DF = st.session_state["perf_df"]
LIT_DF = st.session_state["lit_df"]

# -----------------------------------------------------------------------------
# CALCULATION ENGINE
# -----------------------------------------------------------------------------
def calculate_impacts(route_id, ef_df, routes_df, efficiency_factor=1.0, recycling_rate=0.0):
    """Calculates GWP based on session state and efficiency modifiers."""
    route_data = routes_df[routes_df["route_id"] == route_id].copy()
    if route_data.empty: return None, None

    # Electricity (Scaled by efficiency)
    base_elec_kwh = float(route_data.iloc[0]["electricity_kwh_per_fu"])
    elec_kwh = base_elec_kwh * efficiency_factor
    
    elec_source = route_data.iloc[0]["electricity_source"]
    ef_elec_row = ef_df[ef_df["reagent_name"] == elec_source]
    ef_elec = float(ef_elec_row["GWP_kgCO2_per_kg"].iloc[0]) if not ef_elec_row.empty else 0.0
    gwp_elec = elec_kwh * ef_elec

    # Reagents
    contributions = [{"Component": "Electricity", "Category": "Electricity", "Mass (kg)": 0.0, "GWP": gwp_elec}]
    total_reagent_gwp = 0.0

    for _, row in route_data.iterrows():
        reagent = row["reagent_name"]
        mass = float(row["mass_kg_per_fu"])
        
        # Apply recycling reduction to solvents
        is_solvent = reagent in ["Ethanol", "Formic acid (88%)", "Acetic acid"]
        current_mass = mass * (1 - (recycling_rate/100)) if is_solvent else mass
        
        ef_row = ef_df[ef_df["reagent_name"] == reagent]
        ef_val = float(ef_row["GWP_kgCO2_per_kg"].iloc[0]) if not ef_row.empty else 0.0
        
        gwp_val = current_mass * ef_val
        total_reagent_gwp += gwp_val
        
        if reagent in ["Chitosan", "PDChNF"]: cat = "Polymers"
        elif reagent in ["Zirconium tetrachloride", "2-Aminoterephthalic acid", "Formic acid (88%)", "Ethanol"]: cat = "MOF Reagents"
        else: cat = "Solvents/Other"
            
        contributions.append({"Component": reagent, "Category": cat, "Mass (kg)": current_mass, "GWP": gwp_val})

    total_gwp = gwp_elec + total_reagent_gwp
    
    results = {
        "id": route_id,
        "name": route_data.iloc[0]["route_name"],
        "Total GWP": total_gwp,
        "Electricity GWP": gwp_elec,
        "Non-Electric GWP": total_reagent_gwp,
        "Electricity kWh": elec_kwh,
        "Electricity EF Used": ef_elec
    }
    return results, pd.DataFrame(contributions)

# -----------------------------------------------------------------------------
# VISUALIZATION FUNCTIONS
# -----------------------------------------------------------------------------
def plot_sankey_diagram(results_list):
    """Generates a Sankey diagram showing flow of impacts."""
    # We will aggregate impacts across all active routes for a general flow view
    # Or just visualize the first route (U@Bead) as it's the most complex
    
    target_res = next((r for r in results_list if r["id"] == ID_MOF), results_list[0])
    
    # Values
    elec_gwp = target_res["Electricity GWP"]
    chem_gwp = target_res["Non-Electric GWP"]
    total_gwp = target_res["Total GWP"]
    
    # Nodes: 0=Inputs, 1=Electricity, 2=Chemicals, 3=Bead Production, 4=GWP Impact
    labels = ["Electricity Source", "Chemical Supply", "Lab Synthesis", "Total GWP"]
    colors = ["#FFD700", "#90EE90", "#87CEFA", "#FF6347"]
    
    # Links
    sources = [0, 1, 2, 2]
    targets = [2, 2, 3, 3] # Elec->Syn, Chem->Syn, Syn->GWP
    
    # Simplified: 
    # 0(Elec) -> 2(Syn)
    # 1(Chem) -> 2(Syn)
    # 2(Syn) -> 3(Total)
    
    fig = go.Figure(data=[go.Sankey(
        node = dict(
          pad = 15, thickness = 20,
          line = dict(color = "black", width = 0.5),
          label = labels,
          color = colors
        ),
        link = dict(
          source = [0, 1, 2], 
          target = [2, 2, 3],
          value = [elec_gwp, chem_gwp, total_gwp],
          color = ["rgba(255, 215, 0, 0.4)", "rgba(144, 238, 144, 0.4)", "rgba(135, 206, 250, 0.4)"]
      ))])

    fig.update_layout(title_text=f"Impact Flow: {target_res['name']}", font_size=10, height=400)
    return fig

# -----------------------------------------------------------------------------
# MAIN APP
# -----------------------------------------------------------------------------
def main():
    st.title("Interactive LCA Explorer: Ref-Bead vs U@Bead")
    st.markdown("""
    **Overview:** This dashboard allows interactive analysis of the screening LCA.
    Use the **Control Panel** to simulate scaling effects (efficiency) or recycling scenarios.
    """)

    # --- CALCULATION LOOP ---
    unique_routes = ROUTES_DF["route_id"].unique()
    results_list = []
    dfs_list = []
    
    for rid in unique_routes:
        res, df = calculate_impacts(rid, EF_DF, ROUTES_DF, efficiency_factor=eff_factor, recycling_rate=recycle_rate)
        if res:
            results_list.append(res)
            dfs_list.append(df)
            
    if not results_list:
        st.warning("No valid routes found. Try Resetting to Defaults.")
        return

    perf_map = {row["route_id"]: float(row["capacity_mg_g"]) for _, row in PERF_DF.iterrows()}

    # --- TABS ---
    tab1, tab2, tab3, tab4 = st.tabs(["LCA Results", "Sensitivity Analysis", "Inventory & System", "Literature"])

    # --- TAB 1: RESULTS ---
    with tab1:
        st.header("Life Cycle Impact Assessment")
        
        summary_rows = []
        for r in results_list:
            cap = perf_map.get(r["id"], 0.001)
            summary_rows.append({
                "Bead": r["name"],
                "Total GWP (FU1)": r["Total GWP"],
                "Non-Electric GWP": r["Non-Electric GWP"],
                "GWP per g Cu (FU2)": r["Total GWP"] / cap,
                "Electricity %": (r["Electricity GWP"] / r["Total GWP"]) * 100
            })
        
        sum_df = pd.DataFrame(summary_rows)
        st.dataframe(sum_df.style.format({"Total GWP (FU1)": "{:.2e}", "Non-Electric GWP": "{:.2f}", "GWP per g Cu (FU2)": "{:.2f}", "Electricity %": "{:.1f}%"}))

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Total GWP (Log Scale)")
            fig_log = px.bar(sum_df, x="Bead", y="Total GWP (FU1)", color="Bead", 
                             log_y=True, title="Total GWP (Log Scale)", text_auto='.2s')
            st.plotly_chart(fig_log, use_container_width=True)

        with col2:
            st.subheader("Impact Contribution (%)")
            stack_data = []
            for r in results_list:
                stack_data.append({"Bead": r["name"], "Source": "Electricity", "GWP": r["Electricity GWP"]})
                stack_data.append({"Bead": r["name"], "Source": "Chemicals", "GWP": r["Non-Electric GWP"]})
            
            fig_stack = px.bar(pd.DataFrame(stack_data), x="Bead", y="GWP", color="Source", 
                               title="Breakdown: Electricity vs Chemicals", text_auto='.2s')
            st.plotly_chart(fig_stack, use_container_width=True)
            
        st.subheader("Performance Normalized (FU2)")
        fig_fu2 = px.bar(sum_df, x="Bead", y="GWP per g Cu (FU2)", color="Bead", title="GWP per g Copper Removed", text_auto='.2f')
        st.plotly_chart(fig_fu2, use_container_width=True)

    # --- TAB 2: SENSITIVITY ---
    with tab2:
        st.header("Sensitivity Analysis")
        
        st.subheader("1. Grid Intensity Sensitivity")
        grids = {"QC Hydro": 0.002, "Canada Avg": 0.1197, "UK Grid": 0.225, "EU Avg": 0.25, "US Avg": 0.38, "China Grid": 0.58}
        
        sens_rows = []
        for g_name, g_val in grids.items():
            temp_ef = EF_DF.copy()
            temp_ef.loc[temp_ef["reagent_name"].str.contains("Electricity"), "GWP_kgCO2_per_kg"] = g_val
            for rid in unique_routes:
                res, _ = calculate_impacts(rid, temp_ef, ROUTES_DF, efficiency_factor=eff_factor, recycling_rate=recycle_rate)
                sens_rows.append({"Grid": g_name, "Bead": res["name"], "Total GWP": res["Total GWP"]})
        
        df_sens = pd.DataFrame(sens_rows)
        fig_sens = px.line(df_sens, x="Grid", y="Total GWP", color="Bead", markers=True, title="Total GWP vs Grid Carbon Intensity")
        st.plotly_chart(fig_sens, use_container_width=True)
        
        st.divider()
        st.subheader("2. Batch Scaling Effect")
        st.markdown("Projection of how GWP drops if batch size increases.")
        
        batch_sizes = [0.001, 0.01, 0.1, 1.0, 10.0]
        scale_rows = []
        for rid in unique_routes:
            base_res, _ = calculate_impacts(rid, EF_DF, ROUTES_DF, efficiency_factor=1.0, recycling_rate=0.0)
            base_elec_per_kg = base_res["Electricity kWh"]
            ref_batch_kg = 0.0005 
            
            for b_size in batch_sizes:
                scaling_factor = ref_batch_kg / b_size
                new_elec = base_elec_per_kg * (0.1 + 0.9 * scaling_factor)
                new_gwp = (new_elec * base_res["Electricity EF Used"]) + base_res["Non-Electric GWP"]
                scale_rows.append({"Batch Size (kg)": b_size, "Bead": base_res["name"], "Estimated GWP": new_gwp})
                
        df_scale = pd.DataFrame(scale_rows)
        fig_scale = px.line(df_scale, x="Batch Size (kg)", y="Estimated GWP", color="Bead", 
                            log_x=True, log_y=True, title="Projected GWP vs Batch Size (Log-Log Scale)")
        st.plotly_chart(fig_scale, use_container_width=True)

    # --- TAB 3: INVENTORY ---
    with tab3:
        st.header("Process Inventory & Breakdown")
        
        all_contribs = []
        for i, df in enumerate(dfs_list):
            df["Bead"] = results_list[i]["name"]
            all_contribs.append(df)
        
        if all_contribs:
            df_all = pd.concat(all_contribs)
            
            # 1. Chemical Impacts Only
            st.subheader("A. Chemical Impacts (Excluding Electricity)")
            df_ne = df_all[df_all["Category"] != "Electricity"]
            fig_ne = px.bar(df_ne, x="Bead", y="GWP", color="Component", 
                            title="Chemical GWP Contribution", barmode="group")
            st.plotly_chart(fig_ne, use_container_width=True)

            st.divider()

            # 2. Total Impacts Including Electricity (GROUPED for readability)
            st.subheader("B. Total Process Impacts (Electricity vs Chemicals)")
            st.caption("Side-by-side comparison of electricity vs chemical components.")
            fig_total_breakdown = px.bar(df_all, x="Bead", y="GWP", color="Component", 
                                         title="Total GWP Breakdown", barmode="group")
            st.plotly_chart(fig_total_breakdown, use_container_width=True)
            
            st.divider()

            # 3. Mass Inventory (FACETED for readability)
            st.subheader("C. Mass Inventory")
            st.caption("Separated by bead type to ensure low-mass components in Ref-Bead are visible.")
            fig_mass = px.bar(df_ne, x="Component", y="Mass (kg)", color="Component",
                              facet_col="Bead",  # Separate panels
                              title="Mass Input per kg Product", 
                              matches=None)      # Independent Y-axes
            
            # Ensure Y-axes are independent so Ref-Bead (small) isn't dwarfed by U@Bead (huge)
            fig_mass.update_yaxes(matches=None, showticklabels=True)
            st.plotly_chart(fig_mass, use_container_width=True)
        
        st.subheader("D. Impact Flow (Sankey Diagram)")
        st.plotly_chart(plot_sankey_diagram(results_list), use_container_width=True)

    # --- TAB 4: LITERATURE ---
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
            
        lit_combined = pd.concat([LIT_DF, pd.DataFrame(current_data)])
        
        # Color by Source
        fig_lit = px.bar(lit_combined, x="Material", y="GWP_kgCO2_per_kg", color="Source", 
                         log_y=True, title="Global Warming Potential Comparison (Log Scale)",
                         text="Source")
        
        st.plotly_chart(fig_lit, use_container_width=True)
        st.caption("Log scale used due to high variation between lab-scale and industrial benchmarks.")

if __name__ == "__main__":
    main()
