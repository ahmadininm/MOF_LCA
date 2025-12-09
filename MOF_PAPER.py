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
        return "Error: OpenAI library not installed."
    
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

# Initialize Session State with defaults if empty
if "ef_df" not in st.session_state:
    ef, routes, perf, lit = load_default_data()
    st.session_state["ef_df"] = ef
    st.session_state["routes_df"] = routes
    st.session_state["perf_df"] = perf
    st.session_state["lit_df"] = lit

# -----------------------------------------------------------------------------
# SIDEBAR: DATA EDITOR & AI
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("1. Customize Inputs")
    st.caption("Double-click cells to edit data.")
    
    with st.expander("Emission Factors (EF)", expanded=False):
        st.session_state["ef_df"] = st.data_editor(
            st.session_state["ef_df"], num_rows="dynamic", key="editor_ef"
        )
        
    with st.expander("Route Definitions (Recipe)", expanded=False):
        st.session_state["routes_df"] = st.data_editor(
            st.session_state["routes_df"], num_rows="dynamic", key="editor_routes"
        )
        
    with st.expander("Performance (Adsorption)", expanded=False):
        st.session_state["perf_df"] = st.data_editor(
            st.session_state["perf_df"], num_rows="dynamic", key="editor_perf"
        )

    st.divider()
    st.header("2. AI Assistant")
    ai_query = st.text_input("Ask for data (e.g., 'GWP of acetone')")
    if st.button("Ask AI"):
        with st.spinner("Consulting AI..."):
            st.info(get_ai_suggestion(ai_query))

# Shortcuts for easier access in main code
EF_DF = st.session_state["ef_df"]
ROUTES_DF = st.session_state["routes_df"]
PERF_DF = st.session_state["perf_df"]
LIT_DF = st.session_state["lit_df"]

# -----------------------------------------------------------------------------
# CALCULATION ENGINE
# -----------------------------------------------------------------------------
def calculate_impacts(route_id, ef_df, routes_df, grid_override_val=None):
    """
    Calculates GWP. Allows overriding grid intensity for sensitivity analysis.
    """
    # Filter for the specific route
    route_data = routes_df[routes_df["route_id"] == route_id].copy()
    if route_data.empty: return None, None

    # Get Electricity Info
    elec_kwh = float(route_data.iloc[0]["electricity_kwh_per_fu"])
    elec_source = route_data.iloc[0]["electricity_source"]
    
    # Determine EF for Electricity
    if grid_override_val is not None:
        ef_elec = grid_override_val
    else:
        ef_elec_row = ef_df[ef_df["reagent_name"] == elec_source]
        ef_elec = float(ef_elec_row["GWP_kgCO2_per_kg"].iloc[0]) if not ef_elec_row.empty else 0.0
    
    gwp_elec = elec_kwh * ef_elec

    # Calculate Reagent GWP
    contributions = [{"Component": "Electricity", "Category": "Electricity", "Mass (kg)": 0.0, "GWP": gwp_elec}]
    total_reagent_gwp = 0.0

    for _, row in route_data.iterrows():
        reagent = row["reagent_name"]
        mass = float(row["mass_kg_per_fu"])
        
        ef_row = ef_df[ef_df["reagent_name"] == reagent]
        ef_val = float(ef_row["GWP_kgCO2_per_kg"].iloc[0]) if not ef_row.empty else 0.0
        
        gwp_val = mass * ef_val
        total_reagent_gwp += gwp_val
        
        # Categorize
        if reagent in ["Chitosan", "PDChNF"]: cat = "Polymers"
        elif reagent in ["Zirconium tetrachloride", "2-Aminoterephthalic acid", "Formic acid (88%)", "Ethanol"]: cat = "MOF Reagents"
        else: cat = "Solvents/Other"
            
        contributions.append({"Component": reagent, "Category": cat, "Mass (kg)": mass, "GWP": gwp_val})

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
# PLOTTING FUNCTIONS
# -----------------------------------------------------------------------------
def plot_system_boundary():
    fig = go.Figure()
    x_nodes, y_nodes = [0, 1, 2, 3], [1, 1, 1, 1]
    labels = ["Raw Materials", "Lab Gate", "Synthesis<br>(Elec. Intensive)", "Bead Product"]
    
    fig.add_trace(go.Scatter(x=x_nodes, y=y_nodes, mode="markers+text", 
                             marker=dict(size=50, color=["#D3D3D3", "#FFD700", "#FF6347", "#90EE90"]),
                             text=labels, textposition="bottom center"))

    anns = [
        dict(x=0.5, y=1, text="Transport (Excluded)", ax=0, ay=1),
        dict(x=1.5, y=1, text="Inputs", ax=1, ay=1),
        dict(x=2.5, y=1, text="Processing", ax=2, ay=1),
        dict(x=2, y=0.5, text="Emissions (CO2)", ax=2, ay=1, ayref="y", axref="x"),
    ]
    for a in anns:
        fig.add_annotation(xref="x", yref="y", showarrow=True, arrowhead=2, **a)

    fig.update_layout(title="Figure 10: System Boundary (Gate-to-Gate)", xaxis=dict(visible=False, range=[-0.5, 3.5]), yaxis=dict(visible=False, range=[0, 1.5]), height=300)
    return fig

# -----------------------------------------------------------------------------
# MAIN APP
# -----------------------------------------------------------------------------
def main():
    st.title("Interactive LCA Explorer: Ref-Bead vs U@Bead")
    st.markdown("""
    **Overview:** This tool implements the screening LCA from the paper. 
    **Customization:** Use the sidebar to edit emission factors or recipes for your own study.
    """)

    # 1. BASE CALCULATION
    # -------------------
    # Filter Routes based on what exists in the routes dataframe (dynamic handling)
    unique_routes = ROUTES_DF["route_id"].unique()
    
    results_list = []
    dfs_list = []
    
    for rid in unique_routes:
        res, df = calculate_impacts(rid, EF_DF, ROUTES_DF)
        if res:
            results_list.append(res)
            dfs_list.append(df)
            
    if not results_list:
        st.warning("No valid routes found in Route Definitions.")
        return

    # Extract Performance Data dynamically
    perf_map = {row["route_id"]: row["capacity_mg_g"] for _, row in PERF_DF.iterrows()}

    # TABS
    tab1, tab2, tab3, tab4 = st.tabs(["LCA Results", "Sensitivity Analysis", "Inventory & System", "Literature"])

    # --- TAB 1: RESULTS ---
    with tab1:
        st.header("Life Cycle Impact Assessment")
        
        # Build Summary Dataframe
        summary_rows = []
        for r in results_list:
            cap = perf_map.get(r["id"], 0.001) # Avoid div/0
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
            st.caption("Note: Logarithmic scale emphasizes order of magnitude differences[cite: 84].")

        with col2:
            st.subheader("GWP Contribution (%)")
            # Prepare data for 100% stacked bar
            stack_data = []
            for r in results_list:
                stack_data.append({"Bead": r["name"], "Source": "Electricity", "GWP": r["Electricity GWP"]})
                stack_data.append({"Bead": r["name"], "Source": "Chemicals", "GWP": r["Non-Electric GWP"]})
            
            fig_stack = px.bar(pd.DataFrame(stack_data), x="Bead", y="GWP", color="Source", 
                               title="Percentage Contribution Breakdown", text_auto='.2s')
            st.plotly_chart(fig_stack, use_container_width=True)
            st.caption("Electricity dominates impact (>98%) for both beads.")
            
        st.subheader("Performance Normalized (FU2)")
        fig_fu2 = px.bar(sum_df, x="Bead", y="GWP per g Cu (FU2)", color="Bead", title="GWP per g Copper Removed", text_auto='.2f')
        st.plotly_chart(fig_fu2, use_container_width=True)
        st.caption("Higher capacity of MOF bead partially offsets its higher production footprint[cite: 93].")

    # --- TAB 2: SENSITIVITY ---
    with tab2:
        st.header("Sensitivity Analysis")
        
        st.subheader("1. Grid Intensity Sensitivity")
        st.markdown("How does the location (grid mix) affect the total GWP? ")
        
        grids = {
            "QC Hydro (0.002)": 0.002,
            "Canada Avg (0.12)": 0.1197,
            "UK Grid (0.23)": 0.225,
            "EU Avg (0.25)": 0.25,
            "US Avg (0.38)": 0.38,
            "China Grid (0.58)": 0.58
        }
        
        sens_rows = []
        for g_name, g_val in grids.items():
            for rid in unique_routes:
                res, _ = calculate_impacts(rid, EF_DF, ROUTES_DF, grid_override_val=g_val)
                sens_rows.append({"Grid": g_name, "Bead": res["name"], "Total GWP": res["Total GWP"]})
        
        df_sens = pd.DataFrame(sens_rows)
        fig_sens = px.line(df_sens, x="Grid", y="Total GWP", color="Bead", markers=True, title="Total GWP vs Grid Carbon Intensity")
        st.plotly_chart(fig_sens, use_container_width=True)
        
        st.divider()
        st.subheader("2. Batch Scaling Effect")
        st.markdown("Modeling the reduction in electricity per kg as batch size increases.")
        
        # Model: E_total = (E_fixed / batch_mass) + E_variable
        # Assuming current high values are 90% fixed overhead from freeze dryer
        batch_sizes = [0.001, 0.01, 0.1, 1.0, 10.0] # kg
        scale_rows = []
        
        for rid in unique_routes:
            # Get base electricity from user input (assumed to be the small scale 1kg equivalent)
            base_res, _ = calculate_impacts(rid, EF_DF, ROUTES_DF)
            base_elec_per_kg = base_res["Electricity kWh"]
            
            # Simple scaling model: 
            # Current (0.0004 kg batch) -> huge per kg.
            # E_per_kg_new = Base_E_per_kg * (Reference_Batch / New_Batch)
            ref_batch_kg = 0.0005 # ~0.5g typical lab batch
            
            for b_size in batch_sizes:
                # Apply scaling factor with a floor (variable energy)
                scaling_factor = ref_batch_kg / b_size
                # Assume 10% is variable (linear), 90% is fixed overhead that scales down
                new_elec = base_elec_per_kg * (0.1 + 0.9 * scaling_factor)
                
                # Recalculate GWP
                new_gwp = (new_elec * base_res["Electricity EF Used"]) + base_res["Non-Electric GWP"]
                scale_rows.append({"Batch Size (kg)": b_size, "Bead": base_res["name"], "Estimated GWP": new_gwp})
                
        df_scale = pd.DataFrame(scale_rows)
        fig_scale = px.line(df_scale, x="Batch Size (kg)", y="Estimated GWP", color="Bead", 
                            log_x=True, log_y=True, title="Projected GWP vs Batch Size (Log-Log Scale)")
        st.plotly_chart(fig_scale, use_container_width=True)

    # --- TAB 3: INVENTORY ---
    with tab3:
        st.header("Process Inventory")
        
        # Combine all contribution dfs
        all_contribs = []
        for i, df in enumerate(dfs_list):
            df["Bead"] = results_list[i]["name"]
            all_contribs.append(df)
        
        if all_contribs:
            df_all = pd.concat(all_contribs)
            
            # Non-Electric Bar Chart
            df_ne = df_all[df_all["Category"] != "Electricity"]
            fig_ne = px.bar(df_ne, x="Bead", y="GWP", color="Component", 
                            title="Chemical Impacts Only (Excluding Electricity)", barmode="group")
            st.plotly_chart(fig_ne, use_container_width=True)
            
            # Mass Inventory
            fig_mass = px.bar(df_ne, x="Bead", y="Mass (kg)", color="Component", 
                              title="Mass Input per kg Product", barmode="group")
            st.plotly_chart(fig_mass, use_container_width=True)
        
        st.subheader("System Boundary")
        
        st.plotly_chart(plot_system_boundary(), use_container_width=True)

    # --- TAB 4: LITERATURE ---
    with tab4:
        st.header("Literature Comparison (Figure 8)")
        
        # Combine Lit DF with Current Results
        current_data = []
        for r in results_list:
            current_data.append({
                "Material": f"{r['name']} (This Work)",
                "GWP_kgCO2_per_kg": r["Total GWP"],
                "Type": "This Work",
                "Source": "This Study"
            })
            
        lit_combined = pd.concat([LIT_DF, pd.DataFrame(current_data)])
        
        fig_lit = px.bar(lit_combined, x="Material", y="GWP_kgCO2_per_kg", color="Type", 
                         log_y=True, title="Global Warming Potential Comparison (Log Scale)",
                         text="Source",
                         color_discrete_map={"This Work": "red", "Literature": "blue"})
        st.plotly_chart(fig_lit, use_container_width=True)
        st.caption("Literature sources include Luo et al. (2021), Gu et al. (2018), and Arfasa & Tilahun (2025) .")

if __name__ == "__main__":
    main()
