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
    
    if st.button("🔄 Reset to Defaults"):
        reset_data()
        st.success("Data reset!")
        st.rerun()

    st.divider()
    
    # --- QUICK ADJUST (Sliders) ---
    st.subheader("1. Quick Adjustments")
    st.caption("Easily modify key parameters without editing tables.")
    
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
        help="0.12 = Canada, 0.38 = US, 0.58 = China"
    )
    
    # Update Grid EF in Dataframe if changed
    if new_grid_ef != current_grid_ef:
        st.session_state["ef_df"].loc[
            st.session_state["ef_df"]["reagent_name"] == "Electricity (Canada)", 
            "GWP_kgCO2_per_kg"
        ] = new_grid_ef

    # 2. Electricity Usage (Route 1: Ref)
    current_ref_elec = float(st.session_state["routes_df"].loc[
        st.session_state["routes_df"]["route_id"] == ID_REF, 
        "electricity_kwh_per_fu"
    ].iloc[0])
    
    new_ref_elec = st.number_input(
        "Ref-Bead Electricity (kWh/kg)",
        min_value=0.0, value=current_ref_elec, step=1000.0
    )
    
    if new_ref_elec != current_ref_elec:
        st.session_state["routes_df"].loc[
            st.session_state["routes_df"]["route_id"] == ID_REF, 
            "electricity_kwh_per_fu"
        ] = new_ref_elec
        
    # 3. Electricity Usage (Route 2: MOF)
    current_mof_elec = float(st.session_state["routes_df"].loc[
        st.session_state["routes_df"]["route_id"] == ID_MOF, 
        "electricity_kwh_per_fu"
    ].iloc[0])
    
    new_mof_elec = st.number_input(
        "U@Bead Electricity (kWh/kg)",
        min_value=0.0, value=current_mof_elec, step=1000.0
    )
    
    if new_mof_elec != current_mof_elec:
        st.session_state["routes_df"].loc[
            st.session_state["routes_df"]["route_id"] == ID_MOF, 
            "electricity_kwh_per_fu"
        ] = new_mof_elec

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
def calculate_impacts(route_id, ef_df, routes_df):
    """Calculates GWP based on current session state data."""
    route_data = routes_df[routes_df["route_id"] == route_id].copy()
    if route_data.empty: return None, None

    # Electricity
    elec_kwh = float(route_data.iloc[0]["electricity_kwh_per_fu"])
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
        
        ef_row = ef_df[ef_df["reagent_name"] == reagent]
        ef_val = float(ef_row["GWP_kgCO2_per_kg"].iloc[0]) if not ef_row.empty else 0.0
        gwp_val = mass * ef_val
        total_reagent_gwp += gwp_val
        
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
# PLOTTING
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
    **Overview:** Use the **Control Panel** on the left to adjust grid intensity and electricity usage interactively.
    Results will update automatically.
    """)

    # --- CALCULATION LOOP ---
    unique_routes = ROUTES_DF["route_id"].unique()
    results_list = []
    dfs_list = []
    
    for rid in unique_routes:
        res, df = calculate_impacts(rid, EF_DF, ROUTES_DF)
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
        
        # We calculate this dynamically based on current route recipes, just swapping the grid factor
        sens_rows = []
        for g_name, g_val in grids.items():
            # Use a temporary EF dataframe for calculation
            temp_ef = EF_DF.copy()
            # Find electricity row (assuming standard name) and update it
            temp_ef.loc[temp_ef["reagent_name"].str.contains("Electricity"), "GWP_kgCO2_per_kg"] = g_val
            
            for rid in unique_routes:
                res, _ = calculate_impacts(rid, temp_ef, ROUTES_DF)
                sens_rows.append({"Grid": g_name, "Bead": res["name"], "Total GWP": res["Total GWP"]})
        
        df_sens = pd.DataFrame(sens_rows)
        fig_sens = px.line(df_sens, x="Grid", y="Total GWP", color="Bead", markers=True, title="Total GWP vs Grid Carbon Intensity")
        st.plotly_chart(fig_sens, use_container_width=True)
        
        st.divider()
        st.subheader("2. Batch Scaling Effect")
        st.markdown("Projection of how GWP drops if batch size increases (reducing fixed electricity overhead).")
        
        batch_sizes = [0.001, 0.01, 0.1, 1.0, 10.0] # kg
        scale_rows = []
        
        for rid in unique_routes:
            base_res, _ = calculate_impacts(rid, EF_DF, ROUTES_DF)
            base_elec_per_kg = base_res["Electricity kWh"]
            ref_batch_kg = 0.0005 
            
            for b_size in batch_sizes:
                scaling_factor = ref_batch_kg / b_size
                # Model: 90% fixed overhead scales down, 10% variable stays
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

            # 2. NEW: Total Impacts Including Electricity (Requested Figure)
            st.subheader("B. Total Process Impacts (Including Electricity)")
            st.caption("This chart puts the chemical impacts in perspective with the electricity demand.")
            fig_total_breakdown = px.bar(df_all, x="Bead", y="GWP", color="Component", 
                                         title="Total GWP Breakdown (Electricity vs Components)", barmode="stack")
            st.plotly_chart(fig_total_breakdown, use_container_width=True)
            
            st.divider()

            # 3. Mass Inventory
            st.subheader("C. Mass Inventory")
            fig_mass = px.bar(df_ne, x="Bead", y="Mass (kg)", color="Component", 
                              title="Mass Input per kg Product", barmode="group")
            st.plotly_chart(fig_mass, use_container_width=True)
        
        st.subheader("D. System Boundary")
        st.plotly_chart(plot_system_boundary(), use_container_width=True)

    # --- TAB 4: LITERATURE ---
    with tab4:
        st.header("Literature Comparison")
        
        # Combine Lit DF with Current Results
        current_data = []
        for r in results_list:
            current_data.append({
                "Material": f"{r['name']} (This Work)",
                "GWP_kgCO2_per_kg": r["Total GWP"],
                "Source": "This Work", # Distinct source for coloring
                "Type": "This Work"
            })
            
        lit_combined = pd.concat([LIT_DF, pd.DataFrame(current_data)])
        
        # Updated Figure 8: Colored by Source to distinguish different papers
        fig_lit = px.bar(lit_combined, x="Material", y="GWP_kgCO2_per_kg", color="Source", 
                         log_y=True, title="Global Warming Potential Comparison (Log Scale)",
                         text="Source")
        
        st.plotly_chart(fig_lit, use_container_width=True)
        st.caption("Each color represents a different study/source. Note the log scale.")

if __name__ == "__main__":
    main()
