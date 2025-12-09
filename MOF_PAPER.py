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
    st.caption("Use these sliders to simulate different production scenarios.")
    
    # 1. Grid Intensity
    current_grid_ef = float(st.session_state["ef_df"].loc[
        st.session_state["ef_df"]["reagent_name"] == "Electricity (Canada)", 
        "GWP_kgCO2_per_kg"
    ].iloc[0])
    
    new_grid_ef = st.slider(
        "⚡ Grid Carbon Intensity (kg CO2/kWh)", 
        min_value=0.0, max_value=1.0, 
        value=current_grid_ef, 
        step=0.01,
        help="Controls how 'clean' the electricity is.\n\n"
             "• 0.00: Hydro/Wind/Solar (Zero Carbon)\n"
             "• 0.12: Canada Average (Baseline)\n"
             "• 0.38: US Average\n"
             "• 0.58: China Grid (Coal heavy)\n"
             "• 0.80+: Coal Power Plant"
    )
    
    if new_grid_ef != current_grid_ef:
        st.session_state["ef_df"].loc[
            st.session_state["ef_df"]["reagent_name"] == "Electricity (Canada)", 
            "GWP_kgCO2_per_kg"
        ] = new_grid_ef

    # 2. Process Efficiency (Scaling Factor)
    st.write("**🏭 Process Efficiency**")
    eff_factor = st.slider(
        "Efficiency Multiplier", 
        min_value=0.1, max_value=1.0, value=1.0, 
        help="Simulates scaling up from lab to factory.\n\n"
             "• 1.0 = Lab Scale (Current Baseline). Inefficient, small batches.\n"
             "• 0.5 = Pilot Scale. Uses 50% less electricity per kg.\n"
             "• 0.1 = Industrial Scale. Highly optimized machines."
    )

    # 3. Solvent Recovery
    st.write("**♻️ Solvent Recovery**")
    recycle_rate = st.slider(
        "Recycling Rate (%)", 
        min_value=0, max_value=95, value=0,
        help="Simulates capturing and reusing solvents (Ethanol, Formic Acid).\n\n"
             "• 0%: No recycling (Single use).\n"
             "• 90%: Most solvent is reused, drastically reducing chemical footprint."
    )

    # 4. Yield
    st.write("**🧪 Synthesis Yield**")
    yield_rate = st.slider(
        "Global Yield (%)",
        min_value=10, max_value=100, value=100,
        help="Simulates material losses.\n\n"
             "• 100%: Perfect yield (Baseline assumption).\n"
             "• 50%: Half the product is lost. You need 2x the raw materials and electricity to make the same amount of final beads."
    )

    # 5. Transport Overhead
    st.write("**🚛 Transport & Logistics**")
    transport_overhead = st.slider(
        "Add Transport Overhead (%)",
        min_value=0, max_value=50, value=0,
        help="Adds a percentage to the total GWP to account for shipping raw materials.\n\n"
             "• 0%: Gate-to-Gate (Factory only).\n"
             "• 10-20%: Typical overhead for global supply chains."
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
def calculate_impacts(route_id, ef_df, routes_df, efficiency_factor=1.0, recycling_rate=0.0, yield_rate=100.0, transport_pct=0.0):
    """Calculates GWP based on session state and efficiency modifiers."""
    route_data = routes_df[routes_df["route_id"] == route_id].copy()
    if route_data.empty: return None, None

    # Yield Factor: If yield is 50%, we need 1/0.5 = 2x input
    yield_multiplier = 1.0 / (yield_rate / 100.0)

    # Electricity
    base_elec_kwh = float(route_data.iloc[0]["electricity_kwh_per_fu"])
    # Apply Efficiency (Scale up) AND Yield (Material loss)
    elec_kwh = base_elec_kwh * efficiency_factor * yield_multiplier
    
    elec_source = route_data.iloc[0]["electricity_source"]
    ef_elec_row = ef_df[ef_df["reagent_name"] == elec_source]
    ef_elec = float(ef_elec_row["GWP_kgCO2_per_kg"].iloc[0]) if not ef_elec_row.empty else 0.0
    gwp_elec = elec_kwh * ef_elec

    # Reagents
    contributions = [{"Component": "Electricity", "Category": "Electricity", "Mass (kg)": 0.0, "GWP": gwp_elec}]
    total_reagent_gwp = 0.0

    for _, row in route_data.iterrows():
        reagent = row["reagent_name"]
        base_mass = float(row["mass_kg_per_fu"])
        
        # 1. Apply Yield (need more input if yield is low)
        mass_needed = base_mass * yield_multiplier

        # 2. Apply Recycling (reduce effective GWP impact)
        # Only applies to solvents
        is_solvent = reagent in ["Ethanol", "Formic acid (88%)", "Acetic acid"]
        effective_mass = mass_needed * (1 - (recycling_rate/100)) if is_solvent else mass_needed
        
        ef_row = ef_df[ef_df["reagent_name"] == reagent]
        ef_val = float(ef_row["GWP_kgCO2_per_kg"].iloc[0]) if not ef_row.empty else 0.0
        
        gwp_val = effective_mass * ef_val
        total_reagent_gwp += gwp_val
        
        if reagent in ["Chitosan", "PDChNF"]: cat = "Polymers"
        elif reagent in ["Zirconium tetrachloride", "2-Aminoterephthalic acid", "Formic acid (88%)", "Ethanol"]: cat = "MOF Reagents"
        else: cat = "Solvents/Other"
            
        contributions.append({"Component": reagent, "Category": cat, "Mass (kg)": effective_mass, "GWP": gwp_val})

    # Transport Overhead
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
# VISUALIZATION FUNCTIONS
# -----------------------------------------------------------------------------
def plot_sankey_diagram(results_list):
    """Generates a Sankey diagram showing flow of impacts."""
    # Visualize the first route (U@Bead) as it's the most complex
    target_res = next((r for r in results_list if r["id"] == ID_MOF), results_list[0])
    
    # Values
    elec_gwp = target_res["Electricity GWP"]
    chem_gwp = target_res["Non-Electric GWP"]
    total_gwp = target_res["Total GWP"]
    
    labels = ["Electricity Source", "Chemical Supply", "Lab Synthesis", "Total GWP"]
    colors = ["#FFD700", "#90EE90", "#87CEFA", "#FF6347"]
    
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
    Use the **Control Panel** on the left to simulate scaling effects, recycling, or yield loss.
    """)

    # --- CALCULATION LOOP ---
    unique_routes = ROUTES_DF["route_id"].unique()
    results_list = []
    dfs_list = []
    
    for rid in unique_routes:
        res, df = calculate_impacts(
            rid, EF_DF, ROUTES_DF, 
            efficiency_factor=eff_factor, 
            recycling_rate=recycle_rate,
            yield_rate=yield_rate,
            transport_pct=transport_overhead
        )
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
                res, _ = calculate_impacts(rid, temp_ef, ROUTES_DF, efficiency_factor=eff_factor, recycling_rate=recycle_rate, yield_rate=yield_rate)
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

            # 2. Total Impacts Including Electricity (GROUPED for readability + LOG SCALE)
            st.subheader("B. Total Process Impacts (Electricity vs Chemicals)")
            st.caption("Side-by-side comparison. Note: Logarithmic scale is used to show chemicals clearly next to electricity.")
            fig_total_breakdown = px.bar(df_all, x="Bead", y="GWP", color="Component", 
                                         title="Total GWP Breakdown (Log Scale)", 
                                         barmode="group",
                                         log_y=True) # Added Log Scale for readability
            st.plotly_chart(fig_total_breakdown, use_container_width=True)
            
            st.divider()

            # 3. Mass Inventory (FACETED & UNCOUPLED AXES)
            st.subheader("C. Mass Inventory")
            st.caption("Separated by bead type to ensure low-mass components in Ref-Bead are visible.")
            
            fig_mass = px.bar(
                df_ne, 
                x="Component", 
                y="Mass (kg)", 
                color="Component",
                facet_col="Bead",  # Separate panels
                title="Mass Input per kg Product"
            )
            # FIX: Uncouple Y-axes so Ref-Bead scale is independent of U@Bead
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
        st.caption("Each color represents a distinct data source.")

if __name__ == "__main__":
    main()
