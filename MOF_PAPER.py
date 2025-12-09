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
        context_str += f"- {res['name']}: Total GWP={res['Total GWP']:.2e}, Elec%={res['Electricity %']:.1f}%\n"

    system_prompt = f"""
    You are an expert in Life Cycle Assessment (LCA) for materials science.
    Use the provided data to answer the user's question.
    
    Context Data:
    {context_str}
    
    Guidelines:
    - Be concise and scientific.
    - Explain *why* certain impacts are high (e.g., electricity in lab scale).
    - Suggest practical improvements if asked.
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
    # Reset Custom Grids
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
    
    if st.button("🔄 Reset to Paper Defaults"):
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
        "⚡ Grid Carbon Intensity", 
        min_value=0.0, max_value=1.0, 
        value=current_grid_ef, 
        step=0.01,
        help="kg CO2 per kWh. Controls the cleanliness of the power source used."
    )
    
    if new_grid_ef != current_grid_ef:
        st.session_state["ef_df"].loc[
            st.session_state["ef_df"]["reagent_name"] == "Electricity (Canada)", 
            "GWP_kgCO2_per_kg"
        ] = new_grid_ef

    # Efficiency
    st.write("**🏭 Process Efficiency**")
    eff_factor = st.slider(
        "Efficiency Multiplier", 
        min_value=0.1, max_value=1.0, value=1.0, 
        help="1.0 = Lab Scale (Baseline). Lower values simulate industrial optimization."
    )

    # Recycling
    st.write("**♻️ Solvent Recovery**")
    recycle_rate = st.slider(
        "Recycling Rate (%)", 
        min_value=0, max_value=95, value=0,
        help="Percentage of Ethanol/Formic Acid recovered and reused."
    )

    # Yield
    st.write("**🧪 Global Yield**")
    yield_rate = st.slider(
        "Synthesis Yield (%)",
        min_value=10, max_value=100, value=100,
        help="Material yield. Lower yield = higher waste and input requirements."
    )

    # Transport
    st.write("**🚛 Transport Overhead**")
    transport_overhead = st.slider(
        "Add Transport (%)",
        min_value=0, max_value=50, value=0,
        help="Adds a fixed percentage to account for logistics."
    )

    st.divider()

    # --- TABLES ---
    st.subheader("2. Input Tables")
    with st.expander("Edit Detailed Inputs"):
        st.caption("Edit Emission Factors")
        st.session_state["ef_df"] = st.data_editor(st.session_state["ef_df"], key="ed_ef", num_rows="dynamic")
        
        st.caption("Edit Recipes")
        st.session_state["routes_df"] = st.data_editor(st.session_state["routes_df"], key="ed_routes", num_rows="dynamic")
        
        st.caption("Edit Performance")
        st.session_state["perf_df"] = st.data_editor(st.session_state["perf_df"], key="ed_perf", num_rows="dynamic")

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

    yield_multiplier = 1.0 / (yield_rate / 100.0)

    # Electricity
    base_elec_kwh = float(route_data.iloc[0]["electricity_kwh_per_fu"])
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
        
        # Yield Impact
        mass_needed = base_mass * yield_multiplier

        # Recycling Impact
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
# PLOTTING
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
    st.markdown("""
    **Dashboard:** Adjust scenarios on the left. Use the tabs below to explore Results, Sensitivity, and AI Insights.
    """)

    # --- CALCULATION ---
    unique_routes = ROUTES_DF["route_id"].unique()
    results_list = []
    dfs_list = []
    
    for rid in unique_routes:
        res, df = calculate_impacts(rid, EF_DF, ROUTES_DF, eff_factor, recycle_rate, yield_rate, transport_overhead)
        if res:
            results_list.append(res)
            dfs_list.append(df)
            
    if not results_list:
        st.warning("No routes. Check Inputs.")
        return

    perf_map = {row["route_id"]: float(row["capacity_mg_g"]) for _, row in PERF_DF.iterrows()}

    # Prepare Summary Data
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
        "📊 LCA Results", 
        "📈 Sensitivity Analysis", 
        "📦 Inventory", 
        "📚 Literature", 
        "🤖 AI Insights"
    ])

    # --- TAB 1: RESULTS ---
    with tab1:
        st.header("LCA Results")
        st.dataframe(sum_df.style.format({"Total GWP": "{:.2e}", "Non-Electric GWP": "{:.2f}", "GWP per g Cu": "{:.2f}", "Electricity %": "{:.1f}%"}))

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Total GWP (Log Scale)")
            fig_log = px.bar(sum_df, x="Bead", y="Total GWP", color="Bead", 
                             log_y=True, title="Total GWP (Log Scale)", text_auto='.2s')
            st.plotly_chart(fig_log, use_container_width=True)

        with col2:
            st.subheader("Impact Contribution")
            stack_data = []
            for r in results_list:
                stack_data.append({"Bead": r["name"], "Source": "Electricity", "GWP": r["Electricity GWP"]})
                stack_data.append({"Bead": r["name"], "Source": "Chemicals", "GWP": r["Non-Electric GWP"]})
            fig_stack = px.bar(pd.DataFrame(stack_data), x="Bead", y="GWP", color="Source", 
                               title="Electricity vs Chemicals Breakdown", text_auto='.2s')
            st.plotly_chart(fig_stack, use_container_width=True)
            
        st.subheader("Performance Normalized (FU2)")
        fig_fu2 = px.bar(sum_df, x="Bead", y="GWP per g Cu", color="Bead", title="GWP per g Copper Removed", text_auto='.2f')
        st.plotly_chart(fig_fu2, use_container_width=True)

    # --- TAB 2: SENSITIVITY ---
    with tab2:
        st.header("Sensitivity Analysis")
        
        # 1. Interactive Grid Analysis
        st.subheader("1. Grid Intensity Sensitivity")
        
        col_s1, col_s2 = st.columns([1, 2])
        with col_s1:
            st.write("Add Custom Grid:")
            new_grid_name = st.text_input("Name", "My Local Grid")
            new_grid_val = st.number_input("Intensity (kg CO2/kWh)", 0.0, 1.0, 0.45)
            if st.button("Add to Chart"):
                st.session_state["custom_grids"][new_grid_name] = new_grid_val
                st.success(f"Added {new_grid_name}")
        
        # Calculate for all grids
        sens_rows = []
        for g_name, g_val in st.session_state["custom_grids"].items():
            temp_ef = EF_DF.copy()
            temp_ef.loc[temp_ef["reagent_name"].str.contains("Electricity"), "GWP_kgCO2_per_kg"] = g_val
            for rid in unique_routes:
                res, _ = calculate_impacts(rid, temp_ef, ROUTES_DF, eff_factor, recycle_rate, yield_rate)
                sens_rows.append({
                    "Grid": g_name, 
                    "Grid Value": g_val, 
                    "Bead": res["name"], 
                    "Total GWP": res["Total GWP"]
                })
        
        df_sens = pd.DataFrame(sens_rows).sort_values("Grid Value")
        fig_sens = px.line(df_sens, x="Grid", y="Total GWP", color="Bead", markers=True, 
                           title="Total GWP vs Grid Carbon Intensity",
                           hover_data=["Grid Value"])
        st.plotly_chart(fig_sens, use_container_width=True)
        
        st.divider()
        
        # 2. Scaling Analysis
        st.subheader("2. Batch Scaling Effect")
        ref_batch_kg = st.slider("Reference Lab Batch Size (kg)", 0.0001, 0.0100, 0.0005, step=0.0001, 
                                 help="The actual batch size used in the lab experiment (e.g. 0.5g = 0.0005kg).")
        
        batch_sizes = [0.0005, 0.001, 0.01, 0.1, 1.0, 10.0]
        scale_rows = []
        for rid in unique_routes:
            base_res, _ = calculate_impacts(rid, EF_DF, ROUTES_DF, 1.0, 0.0)
            base_elec = base_res["Electricity kWh"]
            
            for b_size in batch_sizes:
                scale_factor = ref_batch_kg / b_size
                new_elec = base_elec * (0.1 + 0.9 * scale_factor) # Model
                new_gwp = (new_elec * base_res["Electricity EF Used"]) + base_res["Non-Electric GWP"]
                scale_rows.append({"Batch Size (kg)": b_size, "Bead": base_res["name"], "Estimated GWP": new_gwp})
                
        fig_scale = px.line(pd.DataFrame(scale_rows), x="Batch Size (kg)", y="Estimated GWP", color="Bead", 
                            log_x=True, log_y=True, title="Projected GWP vs Batch Size (Log-Log Scale)")
        
        # Add Industrial Target Zone
        fig_scale.add_vrect(x0=1.0, x1=10.0, fillcolor="green", opacity=0.1, annotation_text="Industrial Target")
        st.plotly_chart(fig_scale, use_container_width=True)

    # --- TAB 3: INVENTORY ---
    with tab3:
        st.header("Inventory Breakdown")
        
        all_contribs = []
        for i, df in enumerate(dfs_list):
            df["Bead"] = results_list[i]["name"]
            all_contribs.append(df)
        df_all = pd.concat(all_contribs) if all_contribs else pd.DataFrame()

        if not df_all.empty:
            st.subheader("A. Chemical Impacts")
            df_ne = df_all[df_all["Category"] != "Electricity"]
            fig_ne = px.bar(df_ne, x="Bead", y="GWP", color="Component", title="Chemical GWP (No Elec)", barmode="group")
            st.plotly_chart(fig_ne, use_container_width=True)

            st.divider()

            st.subheader("B. Total Breakdown (Log Scale)")
            fig_breakdown = px.bar(df_all, x="Bead", y="GWP", color="Component", 
                                   title="Total GWP Breakdown (Log Scale)", 
                                   barmode="group", log_y=True)
            st.plotly_chart(fig_breakdown, use_container_width=True)

            st.divider()

            st.subheader("C. Mass Inventory")
            fig_mass = px.bar(df_ne, x="Component", y="Mass (kg)", color="Component", 
                              facet_col="Bead", title="Mass Input per kg Product")
            fig_mass.update_yaxes(matches=None, showticklabels=True)
            st.plotly_chart(fig_mass, use_container_width=True)
        
        st.subheader("D. Impact Flow")
        st.plotly_chart(plot_sankey_diagram(results_list), use_container_width=True)

    # --- TAB 4: LITERATURE ---
    with tab4:
        st.header("Literature Comparison")
        current_data = []
        for r in results_list:
            current_data.append({
                "Material": f"{r['name']} (This Work)",
                "GWP_kgCO2_per_kg": r["Total GWP"],
                "Source": "This Work", "Type": "This Work"
            })
        lit_combined = pd.concat([LIT_DF, pd.DataFrame(current_data)])
        
        fig_lit = px.bar(lit_combined, x="Material", y="GWP_kgCO2_per_kg", color="Source", 
                         log_y=True, title="GWP Comparison (Log Scale)", text="Source")
        st.plotly_chart(fig_lit, use_container_width=True)

    # --- TAB 5: AI INSIGHTS ---
    with tab5:
        st.header("🤖 AI Insights")
        st.caption("Ask questions about the current results displayed in the dashboard.")
        
        col_q1, col_q2 = st.columns([1, 3])
        
        with col_q1:
            preset_q = st.selectbox(
                "Quick Questions", 
                ["(Select One)",
                 "Why is the GWP so high compared to literature?",
                 "Compare Ref-Bead and U@Bead results.",
                 "What is the biggest hotspot?",
                 "How can I reduce the carbon footprint?"]
            )
        
        with col_q2:
            custom_q = st.text_input("Or type your own question:")
        
        final_q = custom_q if custom_q else (preset_q if preset_q != "(Select One)" else "")
        
        if st.button("Analyze Results"):
            if final_q:
                with st.spinner("AI is analyzing your data..."):
                    answer = get_ai_insight(results_list, final_q)
                    st.markdown("### AI Analysis")
                    st.info(answer)
            else:
                st.warning("Please select or type a question.")

if __name__ == "__main__":
    main()
