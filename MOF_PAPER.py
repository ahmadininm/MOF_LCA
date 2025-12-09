import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path
import numpy as np
import graphviz

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
LOGO_DIR = BASE_DIR # Assuming logos are in the root folder

# Route IDs
ID_REF = "ref"
ID_MOF = "mof"

st.set_page_config(page_title="LCA Explorer: Interactive & AI-Enhanced", layout="wide")

# -----------------------------------------------------------------------------
# HARDCODED PROCESS BREAKDOWN (For the new Electricity Figure)
# -----------------------------------------------------------------------------
# Since the CSV only provides total electricity, we approximate the breakdown 
# based on the equipment descriptions in the paper (Table 3/SI).
PROCESS_BREAKDOWN = {
    ID_REF: {
        "Mixing & Dissolution (50°C)": 0.05,
        "Phase Inversion (Pumping)": 0.02,
        "Crosslinking (Heating)": 0.08,
        "Washing": 0.05,
        "Freeze Drying": 0.80  # Lab scale freeze drying is dominant
    },
    ID_MOF: {
        "Support Prep (Mixing/Crosslink)": 0.10,
        "Zr Step (Heating 50°C)": 0.15,
        "Linker Step (Heating 50°C)": 0.15,
        "Washing & Centrifugation": 0.10,
        "Freeze Drying (Final)": 0.50
    }
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
# HEADER & LOGOS
# -----------------------------------------------------------------------------
# Layout: Left (UBC), Center (AME), Right (COV)
col_head1, col_head2, col_head3 = st.columns([1, 2, 1])

def load_logo(filename):
    path = LOGO_DIR / filename
    if path.exists():
        return str(path)
    return None

with col_head1:
    # Left: UBC
    logo_ubc = load_logo("ubc.png")
    if logo_ubc: st.image(logo_ubc, width=150)
    else: st.caption("UBC Logo")

with col_head2:
    # Center: AME
    logo_ame = load_logo("ame.png")
    if logo_ame: 
        # Use columns inside to center perfectly if needed, or simple st.image
        st.image(logo_ame, width=200) 
    else: 
        st.caption("AME Logo")

with col_head3:
    # Right: COV
    logo_cov = load_logo("cov.png")
    if logo_cov: st.image(logo_cov, width=150)
    else: st.caption("Coventry Logo")

st.markdown("---")

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
# PLOTTING FUNCTIONS
# -----------------------------------------------------------------------------
def plot_sankey_diagram_dual(results_list):
    """Generates two Sankey diagrams (one for Ref, one for MOF) stacked."""
    figures = []
    
    for res in results_list:
        elec_gwp = res["Electricity GWP"]
        chem_gwp = res["Non-Electric GWP"]
        total_gwp = res["Total GWP"]
        
        # Determine Color Scheme based on ID
        if res["id"] == ID_REF:
            node_colors = ["#A9A9A9", "#D3D3D3", "#87CEFA", "#4682B4"] # Blues/Greys for Ref
            link_colors = ["rgba(169, 169, 169, 0.4)", "rgba(211, 211, 211, 0.4)", "rgba(135, 206, 250, 0.4)"]
        else:
            node_colors = ["#FFD700", "#90EE90", "#FF7F50", "#CD5C5C"] # Warmer/MOF colors
            link_colors = ["rgba(255, 215, 0, 0.4)", "rgba(144, 238, 144, 0.4)", "rgba(255, 127, 80, 0.4)"]

        labels = ["Electricity Source", "Chemical Supply", "Synthesis Process", "Total GWP"]
        
        fig = go.Figure(data=[go.Sankey(
            node = dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=labels, color=node_colors),
            link = dict(source=[0, 1, 2], target=[2, 2, 3], value=[elec_gwp, chem_gwp, total_gwp], 
                        color=link_colors)
          )])
        fig.update_layout(title_text=f"Impact Flow: {res['name']}", font_size=12, height=350)
        figures.append(fig)
        
    return figures

def plot_electricity_breakdown(results_list):
    """Generates a bar chart showing kWh demand per process step."""
    breakdown_data = []
    
    for res in results_list:
        rid = res["id"]
        total_kwh = res["Electricity kWh"]
        
        # Get specific steps from the hardcoded dictionary
        steps = PROCESS_BREAKDOWN.get(rid, {"General Process": 1.0})
        
        for step_name, fraction in steps.items():
            kwh_step = total_kwh * fraction
            breakdown_data.append({
                "Bead": res["name"],
                "Process Step": step_name,
                "Electricity Demand (kWh/kg)": kwh_step
            })
            
    df_elec = pd.DataFrame(breakdown_data)
    fig = px.bar(
        df_elec, 
        x="Bead", 
        y="Electricity Demand (kWh/kg)", 
        color="Process Step",
        title="Electricity Demand per Process Step",
        text_auto='.1f'
    )
    return fig

def plot_system_boundary():
    """Creates a Graphviz diagram for the system boundary."""
    dot = graphviz.Digraph()
    dot.attr(rankdir='LR', size='8,5')
    dot.attr('node', shape='box', style='filled', color='lightblue')
    
    with dot.subgraph(name='cluster_0') as c:
        c.attr(label='Cradle-to-Gate System Boundary')
        c.attr(color='black')
        c.node('Raw Materials', 'Raw Material Extraction\n(Chitin, Zr, Linkers)')
        c.node('Transport', 'Transport to Lab')
        c.node('Synthesis', 'Bead Synthesis\n(Mixing, Crosslinking, Drying)')
        
        c.edge('Raw Materials', 'Transport')
        c.edge('Transport', 'Synthesis')
        
    dot.node('Output', 'Final Dry Bead (1 kg)', shape='oval', color='lightgreen')
    dot.edge('Synthesis', 'Output')
    
    dot.node('Electricity', 'Grid Electricity', shape='ellipse', color='yellow')
    dot.edge('Electricity', 'Synthesis', style='dashed')
    
    return dot

# -----------------------------------------------------------------------------
# MAIN APP
# -----------------------------------------------------------------------------
def main():
    st.title("Interactive LCA Explorer: Ref-Bead vs U@Bead")
    st.markdown("""
    **Dashboard:** Adjust scenarios on the left. Explore results, sensitivity, and AI insights below.
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
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 LCA Results", 
        "⚡ Electricity Breakdown",
        "🌊 Impact Flow", 
        "📈 Scaling Analysis", 
        "🌐 System Boundary",
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
            st.subheader("Performance Normalized (FU2)")
            fig_fu2 = px.bar(sum_df, x="Bead", y="GWP per g Cu", color="Bead", title="GWP per g Copper Removed", text_auto='.2f')
            st.plotly_chart(fig_fu2, use_container_width=True)

    # --- TAB 2: ELECTRICITY BREAKDOWN ---
    with tab2:
        st.header("Process Energy Analysis")
        st.markdown("Detailed breakdown of electricity consumption per synthesis step.")
        fig_elec_break = plot_electricity_breakdown(results_list)
        st.plotly_chart(fig_elec_break, use_container_width=True)

    # --- TAB 3: IMPACT FLOW ---
    with tab3:
        st.header("Impact Flow (Sankey Diagrams)")
        sankey_figs = plot_sankey_diagram_dual(results_list)
        
        col_san1, col_san2 = st.columns(2)
        if len(sankey_figs) > 0:
            with col_san1: st.plotly_chart(sankey_figs[0], use_container_width=True)
        if len(sankey_figs) > 1:
            with col_san2: st.plotly_chart(sankey_figs[1], use_container_width=True)
            
        st.divider()
        st.subheader("Mass Inventory")
        all_contribs = []
        for i, df in enumerate(dfs_list):
            df["Bead"] = results_list[i]["name"]
            all_contribs.append(df)
        df_all = pd.concat(all_contribs) if all_contribs else pd.DataFrame()
        
        df_ne = df_all[df_all["Category"] != "Electricity"]
        fig_mass = px.bar(df_ne, x="Component", y="Mass (kg)", color="Component", 
                          facet_col="Bead", title="Mass Input per kg Product")
        st.plotly_chart(fig_mass, use_container_width=True)

    # --- TAB 4: SCALING & SENSITIVITY ---
    with tab4:
        st.header("Scaling & Sensitivity")
        
        # 1. Batch Scaling (Corrected to 100kg, Power Law)
        st.subheader("1. Batch Size Scaling Projection")
        st.markdown("Projected reduction in GWP as production scales from Lab (g) to Pilot (kg).")
        
        # Range from 0.001 kg (1g) to 100 kg
        batch_sizes = np.logspace(np.log10(0.0005), np.log10(100), num=20)
        ref_lab_batch = 0.0005 # 0.5g lab batch
        scaling_exponent = 0.6 # Typical rule of thumb for equipment scaling (0.6 power rule)
        
        scale_rows = []
        for rid in unique_routes:
            base_res, _ = calculate_impacts(rid, EF_DF, ROUTES_DF, 1.0, 0.0)
            base_elec_total = base_res["Electricity kWh"]
            base_chem_gwp = base_res["Non-Electric GWP"]
            elec_ef = base_res["Electricity EF Used"]
            
            for b_size in batch_sizes:
                # Scale Factor Calculation: Energy/kg decreases as batch size increases
                # E_spec_new = E_spec_lab * (Batch_new / Batch_lab)^(p - 1)
                # If p=0.6, exponent is -0.4
                scale_factor = (b_size / ref_lab_batch) ** (scaling_exponent - 1)
                
                # Apply scaling only to electricity (Chemicals remain roughly linear per kg)
                new_elec_spec = base_elec_total * scale_factor
                
                # Floor the electricity scaling (cannot go to zero, assume industrial efficiency limit ~5% of lab)
                min_limit = base_elec_total * 0.05 
                new_elec_spec = max(new_elec_spec, min_limit)
                
                new_gwp = (new_elec_spec * elec_ef) + base_chem_gwp
                
                scale_rows.append({"Batch Size (kg)": b_size, "Bead": base_res["name"], "Estimated GWP": new_gwp})
                
        fig_scale = px.line(pd.DataFrame(scale_rows), x="Batch Size (kg)", y="Estimated GWP", color="Bead", 
                            log_x=True, log_y=True, title="Projected GWP vs Batch Size (Power Law Model)")
        
        # Add Industrial Target Zone
        fig_scale.add_vrect(x0=10.0, x1=100.0, fillcolor="green", opacity=0.1, annotation_text="Pilot/Ind. Scale")
        st.plotly_chart(fig_scale, use_container_width=True)

        st.divider()

        # 2. Grid Sensitivity
        st.subheader("2. Grid Intensity Sensitivity")
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
                           title="Total GWP vs Grid Carbon Intensity")
        st.plotly_chart(fig_sens, use_container_width=True)

    # --- TAB 5: SYSTEM BOUNDARY ---
    with tab5:
        st.header("System Boundary")
        st.markdown("Visual representation of the Cradle-to-Gate Life Cycle Assessment scope.")
        
        try:
            diag = plot_system_boundary()
            st.graphviz_chart(diag)
        except Exception as e:
            st.error(f"Could not render diagram (Graphviz missing?): {e}")

    # --- TAB 6: LITERATURE ---
    with tab6:
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

    # --- TAB 7: AI INSIGHTS ---
    with tab7:
        st.header("🤖 AI Insights")
        st.markdown("Ask detailed questions about the LCA results, hotspots, or optimization strategies.")
        
        # Quick Chips/Buttons (No Dropdown)
        st.write("Suggested Questions:")
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        q_clicked = None
        
        if col_btn1.button("🔥 Identify Hotspots"): q_clicked = "What are the major environmental hotspots in these beads?"
        if col_btn2.button("🏭 Scale-up Advice"): q_clicked = "How can we reduce the impact during industrial scale-up?"
        if col_btn3.button("🆚 Compare Beads"): q_clicked = "Compare the Ref-Bead and U@Bead in terms of efficiency."

        # Text Input
        user_text = st.text_area("Or type your own question here:", height=100)
        
        final_q = user_text if user_text else q_clicked
        
        # Context Selection
        context_opt = st.radio("Focus Answer On:", ["Both Beads", "Ref-Bead Only", "MOF Bead Only"], horizontal=True)
        
        if st.button("Analyze with AI", type="primary"):
            if final_q:
                # Filter context based on selection
                context_data = results_list
                if context_opt == "Ref-Bead Only": context_data = [r for r in results_list if r['id'] == ID_REF]
                if context_opt == "MOF Bead Only": context_data = [r for r in results_list if r['id'] == ID_MOF]

                with st.spinner("AI is analyzing data..."):
                    answer = get_ai_insight(context_data, final_q)
                    st.markdown("### 💡 AI Response")
                    st.success(answer)
            else:
                st.warning("Please select or type a question.")

if __name__ == "__main__":
    main()
