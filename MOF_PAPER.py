import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import numpy as np
import io
import graphviz

# Attempt to import OpenAI
try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    OpenAI = None
    _OPENAI_AVAILABLE = False

# -----------------------------------------------------------------------------
# EMBEDDED DATA (Copied from your files)
# -----------------------------------------------------------------------------
CSV_EMISSIONS = """reagent_name,GWP_kgCO2_per_kg,source_note
Chitosan,41.8,"Table 1: Huang et al. 2025"
PDChNF,36.65,"Table 1: Huang et al. 2025"
Zirconium tetrachloride,5.0,"Table 1: Proxy based on Osterwalder et al."
2-Aminoterephthalic acid,1.98,"Table 1: Proxy based on PTA"
Formic acid (88%),2.51,"Table 1: Petrochemical route"
Ethanol,1.24,"Table 1: Fossil route"
Acetic acid,1.20,"Table 1: Nicholson et al."
Sodium hydroxide,0.83,"Table 1: Rincon et al."
Epichlorohydrin,1.17,"Table 1: Madej et al."
Sulfuric acid,0.14,"Table 1: City of Winnipeg"
Electricity (Canada),0.1197,"Table 1: Climate Transparency Report"
"""

CSV_ROUTES = """route_id,route_name,product_name,reagent_name,mass_kg_per_fu,electricity_kwh_per_fu,electricity_source,notes
ref,Ref-Bead (Polymer),PDChNF-Chitosan Bead,Chitosan,0.8571,93600,Electricity (Canada),"Mass fraction 6:1 Chitosan:PDChNF. Elec: 9.36e4 kWh/kg."
ref,Ref-Bead (Polymer),PDChNF-Chitosan Bead,PDChNF,0.1429,93600,Electricity (Canada),""
ref,Ref-Bead (Polymer),PDChNF-Chitosan Bead,Acetic acid,0.150,93600,Electricity (Canada),"Approx 0.15 kg/kg bead."
mof,U@Bead (MOF-Functionalised),UiO-66-NH2 Composite,Chitosan,0.7457,156000,Electricity (Canada),"Support scaled by 0.87 (13wt% MOF). Elec: 1.56e5 kWh/kg."
mof,U@Bead (MOF-Functionalised),UiO-66-NH2 Composite,PDChNF,0.1243,156000,Electricity (Canada),"Support scaled by 0.87."
mof,U@Bead (MOF-Functionalised),UiO-66-NH2 Composite,Acetic acid,0.1305,156000,Electricity (Canada),"Support scaled by 0.87."
mof,U@Bead (MOF-Functionalised),UiO-66-NH2 Composite,Zirconium tetrachloride,1.0833,156000,Electricity (Canada),"0.65g per 0.6g batch -> 1.08 kg/kg."
mof,U@Bead (MOF-Functionalised),UiO-66-NH2 Composite,2-Aminoterephthalic acid,0.7833,156000,Electricity (Canada),"0.47g per 0.6g batch -> 0.78 kg/kg."
mof,U@Bead (MOF-Functionalised),UiO-66-NH2 Composite,Formic acid (88%),30.42,156000,Electricity (Canada),"17mL (20.7g) per 0.6g batch -> 30.42 kg/kg."
mof,U@Bead (MOF-Functionalised),UiO-66-NH2 Composite,Ethanol,65.75,156000,Electricity (Canada),"50mL (39.5g) per 0.6g batch -> 65.75 kg/kg."
"""

CSV_LIT = """Material,GWP_kgCO2_per_kg,Source,Type
"UiO-66-NH2 (solvothermal 1)",353,"Luo et al. 2021",Literature
"UiO-66-NH2 (solvothermal 2)",180,"Luo et al. 2021",Literature
"UiO-66-NH2 (aqueous)",43,"Luo et al. 2021",Literature
"UiO-66 (Zr) Commercial",273.8,"Dutta et al. 2024",Literature
"Activated carbon (coal)",18.28,"Gu et al. 2018",Literature
"Activated carbon (wood)",8.6,"Gu et al. 2018",Literature
"Ni-Fe LDH / chitosan (coal)",62.58,"Bisaria et al. 2023",Literature
"Ni-Fe LDH / chitosan (renewables)",31.21,"Bisaria et al. 2023",Literature
"Biochar (meta-analysis)",1.2,"Arfasa and Tilahun 2025",Literature
"Biomass-derived adsorbents",2.85,"Arfasa and Tilahun 2025",Literature
"MOF adsorbents (mean)",25,"Arfasa and Tilahun 2025",Literature
"""

CSV_PERF = """material_id,route_id,capacity_mg_g,notes
mat_ref,ref,77,"Capacity from Figure 4d: 77 mg/g"
mat_mof,mof,116,"Capacity from Figure 4d: 116 mg/g"
"""

# -----------------------------------------------------------------------------
# CONSTANTS & SETUP
# -----------------------------------------------------------------------------
st.set_page_config(page_title="LCA Explorer: Ref-Bead vs U@Bead", layout="wide")

# Route IDs
ID_REF = "ref"
ID_MOF = "mof"

# -----------------------------------------------------------------------------
# AI HELPER FUNCTION
# -----------------------------------------------------------------------------
def get_ai_insight(context_data, user_question):
    if not _OPENAI_AVAILABLE:
        return "Error: OpenAI library not installed."
    
    api_key = st.secrets.get("openai_api_key2")
    if not api_key:
        return "Error: API Key 'openai_api_key2' not found in secrets."

    client = OpenAI(api_key=api_key)
    
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
    - Explain *why* certain impacts are high (e.g., electricity in lab scale due to freeze drying).
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
# DATA LOADING
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    """Load data from embedded strings."""
    return (
        pd.read_csv(io.StringIO(CSV_EMISSIONS)),
        pd.read_csv(io.StringIO(CSV_ROUTES)),
        pd.read_csv(io.StringIO(CSV_PERF)),
        pd.read_csv(io.StringIO(CSV_LIT))
    )

def reset_data():
    ef, routes, perf, lit = load_data()
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

if "ef_df" not in st.session_state:
    reset_data()

EF_DF = st.session_state["ef_df"]
ROUTES_DF = st.session_state["routes_df"]
PERF_DF = st.session_state["perf_df"]
LIT_DF = st.session_state["lit_df"]

# -----------------------------------------------------------------------------
# CALCULATION ENGINE
# -----------------------------------------------------------------------------
def calculate_impacts(route_id, ef_df, routes_df, efficiency_factor=1.0, recycling_rate=0.0, yield_rate=100.0, transport_pct=0.0):
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
# FIGURES & PLOTS
# -----------------------------------------------------------------------------
def plot_sankey_diagram(results_list):
    """Generates two Stacked Sankey diagrams."""
    figs = []
    
    for res in results_list:
        elec_gwp = res["Electricity GWP"]
        chem_gwp = res["Non-Electric GWP"]
        total_gwp = res["Total GWP"]
        
        labels = ["Electricity", "Chemicals", "Synthesis", f"Total: {res['name']}"]
        colors = ["#FFD700", "#90EE90", "#87CEFA", "#FF6347"]
        
        fig = go.Figure(data=[go.Sankey(
            node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=labels, color=colors),
            link=dict(source=[0, 1, 2], target=[2, 2, 3], value=[elec_gwp, chem_gwp, total_gwp], 
                      color=["rgba(255, 215, 0, 0.4)", "rgba(144, 238, 144, 0.4)", "rgba(135, 206, 250, 0.4)"])
        )])
        fig.update_layout(title_text=f"Impact Flow: {res['name']}", font_size=12, height=300)
        figs.append(fig)
        
    return figs

def plot_electricity_breakdown():
    """Generates the stacked bar chart for electricity step breakdown."""
    # Hardcoded breakdown based on lab equipment data
    data = [
        # Ref-Bead
        {"Bead": "Ref-Bead", "Step": "Microfluidization", "kWh_per_kg": 1238},
        {"Bead": "Ref-Bead", "Step": "Mixing & Heating", "kWh_per_kg": 18750},
        {"Bead": "Ref-Bead", "Step": "Freeze Drying", "kWh_per_kg": 73600},
        
        # U@Bead
        {"Bead": "U@Bead", "Step": "Microfluidization", "kWh_per_kg": 825},
        {"Bead": "U@Bead", "Step": "Mixing & Heating", "kWh_per_kg": 12500},
        {"Bead": "U@Bead", "Step": "Freeze Drying (Support)", "kWh_per_kg": 49066},
        {"Bead": "U@Bead", "Step": "MOF Synthesis (Heat)", "kWh_per_kg": 25000},
        {"Bead": "U@Bead", "Step": "Freeze Drying (Final)", "kWh_per_kg": 49066},
    ]
    df_elec = pd.DataFrame(data)
    
    fig = px.bar(df_elec, x="Bead", y="kWh_per_kg", color="Step", 
                 title="Electricity Demand per Process Step", 
                 labels={"kWh_per_kg": "Energy Intensity (kWh/kg)"},
                 color_discrete_sequence=px.colors.qualitative.Pastel)
    return fig

def create_system_boundary_diagram():
    """Creates a Graphviz chart for the system boundary."""
    g = graphviz.Digraph()
    g.attr(rankdir='LR', dpi='100')
    
    with g.subgraph(name='cluster_0') as c:
        c.attr(style='dashed', label='Cradle-to-Gate System Boundary')
        c.node('Raw', 'Raw Materials\n(Chitin, Acids, Salts)', shape='box')
        c.node('Ref', 'Ref-Bead Synthesis\n(Dissolution, MF, FD)', shape='box')
        c.node('MOF', 'MOF Functionalization\n(Solvothermal Growth)', shape='box')
        c.node('Prod', 'Final Adsorbent', shape='ellipse', style='filled', fillcolor='lightblue')
        
        c.edge('Raw', 'Ref', label='Inputs')
        c.edge('Ref', 'MOF', label='Intermediates')
        c.edge('MOF', 'Prod', label='Product')
        
    g.node('Use', 'Water Treatment\n(Use Phase)', shape='diamond', style='dashed')
    g.edge('Prod', 'Use', label='Excluded', style='dotted')
    
    return g

# -----------------------------------------------------------------------------
# MAIN APP
# -----------------------------------------------------------------------------
def main():
    # --- LOGO LAYOUT ---
    # Top layout for logos: UBC (Left), AME (Center), Coventry (Right)
    col_l, col_m, col_r = st.columns([1, 1, 1])
    with col_l:
        st.image("ubc.png", width=150) # Ensure ubc.png is in the folder
    with col_m:
        col_m.write("") # Spacer
        st.image("ame.png", width=150) # Ensure ame.png is in the folder
    with col_r:
        st.image("cov.png", width=150) # Ensure cov.png is in the folder

    st.divider()
    
    st.title("Interactive LCA Explorer: Ref-Bead vs U@Bead")
    st.markdown("**Dashboard:** Adjust scenarios on the left to see how industrial scaling affects carbon footprint.")

    # --- SIDEBAR INPUTS ---
    with st.sidebar:
        st.header("Control Panel")
        if st.button("🔄 Reset to Defaults"):
            reset_data()
            st.rerun()

        st.subheader("Scenario Parameters")
        
        # Grid Intensity
        current_grid_ef = float(EF_DF.loc[EF_DF["reagent_name"] == "Electricity (Canada)", "GWP_kgCO2_per_kg"].iloc[0])
        new_grid_ef = st.slider("⚡ Grid Carbon Intensity", 0.0, 1.0, current_grid_ef, 0.01)
        if new_grid_ef != current_grid_ef:
            st.session_state["ef_df"].loc[st.session_state["ef_df"]["reagent_name"] == "Electricity (Canada)", "GWP_kgCO2_per_kg"] = new_grid_ef

        eff_factor = st.slider("Efficiency Multiplier", 0.1, 1.0, 1.0, help="1.0 = Lab Scale. 0.1 = Industrial Target.")
        recycle_rate = st.slider("Solvent Recovery (%)", 0, 95, 0)
        yield_rate = st.slider("Synthesis Yield (%)", 10, 100, 100)
        transport_overhead = st.slider("Transport Overhead (%)", 0, 50, 0)
        
        st.divider()
        with st.expander("Edit Inputs Data"):
            st.session_state["routes_df"] = st.data_editor(st.session_state["routes_df"], key="ed_routes")

    # --- CALCULATIONS ---
    unique_routes = ROUTES_DF["route_id"].unique()
    results_list = []
    dfs_list = []
    
    for rid in unique_routes:
        res, df = calculate_impacts(rid, EF_DF, ROUTES_DF, eff_factor, recycle_rate, yield_rate, transport_overhead)
        if res:
            results_list.append(res)
            dfs_list.append(df)
            
    if not results_list:
        st.error("No valid routes found.")
        return

    # Summary Data
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
        "📊 LCA Results", 
        "📈 Sensitivity Analysis", 
        "📦 Inventory & Flow", 
        "📚 Literature", 
        "🤖 AI Insights"
    ])

    # --- TAB 1: RESULTS ---
    with tab1:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("Total GWP (Log Scale)")
            fig_log = px.bar(sum_df, x="Bead", y="Total GWP", color="Bead", log_y=True, text_auto='.2s')
            st.plotly_chart(fig_log, use_container_width=True)
        with col2:
            st.subheader("GWP per g Copper Removed")
            fig_fu2 = px.bar(sum_df, x="Bead", y="GWP per g Cu", color="Bead", text_auto='.2f')
            st.plotly_chart(fig_fu2, use_container_width=True)
            
        st.divider()
        st.subheader("Electricity Demand Breakdown")
        st.plotly_chart(plot_electricity_breakdown(), use_container_width=True)

    # --- TAB 2: SENSITIVITY ---
    with tab2:
        st.header("Sensitivity Analysis")
        
        # 1. Batch Scaling
        st.subheader("1. Industrial Scaling Projection")
        st.markdown("Projected reduction in GWP as batch size increases from Lab scale (grams) to Industrial scale (100 kg).")
        
        # Generate Log Space X-axis from 0.0001 kg to 100 kg
        batch_sizes = np.logspace(np.log10(0.0001), np.log10(100), 50) 
        ref_batch_kg = 0.0004 # 0.4g lab scale
        
        scale_rows = []
        for rid in unique_routes:
            base_res, _ = calculate_impacts(rid, EF_DF, ROUTES_DF, 1.0, 0.0)
            base_elec = base_res["Electricity kWh"]
            
            for b_size in batch_sizes:
                # Scaling Law: Power Law approximation E ~ E0 * (m/m0)^-b
                # Simplified user model logic: starts at 100%, decays to 10% at high scale
                # Using a logistic-like decay or simple power law
                ratio = b_size / ref_batch_kg
                scaling_factor = 0.1 + 0.9 * (ratio ** -0.3) # -0.3 is a typical chemical scaling exponent
                
                # Cap scaling at 1.0 (lab scale) for very small batches to avoid infinity
                scaling_factor = min(scaling_factor, 1.5)
                
                new_elec = base_elec * scaling_factor
                new_gwp = (new_elec * base_res["Electricity EF Used"]) + base_res["Non-Electric GWP"]
                
                scale_rows.append({"Batch Size (kg)": b_size, "Bead": base_res["name"], "Projected GWP": new_gwp})
        
        fig_scale = px.line(pd.DataFrame(scale_rows), x="Batch Size (kg)", y="Projected GWP", color="Bead", 
                            log_x=True, log_y=True, title="Scaling Projection (0.1g to 100kg)")
        fig_scale.add_vrect(x0=10, x1=100, fillcolor="green", opacity=0.1, annotation_text="Industrial Goal")
        st.plotly_chart(fig_scale, use_container_width=True)
        
        st.divider()

        # 2. Grid Sensitivity
        st.subheader("2. Grid Intensity Sensitivity")
        sens_rows = []
        for g_name, g_val in st.session_state["custom_grids"].items():
            temp_ef = EF_DF.copy()
            temp_ef.loc[temp_ef["reagent_name"].str.contains("Electricity"), "GWP_kgCO2_per_kg"] = g_val
            for rid in unique_routes:
                res, _ = calculate_impacts(rid, temp_ef, ROUTES_DF, eff_factor, recycle_rate)
                sens_rows.append({"Grid": g_name, "Grid Value": g_val, "Bead": res["name"], "Total GWP": res["Total GWP"]})
        
        df_sens = pd.DataFrame(sens_rows).sort_values("Grid Value")
        fig_sens = px.line(df_sens, x="Grid", y="Total GWP", color="Bead", markers=True)
        st.plotly_chart(fig_sens, use_container_width=True)

    # --- TAB 3: INVENTORY ---
    with tab3:
        st.header("Inventory & Process Flow")
        
        col_i1, col_i2 = st.columns([1, 1])
        with col_i1:
            st.subheader("System Boundary")
            st.graphviz_chart(create_system_boundary_diagram())
            
        with col_i2:
            st.subheader("Impact Contribution")
            df_all = pd.concat([df.assign(Bead=results_list[i]["name"]) for i, df in enumerate(dfs_list)])
            fig_breakdown = px.bar(df_all, x="Bead", y="GWP", color="Component", log_y=True)
            st.plotly_chart(fig_breakdown, use_container_width=True)
            
        st.divider()
        st.subheader("Impact Flows (Sankey)")
        sankey_figs = plot_sankey_diagram(results_list)
        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(sankey_figs[0], use_container_width=True)
        with c2: 
            if len(sankey_figs) > 1: st.plotly_chart(sankey_figs[1], use_container_width=True)

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
        fig_lit = px.bar(lit_combined, x="Material", y="GWP_kgCO2_per_kg", color="Source", log_y=True)
        st.plotly_chart(fig_lit, use_container_width=True)

    # --- TAB 5: AI INSIGHTS ---
    with tab5:
        st.header("🤖 AI Insights")
        
        # Helper to set text area value
        if "ai_question" not in st.session_state: st.session_state.ai_question = ""
        
        def set_q(q): st.session_state.ai_question = q
        
        st.write("Quick Questions:")
        bq1, bq2, bq3, bq4 = st.columns(4)
        if bq1.button("Why is GWP high?"): set_q("Why is the GWP so high compared to literature?")
        if bq2.button("Compare Beads"): set_q("Compare Ref-Bead and U@Bead results.")
        if bq3.button("Hotspots?"): set_q("What is the biggest hotspot?")
        if bq4.button("How to reduce?"): set_q("How can I reduce the carbon footprint?")
        
        user_input = st.text_area("Ask any question about the LCA results:", value=st.session_state.ai_question, height=100)
        
        if st.button("Analyze Results"):
            if user_input:
                with st.spinner("AI is analyzing..."):
                    answer = get_ai_insight(results_list, user_input)
                    st.markdown("### Analysis")
                    st.info(answer)
            else:
                st.warning("Please enter a question.")

if __name__ == "__main__":
    main()
