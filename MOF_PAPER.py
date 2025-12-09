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
# LOGO DISPLAY
# -----------------------------------------------------------------------------
def render_header():
    col1, col2, col3 = st.columns([1, 1, 1])
    
    # Placeholders for logos - assuming they exist in DATA_DIR
    # You need to ensure ubc.png, ame.png, cov.png are in the 'data' folder
    with col1:
        if (DATA_DIR / "ubc.png").exists():
            st.image(str(DATA_DIR / "ubc.png"), width=150)
        else:
            st.write("UBC")
            
    with col2:
        if (DATA_DIR / "ame.png").exists():
            st.image(str(DATA_DIR / "ame.png"), width=150)
        else:
            st.write("AME")
            
    with col3:
        if (DATA_DIR / "cov.png").exists():
            st.image(str(DATA_DIR / "cov.png"), width=150)
        else:
            st.write("Coventry")

# -----------------------------------------------------------------------------
# AI HELPER FUNCTION
# -----------------------------------------------------------------------------
def get_ai_insight(context_data, user_question, current_params, focus_route="All"):
    """
    Sends calculation results + user question to OpenAI for analysis.
    """
    if not _OPENAI_AVAILABLE:
        return "Error: OpenAI library not installed."
    
    api_key = st.secrets.get("openai_api_key2")
    if not api_key:
        return "Error: API Key 'openai_api_key2' not found in secrets."

    client = OpenAI(api_key=api_key)
    
    # Filter context based on user selection
    if focus_route != "All":
        context_data = [r for r in context_data if r['name'] == focus_route]

    # Prepare a summary of the current results
    context_str = "Current Simulation Results:\n"
    for res in context_data:
        context_str += f"- {res['name']}: Total GWP={res['Total GWP']:.2e} kg CO2e, Elec%={res['Electricity %']:.1f}%\n"
    
    settings_str = f"""
    Current Settings:
    - Grid Intensity: {current_params['grid']} kg CO2/kWh
    - Yield: {current_params['yield']}%
    - Drying Time: {current_params['dry_time']} h
    """

    system_prompt = f"""
    You are an expert in Life Cycle Assessment (LCA) for materials.
    Answer the user's question based strictly on the provided simulation data.
    
    Context:
    {context_str}
    {settings_str}
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
def load_default_data():
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
    ef, routes, perf, lit = load_default_data()
    st.session_state["ef_df"] = ef
    st.session_state["routes_df"] = routes
    st.session_state["perf_df"] = perf
    st.session_state["lit_df"] = lit
    st.session_state["custom_grids"] = {
        "QC Hydro": 0.002, "Canada Avg": 0.1197, "UK Grid": 0.225, 
        "EU Avg": 0.25, "US Avg": 0.38, "China Grid": 0.58
    }

if "ef_df" not in st.session_state:
    reset_data()

# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Control Panel")
    if st.button("🔄 Reset Defaults"):
        reset_data()
        st.experimental_rerun()

    st.divider()
    st.subheader("1. Scenario Parameters")
    
    # Grid
    current_grid = float(st.session_state["ef_df"].loc[st.session_state["ef_df"]["reagent_name"]=="Electricity (Canada)", "GWP_kgCO2_per_kg"].iloc[0])
    new_grid = st.slider("⚡ Grid Intensity (kg CO2/kWh)", 0.0, 1.0, current_grid, 0.01)
    if new_grid != current_grid:
        st.session_state["ef_df"].loc[st.session_state["ef_df"]["reagent_name"]=="Electricity (Canada)", "GWP_kgCO2_per_kg"] = new_grid

    # Yield
    st.write("**🧪 Synthesis Yield**")
    yield_rate = st.slider("Global Yield (%)", 10, 100, 100)

    # Drying
    st.write("**⏳ Drying Duration**")
    dry_time = st.slider("Freeze Drying Time (h)", 1, 48, 16)

    # Solvent
    st.write("**💧 Solvent Usage**")
    solvent_vol_factor = st.slider("Volume Multiplier", 0.1, 2.0, 1.0)

    # Recycling
    st.write("**♻️ Solvent Recovery**")
    recycle_rate = st.slider("Recycling Rate (%)", 0, 95, 0)

    # Transport
    st.write("**🚛 Logistics**")
    transport_overhead = st.slider("Transport Overhead (%)", 0, 50, 0)

    st.divider()
    st.subheader("2. Input Tables")
    with st.expander("Edit Data"):
        st.session_state["ef_df"] = st.data_editor(st.session_state["ef_df"], num_rows="dynamic")
        st.session_state["routes_df"] = st.data_editor(st.session_state["routes_df"], num_rows="dynamic")
        st.session_state["perf_df"] = st.data_editor(st.session_state["perf_df"], num_rows="dynamic")

# Shortcuts
EF_DF = st.session_state["ef_df"]
ROUTES_DF = st.session_state["routes_df"]
PERF_DF = st.session_state["perf_df"]
LIT_DF = st.session_state["lit_df"]

current_params = {
    "grid": new_grid, "yield": yield_rate, 
    "dry_time": dry_time, "solvent_vol": solvent_vol_factor
}

# -----------------------------------------------------------------------------
# CALCULATION
# -----------------------------------------------------------------------------
def calculate_impacts(route_id, ef_df, routes_df, dry_time_h, solvent_factor, recycling_rate, yield_rate, transport_pct):
    route_data = routes_df[routes_df["route_id"] == route_id].copy()
    if route_data.empty: return None, None

    yield_multiplier = 1.0 / (yield_rate / 100.0)

    # Electricity Scaling (Linear with drying time approx)
    base_elec_kwh = float(route_data.iloc[0]["electricity_kwh_per_fu"])
    time_factor = dry_time_h / 16.0
    elec_kwh = base_elec_kwh * time_factor * yield_multiplier
    
    elec_src = route_data.iloc[0]["electricity_source"]
    ef_elec = float(ef_df[ef_df["reagent_name"] == elec_src]["GWP_kgCO2_per_kg"].iloc[0])
    gwp_elec = elec_kwh * ef_elec

    contributions = [{"Component": "Electricity", "Category": "Electricity", "Mass (kg)": 0.0, "GWP": gwp_elec}]
    total_reagent_gwp = 0.0

    for _, row in route_data.iterrows():
        reagent = row["reagent_name"]
        base_mass = float(row["mass_kg_per_fu"])
        mass_needed = base_mass * yield_multiplier

        is_solvent = reagent in ["Ethanol", "Formic acid (88%)", "Acetic acid"]
        if is_solvent: mass_needed *= solvent_factor
        effective_mass = mass_needed * (1 - (recycling_rate/100)) if is_solvent else mass_needed
        
        ef_val = float(ef_df[ef_df["reagent_name"] == reagent]["GWP_kgCO2_per_kg"].iloc[0])
        gwp_val = effective_mass * ef_val
        total_reagent_gwp += gwp_val
        
        if reagent in ["Chitosan", "PDChNF"]: cat = "Polymers"
        elif reagent in ["Zirconium tetrachloride", "2-Aminoterephthalic acid", "Formic acid (88%)", "Ethanol"]: cat = "MOF Reagents"
        else: cat = "Solvents/Other"
            
        contributions.append({"Component": reagent, "Category": cat, "Mass (kg)": effective_mass, "GWP": gwp_val})

    trans_gwp = (gwp_elec + total_reagent_gwp) * (transport_pct / 100.0)
    if trans_gwp > 0:
        contributions.append({"Component": "Transport", "Category": "Logistics", "Mass (kg)": 0.0, "GWP": trans_gwp})

    return {
        "id": route_id, "name": route_data.iloc[0]["route_name"],
        "Total GWP": gwp_elec + total_reagent_gwp + trans_gwp,
        "Electricity GWP": gwp_elec, "Non-Electric GWP": total_reagent_gwp + trans_gwp,
        "Electricity %": (gwp_elec / (gwp_elec + total_reagent_gwp + trans_gwp)) * 100
    }, pd.DataFrame(contributions)

# -----------------------------------------------------------------------------
# PLOTTING FUNCTIONS
# -----------------------------------------------------------------------------
def plot_sankey_diagram(res):
    """Creates a Sankey diagram for a single route result."""
    labels = ["Electricity", "Chemicals", "Process", "Total GWP"]
    colors = ["#FFD700", "#90EE90", "#87CEFA", "#FF6347"]
    
    fig = go.Figure(data=[go.Sankey(
        node = dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=labels, color=colors),
        link = dict(
            source=[0, 1, 2], target=[2, 2, 3],
            value=[res['Electricity GWP'], res['Non-Electric GWP'], res['Total GWP']],
            color=["rgba(255, 215, 0, 0.4)", "rgba(144, 238, 144, 0.4)", "rgba(135, 206, 250, 0.4)"]
        )
    )])
    fig.update_layout(title_text=f"Impact Flow: {res['name']}", height=300, margin=dict(l=10, r=10, t=40, b=10))
    return fig

def plot_system_boundary():
    """Simple flowchart for System Boundary."""
    fig = go.Figure()
    
    # Boxes
    fig.add_shape(type="rect", x0=0, y0=0, x1=1, y1=1, line=dict(color="RoyalBlue"), fillcolor="LightSkyBlue")
    fig.add_annotation(x=0.5, y=0.5, text="Raw Materials", showarrow=False)
    
    fig.add_shape(type="rect", x0=2, y0=0, x1=3, y1=1, line=dict(color="RoyalBlue"), fillcolor="LightSkyBlue")
    fig.add_annotation(x=2.5, y=0.5, text="Lab Synthesis", showarrow=False)
    
    fig.add_shape(type="rect", x0=4, y0=0, x1=5, y1=1, line=dict(color="Green"), fillcolor="LightGreen")
    fig.add_annotation(x=4.5, y=0.5, text="Final Bead", showarrow=False)
    
    # Arrows
    fig.add_annotation(x=2, y=0.5, ax=1, ay=0.5, xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=2)
    fig.add_annotation(x=4, y=0.5, ax=3, ay=0.5, xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=2)
    
    # Inputs
    fig.add_annotation(x=2.5, y=1, ax=2.5, ay=1.5, xref="x", yref="y", axref="x", ayref="y", text="Electricity", showarrow=True)
    
    fig.update_layout(
        title="Figure: Simple System Boundary (Gate-to-Gate)",
        xaxis=dict(range=[-0.5, 5.5], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(range=[-0.5, 2], showgrid=False, zeroline=False, visible=False),
        height=200
    )
    return fig

def plot_electricity_steps():
    """New Figure: Breakdown of Electricity by Step (Hardcoded from Paper Data)."""
    data = [
        {"Step": "Microfluidizer", "kWh/kg": 1240, "Bead": "Ref-Bead"},
        {"Step": "Mixing/Crosslink", "kWh/kg": 18760, "Bead": "Ref-Bead"},
        {"Step": "Freeze Drying (Ref)", "kWh/kg": 73600, "Bead": "Ref-Bead"},
        {"Step": "Support Prep", "kWh/kg": 81500, "Bead": "U@Bead"},
        {"Step": "MOF Synthesis", "kWh/kg": 25000, "Bead": "U@Bead"},
        {"Step": "Freeze Drying (MOF)", "kWh/kg": 49100, "Bead": "U@Bead"},
    ]
    df = pd.DataFrame(data)
    fig = px.bar(df, x="kWh/kg", y="Step", color="Bead", orientation='h', 
                 title="Electricity Demand per Process Step", barmode="group")
    return fig

# -----------------------------------------------------------------------------
# MAIN APP
# -----------------------------------------------------------------------------
def main():
    render_header()
    st.title("Interactive LCA Explorer: Ref-Bead vs U@Bead")
    st.markdown("Use the **Control Panel** to adjust scenarios.")

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
        st.warning("No routes found.")
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
            "Electricity %": r["Electricity %"]
        })
    sum_df = pd.DataFrame(summary_rows)

    # --- TABS ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Results", "📈 Sensitivity", "📦 Inventory", "📚 Literature", "🤖 AI Insights"
    ])

    # --- TAB 1: RESULTS ---
    with tab1:
        st.dataframe(sum_df.style.format({"Total GWP": "{:.2e}", "Non-Electric GWP": "{:.2f}", "GWP per g Cu": "{:.2f}"}))
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(px.bar(sum_df, x="Bead", y="Total GWP", color="Bead", log_y=True, title="Total GWP (Log Scale)", text_auto='.2s'), use_container_width=True)
        with col2:
            st.plotly_chart(px.bar(sum_df, x="Bead", y="GWP per g Cu", color="Bead", title="Performance Normalized (FU2)", text_auto='.2f'), use_container_width=True)

    # --- TAB 2: SENSITIVITY ---
    with tab2:
        st.subheader("1. Grid Intensity")
        grids = st.session_state["custom_grids"]
        sens_rows = []
        for g_name, g_val in grids.items():
            temp_ef = EF_DF.copy()
            temp_ef.loc[temp_ef["reagent_name"].str.contains("Electricity"), "GWP_kgCO2_per_kg"] = g_val
            for rid in unique_routes:
                r, _ = calculate_impacts(rid, temp_ef, ROUTES_DF, dry_time, solvent_vol_factor, recycle_rate, yield_rate, transport_overhead)
                sens_rows.append({"Grid": g_name, "Grid Val": g_val, "Bead": r["name"], "Total GWP": r["Total GWP"]})
        
        st.plotly_chart(px.line(pd.DataFrame(sens_rows).sort_values("Grid Val"), x="Grid", y="Total GWP", color="Bead", markers=True, title="GWP vs Grid Intensity"), use_container_width=True)

        st.divider()
        st.subheader("2. Batch Scaling (1kg - 100kg)")
        st.caption("Simulating the drop in electricity intensity as batch size increases towards pilot scale.")
        
        batch_sizes = np.linspace(1, 100, 20) # 1 to 100 kg linear
        scale_rows = []
        ref_batch_kg = 0.0005 # Lab scale
        
        for rid in unique_routes:
            base_res, _ = calculate_impacts(rid, EF_DF, ROUTES_DF, dry_time, solvent_vol_factor, recycle_rate, yield_rate, 0)
            base_elec = base_res["Electricity kWh"] # This is the HIGH per-kg value from lab
            
            for b_size in batch_sizes:
                # Model: Variable energy (10%) stays constant per kg. Fixed overhead (90%) dilutes.
                # Factor = 0.1 + 0.9 * (RefBatch / NewBatch)
                # Since RefBatch is tiny (0.0005) and NewBatch is huge (1+), this factor drops to ~0.1 quickly.
                scale_factor = 0.1 + 0.9 * (ref_batch_kg / b_size)
                
                new_elec = base_elec * scale_factor
                new_gwp = (new_elec * base_res["Electricity EF Used"]) + base_res["Non-Electric GWP"]
                scale_rows.append({"Batch Size (kg)": b_size, "Bead": base_res["name"], "Estimated GWP": new_gwp})
                
        fig_scale = px.line(pd.DataFrame(scale_rows), x="Batch Size (kg)", y="Estimated GWP", color="Bead", 
                            title="Projected GWP Scaling (1-100 kg)", log_y=True)
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
            st.plotly_chart(px.bar(df_all[df_all["Category"]!="Electricity"], x="Bead", y="GWP", color="Component", title="Chemical Impacts Only"), use_container_width=True)
        with col_i2:
            st.plotly_chart(px.bar(df_all, x="Bead", y="GWP", color="Component", title="Total Breakdown (Log Scale)", log_y=True), use_container_width=True)

        st.subheader("Detailed Electricity Breakdown")
        st.plotly_chart(plot_electricity_steps(), use_container_width=True)

        st.subheader("Mass Inventory")
        fig_mass = px.bar(df_all[df_all["Category"]!="Electricity"], x="Component", y="Mass (kg)", color="Component", facet_col="Bead", title="Mass Input per kg")
        fig_mass.update_yaxes(matches=None, showticklabels=True)
        st.plotly_chart(fig_mass, use_container_width=True)

        st.subheader("Impact Flows")
        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(plot_sankey_diagram(results_list[0]), use_container_width=True)
        with c2: st.plotly_chart(plot_sankey_diagram(results_list[1]), use_container_width=True)
        
        st.subheader("System Boundary")
        st.plotly_chart(plot_system_boundary(), use_container_width=True)

    # --- TAB 4: LITERATURE ---
    with tab4:
        curr_data = [{"Material": f"{r['name']} (This Work)", "GWP_kgCO2_per_kg": r["Total GWP"], "Source": "This Work"} for r in results_list]
        lit_df = pd.concat([LIT_DF, pd.DataFrame(curr_data)])
        st.plotly_chart(px.bar(lit_df, x="Material", y="GWP_kgCO2_per_kg", color="Source", log_y=True, title="Literature Benchmark"), use_container_width=True)

    # --- TAB 5: AI ---
    with tab5:
        st.header("🤖 AI Insights")
        
        # Context Selection
        route_opts = ["All"] + [r['name'] for r in results_list]
        focus = st.radio("Focus Analysis On:", route_opts, horizontal=True)
        
        # Clickable Questions
        col_btns, col_text = st.columns([1, 2])
        user_q = ""
        
        with col_btns:
            st.caption("Quick Ask:")
            if st.button("🔥 Identify Hotspots"): user_q = "What are the top 3 GWP contributors?"
            if st.button("🏭 Scaling Potential"): user_q = "How much will GWP drop if we scale to 100kg?"
            if st.button("⚖️ Compare Beads"): user_q = "Is the MOF bead worth the extra environmental cost?"
            if st.button("🔮 Future Prediction"): user_q = "Predict GWP in 2030 with green energy."

        with col_text:
            final_q = st.text_area("Your Question:", value=user_q, height=100)
            if st.button("Analyze"):
                with st.spinner("Analyzing..."):
                    ans = get_ai_insight(results_list, final_q, current_params, focus)
                    st.markdown(ans)

if __name__ == "__main__":
    main()
