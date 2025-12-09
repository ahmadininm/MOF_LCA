import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

# -----------------------------------------------------------------------------
# CONSTANTS & SETUP
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# Route IDs matching the CSV
ID_REF = "ref"
ID_MOF = "mof"

st.set_page_config(page_title="Screening LCA Explorer", layout="wide")

# -----------------------------------------------------------------------------
# DATA LOADING
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    """Load all necessary CSV files from the data directory."""
    try:
        ef = pd.read_csv(DATA_DIR / "emission_factors.csv")
        routes = pd.read_csv(DATA_DIR / "lca_routes.csv")
        perf = pd.read_csv(DATA_DIR / "performance.csv")
        lit = pd.read_csv(DATA_DIR / "literature.csv")
        return ef, routes, perf, lit
    except Exception as e:
        st.error(f"Error loading data: {e}. Please ensure CSV files are in the 'data' folder.")
        return None, None, None, None

EF_DF, ROUTES_DF, PERF_DF, LIT_DF = load_data()

# -----------------------------------------------------------------------------
# CALCULATION ENGINE
# -----------------------------------------------------------------------------
def calculate_impacts(route_id, ef_df, routes_df):
    """
    Calculates GWP for a given route based on mass balances and emission factors.
    Returns a dictionary of results and a dataframe of contributions.
    """
    # Filter for the specific route
    route_data = routes_df[routes_df["route_id"] == route_id].copy()
    
    if route_data.empty:
        return None, None

    # Get Electricity Info (Same for all rows in a route usually)
    elec_kwh = route_data.iloc[0]["electricity_kwh_per_fu"]
    elec_source = route_data.iloc[0]["electricity_source"]
    
    # Get Electricity EF
    ef_elec_row = ef_df[ef_df["reagent_name"] == elec_source]
    ef_elec = float(ef_elec_row["GWP_kgCO2_per_kg"].iloc[0]) if not ef_elec_row.empty else 0.0
    
    # Calculate Electricity GWP
    gwp_elec = elec_kwh * ef_elec

    # Calculate Reagent GWP
    contributions = []
    
    # Add Electricity as a component
    contributions.append({
        "Component": "Electricity",
        "Category": "Electricity",
        "Mass (kg)": 0.0,
        "GWP (kg CO2e)": gwp_elec
    })

    total_reagent_gwp = 0.0

    for _, row in route_data.iterrows():
        reagent = row["reagent_name"]
        mass = row["mass_kg_per_fu"]
        
        # Lookup EF
        ef_row = ef_df[ef_df["reagent_name"] == reagent]
        ef_val = float(ef_row["GWP_kgCO2_per_kg"].iloc[0]) if not ef_row.empty else 0.0
        
        gwp_val = mass * ef_val
        total_reagent_gwp += gwp_val
        
        # Categorize for charts
        if reagent in ["Chitosan", "PDChNF"]:
            cat = "Polymers"
        elif reagent in ["Zirconium tetrachloride", "2-Aminoterephthalic acid", "Formic acid (88%)", "Ethanol"]:
            cat = "MOF Reagents"
        else:
            cat = "Solvents/Other"
            
        contributions.append({
            "Component": reagent,
            "Category": cat,
            "Mass (kg)": mass,
            "GWP (kg CO2e)": gwp_val
        })

    total_gwp = gwp_elec + total_reagent_gwp
    
    results = {
        "id": route_id,
        "name": route_data.iloc[0]["route_name"],
        "Total GWP": total_gwp,
        "Electricity GWP": gwp_elec,
        "Non-Electric GWP": total_reagent_gwp,
        "Electricity kWh": elec_kwh
    }
    
    return results, pd.DataFrame(contributions)

# -----------------------------------------------------------------------------
# PLOTTING FUNCTIONS
# -----------------------------------------------------------------------------
def plot_system_boundary():
    """Draws the system boundary (Figure 10 replacement) using Plotly."""
    fig = go.Figure()

    # Nodes
    x_nodes = [0, 1, 2, 3]
    y_nodes = [1, 1, 1, 1]
    labels = ["Raw Materials", "Lab Gate", "Synthesis<br>(Elec. Intensive)", "Bead Product"]
    
    fig.add_trace(go.Scatter(
        x=x_nodes, y=y_nodes,
        mode="markers+text",
        marker=dict(size=50, color=["#D3D3D3", "#FFD700", "#FF6347", "#90EE90"]),
        text=labels, textposition="bottom center"
    ))

    # Edges
    annotations = [
        dict(x=0.5, y=1, xref="x", yref="y", text="Transport (Excluded)", showarrow=True, arrowhead=2, ax=0, ay=1),
        dict(x=1.5, y=1, xref="x", yref="y", text="Inputs", showarrow=True, arrowhead=2, ax=1, ay=1),
        dict(x=2.5, y=1, xref="x", yref="y", text="Processing", showarrow=True, arrowhead=2, ax=2, ay=1),
        dict(x=2, y=0.5, xref="x", yref="y", text="Emissions (CO2)", showarrow=True, arrowhead=2, ax=2, ay=1, ayref="y", axref="x"),
    ]

    fig.update_layout(
        title="Figure 10: System Boundary (Gate-to-Gate)",
        xaxis=dict(showgrid=False, zeroline=False, visible=False, range=[-0.5, 3.5]),
        yaxis=dict(showgrid=False, zeroline=False, visible=False, range=[0, 1.5]),
        annotations=annotations,
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

# -----------------------------------------------------------------------------
# MAIN APP LAYOUT
# -----------------------------------------------------------------------------
def main():
    if EF_DF is None: return

    st.title("Screening LCA: Ref-Bead vs U@Bead")
    st.markdown("""
    [cite_start]**Data Source:** All calculations are strictly derived from the *Supplementary Information* calculations provided in the uploaded text [cite: 114-286].
    
    **Scope:** Gate-to-gate screening LCA focused on laboratory synthesis conditions.
    """)

    # --- Pre-Calculate Results for Both Routes ---
    res_ref, df_ref = calculate_impacts(ID_REF, EF_DF, ROUTES_DF)
    res_mof, df_mof = calculate_impacts(ID_MOF, EF_DF, ROUTES_DF)

    # --- Get Adsorption Capacity ---
    cap_ref = float(PERF_DF[PERF_DF["route_id"] == ID_REF]["capacity_mg_g"].iloc[0]) # 77
    cap_mof = float(PERF_DF[PERF_DF["route_id"] == ID_MOF]["capacity_mg_g"].iloc[0]) # 116

    # --- Calculate FU2 (Per g Cu) ---
    # Total GWP (kg CO2) / (Capacity (g Cu/kg bead))
    gwp_fu2_ref = res_ref["Total GWP"] / cap_ref
    gwp_fu2_mof = res_mof["Total GWP"] / cap_mof

    # --- TABS ---
    tab1, tab2, tab3 = st.tabs(["LCA Results & Comparison", "Inventory & System", "Literature Benchmarks"])

    # -------------------------------------------------------------------------
    # TAB 1: RESULTS
    # -------------------------------------------------------------------------
    with tab1:
        st.header("LCA Results")
        
        # Comparison Dataframe
        comp_data = {
            "Metric": ["Electricity Use (kWh/kg)", "Total GWP (kg CO2e/kg bead)", "Non-Electric GWP (kg CO2e/kg)", "GWP per g Cu Removed (kg CO2e/g)"],
            "Ref-Bead": [res_ref["Electricity kWh"], res_ref["Total GWP"], res_ref["Non-Electric GWP"], gwp_fu2_ref],
            "U@Bead": [res_mof["Electricity kWh"], res_mof["Total GWP"], res_mof["Non-Electric GWP"], gwp_fu2_mof]
        }
        st.dataframe(pd.DataFrame(comp_data).style.format("{:.2e}", subset=["Ref-Bead", "U@Bead"]))

        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("1. Total GWP (FU1)")
            # Comparison Plot FU1
            df_fu1 = pd.DataFrame([
                {"Bead": "Ref-Bead", "GWP": res_ref["Total GWP"]},
                {"Bead": "U@Bead", "GWP": res_mof["Total GWP"]}
            ])
            fig_fu1 = px.bar(df_fu1, x="Bead", y="GWP", color="Bead", title="Total GWP per kg Bead", barmode="group")
            st.plotly_chart(fig_fu1, use_container_width=True)
            st.caption("Dominated by electricity (~99%).")

        with col2:
            st.subheader("2. Performance Normalized (FU2)")
            # Comparison Plot FU2
            df_fu2 = pd.DataFrame([
                {"Bead": "Ref-Bead", "GWP": gwp_fu2_ref},
                {"Bead": "U@Bead", "GWP": gwp_fu2_mof}
            ])
            fig_fu2 = px.bar(df_fu2, x="Bead", y="GWP", color="Bead", title="GWP per g Cu Removed", barmode="group")
            st.plotly_chart(fig_fu2, use_container_width=True)
            st.caption("Gap narrows due to higher capacity of MOF bead (116 vs 77 mg/g).")

        st.divider()
        st.subheader("3. Non-Electric Impacts (Chemicals Only)")
        
        # Combine Non-Electric Data
        ne_ref = df_ref[df_ref["Category"] != "Electricity"].copy()
        ne_ref["Bead"] = "Ref-Bead"
        ne_mof = df_mof[df_mof["Category"] != "Electricity"].copy()
        ne_mof["Bead"] = "U@Bead"
        
        df_ne = pd.concat([ne_ref, ne_mof])
        
        # Grouped bar chart by component
        fig_ne = px.bar(df_ne, x="Bead", y="GWP (kg CO2e)", color="Component", 
                        title="Chemical Impacts Only (Excluding Electricity)", barmode="group")
        st.plotly_chart(fig_ne, use_container_width=True)
        st.caption("Shows the additional burden of MOF reagents (Ethanol, Formic, ZrCl4).")

    # -------------------------------------------------------------------------
    # TAB 2: INVENTORY
    # -------------------------------------------------------------------------
    with tab2:
        st.header("Inventory Analysis")
        
        st.subheader("Reagent Mass per kg Bead")
        # Mass Chart
        df_mass = df_ne.copy() # Reuse non-electric DF which has masses
        fig_mass = px.bar(df_mass, x="Bead", y="Mass (kg)", color="Component", barmode="group", title="Mass Inventory per kg")
        st.plotly_chart(fig_mass, use_container_width=True)
        
        st.subheader("Electricity Breakdown")
        # Hardcoded breakdown from text citations for visualization
        elec_data = [
            {"Step": "Microfluidisation", "kWh": 1240, "Bead": "Ref-Bead"},
            {"Step": "Hotplate (Mix/Cross)", "kWh": 18760, "Bead": "Ref-Bead"},
            {"Step": "Freeze Drying", "kWh": 73600, "Bead": "Ref-Bead"},
            
            {"Step": "Support Prep (Total)", "kWh": 81500, "Bead": "U@Bead"},
            {"Step": "MOF Synthesis (Stir)", "kWh": 25000, "Bead": "U@Bead"},
            {"Step": "MOF Freeze Drying", "kWh": 49100, "Bead": "U@Bead"},
        ]
        fig_elec = px.bar(pd.DataFrame(elec_data), y="Step", x="kWh", color="Step", orientation='h', facet_col="Bead", title="Electricity Consumption by Step")
        st.plotly_chart(fig_elec, use_container_width=True)
        st.caption("Freeze drying is the primary hotspot.")
        
        st.subheader("System Boundary")
        st.plotly_chart(plot_system_boundary(), use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 3: LITERATURE
    # -------------------------------------------------------------------------
    with tab3:
        st.header("Literature Comparison (Figure 8)")
        
        # Prepare Data
        # Add This Work results to the Lit DF
        this_work = [
            {"Material": "Ref-Bead (This Work)", "GWP_kgCO2_per_kg": res_ref["Total GWP"], "Type": "This Work"},
            {"Material": "U@Bead (This Work)", "GWP_kgCO2_per_kg": res_mof["Total GWP"], "Type": "This Work"}
        ]
        lit_plot_df = pd.concat([LIT_DF, pd.DataFrame(this_work)])
        
        fig_lit = px.bar(lit_plot_df, x="Material", y="GWP_kgCO2_per_kg", color="Type", 
                         log_y=True, title="GWP Comparison (Log Scale)",
                         color_discrete_map={"This Work": "red", "Literature": "blue"})
        st.plotly_chart(fig_lit, use_container_width=True)
        st.caption("Note: 'This Work' values are high due to unscaled lab electricity allocation.")

if __name__ == "__main__":
    main()
