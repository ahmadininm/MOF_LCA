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
LOGO_DIR = BASE_DIR / "assets"  # put cov.png (left) and ubc.png (right) in ./assets

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
        context_str += (
            f"- {res['name']}: "
            f"Total GWP={res['Total GWP']:.2e}, "
            f"Elec GWP={res['Electricity GWP']:.2e}, "
            f"Elec%={res['Electricity %']:.1f}%\n"
        )

    system_prompt = f"""
You are an expert in Life Cycle Assessment (LCA) for materials science.
Use the provided data to answer the user's question.

Context Data:
{context_str}

Guidelines:
- Be concise and scientific.
- Comment on why certain impacts are high (for example electricity at lab scale).
- If the user asks about optimisation, suggest practical process improvements, especially around electricity demand, batch size and grid mix.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_question},
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"AI Error: {e}"


# -----------------------------------------------------------------------------
# ELECTRICITY STEP DATA (LAB + CAPACITY)
# -----------------------------------------------------------------------------
def get_default_elec_steps_df() -> pd.DataFrame:
    """
    Default per-step electricity inventory at lab scale, including:
    - route_id / bead / step
    - lab_batch_kg: actual mass per batch in the experiment
    - kWh_per_kg: lab-scale electricity intensity per kg bead
    - capacity_kg: effective equipment capacity per batch (defaults to lab_batch_kg,
      so there is no scaling unless the user changes it).
    """
    data = [
        # Ref-Bead (PDChNF–chitosan reference bead), lab batch = 0.0004 kg
        {
            "route_id": ID_REF,
            "Bead": "Ref-Bead (Polymer)",
            "Step": "Microfluidiser",
            "kWh_per_kg": 1.24e3,
            "lab_batch_kg": 0.0004,
        },
        {
            "route_id": ID_REF,
            "Bead": "Ref-Bead (Polymer)",
            "Step": "Hotplate mixing",
            "kWh_per_kg": 9.38e3,
            "lab_batch_kg": 0.0004,
        },
        {
            "route_id": ID_REF,
            "Bead": "Ref-Bead (Polymer)",
            "Step": "Hotplate crosslinking",
            "kWh_per_kg": 9.38e3,
            "lab_batch_kg": 0.0004,
        },
        {
            "route_id": ID_REF,
            "Bead": "Ref-Bead (Polymer)",
            "Step": "Freeze-drying",
            "kWh_per_kg": 7.36e4,
            "lab_batch_kg": 0.0004,
        },
        # U@Bead (MOF-functionalised bead), lab batch = 0.0006 kg
        {
            "route_id": ID_MOF,
            "Bead": "U@Bead (MOF-Functionalised)",
            "Step": "Carry-over (Ref support)",
            "kWh_per_kg": 8.15e4,
            "lab_batch_kg": 0.0006,
        },
        {
            "route_id": ID_MOF,
            "Bead": "U@Bead (MOF-Functionalised)",
            "Step": "UiO stirring (Zr step)",
            "kWh_per_kg": 1.25e4,
            "lab_batch_kg": 0.0006,
        },
        {
            "route_id": ID_MOF,
            "Bead": "U@Bead (MOF-Functionalised)",
            "Step": "UiO stirring (linker step)",
            "kWh_per_kg": 1.25e4,
            "lab_batch_kg": 0.0006,
        },
        {
            "route_id": ID_MOF,
            "Bead": "U@Bead (MOF-Functionalised)",
            "Step": "Second freeze-drying",
            "kWh_per_kg": 4.91e4,
            "lab_batch_kg": 0.0006,
        },
    ]
    df = pd.DataFrame(data)
    # No scaling by default: capacity_kg = lab_batch_kg
    df["capacity_kg"] = df["lab_batch_kg"]
    return df


def get_electricity_step_data() -> pd.DataFrame:
    """
    Returns the current per-step electricity table (with lab batch and capacity).
    """
    if "elec_steps_df" not in st.session_state:
        st.session_state["elec_steps_df"] = get_default_elec_steps_df()
    return st.session_state["elec_steps_df"]


# -----------------------------------------------------------------------------
# DATA LOADING & SESSION STATE
# -----------------------------------------------------------------------------
@st.cache_data
def load_default_data():
    """Load default CSV files from disk."""
    def read_safe(path: Path) -> pd.DataFrame:
        if not path.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(path, encoding="utf-8")
        except Exception:
            return pd.read_csv(path, encoding="latin1")

    return (
        read_safe(DATA_DIR / "emission_factors.csv"),
        read_safe(DATA_DIR / "lca_routes.csv"),
        read_safe(DATA_DIR / "performance.csv"),
        read_safe(DATA_DIR / "literature.csv"),
    )


def reset_data():
    """Resets session state to default CSV values and default electricity steps."""
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
        "China Grid": 0.58,
    }
    # Reset electricity steps (lab-scale, no scaling)
    st.session_state["elec_steps_df"] = get_default_elec_steps_df()


# Initialise Session State
if "ef_df" not in st.session_state:
    reset_data()

# Shortcuts
EF_DF = st.session_state["ef_df"]
ROUTES_DF = st.session_state["routes_df"]
PERF_DF = st.session_state["perf_df"]
LIT_DF = st.session_state["lit_df"]


# -----------------------------------------------------------------------------
# HEADER LOGOS
# -----------------------------------------------------------------------------
def render_header_logos():
    """
    Show cov.png on the top left and ubc.png on the top right of all pages.
    """
    cov_path = LOGO_DIR / "cov.png"
    ubc_path = LOGO_DIR / "ubc.png"

    col_left, col_right = st.columns([1, 1])

    with col_left:
        if cov_path.exists():
            st.image(str(cov_path))
    with col_right:
        if ubc_path.exists():
            st.image(str(ubc_path), use_column_width=False)


# -----------------------------------------------------------------------------
# CALCULATION ENGINE
# -----------------------------------------------------------------------------
def calculate_impacts(
    route_id,
    ef_df,
    routes_df,
    efficiency_factor: float = 1.0,
    recycling_rate: float = 0.0,
    yield_rate: float = 100.0,
    transport_pct: float = 0.0,
):
    """Calculates GWP based on session state and efficiency modifiers."""
    route_data = routes_df[routes_df["route_id"] == route_id].copy()
    if route_data.empty:
        return None, None

    yield_multiplier = 1.0 / (yield_rate / 100.0)

    # Electricity (already possibly scaled by capacity utilisation)
    base_elec_kwh = float(route_data.iloc[0]["electricity_kwh_per_fu"])
    elec_kwh = base_elec_kwh * efficiency_factor * yield_multiplier

    elec_source = route_data.iloc[0]["electricity_source"]
    ef_elec_row = ef_df[ef_df["reagent_name"] == elec_source]
    ef_elec = float(ef_elec_row["GWP_kgCO2_per_kg"].iloc[0]) if not ef_elec_row.empty else 0.0
    gwp_elec = elec_kwh * ef_elec

    # Reagents
    contributions = [
        {
            "Component": "Electricity",
            "Category": "Electricity",
            "Mass (kg)": 0.0,
            "GWP": gwp_elec,
        }
    ]
    total_reagent_gwp = 0.0

    for _, row in route_data.iterrows():
        reagent = row["reagent_name"]
        base_mass = float(row["mass_kg_per_fu"])

        # Yield impact
        mass_needed = base_mass * yield_multiplier

        # Recycling impact (for selected solvents)
        is_solvent = reagent in ["Ethanol", "Formic acid (88%)", "Acetic acid"]
        effective_mass = mass_needed * (1 - (recycling_rate / 100.0)) if is_solvent else mass_needed

        ef_row = ef_df[ef_df["reagent_name"] == reagent]
        ef_val = float(ef_row["GWP_kgCO2_per_kg"].iloc[0]) if not ef_row.empty else 0.0

        gwp_val = effective_mass * ef_val
        total_reagent_gwp += gwp_val

        if reagent in ["Chitosan", "PDChNF"]:
            cat = "Polymers"
        elif reagent in ["Zirconium tetrachloride", "2-Aminoterephthalic acid", "Formic acid (88%)", "Ethanol"]:
            cat = "MOF Reagents"
        else:
            cat = "Solvents/Other"

        contributions.append(
            {
                "Component": reagent,
                "Category": cat,
                "Mass (kg)": effective_mass,
                "GWP": gwp_val,
            }
        )

    # Transport
    raw_total_gwp = gwp_elec + total_reagent_gwp
    transport_gwp = raw_total_gwp * (transport_pct / 100.0)

    if transport_gwp > 0:
        contributions.append(
            {
                "Component": "Transport",
                "Category": "Logistics",
                "Mass (kg)": 0.0,
                "GWP": transport_gwp,
            }
        )

    final_total_gwp = raw_total_gwp + transport_gwp

    results = {
        "id": route_id,
        "name": route_data.iloc[0]["route_name"],
        "Total GWP": final_total_gwp,
        "Electricity GWP": gwp_elec,
        "Non-Electric GWP": total_reagent_gwp + transport_gwp,
        "Electricity kWh": elec_kwh,
        "Electricity EF Used": ef_elec,
    }
    return results, pd.DataFrame(contributions)


# -----------------------------------------------------------------------------
# PLOTTING HELPERS
# -----------------------------------------------------------------------------
def plot_sankey_diagram(results_list, route_id=None):
    """Build a simple Sankey diagram for a given route (Ref or MOF)."""
    if not results_list:
        return go.Figure()

    if route_id is not None:
        target_res = next((r for r in results_list if r["id"] == route_id), None)
        if target_res is None:
            target_res = results_list[0]
    else:
        target_res = results_list[0]

    elec_gwp = target_res["Electricity GWP"]
    chem_gwp = target_res["Non-Electric GWP"]
    total_gwp = target_res["Total GWP"]

    labels = ["Electricity Source", "Chemical Supply", "Lab Synthesis", "Total GWP"]
    colors = ["#FFD700", "#90EE90", "#87CEFA", "#FF6347"]

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=labels,
                    color=colors,
                ),
                link=dict(
                    source=[0, 1, 2],
                    target=[2, 2, 3],
                    value=[elec_gwp, chem_gwp, total_gwp],
                    color=[
                        "rgba(255, 215, 0, 0.4)",
                        "rgba(144, 238, 144, 0.4)",
                        "rgba(135, 206, 250, 0.4)",
                    ],
                ),
            )
        ]
    )
    fig.update_layout(
        title_text=f"Impact flow: {target_res['name']}",
        font_size=10,
        height=400,
    )
    return fig


def create_system_boundary_figure() -> go.Figure:
    """
    Simple system boundary diagram for the gate to gate LCA scope.
    """
    fig = go.Figure()

    # Main system boundary box
    fig.add_shape(
        type="rect",
        x0=0.25,
        y0=0.2,
        x1=0.75,
        y1=0.8,
        line=dict(color="black", width=2),
        fillcolor="rgba(144, 238, 144, 0.1)",
    )

    # Upstream box (excluded)
    fig.add_shape(
        type="rect",
        x0=0.02,
        y0=0.35,
        x1=0.20,
        y1=0.65,
        line=dict(color="grey", width=1),
        fillcolor="rgba(200, 200, 200, 0.1)",
    )
    fig.add_annotation(
        x=0.11,
        y=0.5,
        text="Upstream:\nFisheries,\ncrab processing,\nchitin purification\n(excluded)",
        showarrow=False,
        font=dict(size=10),
    )

    # Downstream box (excluded)
    fig.add_shape(
        type="rect",
        x0=0.80,
        y0=0.35,
        x1=0.98,
        y1=0.65,
        line=dict(color="grey", width=1),
        fillcolor="rgba(200, 200, 200, 0.1)",
    )
    fig.add_annotation(
        x=0.89,
        y=0.5,
        text="Use phase,\nregeneration,\nend of life\n(excluded)",
        showarrow=False,
        font=dict(size=10),
    )

    # Main system annotation
    fig.add_annotation(
        x=0.5,
        y=0.6,
        text=(
            "System boundary (included):\n"
            "Lab scale bead synthesis\n"
            "PDChNF–chitosan bead (Ref-Bead)\n"
            "+ UiO-66-NH₂ growth and NaOH step\n"
            "(U@Bead-2step-aUiO)"
        ),
        showarrow=False,
        font=dict(size=11),
    )

    # Arrows
    fig.add_annotation(x=0.22, y=0.5, ax=0.25, ay=0.5, showarrow=True, arrowhead=2)
    fig.add_annotation(x=0.75, y=0.5, ax=0.78, ay=0.5, showarrow=True, arrowhead=2)

    fig.update_xaxes(visible=False, range=[0, 1])
    fig.update_yaxes(visible=False, range=[0, 1])

    fig.update_layout(
        title="System boundary for the screening LCA (gate to gate, lab scale)",
        height=350,
        margin=dict(l=20, r=20, t=70, b=20),
    )
    return fig


# -----------------------------------------------------------------------------
# MAIN APP
# -----------------------------------------------------------------------------
def main():
    # Logos on top of everything
    render_header_logos()

    st.title("Interactive LCA Explorer: Ref-Bead vs U@Bead")
    st.markdown(
        """
**Dashboard:** Adjust scenarios on the left. Use the tabs below to explore results, sensitivity and AI supported interpretation.
"""
    )

    # -------------------------------------------------------------------------
    # SIDEBAR: USER INPUTS
    # -------------------------------------------------------------------------
    with st.sidebar:
        st.header("Control Panel")

        if st.button("Reset to paper defaults"):
            reset_data()
            st.success("Data reset to defaults from the LCA section.")
            st.rerun()

        st.divider()

        # --- QUICK ADJUST ---
        st.subheader("1. Scenario parameters")

        # Grid Intensity
        current_grid_ef = float(
            st.session_state["ef_df"]
            .loc[
                st.session_state["ef_df"]["reagent_name"] == "Electricity (Canada)",
                "GWP_kgCO2_per_kg",
            ]
            .iloc[0]
        )

        new_grid_ef = st.slider(
            "Grid carbon intensity (kg CO₂ per kWh)",
            min_value=0.0,
            max_value=1.0,
            value=current_grid_ef,
            step=0.01,
            help="Controls the cleanliness of the power source used in the LCA.",
        )

        if new_grid_ef != current_grid_ef:
            st.session_state["ef_df"].loc[
                st.session_state["ef_df"]["reagent_name"] == "Electricity (Canada)",
                "GWP_kgCO2_per_kg",
            ] = new_grid_ef

        # Efficiency
        st.write("**Process efficiency**")
        eff_factor = st.slider(
            "Efficiency multiplier",
            min_value=0.1,
            max_value=1.0,
            value=1.0,
            help="1.0 = lab scale baseline. Lower values approximate improved industrial efficiency.",
        )

        # Recycling
        st.write("**Solvent recovery**")
        recycle_rate = st.slider(
            "Recycling rate (%)",
            min_value=0,
            max_value=95,
            value=0,
            help="Percentage of ethanol and formic acid recovered and reused.",
        )

        # Yield
        st.write("**Global yield**")
        yield_rate = st.slider(
            "Synthesis yield (%)",
            min_value=10,
            max_value=100,
            value=100,
            help="Lower yield increases the required inputs per kg of bead.",
        )

        # Transport
        st.write("**Transport overhead**")
        transport_overhead = st.slider(
            "Add transport (%)",
            min_value=0,
            max_value=50,
            value=0,
            help="Adds a fixed percentage to account for logistics emissions.",
        )

        st.divider()

        # --- TABLES ---
        st.subheader("2. Input tables")
        with st.expander("Edit detailed inputs"):
            st.caption("Emission factors")
            st.session_state["ef_df"] = st.data_editor(
                st.session_state["ef_df"], key="ed_ef", num_rows="dynamic"
            )

            st.caption("Bead recipes and electricity")
            st.session_state["routes_df"] = st.data_editor(
                st.session_state["routes_df"], key="ed_routes", num_rows="dynamic"
            )

            st.caption("Performance data")
            st.session_state["perf_df"] = st.data_editor(
                st.session_state["perf_df"], key="ed_perf", num_rows="dynamic"
            )

    # Refresh shortcuts after any edits
    ef_df = st.session_state["ef_df"]
    routes_df = st.session_state["routes_df"]
    perf_df = st.session_state["perf_df"]
    lit_df = st.session_state["lit_df"]

    # -------------------------------------------------------------------------
    # ELECTRICITY CAPACITY SCALING (PER STEP)
    # -------------------------------------------------------------------------
    elec_steps_df = get_electricity_step_data()

    # Apply capacity-based scaling to electricity_kwh_per_fu for each route
    routes_df_effective = routes_df.copy()
    scaled_elec_by_route = {}

    for rid in routes_df_effective["route_id"].unique():
        step_rows = elec_steps_df[elec_steps_df["route_id"] == rid]
        if step_rows.empty:
            continue

        scaled_specific_steps = []
        for _, row in step_rows.iterrows():
            k_lab = float(row["kWh_per_kg"])
            lab_batch = float(row["lab_batch_kg"])
            capacity = float(row.get("capacity_kg", lab_batch))
            if capacity <= 0:
                capacity = lab_batch
            # scaling_factor = lab_batch / capacity
            scaling_factor = lab_batch / capacity
            k_scaled = k_lab * scaling_factor
            scaled_specific_steps.append(k_scaled)

        total_scaled_elec = float(np.sum(scaled_specific_steps))
        scaled_elec_by_route[rid] = total_scaled_elec
        routes_df_effective.loc[
            routes_df_effective["route_id"] == rid, "electricity_kwh_per_fu"
        ] = total_scaled_elec

    # Use the effective routes dataframe for all subsequent calculations
    routes_df = routes_df_effective

    # -------------------------------------------------------------------------
    # CALCULATIONS
    # -------------------------------------------------------------------------
    unique_routes = routes_df["route_id"].unique()
    results_list = []
    dfs_list = []

    for rid in unique_routes:
        res, df = calculate_impacts(
            rid,
            ef_df,
            routes_df,
            efficiency_factor=eff_factor,
            recycling_rate=recycle_rate,
            yield_rate=yield_rate,
            transport_pct=transport_overhead,
        )
        if res:
            # add Electricity % once so that AI tab can use it
            if res["Total GWP"] > 0:
                res["Electricity %"] = (res["Electricity GWP"] / res["Total GWP"]) * 100.0
            else:
                res["Electricity %"] = 0.0

            results_list.append(res)
            dfs_list.append(df)

    if not results_list:
        st.warning("No routes found. Check the input tables in the sidebar.")
        return

    perf_map = {
        row["route_id"]: float(row["capacity_mg_g"]) for _, row in perf_df.iterrows()
    }

    # Prepare summary data
    summary_rows = []
    for r in results_list:
        cap = perf_map.get(r["id"], 0.001)  # mg/g
        summary_rows.append(
            {
                "Bead": r["name"],
                "Total GWP": r["Total GWP"],
                "Non-Electric GWP": r["Non-Electric GWP"],
                "GWP per g Cu": r["Total GWP"] / cap,
                "Electricity %": (r["Electricity GWP"] / r["Total GWP"]) * 100.0
                if r["Total GWP"] > 0
                else 0.0,
            }
        )
    sum_df = pd.DataFrame(summary_rows)

    # -------------------------------------------------------------------------
    # TABS
    # -------------------------------------------------------------------------
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "LCA results",
            "Sensitivity and scaling",
            "Inventory and flows",
            "Literature comparison",
            "AI insights",
        ]
    )

    # --- TAB 1: RESULTS ---
    with tab1:
        st.header("LCA results")

        st.dataframe(
            sum_df.style.format(
                {
                    "Total GWP": "{:.2e}",
                    "Non-Electric GWP": "{:.2f}",
                    "GWP per g Cu": "{:.2f}",
                    "Electricity %": "{:.1f}%",
                }
            )
        )

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Total GWP per kg bead (log scale, FU1)")
            fig_log = px.bar(
                sum_df,
                x="Bead",
                y="Total GWP",
                color="Bead",
                log_y=True,
                title="Total GWP per kg bead (FU1)",
                text_auto=".2s",
            )
            st.plotly_chart(fig_log, use_container_width=True)

        with col2:
            st.subheader("Electricity versus chemicals")
            stack_data = []
            for r in results_list:
                stack_data.append(
                    {
                        "Bead": r["name"],
                        "Source": "Electricity",
                        "GWP": r["Electricity GWP"],
                    }
                )
                stack_data.append(
                    {
                        "Bead": r["name"],
                        "Source": "Chemicals and transport",
                        "GWP": r["Non-Electric GWP"],
                    }
                )
            fig_stack = px.bar(
                pd.DataFrame(stack_data),
                x="Bead",
                y="GWP",
                color="Source",
                title="Electricity versus non electricity contributions",
                text_auto=".2s",
            )
            st.plotly_chart(fig_stack, use_container_width=True)

        st.subheader("Performance normalised impacts (FU2)")
        fig_fu2 = px.bar(
            sum_df,
            x="Bead",
            y="GWP per g Cu",
            color="Bead",
            title="GWP per g Cu removed (FU2)",
            text_auto=".2f",
        )
        st.plotly_chart(fig_fu2, use_container_width=True)

        st.subheader("System boundary (schematic)")
        st.plotly_chart(create_system_boundary_figure(), use_container_width=True)

    # --- TAB 2: SENSITIVITY ---
    with tab2:
        st.header("Sensitivity and scaling")

        # 0. Equipment capacity and utilisation
        st.subheader("0. Equipment capacity and utilisation")
        st.markdown(
            "The table below specifies, for each electricity-using step, the laboratory batch size "
            "and an effective equipment capacity per batch. By default, capacity equals the lab "
            "batch (no scaling). Increasing the capacity reduces electricity per kg in the main LCA "
            "results while keeping the functional unit at 1 kg of beads."
        )

        elec_steps_df = get_electricity_step_data()
        st.caption("Edit 'capacity_kg' to explore more realistic utilisation scenarios.")
        st.session_state["elec_steps_df"] = st.data_editor(
            elec_steps_df,
            key="ed_elec_steps",
            num_rows="fixed",
            disabled=["route_id", "Bead", "Step", "lab_batch_kg", "kWh_per_kg"],
        )

        st.divider()

        # 1. Grid intensity sensitivity
        st.subheader("1. Grid intensity sensitivity")

        col_s1, col_s2 = st.columns([1, 2])
        with col_s1:
            st.write("Add custom grid point:")
            new_grid_name = st.text_input("Grid name", "My local grid")
            new_grid_val = st.number_input(
                "Intensity (kg CO₂/kWh)",
                min_value=0.0,
                max_value=1.0,
                value=0.45,
            )
            if st.button("Add grid to chart"):
                st.session_state["custom_grids"][new_grid_name] = new_grid_val
                st.success(f"Added grid: {new_grid_name}")

        # Calculate for all grids
        sens_rows = []
        for g_name, g_val in st.session_state["custom_grids"].items():
            temp_ef = ef_df.copy()
            temp_ef.loc[
                temp_ef["reagent_name"].str.contains("Electricity"), "GWP_kgCO2_per_kg"
            ] = g_val
            for rid in unique_routes:
                res, _ = calculate_impacts(
                    rid,
                    temp_ef,
                    routes_df,
                    efficiency_factor=eff_factor,
                    recycling_rate=recycle_rate,
                    yield_rate=yield_rate,
                    transport_pct=transport_overhead,
                )
                sens_rows.append(
                    {
                        "Grid": g_name,
                        "Grid Value": g_val,
                        "Bead": res["name"],
                        "Total GWP": res["Total GWP"],
                    }
                )

        df_sens = pd.DataFrame(sens_rows).sort_values("Grid Value")
        fig_sens = px.line(
            df_sens,
            x="Grid",
            y="Total GWP",
            color="Bead",
            markers=True,
            title="Total GWP versus grid carbon intensity",
            hover_data=["Grid Value"],
        )
        st.plotly_chart(fig_sens, use_container_width=True)

        st.divider()

        # 2. Batch scaling effect (fixed curve up to 100 kg)
        st.subheader("2. Batch scaling effect (electricity driven)")

        # lab batch sizes from the original LCA inventory (kg per batch)
        LAB_BATCH_REF_KG = 0.0004
        LAB_BATCH_MOF_KG = 0.0006

        batch_sizes = np.logspace(np.log10(0.0004), np.log10(100.0), num=40)
        scale_rows = []

        for rid in unique_routes:
            base_res, _ = calculate_impacts(
                rid,
                ef_df,
                routes_df,
                efficiency_factor=1.0,
                recycling_rate=0.0,
                yield_rate=100.0,
                transport_pct=transport_overhead,
            )
            if base_res is None:
                continue

            base_elec_intensity = base_res["Electricity kWh"]
            if rid == ID_REF:
                lab_batch = LAB_BATCH_REF_KG
            elif rid == ID_MOF:
                lab_batch = LAB_BATCH_MOF_KG
            else:
                lab_batch = LAB_BATCH_REF_KG

            for b_size in batch_sizes:
                # Assume kWh per batch roughly constant, so kWh/kg scales with lab_batch / b_size
                scale_factor = lab_batch / b_size
                new_elec_intensity = base_elec_intensity * scale_factor
                new_gwp = new_elec_intensity * base_res["Electricity EF Used"] + base_res["Non-Electric GWP"]
                scale_rows.append(
                    {
                        "Batch size (kg)": b_size,
                        "Bead": base_res["name"],
                        "Estimated GWP": new_gwp,
                    }
                )

        df_scale = pd.DataFrame(scale_rows)
        fig_scale = px.line(
            df_scale,
            x="Batch size (kg)",
            y="Estimated GWP",
            color="Bead",
            log_x=True,
            log_y=True,
            markers=True,
            title="Projected GWP versus batch size (log log)",
        )
        fig_scale.update_xaxes(range=[np.log10(0.0004), np.log10(100.0)])
        st.plotly_chart(fig_scale, use_container_width=True)

        st.divider()

        # 3. Electricity demand per process step (scaled by current capacities)
        st.subheader("3. Electricity demand per process step")
        elec_step_df_plot = get_electricity_step_data().copy()
        # Compute scaled kWh/kg based on current capacity assumptions
        def _scaled_kwh(row):
            cap = row["capacity_kg"] if row["capacity_kg"] > 0 else row["lab_batch_kg"]
            return row["kWh_per_kg"] * (row["lab_batch_kg"] / cap)

        elec_step_df_plot["kWh_per_kg_scaled"] = elec_step_df_plot.apply(_scaled_kwh, axis=1)

        fig_steps = px.bar(
            elec_step_df_plot,
            x="Step",
            y="kWh_per_kg_scaled",
            color="Bead",
            barmode="group",
            log_y=True,
            title="Electricity demand by process step (kWh per kg bead, with current capacities)",
            text_auto=".2s",
        )
        fig_steps.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig_steps, use_container_width=True)

    # --- TAB 3: INVENTORY ---
    with tab3:
        st.header("Inventory and impact breakdown")

        all_contribs = []
        for i, df in enumerate(dfs_list):
            df = df.copy()
            df["Bead"] = results_list[i]["name"]
            all_contribs.append(df)
        df_all = pd.concat(all_contribs) if all_contribs else pd.DataFrame()

        if not df_all.empty:
            st.subheader("A. Chemical impacts (excluding electricity)")
            df_ne = df_all[df_all["Category"] != "Electricity"]
            fig_ne = px.bar(
                df_ne,
                x="Bead",
                y="GWP",
                color="Component",
                title="Chemical GWP (no electricity)",
                barmode="group",
            )
            st.plotly_chart(fig_ne, use_container_width=True)

            st.divider()

            st.subheader("B. Total breakdown (log scale)")
            fig_breakdown = px.bar(
                df_all,
                x="Bead",
                y="GWP",
                color="Component",
                title="Total GWP breakdown (log scale)",
                barmode="group",
                log_y=True,
            )
            st.plotly_chart(fig_breakdown, use_container_width=True)

            st.divider()

            st.subheader("C. Mass inventory per kg bead")
            fig_mass = px.bar(
                df_ne,
                x="Component",
                y="Mass (kg)",
                color="Component",
                facet_col="Bead",
                title="Mass input per kg product",
            )
            fig_mass.update_yaxes(matches=None, showticklabels=True)
            st.plotly_chart(fig_mass, use_container_width=True)

        st.subheader("D. Impact flow (Sankey diagrams)")
        st.markdown("**Ref-Bead (polymer only)**")
        st.plotly_chart(
            plot_sankey_diagram(results_list, route_id=ID_REF),
            use_container_width=True,
        )

        st.markdown("**U@Bead (MOF functionalised)**")
        st.plotly_chart(
            plot_sankey_diagram(results_list, route_id=ID_MOF),
            use_container_width=True,
        )

    # --- TAB 4: LITERATURE ---
    with tab4:
        st.header("Literature comparison")

        current_data = []
        for r in results_list:
            current_data.append(
                {
                    "Material": f"{r['name']} (this work)",
                    "GWP_kgCO2_per_kg": r["Total GWP"],
                    "Source": "This work",
                    "Type": "This work",
                }
            )
        lit_combined = pd.concat([lit_df, pd.DataFrame(current_data)])

        fig_lit = px.bar(
            lit_combined,
            x="Material",
            y="GWP_kgCO2_per_kg",
            color="Source",
            log_y=True,
            title="GWP comparison with literature (log scale)",
            text="Source",
        )
        fig_lit.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_lit, use_container_width=True)

    # --- TAB 5: AI INSIGHTS ---
    with tab5:
        st.header("AI insights")
        st.caption(
            "Ask questions about the current results. You can focus on one bead or consider both together."
        )

        # Optional route focus
        route_name_map = {r["id"]: r["name"] for r in results_list}
        focus_options = ["All routes"]
        for rid in unique_routes:
            if rid in route_name_map:
                focus_options.append(route_name_map[rid])

        focus_choice = st.radio(
            "Route focus (optional)",
            options=focus_options,
            index=0,
            help="Select a bead to focus the AI summary, or keep 'All routes'.",
        )

        if focus_choice == "All routes":
            ai_context_results = results_list
        else:
            chosen_id = None
            for rid, name in route_name_map.items():
                if name == focus_choice:
                    chosen_id = rid
                    break
            if chosen_id is None:
                ai_context_results = results_list
            else:
                ai_context_results = [r for r in results_list if r["id"] == chosen_id]

        st.write("Sample questions (click to populate the box):")
        col_q1, col_q2 = st.columns(2)
        sample_questions = [
            "Why is the GWP so high compared to literature?",
            "Compare Ref-Bead and U@Bead results.",
            "What is the biggest hotspot in this scenario?",
            "How can I reduce the carbon footprint of bead production?",
        ]

        # Ensure the session key exists for the text area
        if "ai_custom_q" not in st.session_state:
            st.session_state["ai_custom_q"] = ""

        for i, q in enumerate(sample_questions):
            col = col_q1 if i % 2 == 0 else col_q2
            with col:
                if st.button(q, key=f"sample_q_{i}"):
                    st.session_state["ai_custom_q"] = q

        user_q = st.text_area(
            "Type your question about the LCA results:",
            key="ai_custom_q",
            height=140,
        )

        if st.button("Analyse results"):
            if user_q.strip():
                with st.spinner("AI is analysing your data..."):
                    answer = get_ai_insight(ai_context_results, user_q)
                    st.markdown("### AI analysis")
                    st.info(answer)
            else:
                st.warning("Please enter a question or click one of the sample prompts.")


if __name__ == "__main__":
    main()
