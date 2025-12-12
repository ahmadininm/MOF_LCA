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
LOGO_DIR = BASE_DIR / "assets"  # not used in this version, logos loaded from DATA_DIR

# Route IDs
ID_REF = "ref"
ID_MOF = "mof"

# Polymer fraction of Ref-Bead carried into U@Bead (for electricity allocation)
POLYMER_FRACTION_MOF = 0.87

# Default equipment parameters for electricity-intensive steps
# Masses are in grams; power in kW; time in hours.
EQUIPMENT_DEFAULTS = {
    "ref_micro": {
        "label": "Ref-Bead microfluidiser (PDChNF fibrillation)",
        "route_id": ID_REF,
        "power_kw": 1.5,
        "time_h": 0.33,
        "batch_mass_g": 0.4,    # dry bead mass per batch
        "alloc_mass_g": 0.4,    # allocation mass / effective capacity
    },
    "ref_mix": {
        "label": "Ref-Bead hotplate mixing (50 °C)",
        "route_id": ID_REF,
        "power_kw": 0.625,
        "time_h": 6.0,
        "batch_mass_g": 0.4,
        "alloc_mass_g": 0.4,
    },
    "ref_cross": {
        "label": "Ref-Bead hotplate crosslinking (50 °C)",
        "route_id": ID_REF,
        "power_kw": 0.625,
        "time_h": 6.0,
        "batch_mass_g": 0.4,
        "alloc_mass_g": 0.4,
    },
    "ref_freeze": {
        "label": "Ref-Bead freeze-drying",
        "route_id": ID_REF,
        "power_kw": 1.84,
        "time_h": 16.0,
        "batch_mass_g": 0.4,
        "alloc_mass_g": 0.4,
    },
    "mof_stir_zr": {
        "label": "UiO stirring (Zr step)",
        "route_id": ID_MOF,
        "power_kw": 0.625,
        "time_h": 12.0,
        "batch_mass_g": 0.6,
        "alloc_mass_g": 0.6,
    },
    "mof_stir_linker": {
        "label": "UiO stirring (linker step)",
        "route_id": ID_MOF,
        "power_kw": 0.625,
        "time_h": 12.0,
        "batch_mass_g": 0.6,
        "alloc_mass_g": 0.6,
    },
    "mof_freeze": {
        "label": "Second freeze-drying (U@Bead)",
        "route_id": ID_MOF,
        "power_kw": 1.84,
        "time_h": 16.0,
        "batch_mass_g": 0.6,
        "alloc_mass_g": 0.6,
    },
}

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
    """Resets session state to default CSV values and equipment settings."""
    ef, routes, perf, lit = load_default_data()
    st.session_state["ef_df"] = ef
    st.session_state["routes_df"] = routes
    st.session_state["perf_df"] = perf
    st.session_state["lit_df"] = lit

    # Reset custom grids
    st.session_state["custom_grids"] = {
        "QC Hydro": 0.002,
        "Canada Avg": 0.1197,
        "UK Grid": 0.225,
        "EU Avg": 0.25,
        "US Avg": 0.38,
        "China Grid": 0.58,
    }

    # Reset equipment parameters
    st.session_state["equipment_params"] = {k: dict(v) for k, v in EQUIPMENT_DEFAULTS.items()}

    # Clear widget states related to equipment controls
    for key in list(st.session_state.keys()):
        if key.startswith("equip_"):
            del st.session_state[key]


# Initialise Session State
if "ef_df" not in st.session_state:
    reset_data()

if "equipment_params" not in st.session_state:
    st.session_state["equipment_params"] = {k: dict(v) for k, v in EQUIPMENT_DEFAULTS.items()}

# Shortcuts (not used directly in main, but kept for completeness)
EF_DF = st.session_state["ef_df"]
ROUTES_DF = st.session_state["routes_df"]
PERF_DF = st.session_state["perf_df"]
LIT_DF = st.session_state["lit_df"]


# -----------------------------------------------------------------------------
# HEADER LOGOS
# -----------------------------------------------------------------------------
def render_header_logos():
    cov_path = DATA_DIR / "cov.png"
    ubc_path = DATA_DIR / "ubc.png"

    col_left, col_spacer, col_right = st.columns([1, 2, 1])
    with col_left:
        if cov_path.exists():
            st.image(str(cov_path))
    with col_right:
        if ubc_path.exists():
            st.image(str(ubc_path))


# -----------------------------------------------------------------------------
# EQUIPMENT-BASED ELECTRICITY CALCULATIONS
# -----------------------------------------------------------------------------
def compute_electricity_from_equipment(equipment_params):
    """
    Compute electricity intensity (kWh/kg bead) for each route and step
    from equipment power, time and allocation mass.
    """
    step_rows = []
    ref_total = 0.0
    mof_specific_total = 0.0

    # Ref-Bead steps
    for step_id, cfg in equipment_params.items():
        if cfg.get("route_id") == ID_REF:
            mass_g = cfg.get("alloc_mass_g", cfg.get("batch_mass_g", 0.0))
            mass_kg = max(mass_g, 1e-9) / 1000.0
            power_kw = cfg.get("power_kw", 0.0)
            time_h = cfg.get("time_h", 0.0)
            kwh_per_kg = power_kw * time_h / mass_kg
            ref_total += kwh_per_kg
            step_rows.append(
                {
                    "route_id": ID_REF,
                    "Bead": "Ref-Bead (Polymer)",
                    "Step": cfg.get("label", step_id),
                    "kWh_per_kg": kwh_per_kg,
                }
            )

    # Carry-over of Ref-Bead electricity into U@Bead (87 wt% polymer support)
    carry_over = POLYMER_FRACTION_MOF * ref_total
    step_rows.append(
        {
            "route_id": ID_MOF,
            "Bead": "U@Bead (MOF-Functionalised)",
            "Step": "Carry-over (Ref support)",
            "kWh_per_kg": carry_over,
        }
    )

    # MOF-specific steps
    for step_id, cfg in equipment_params.items():
        if cfg.get("route_id") == ID_MOF:
            mass_g = cfg.get("alloc_mass_g", cfg.get("batch_mass_g", 0.0))
            mass_kg = max(mass_g, 1e-9) / 1000.0
            power_kw = cfg.get("power_kw", 0.0)
            time_h = cfg.get("time_h", 0.0)
            kwh_per_kg = power_kw * time_h / mass_kg
            mof_specific_total += kwh_per_kg
            step_rows.append(
                {
                    "route_id": ID_MOF,
                    "Bead": "U@Bead (MOF-Functionalised)",
                    "Step": cfg.get("label", step_id),
                    "kWh_per_kg": kwh_per_kg,
                }
            )

    route_totals = {
        ID_REF: ref_total,
        ID_MOF: carry_over + mof_specific_total,
    }

    return route_totals, pd.DataFrame(step_rows)


def sync_equipment_params_from_widgets():
    """
    Sync equipment parameter values from widget state into session_state['equipment_params'].
    """
    if "equipment_params" not in st.session_state:
        st.session_state["equipment_params"] = {k: dict(v) for k, v in EQUIPMENT_DEFAULTS.items()}

    equipment_params = st.session_state["equipment_params"]

    for step_id, cfg in equipment_params.items():
        for field in ("power_kw", "time_h", "batch_mass_g", "alloc_mass_g"):
            widget_key = f"equip_{step_id}_{field}"
            if widget_key in st.session_state:
                val = st.session_state[widget_key]
                try:
                    cfg[field] = float(val)
                except (TypeError, ValueError):
                    pass

    st.session_state["equipment_params"] = equipment_params


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

    # Electricity
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

        # Recycling impact
        is_solvent = reagent in ["Ethanol", "Formic acid (88%)", "Acetic acid"]
        if is_solvent:
            effective_mass = mass_needed * (1 - (recycling_rate / 100.0))
        else:
            effective_mass = mass_needed

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
def plot_sankey_diagram(results_list, contrib_df_all=None, step_df=None, route_id=None):
    """
    Build an itemised Sankey diagram for a given route.
    - Electricity is broken down by process step (microfluidiser, mixing, freeze-drying, etc).
    - Chemical supply is broken down by individual components.
    Falls back to the original aggregate diagram if detailed data are missing.
    """
    if not results_list:
        return go.Figure()

    # Pick target route
    if route_id is not None:
        target_res = next((r for r in results_list if r["id"] == route_id), None)
        if target_res is None:
            target_res = results_list[0]
    else:
        target_res = results_list[0]

    bead_name = target_res["name"]
    route_key = target_res["id"]
    total_gwp = target_res["Total GWP"]
    elec_gwp_total = target_res["Electricity GWP"]

    # If we have both component contributions and step-level electricity, do itemised diagram
    chem_df = None
    elec_steps_df = None

    if contrib_df_all is not None and "Bead" in contrib_df_all.columns:
        chem_df = contrib_df_all[contrib_df_all["Bead"] == bead_name].copy()
        if "Category" in chem_df.columns:
            chem_df = chem_df[chem_df["Category"] != "Electricity"]
        chem_df = chem_df[chem_df["GWP"] > 0]

    if step_df is not None and "route_id" in step_df.columns:
        elec_steps_df = step_df[step_df["route_id"] == route_key].copy()
        if not elec_steps_df.empty:
            elec_steps_df = elec_steps_df[elec_steps_df["kWh_per_kg"] > 0]

    if chem_df is not None and not chem_df.empty and elec_steps_df is not None and not elec_steps_df.empty:
        # Build nodes: chemicals, electricity steps, lab synthesis, total
        chem_labels = list(dict.fromkeys(chem_df["Component"].tolist()))

        # Electricity step labels, prefixed so they are clearly electricity
        elec_step_labels = []
        for s in elec_steps_df["Step"].tolist():
            elec_step_labels.append(f"Electricity: {s}")

        lab_label = f"{bead_name} synthesis"
        total_label = "Total GWP"

        node_labels = chem_labels + elec_step_labels + [lab_label, total_label]

        # Colour mapping: chemicals green, electricity yellow, lab blue, total red
        node_colors = []
        for lbl in node_labels:
            low = lbl.lower()
            if lbl == lab_label:
                node_colors.append("#87CEFA")  # light blue
            elif lbl == total_label:
                node_colors.append("#FF6347")  # red
            elif low.startswith("electricity:"):
                node_colors.append("#FFD700")  # yellow
            elif "transport" in low or "logistics" in low:
                node_colors.append("#A9A9A9")  # grey
            else:
                node_colors.append("#90EE90")  # light green for chemicals

        idx = {lbl: i for i, lbl in enumerate(node_labels)}

        sources = []
        targets = []
        values = []

        # Chemicals: each component -> lab synthesis
        for _, row in chem_df.iterrows():
            comp = row["Component"]
            gwp_val = float(row["GWP"])
            if gwp_val <= 0:
                continue
            if comp not in idx:
                # Should not happen, but guard
                continue
            sources.append(idx[comp])
            targets.append(idx[lab_label])
            values.append(gwp_val)

        # Electricity: split total electricity GWP across steps by kWh share
        kwh_vals = elec_steps_df["kWh_per_kg"].to_numpy(dtype=float)
        kwh_sum = float(kwh_vals.sum())
        if kwh_sum > 0 and elec_gwp_total > 0:
            for step_name, kwh in zip(elec_steps_df["Step"], kwh_vals):
                frac = kwh / kwh_sum
                gwp_step = elec_gwp_total * frac
                lbl = f"Electricity: {step_name}"
                sources.append(idx[lbl])
                targets.append(idx[lab_label])
                values.append(gwp_step)

        # Lab -> total GWP (use sum of all incoming flows to avoid mismatch)
        total_from_components = float(sum(values))
        sources.append(idx[lab_label])
        targets.append(idx[total_label])
        values.append(total_from_components)

        fig = go.Figure(
            data=[
                go.Sankey(
                    node=dict(
                        pad=15,
                        thickness=20,
                        line=dict(color="black", width=0.5),
                        label=node_labels,
                        color=node_colors,
                    ),
                    link=dict(
                        source=sources,
                        target=targets,
                        value=values,
                    ),
                )
            ]
        )

        fig.update_layout(
            title_text=f"Impact flow (itemised): {bead_name}",
            font_size=10,
            height=400,
        )
        return fig

    # Fallback: original aggregate version if we do not have enough detail
    elec_gwp = target_res["Electricity GWP"]
    chem_gwp = target_res["Non-Electric GWP"]

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


def render_system_boundary_graphviz():
    """
    Renders the system boundary using Graphviz to show included vs excluded steps.
    """
    dot = """
    digraph {
        rankdir=LR;
        bgcolor="transparent";
        node [shape=box, style="filled,rounded", fontname="Sans-Serif", fontsize=10];
        edge [fontname="Sans-Serif", fontsize=9, color="#666666"];

        # Excluded Upstream
        subgraph cluster_upstream {
            label = "Upstream (Excluded)";
            style = "dashed";
            color = "#808080";
            fontcolor = "#808080";
            node [fillcolor="#f9f9f9", color="#808080", fontcolor="#808080"];
            
            Raw [label="Raw Materials\n(Fishery waste, Mining, Fossil fuels)"];
            Trans [label="Transport\n(to Laboratory)"];
        }

        # Included System Boundary
        subgraph cluster_gate {
            label = "System Boundary (Included: Gate-to-Gate)";
            style = "solid";
            color = "#2E8B57";  # SeaGreen
            penwidth = 2;
            fontcolor = "#2E8B57";
            node [fillcolor="#E8F5E9", color="#2E8B57", fontcolor="black"];
            
            Inputs [label="Inputs:\nElectricity, Water,\nReagents, Solvents"];
            
            subgraph cluster_ref {
                label = "Ref-Bead Process";
                style = "dotted";
                color = "#2E8B57";
                Process1 [label="Polymer Dissolution\n& Mixing"];
                Process2 [label="Crosslinking\n& Washing"];
                Process3 [label="Freeze-Drying 1"];
            }
            
            subgraph cluster_mof {
                label = "MOF Process (Add-on)";
                style = "dotted";
                color = "#2E8B57";
                Process4 [label="UiO-66-NH₂\nGrowth (Zr + Linker)"];
                Process5 [label="Freeze-Drying 2"];
            }
            
            Product [label="Final Dry Bead\n(at Lab Gate)"];
        }

        # Excluded Downstream
        subgraph cluster_downstream {
            label = "Downstream (Excluded)";
            style = "dashed";
            color = "#808080";
            fontcolor = "#808080";
            node [fillcolor="#f9f9f9", color="#808080", fontcolor="#808080"];
            
            Use [label="Use Phase\n(Copper Removal)"];
            EOL [label="End of Life\n(Disposal/Regeneration)"];
        }

        # Edges
        Raw -> Trans;
        Trans -> Inputs;
        
        Inputs -> Process1;
        Inputs -> Process4;
        
        Process1 -> Process2;
        Process2 -> Process3;
        
        Process3 -> Product [label="Ref-Bead", style="dashed"];
        Process3 -> Process4 [label="Support"];
        Process4 -> Process5;
        Process5 -> Product [label="U@Bead"];
        
        Product -> Use;
        Use -> EOL;
    }
    """
    try:
        st.graphviz_chart(dot, use_container_width=True)
        st.caption("Figure: Gate-to-gate system boundary. Green box indicates processes included in the study.")
    except Exception as e:
        st.error(f"Graphviz rendering failed: {e}")


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
    # EQUIPMENT SCENARIOS: BASELINE VS SCALED
    # -------------------------------------------------------------------------
    # Sync equipment parameters from widgets (scaled scenario)
    sync_equipment_params_from_widgets()
    equipment_params_scaled = st.session_state["equipment_params"]

    # Baseline (default equipment settings) and scaled (current equipment tab)
    route_elec_baseline, step_df_baseline = compute_electricity_from_equipment(EQUIPMENT_DEFAULTS)
    route_elec_scaled, step_df_scaled = compute_electricity_from_equipment(equipment_params_scaled)

    # Build two route tables: baseline and scaled
    routes_base = routes_df.copy()
    routes_scaled = routes_df.copy()

    for rid, kwh_per_kg in route_elec_baseline.items():
        routes_base.loc[routes_base["route_id"] == rid, "electricity_kwh_per_fu"] = kwh_per_kg

    for rid, kwh_per_kg in route_elec_scaled.items():
        routes_scaled.loc[routes_scaled["route_id"] == rid, "electricity_kwh_per_fu"] = kwh_per_kg

    # -------------------------------------------------------------------------
    # CALCULATIONS FOR BOTH SCENARIOS
    # -------------------------------------------------------------------------
    unique_routes = routes_df["route_id"].unique()

    # Baseline
    base_results_list = []
    base_dfs_list = []

    for rid in unique_routes:
        res, df = calculate_impacts(
            rid,
            ef_df,
            routes_base,
            efficiency_factor=eff_factor,
            recycling_rate=recycle_rate,
            yield_rate=yield_rate,
            transport_pct=transport_overhead,
        )

        if res:
            if res["Total GWP"] > 0:
                res["Electricity %"] = (res["Electricity GWP"] / res["Total GWP"]) * 100.0
            else:
                res["Electricity %"] = 0.0

            base_results_list.append(res)
            base_dfs_list.append(df)

    if not base_results_list:
        st.warning("No routes found. Check the input tables in the sidebar.")
        return

    # Scaled (equipment tab)
    scaled_results_list = []
    scaled_dfs_list = []

    for rid in unique_routes:
        res, df = calculate_impacts(
            rid,
            ef_df,
            routes_scaled,
            efficiency_factor=eff_factor,
            recycling_rate=recycle_rate,
            yield_rate=yield_rate,
            transport_pct=transport_overhead,
        )

        if res:
            if res["Total GWP"] > 0:
                res["Electricity %"] = (res["Electricity GWP"] / res["Total GWP"]) * 100.0
            else:
                res["Electricity %"] = 0.0

            scaled_results_list.append(res)
            scaled_dfs_list.append(df)

    # Performance map (same for both)
    perf_map = {
        row["route_id"]: float(row["capacity_mg_g"]) for _, row in perf_df.iterrows()
    }

    # Summary tables
    base_summary_rows = []
    for r in base_results_list:
        cap = perf_map.get(r["id"], 0.001)  # mg/g
        base_summary_rows.append(
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

    scaled_summary_rows = []
    for r in scaled_results_list:
        cap = perf_map.get(r["id"], 0.001)  # mg/g
        scaled_summary_rows.append(
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

    base_sum_df = pd.DataFrame(base_summary_rows)
    scaled_sum_df = pd.DataFrame(scaled_summary_rows)

    # For AI and some downstream uses we keep scaled as the "current" scenario
    results_list = scaled_results_list
    dfs_list = scaled_dfs_list

    # -------------------------------------------------------------------------
    # TABS
    # -------------------------------------------------------------------------
    tab_scale, tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Scaling (equipment)",
            "LCA results",
            "Sensitivity and scaling",
            "Inventory and flows",
            "Literature comparison",
            "AI insights",
        ]
    )

    # --- NEW TAB: EQUIPMENT SCALING ---
    with tab_scale:
        st.header("Equipment utilisation and electricity scaling")

        st.markdown(
            """
The electricity intensities used in the LCA are derived from the power and runtime of each
unit operation, divided by an allocation mass (effective batch capacity):

`kWh/kg = (power × time per batch) / allocation mass`.

By increasing the allocation mass relative to the actual batch size, you can explore how
better utilisation of the microfluidiser, stirred reactors and freeze dryer would lower
the electricity per kilogram of bead.
"""
        )

        eq = st.session_state["equipment_params"]

        st.subheader("Ref-Bead (polymer) steps")

        for step_id in ["ref_micro", "ref_mix", "ref_cross", "ref_freeze"]:
            cfg = eq.get(step_id)
            if cfg is None:
                continue

            st.markdown(f"**{cfg['label']}**")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.number_input(
                    "Power (kW)",
                    min_value=0.0,
                    value=float(cfg["power_kw"]),
                    step=0.01,
                    key=f"equip_{step_id}_power_kw",
                )
            with col2:
                st.number_input(
                    "Time per batch (h)",
                    min_value=0.0,
                    value=float(cfg["time_h"]),
                    step=0.25,
                    key=f"equip_{step_id}_time_h",
                )
            with col3:
                st.number_input(
                    "Bead mass per batch (g)",
                    min_value=0.0001,
                    value=float(cfg["batch_mass_g"]),
                    step=0.01,
                    key=f"equip_{step_id}_batch_mass_g",
                )
            with col4:
                st.number_input(
                    "Allocation mass / capacity (g)",
                    min_value=0.0001,
                    value=float(cfg["alloc_mass_g"]),
                    step=0.01,
                    key=f"equip_{step_id}_alloc_mass_g",
                )

        st.subheader("U@Bead (MOF-functionalised) specific steps")

        for step_id in ["mof_stir_zr", "mof_stir_linker", "mof_freeze"]:
            cfg = eq.get(step_id)
            if cfg is None:
                continue

            st.markdown(f"**{cfg['label']}**")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.number_input(
                    "Power (kW)",
                    min_value=0.0,
                    value=float(cfg["power_kw"]),
                    step=0.01,
                    key=f"equip_{step_id}_power_kw",
                )
            with col2:
                st.number_input(
                    "Time per batch (h)",
                    min_value=0.0,
                    value=float(cfg["time_h"]),
                    step=0.25,
                    key=f"equip_{step_id}_time_h",
                )
            with col3:
                st.number_input(
                    "Bead mass per batch (g)",
                    min_value=0.0001,
                    value=float(cfg["batch_mass_g"]),
                    step=0.01,
                    key=f"equip_{step_id}_batch_mass_g",
                )
            with col4:
                st.number_input(
                    "Allocation mass / capacity (g)",
                    min_value=0.0001,
                    value=float(cfg["alloc_mass_g"]),
                    step=0.01,
                    key=f"equip_{step_id}_alloc_mass_g",
                )

        # Show resulting electricity intensities from current settings
        route_totals_scaled, step_df_scaled_current = compute_electricity_from_equipment(
            st.session_state["equipment_params"]
        )
        route_totals_base, step_df_base_current = compute_electricity_from_equipment(EQUIPMENT_DEFAULTS)

        st.markdown("### Electricity intensity per kilogram of bead (baseline vs scaled)")
        col_e1, col_e2 = st.columns(2)
        with col_e1:
            st.markdown("**Baseline equipment (defaults)**")
            summary_elec_base = pd.DataFrame(
                {
                    "Bead": ["Ref-Bead (Polymer)", "U@Bead (MOF-Functionalised)"],
                    "Electricity kWh/kg": [
                        route_totals_base.get(ID_REF, np.nan),
                        route_totals_base.get(ID_MOF, np.nan),
                    ],
                }
            )
            st.dataframe(summary_elec_base.style.format({"Electricity kWh/kg": "{:.2e}"}))
        with col_e2:
            st.markdown("**Scaled equipment (current settings)**")
            summary_elec_scaled = pd.DataFrame(
                {
                    "Bead": ["Ref-Bead (Polymer)", "U@Bead (MOF-Functionalised)"],
                    "Electricity kWh/kg": [
                        route_totals_scaled.get(ID_REF, np.nan),
                        route_totals_scaled.get(ID_MOF, np.nan),
                    ],
                }
            )
            st.dataframe(summary_elec_scaled.style.format({"Electricity kWh/kg": "{:.2e}"}))

        st.markdown("### Step-level electricity breakdown (kWh per kg bead)")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown("**Baseline equipment (defaults)**")
            st.dataframe(step_df_baseline.style.format({"kWh_per_kg": "{:.2e}"}))
        with col_s2:
            st.markdown("**Scaled equipment (current settings)**")
            st.dataframe(step_df_scaled.style.format({"kWh_per_kg": "{:.2e}"}))

    # --- TAB 1: RESULTS ---
    with tab1:
        st.header("LCA results")

        # Summary tables
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.subheader("Baseline (default equipment)")
            st.dataframe(
                base_sum_df.style.format(
                    {
                        "Total GWP": "{:.2e}",
                        "Non-Electric GWP": "{:.2f}",
                        "GWP per g Cu": "{:.2f}",
                        "Electricity %": "{:.1f}%",
                    }
                )
            )
        with col_t2:
            st.subheader("Scaled (from equipment tab)")
            st.dataframe(
                scaled_sum_df.style.format(
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
            st.subheader("Total GWP per kg bead (baseline, FU1)")
            fig_log_base = px.bar(
                base_sum_df,
                x="Bead",
                y="Total GWP",
                color="Bead",
                title="Total GWP per kg bead (baseline, FU1)",
                text_auto=".2s",
            )
            fig_log_base.update_yaxes(rangemode="tozero")
            st.plotly_chart(fig_log_base, use_container_width=True)

        with col2:
            st.subheader("Total GWP per kg bead (scaled, FU1)")
            fig_log_scaled = px.bar(
                scaled_sum_df,
                x="Bead",
                y="Total GWP",
                color="Bead",
                title="Total GWP per kg bead (scaled, FU1)",
                text_auto=".2s",
            )
            fig_log_scaled.update_yaxes(rangemode="tozero")
            st.plotly_chart(fig_log_scaled, use_container_width=True)

        # Electricity vs chemicals
        col_ec1, col_ec2 = st.columns(2)

        with col_ec1:
            st.subheader("Electricity versus chemicals (baseline)")
            stack_data_base = []
            for r in base_results_list:
                stack_data_base.append(
                    {
                        "Bead": r["name"],
                        "Source": "Electricity",
                        "GWP": r["Electricity GWP"],
                    }
                )
                stack_data_base.append(
                    {
                        "Bead": r["name"],
                        "Source": "Chemicals and transport",
                        "GWP": r["Non-Electric GWP"],
                    }
                )

            fig_stack_base = px.bar(
                pd.DataFrame(stack_data_base),
                x="Bead",
                y="GWP",
                color="Source",
                title="Electricity versus non electricity contributions (baseline)",
                text_auto=".2s",
            )
            st.plotly_chart(fig_stack_base, use_container_width=True)

        with col_ec2:
            st.subheader("Electricity versus chemicals (scaled)")
            stack_data_scaled = []
            for r in scaled_results_list:
                stack_data_scaled.append(
                    {
                        "Bead": r["name"],
                        "Source": "Electricity",
                        "GWP": r["Electricity GWP"],
                    }
                )
                stack_data_scaled.append(
                    {
                        "Bead": r["name"],
                        "Source": "Chemicals and transport",
                        "GWP": r["Non-Electric GWP"],
                    }
                )

            fig_stack_scaled = px.bar(
                pd.DataFrame(stack_data_scaled),
                x="Bead",
                y="GWP",
                color="Source",
                title="Electricity versus non electricity contributions (scaled)",
                text_auto=".2s",
            )
            st.plotly_chart(fig_stack_scaled, use_container_width=True)

        # Performance normalised impacts
        col_fu1, col_fu2 = st.columns(2)
        with col_fu1:
            st.subheader("Performance normalised impacts (baseline, FU2)")
            fig_fu2_base = px.bar(
                base_sum_df,
                x="Bead",
                y="GWP per g Cu",
                color="Bead",
                title="GWP per g Cu removed (baseline, FU2)",
                text_auto=".2f",
            )
            st.plotly_chart(fig_fu2_base, use_container_width=True)

        with col_fu2:
            st.subheader("Performance normalised impacts (scaled, FU2)")
            fig_fu2_scaled = px.bar(
                scaled_sum_df,
                x="Bead",
                y="GWP per g Cu",
                color="Bead",
                title="GWP per g Cu removed (scaled, FU2)",
                text_auto=".2f",
            )
            st.plotly_chart(fig_fu2_scaled, use_container_width=True)

        st.subheader("System boundary (schematic)")
        render_system_boundary_graphviz()

    # --- TAB 2: SENSITIVITY ---
    with tab2:
        st.header("Sensitivity and scaling")

        # 1. Grid intensity sensitivity
        st.subheader("1. Grid intensity sensitivity")
        col_sens1, col_sens2 = st.columns(2)

        # Calculate for all grids for both baseline and scaled
        sens_rows_base = []
        sens_rows_scaled = []

        for g_name, g_val in st.session_state["custom_grids"].items():
            temp_ef = ef_df.copy()
            temp_ef.loc[
                temp_ef["reagent_name"].str.contains("Electricity"),
                "GWP_kgCO2_per_kg",
            ] = g_val

            for rid in unique_routes:
                # Baseline
                res_base, _ = calculate_impacts(
                    rid,
                    temp_ef,
                    routes_base,
                    efficiency_factor=eff_factor,
                    recycling_rate=recycle_rate,
                    yield_rate=yield_rate,
                    transport_pct=transport_overhead,
                )
                sens_rows_base.append(
                    {
                        "Grid": g_name,
                        "Grid Value": g_val,
                        "Bead": res_base["name"],
                        "Total GWP": res_base["Total GWP"],
                    }
                )

                # Scaled
                res_scaled, _ = calculate_impacts(
                    rid,
                    temp_ef,
                    routes_scaled,
                    efficiency_factor=eff_factor,
                    recycling_rate=recycle_rate,
                    yield_rate=yield_rate,
                    transport_pct=transport_overhead,
                )
                sens_rows_scaled.append(
                    {
                        "Grid": g_name,
                        "Grid Value": g_val,
                        "Bead": res_scaled["name"],
                        "Total GWP": res_scaled["Total GWP"],
                    }
                )

        df_sens_base = pd.DataFrame(sens_rows_base).sort_values("Grid Value")
        df_sens_scaled = pd.DataFrame(sens_rows_scaled).sort_values("Grid Value")

        with col_sens1:
            st.markdown("**Baseline (default equipment)**")
            fig_sens_base = px.line(
                df_sens_base,
                x="Grid",
                y="Total GWP",
                color="Bead",
                markers=True,
                title="Total GWP versus grid carbon intensity (baseline)",
                hover_data=["Grid Value"],
            )
            st.plotly_chart(fig_sens_base, use_container_width=True)

        with col_sens2:
            st.markdown("**Scaled (equipment tab)**")
            fig_sens_scaled = px.line(
                df_sens_scaled,
                x="Grid",
                y="Total GWP",
                color="Bead",
                markers=True,
                title="Total GWP versus grid carbon intensity (scaled)",
                hover_data=["Grid Value"],
            )
            st.plotly_chart(fig_sens_scaled, use_container_width=True)

        st.divider()

        # 2. Batch scaling effect (fixed curve up to 100 kg)
        st.subheader("2. Batch scaling effect (electricity driven)")

        # lab batch sizes from the LCA inventory (kg per batch)
        LAB_BATCH_REF_KG = 0.0004
        LAB_BATCH_MOF_KG = 0.0006

        batch_sizes = np.logspace(np.log10(0.0004), np.log10(100.0), num=40)

        scale_rows_base = []
        scale_rows_scaled = []

        for rid in unique_routes:
            # Baseline
            base_res, _ = calculate_impacts(
                rid,
                ef_df,
                routes_base,
                efficiency_factor=1.0,
                recycling_rate=0.0,
                yield_rate=100.0,
                transport_pct=transport_overhead,
            )

            # Scaled
            scaled_res, _ = calculate_impacts(
                rid,
                ef_df,
                routes_scaled,
                efficiency_factor=1.0,
                recycling_rate=0.0,
                yield_rate=100.0,
                transport_pct=transport_overhead,
            )

            if base_res is None or scaled_res is None:
                continue

            base_elec_intensity = base_res["Electricity kWh"]
            scaled_elec_intensity = scaled_res["Electricity kWh"]

            if rid == ID_REF:
                lab_batch = LAB_BATCH_REF_KG
            elif rid == ID_MOF:
                lab_batch = LAB_BATCH_MOF_KG
            else:
                lab_batch = LAB_BATCH_REF_KG

            for b_size in batch_sizes:
                scale_factor = lab_batch / b_size

                # Baseline
                new_elec_intensity_base = base_elec_intensity * scale_factor
                new_gwp_base = new_elec_intensity_base * base_res["Electricity EF Used"] + base_res["Non-Electric GWP"]
                scale_rows_base.append(
                    {
                        "Batch size (kg)": b_size,
                        "Bead": base_res["name"],
                        "Estimated GWP": new_gwp_base,
                    }
                )

                # Scaled
                new_elec_intensity_scaled = scaled_elec_intensity * scale_factor
                new_gwp_scaled = new_elec_intensity_scaled * scaled_res["Electricity EF Used"] + scaled_res["Non-Electric GWP"]
                scale_rows_scaled.append(
                    {
                        "Batch size (kg)": b_size,
                        "Bead": scaled_res["name"],
                        "Estimated GWP": new_gwp_scaled,
                    }
                )

        df_scale_base = pd.DataFrame(scale_rows_base)
        df_scale_scaled = pd.DataFrame(scale_rows_scaled)

        col_bs1, col_bs2 = st.columns(2)
        with col_bs1:
            st.markdown("**Baseline (default equipment)**")
            fig_scale_base = px.line(
                df_scale_base,
                x="Batch size (kg)",
                y="Estimated GWP",
                color="Bead",
                log_x=True,
                log_y=True,
                markers=True,
                title="Projected GWP versus batch size (baseline, log–log)",
            )
            fig_scale_base.update_xaxes(range=[np.log10(0.0004), np.log10(100.0)])
            st.plotly_chart(fig_scale_base, use_container_width=True)

        with col_bs2:
            st.markdown("**Scaled (equipment tab)**")
            fig_scale_scaled = px.line(
                df_scale_scaled,
                x="Batch size (kg)",
                y="Estimated GWP",
                color="Bead",
                log_x=True,
                log_y=True,
                markers=True,
                title="Projected GWP versus batch size (scaled, log–log)",
            )
            fig_scale_scaled.update_xaxes(range=[np.log10(0.0004), np.log10(100.0)])
            st.plotly_chart(fig_scale_scaled, use_container_width=True)

        st.divider()

        # 3. Electricity demand per process step
        st.subheader("3. Electricity demand per process step")

        col_step1, col_step2 = st.columns(2)

        with col_step1:
            st.markdown("**Baseline (default equipment)**")
            fig_steps_base = px.bar(
                step_df_baseline,
                x="Step",
                y="kWh_per_kg",
                color="Bead",
                barmode="group",
                title="Electricity demand by process step (baseline, kWh per kg bead)",
                text_auto=".2s",
            )
            fig_steps_base.update_yaxes(rangemode="tozero")
            fig_steps_base.update_layout(xaxis_tickangle=-30)
            st.plotly_chart(fig_steps_base, use_container_width=True)

        with col_step2:
            st.markdown("**Scaled (equipment tab)**")
            fig_steps_scaled = px.bar(
                step_df_scaled,
                x="Step",
                y="kWh_per_kg",
                color="Bead",
                barmode="group",
                title="Electricity demand by process step (scaled, kWh per kg bead)",
                text_auto=".2s",
            )
            fig_steps_scaled.update_yaxes(rangemode="tozero")
            fig_steps_scaled.update_layout(xaxis_tickangle=-30)
            st.plotly_chart(fig_steps_scaled, use_container_width=True)

    # --- TAB 3: INVENTORY ---
    with tab3:
        st.header("Inventory and impact breakdown")

        # Baseline contributions
        all_contribs_base = []
        for i, df in enumerate(base_dfs_list):
            df_b = df.copy()
            df_b["Bead"] = base_results_list[i]["name"]
            all_contribs_base.append(df_b)
        df_all_base = pd.concat(all_contribs_base) if all_contribs_base else pd.DataFrame()

        # Scaled contributions
        all_contribs_scaled = []
        for i, df in enumerate(scaled_dfs_list):
            df_s = df.copy()
            df_s["Bead"] = scaled_results_list[i]["name"]
            all_contribs_scaled.append(df_s)
        df_all_scaled = pd.concat(all_contribs_scaled) if all_contribs_scaled else pd.DataFrame()

        if not df_all_base.empty and not df_all_scaled.empty:
            # A. Chemical impacts (excluding electricity)
            st.subheader("A. Chemical impacts (excluding electricity)")
            col_inv1, col_inv2 = st.columns(2)

            df_ne_base = df_all_base[df_all_base["Category"] != "Electricity"]
            df_ne_scaled = df_all_scaled[df_all_scaled["Category"] != "Electricity"]

            with col_inv1:
                st.markdown("**Baseline (default equipment)**")
                fig_ne_base = px.bar(
                    df_ne_base,
                    x="Bead",
                    y="GWP",
                    color="Component",
                    title="Chemical GWP (no electricity, baseline)",
                    barmode="group",
                )
                st.plotly_chart(fig_ne_base, use_container_width=True)

            with col_inv2:
                st.markdown("**Scaled (equipment tab)**")
                fig_ne_scaled = px.bar(
                    df_ne_scaled,
                    x="Bead",
                    y="GWP",
                    color="Component",
                    title="Chemical GWP (no electricity, scaled)",
                    barmode="group",
                )
                st.plotly_chart(fig_ne_scaled, use_container_width=True)

            st.divider()

            # B. Total breakdown
            st.subheader("B. Total breakdown")
            col_tb1, col_tb2 = st.columns(2)

            with col_tb1:
                st.markdown("**Baseline (default equipment)**")
                fig_breakdown_base = px.bar(
                    df_all_base,
                    x="Bead",
                    y="GWP",
                    color="Component",
                    title="Total GWP breakdown (baseline)",
                    barmode="group",
                )
                st.plotly_chart(fig_breakdown_base, use_container_width=True)

            with col_tb2:
                st.markdown("**Scaled (equipment tab)**")
                fig_breakdown_scaled = px.bar(
                    df_all_scaled,
                    x="Bead",
                    y="GWP",
                    color="Component",
                    title="Total GWP breakdown (scaled)",
                    barmode="group",
                )
                st.plotly_chart(fig_breakdown_scaled, use_container_width=True)

            st.divider()

            # C. Mass inventory per kg bead (same in both cases)
            st.subheader("C. Mass inventory per kg bead")
            fig_mass = px.bar(
                df_ne_base,
                x="Component",
                y="Mass (kg)",
                color="Component",
                facet_col="Bead",
                title="Mass input per kg product",
            )
            fig_mass.update_yaxes(matches=None, showticklabels=True)
            st.plotly_chart(fig_mass, use_container_width=True)

            # D. Impact flow (Sankey diagrams)
            st.subheader("D. Impact flow (Sankey diagrams)")

            st.markdown("**Ref-Bead (polymer only)**")
            col_ref1, col_ref2 = st.columns(2)
            with col_ref1:
                st.markdown("Baseline")
                st.plotly_chart(
                    plot_sankey_diagram(
                        base_results_list,
                        contrib_df_all=df_all_base,
                        step_df=step_df_baseline,
                        route_id=ID_REF,
                    ),
                    use_container_width=True,
                )
            with col_ref2:
                st.markdown("Scaled")
                st.plotly_chart(
                    plot_sankey_diagram(
                        scaled_results_list,
                        contrib_df_all=df_all_scaled,
                        step_df=step_df_scaled,
                        route_id=ID_REF,
                    ),
                    use_container_width=True,
                )

            st.markdown("**U@Bead (MOF functionalised)**")
            col_mof1, col_mof2 = st.columns(2)
            with col_mof1:
                st.markdown("Baseline")
                st.plotly_chart(
                    plot_sankey_diagram(
                        base_results_list,
                        contrib_df_all=df_all_base,
                        step_df=step_df_baseline,
                        route_id=ID_MOF,
                    ),
                    use_container_width=True,
                )
            with col_mof2:
                st.markdown("Scaled")
                st.plotly_chart(
                    plot_sankey_diagram(
                        scaled_results_list,
                        contrib_df_all=df_all_scaled,
                        step_df=step_df_scaled,
                        route_id=ID_MOF,
                    ),
                    use_container_width=True,
                )

    # --- TAB 4: LITERATURE ---
    with tab4:
        st.header("Literature comparison")

        # Baseline
        current_data_base = []
        for r in base_results_list:
            current_data_base.append(
                {
                    "Material": f"{r['name']} (baseline, this work)",
                    "GWP_kgCO2_per_kg": r["Total GWP"],
                    "Source": "This work (baseline)",
                    "Type": "This work",
                }
            )

        # Scaled
        current_data_scaled = []
        for r in scaled_results_list:
            current_data_scaled.append(
                {
                    "Material": f"{r['name']} (scaled, this work)",
                    "GWP_kgCO2_per_kg": r["Total GWP"],
                    "Source": "This work (scaled)",
                    "Type": "This work",
                }
            )

        col_lit1, col_lit2 = st.columns(2)

        with col_lit1:
            st.subheader("Baseline (default equipment)")
            lit_combined_base = pd.concat([lit_df, pd.DataFrame(current_data_base)])
            fig_lit_base = px.bar(
                lit_combined_base,
                x="Material",
                y="GWP_kgCO2_per_kg",
                color="Source",
                title="GWP comparison with literature (baseline)",
                text="Source",
            )
            fig_lit_base.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_lit_base, use_container_width=True)

        with col_lit2:
            st.subheader("Scaled (equipment tab)")
            lit_combined_scaled = pd.concat([lit_df, pd.DataFrame(current_data_scaled)])
            fig_lit_scaled = px.bar(
                lit_combined_scaled,
                x="Material",
                y="GWP_kgCO2_per_kg",
                color="Source",
                title="GWP comparison with literature (scaled)",
                text="Source",
            )
            fig_lit_scaled.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_lit_scaled, use_container_width=True)

    # --- TAB 5: AI INSIGHTS ---
    with tab5:
        st.header("AI insights")
        st.caption(
            "Ask questions about the current results. You can focus on one bead or consider both together."
        )

        # Optional route focus – uses the scaled (current) scenario
        route_name_map = {r["id"]: r["name"] for r in scaled_results_list}
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
            ai_context_results = scaled_results_list
        else:
            chosen_id = None
            for rid, name in route_name_map.items():
                if name == focus_choice:
                    chosen_id = rid
                    break

            if chosen_id is None:
                ai_context_results = scaled_results_list
            else:
                ai_context_results = [r for r in scaled_results_list if r["id"] == chosen_id]

        st.write("Sample questions (click to populate the box):")
        col_q1, col_q2 = st.columns(2)
        sample_questions = [
            "How does changing equipment utilisation affect GWP?",
            "Compare baseline and scaled scenarios for Ref-Bead and U@Bead.",
            "What is the biggest hotspot in the current scaled scenario?",
            "Which equipment changes give the largest reductions in electricity demand?",
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

