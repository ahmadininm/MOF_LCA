# File: app.py
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Attempt to import the new style OpenAI client
try:
    from openai import OpenAI  # type: ignore

    _OPENAI_AVAILABLE = True
except Exception:
    OpenAI = None  # type: ignore
    _OPENAI_AVAILABLE = False


# --------------------------------------------------------------------------------------
# PATHS AND CONSTANTS
# --------------------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

IMPACT_COLUMNS_REAGENTS = [
    "GWP_kgCO2_per_kg",
    "CED_MJ_per_kg",
    "Water_m3_per_kg",
]
IMPACT_COLUMNS_ELEC = [
    "GWP_kgCO2_per_kWh",
    "CED_MJ_per_kWh",
    "Water_m3_per_kWh",
]

COST_COL_REAGENT = "Cost_USD_per_kg"
COST_COL_ELEC = "Cost_USD_per_kWh"

# Route identifiers used for the bead case study in V11
ROUTE_ID_REF = "ref_bead"
ROUTE_ID_MOF = "u_mof_bead"

# Cu removal per kg bead, from V11 Table 2
CU_REMOVED_G_PER_KG = {
    ROUTE_ID_REF: 77.0,
    ROUTE_ID_MOF: 116.0,
}

# Friendly labels for the bead routes
ROUTE_LABELS = {
    ROUTE_ID_REF: "Ref-Bead",
    ROUTE_ID_MOF: "U@Bead-2step-aUiO",
}

# Mapping from reagent name to contribution category for the bead figures
REAGENT_CATEGORY_MAP = {
    "RefBead_polymers": "Polymers",
    "UBead_polymers": "Polymers",
    "MOF_specific_reagents": "MOF-specific reagents",
    "Other_chemicals": "Other chemicals",
}


# --------------------------------------------------------------------------------------
# DATA CLASSES
# --------------------------------------------------------------------------------------


@dataclass
class SynthesisReagent:
    name: str
    role: str
    mass_kg: float


@dataclass
class SynthesisRoute:
    route_id: str
    name: str
    product_name: str
    functional_unit_mass_kg: float
    reagents: List[SynthesisReagent] = field(default_factory=list)
    electricity_kWh: float = 0.0
    electricity_mix: str = "CA_grid"
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "route_id": self.route_id,
            "name": self.name,
            "product_name": self.product_name,
            "functional_unit_mass_kg": self.functional_unit_mass_kg,
            "electricity_kWh": self.electricity_kWh,
            "electricity_mix": self.electricity_mix,
            "notes": self.notes,
            "reagents": [
                {"name": r.name, "role": r.role, "mass_kg": r.mass_kg}
                for r in self.reagents
            ],
        }


# --------------------------------------------------------------------------------------
# IMPACT DATA HANDLING
# --------------------------------------------------------------------------------------


@st.cache_data
def load_reagents_table() -> pd.DataFrame:
    path = DATA_DIR / "reagents.csv"
    if not path.exists():
        st.error(f"Could not find reagents data at {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    required_cols = [
        "reagent_name",
        "GWP_kgCO2_per_kg",
        "CED_MJ_per_kg",
        "Water_m3_per_kg",
        COST_COL_REAGENT,
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"reagents.csv is missing required columns: {missing}")
    return df


@st.cache_data
def load_electricity_mix_table() -> pd.DataFrame:
    path = DATA_DIR / "electricity_mix.csv"
    if not path.exists():
        st.error(f"Could not find electricity mix data at {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    required_cols = [
        "grid_name",
        "GWP_kgCO2_per_kWh",
        "CED_MJ_per_kWh",
        "Water_m3_per_kWh",
        COST_COL_ELEC,
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"electricity_mix.csv is missing required columns: {missing}")
    return df


def list_reagents(df: pd.DataFrame) -> List[str]:
    if df.empty or "reagent_name" not in df.columns:
        return []
    names = sorted(df["reagent_name"].astype(str).unique())
    return names


def list_electricity_grids(df: pd.DataFrame) -> List[str]:
    if df.empty or "grid_name" not in df.columns:
        return []
    names = sorted(df["grid_name"].astype(str).unique())
    return names


def get_reagent_row(
    reagents_df: pd.DataFrame, reagent_name: str
) -> Optional[pd.Series]:
    if reagents_df.empty:
        return None
    matches = reagents_df[reagents_df["reagent_name"] == reagent_name]
    if matches.empty:
        return None
    return matches.iloc[0]


def get_electricity_row(
    elec_df: pd.DataFrame, grid_name: str
) -> Optional[pd.Series]:
    if elec_df.empty:
        return None
    matches = elec_df[elec_df["grid_name"] == grid_name]
    if matches.empty:
        return None
    return matches.iloc[0]


# --------------------------------------------------------------------------------------
# INVENTORY AND ROUTES
# --------------------------------------------------------------------------------------


@st.cache_data
def load_example_routes() -> Dict[str, SynthesisRoute]:
    path = DATA_DIR / "example_routes.csv"
    if not path.exists():
        st.error(f"Could not find example routes at {path}")
        return {}

    df = pd.read_csv(path)
    required_cols = [
        "route_id",
        "name",
        "product_name",
        "functional_unit_mass_kg",
        "reagent_name",
        "reagent_role",
        "reagent_mass_kg",
        "electricity_kWh",
        "electricity_mix",
        "notes",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"example_routes.csv is missing required columns: {missing}")
        return {}

    routes: Dict[str, SynthesisRoute] = {}
    for route_id, group in df.groupby("route_id"):
        group = group.copy()
        group["reagent_name"] = group["reagent_name"].fillna("").astype(str)
        group["reagent_role"] = group["reagent_role"].fillna("").astype(str)

        name = str(group["name"].iloc[0])
        product_name = str(group["product_name"].iloc[0])
        fu_mass = float(group["functional_unit_mass_kg"].iloc[0])

        elec_vals = group["electricity_kWh"].dropna()
        electricity_kWh = float(elec_vals.iloc[0]) if not elec_vals.empty else 0.0

        mix_vals = group["electricity_mix"].dropna()
        electricity_mix = str(mix_vals.iloc[0]) if not mix_vals.empty else "CA_grid"

        note_vals = group["notes"].dropna().astype(str).unique().tolist()
        notes = " | ".join(note_vals)

        reagents: List[SynthesisReagent] = []
        for _, row in group.iterrows():
            rname = row["reagent_name"].strip()
            if not rname:
                continue
            try:
                mass_kg = float(row["reagent_mass_kg"])
            except Exception:
                continue
            reagents.append(
                SynthesisReagent(
                    name=rname,
                    role=row["reagent_role"].strip(),
                    mass_kg=mass_kg,
                )
            )

        routes[str(route_id)] = SynthesisRoute(
            route_id=str(route_id),
            name=name,
            product_name=product_name,
            functional_unit_mass_kg=fu_mass,
            reagents=reagents,
            electricity_kWh=electricity_kWh,
            electricity_mix=electricity_mix,
            notes=notes,
        )

    return routes


def add_user_route_to_state(route: SynthesisRoute) -> None:
    if "user_routes" not in st.session_state:
        st.session_state["user_routes"] = {}
    st.session_state["user_routes"][route.route_id] = route


def get_all_routes() -> Dict[str, SynthesisRoute]:
    routes = dict(load_example_routes())
    user_routes: Dict[str, SynthesisRoute] = st.session_state.get("user_routes", {})
    routes.update(user_routes)
    return routes


def make_new_route_from_ui(
    reagents_df: pd.DataFrame,
    elec_df: pd.DataFrame,
) -> Optional[SynthesisRoute]:
    st.subheader("Create a new synthesis route")

    with st.form("new_route_form", clear_on_submit=False):
        col_a, col_b = st.columns(2)
        name = col_a.text_input("Route name", value="Custom route")
        product_name = col_b.text_input("Product name", value="Custom sorbent")
        fu_mass = col_a.number_input(
            "Functional unit mass [kg product]", min_value=0.001, value=1.0, step=0.1
        )

        grid_options = list_electricity_grids(elec_df)
        if not grid_options:
            grid_options = ["CA_grid"]
        grid = col_b.selectbox("Electricity mix", options=grid_options, index=0)
        electricity_kWh = col_b.number_input(
            "Electricity use per functional unit [kWh]",
            min_value=0.0,
            value=5.0,
            step=0.5,
        )

        st.markdown("**Reagents for this route**")
        reagent_names = list_reagents(reagents_df)
        if not reagent_names:
            st.info("No reagents loaded from reagents.csv")
            reagent_names = []

        default_rows = st.number_input(
            "Number of reagent lines",
            min_value=1,
            max_value=15,
            value=5,
            step=1,
        )

        reagents: List[SynthesisReagent] = []
        for i in range(int(default_rows)):
            st.write(f"Reagent {i + 1}")
            c1, c2, c3 = st.columns([3, 2, 2])
            rname = c1.selectbox(
                f"Reagent name {i + 1}",
                options=[""] + reagent_names,
                index=0,
                key=f"new_route_reagent_name_{i}",
            )
            role = c2.text_input(
                f"Role {i + 1}", value="", key=f"new_route_reagent_role_{i}"
            )
            mass = c3.number_input(
                f"Mass [kg] {i + 1}",
                min_value=0.0,
                value=0.0,
                step=0.01,
                key=f"new_route_reagent_mass_{i}",
            )
            if rname and mass > 0:
                reagents.append(
                    SynthesisReagent(
                        name=rname,
                        role=role,
                        mass_kg=float(mass),
                    )
                )

        notes = st.text_area(
            "Route notes (optional)",
            "User defined route created in the app.",
        )

        submitted = st.form_submit_button("Add route")
        if submitted:
            if not name.strip():
                st.warning("Please provide a route name.")
                return None
            if not reagents:
                st.warning("Please define at least one reagent with mass > 0.")
                return None

            route_id = f"user_route_{len(st.session_state.get('user_routes', {})) + 1}"
            new_route = SynthesisRoute(
                route_id=route_id,
                name=name.strip(),
                product_name=product_name.strip() or name.strip(),
                functional_unit_mass_kg=float(fu_mass),
                reagents=reagents,
                electricity_kWh=float(electricity_kWh),
                electricity_mix=grid,
                notes=notes.strip(),
            )
            add_user_route_to_state(new_route)
            st.success(f"Added route '{new_route.name}' with id {new_route.route_id}.")
            return new_route

    return None


# --------------------------------------------------------------------------------------
# LCA ENGINE
# --------------------------------------------------------------------------------------


def calculate_route_impacts(
    route: SynthesisRoute,
    reagents_df: pd.DataFrame,
    elec_df: pd.DataFrame,
) -> Tuple[pd.Series, pd.DataFrame]:
    """Return total impacts per functional unit and a contribution table."""
    data_rows: List[Dict[str, Any]] = []

    # Reagents
    for r in route.reagents:
        row = get_reagent_row(reagents_df, r.name)
        if row is None:
            data_rows.append(
                {
                    "component": r.name,
                    "type": "reagent",
                    "mass_kg": r.mass_kg,
                    "GWP_kgCO2": np.nan,
                    "CED_MJ": np.nan,
                    "Water_m3": np.nan,
                    "Cost_USD": np.nan,
                    "note": "Reagent not found in reagents.csv",
                }
            )
            continue

        gwp = float(row["GWP_kgCO2_per_kg"]) * r.mass_kg
        ced = float(row["CED_MJ_per_kg"]) * r.mass_kg
        water = float(row["Water_m3_per_kg"]) * r.mass_kg
        cost = float(row.get(COST_COL_REAGENT, 0.0)) * r.mass_kg

        data_rows.append(
            {
                "component": r.name,
                "type": "reagent",
                "mass_kg": r.mass_kg,
                "GWP_kgCO2": gwp,
                "CED_MJ": ced,
                "Water_m3": water,
                "Cost_USD": cost,
                "note": "",
            }
        )

    # Electricity
    if route.electricity_kWh > 0:
        erow = get_electricity_row(elec_df, route.electricity_mix)
        if erow is not None:
            gwp_e = float(erow["GWP_kgCO2_per_kWh"]) * route.electricity_kWh
            ced_e = float(erow["CED_MJ_per_kWh"]) * route.electricity_kWh
            water_e = float(erow["Water_m3_per_kWh"]) * route.electricity_kWh
            cost_e = float(erow.get(COST_COL_ELEC, 0.0)) * route.electricity_kWh
            data_rows.append(
                {
                    "component": f"Electricity ({route.electricity_mix})",
                    "type": "electricity",
                    "mass_kg": np.nan,
                    "GWP_kgCO2": gwp_e,
                    "CED_MJ": ced_e,
                    "Water_m3": water_e,
                    "Cost_USD": cost_e,
                    "note": "",
                }
            )
        else:
            data_rows.append(
                {
                    "component": f"Electricity ({route.electricity_mix})",
                    "type": "electricity",
                    "mass_kg": np.nan,
                    "GWP_kgCO2": np.nan,
                    "CED_MJ": np.nan,
                    "Water_m3": np.nan,
                    "Cost_USD": np.nan,
                    "note": "Electricity mix not found in electricity_mix.csv",
                }
            )

    contrib_df = pd.DataFrame(data_rows)

    totals = pd.Series(
        {
            "GWP_kgCO2": contrib_df["GWP_kgCO2"].sum(skipna=True),
            "CED_MJ": contrib_df["CED_MJ"].sum(skipna=True),
            "Water_m3": contrib_df["Water_m3"].sum(skipna=True),
            "Cost_USD": contrib_df["Cost_USD"].sum(skipna=True),
        }
    )

    return totals, contrib_df


def compare_routes_impacts(
    routes: List[SynthesisRoute],
    reagents_df: pd.DataFrame,
    elec_df: pd.DataFrame,
) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for r in routes:
        totals, _ = calculate_route_impacts(r, reagents_df, elec_df)
        records.append(
            {
                "route_id": r.route_id,
                "route_name": r.name,
                "product_name": r.product_name,
                "GWP_kgCO2_per_FU": totals["GWP_kgCO2"],
                "CED_MJ_per_FU": totals["CED_MJ"],
                "Water_m3_per_FU": totals["Water_m3"],
                "Cost_USD_per_FU": totals["Cost_USD"],
            }
        )
    return pd.DataFrame(records)


# --------------------------------------------------------------------------------------
# PERFORMANCE MODEL
# --------------------------------------------------------------------------------------


@st.cache_data
def load_performance_data() -> pd.DataFrame:
    path = DATA_DIR / "example_performance.csv"
    if not path.exists():
        st.error(f"Could not find performance data at {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    required_cols = [
        "material_id",
        "material_name",
        "linked_route_id",
        "q_max_mg_per_g",
        "removal_efficiency_at_given_conditions_percent",
        "lifetime_cycles_estimate",
        "notes",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"example_performance.csv is missing required columns: {missing}")
    return df


def get_material_performance(
    perf_df: pd.DataFrame, material_id: str
) -> Optional[pd.Series]:
    if perf_df.empty:
        return None
    matches = perf_df[perf_df["material_id"] == material_id]
    if matches.empty:
        return None
    return matches.iloc[0]


def estimate_sorbent_mass_per_m3(
    C0_mg_L: float,
    C_target_mg_L: float,
    q_max_mg_g: float,
    cycles: int,
) -> float:
    if q_max_mg_g <= 0:
        return float("nan")
    if cycles < 1:
        cycles = 1
    deltaC = max(C0_mg_L - C_target_mg_L, 0.0)
    mass_single_use_g = deltaC * 1000.0 / q_max_mg_g
    mass_g = mass_single_use_g / float(cycles)
    return mass_g / 1000.0  # kg per m3


def impacts_per_m3_from_route_and_mass(
    route: SynthesisRoute,
    reagents_df: pd.DataFrame,
    elec_df: pd.DataFrame,
    mass_per_m3_kg: float,
) -> pd.Series:
    totals, _ = calculate_route_impacts(route, reagents_df, elec_df)
    fu_mass = max(route.functional_unit_mass_kg, 1e-9)
    factor = mass_per_m3_kg / fu_mass
    return pd.Series(
        {
            "GWP_kgCO2_per_m3": totals["GWP_kgCO2"] * factor,
            "CED_MJ_per_m3": totals["CED_MJ"] * factor,
            "Water_m3_per_m3": totals["Water_m3"] * factor,
            "Cost_USD_per_m3": totals["Cost_USD"] * factor,
        }
    )


def compute_performance_scenario(
    route: SynthesisRoute,
    perf_row: pd.Series,
    reagents_df: pd.DataFrame,
    elec_df: pd.DataFrame,
    C0_mg_L: float,
    C_target_mg_L: float,
    volume_m3: float,
    cycles_used: int,
) -> Dict[str, Any]:
    qmax = float(perf_row["q_max_mg_per_g"])
    lifetime_cycles = max(int(perf_row["lifetime_cycles_estimate"]), 1)
    cycles_effective = min(int(cycles_used), lifetime_cycles)
    if cycles_effective < 1:
        cycles_effective = 1

    mass_per_m3_kg = estimate_sorbent_mass_per_m3(
        C0_mg_L=C0_mg_L,
        C_target_mg_L=C_target_mg_L,
        q_max_mg_g=qmax,
        cycles=cycles_effective,
    )
    impacts_per_m3 = impacts_per_m3_from_route_and_mass(
        route, reagents_df, elec_df, mass_per_m3_kg
    )

    factor_volume = max(volume_m3, 0.0)
    impacts_total = impacts_per_m3 * factor_volume

    deltaC = max(C0_mg_L - C_target_mg_L, 0.0)
    mg_removed_total = deltaC * 1000.0 * volume_m3
    g_removed_total = mg_removed_total / 1000.0 if mg_removed_total > 0 else float(
        "nan"
    )

    gwp_intensity = (
        impacts_total["GWP_kgCO2_per_m3"] / g_removed_total
        if g_removed_total and not math.isnan(g_removed_total)
        else float("nan")
    )

    return {
        "mass_per_m3_kg": mass_per_m3_kg,
        "impacts_per_m3": impacts_per_m3,
        "impacts_total": impacts_total,
        "g_removed_total": g_removed_total,
        "gwp_per_g_removed": gwp_intensity,
        "cycles_effective": cycles_effective,
        "lifetime_cycles": lifetime_cycles,
    }


# --------------------------------------------------------------------------------------
# AI HELPERS
# --------------------------------------------------------------------------------------


def get_openai_client() -> Tuple[Optional[Any], Optional[str]]():
    if not _OPENAI_AVAILABLE:
        return None, "openai package is not installed. Install it to enable AI features."

    try:
        api_key = st.secrets["openai_api_key2"]
    except Exception:
        return None, (
            "OpenAI API key 'openai_api_key2' not found in Streamlit secrets. "
            "Add it to .streamlit/secrets.toml to enable AI features."
        )

    if not api_key:
        return None, "OpenAI API key is empty."

    try:
        client = OpenAI(api_key=api_key)
    except Exception as e:
        return None, f"Could not create OpenAI client: {e}"
    return client, None


def _build_system_prompt() -> str:
    return (
        "You are a materials sustainability expert helping to interpret screening life cycle "
        "assessment and adsorption performance results for MOF polymer sorbents.\n\n"
        "Very important rules:\n"
        "1. Never invent numerical impact factors or adsorption capacities.\n"
        "2. Never overwrite or quietly change numeric values given in the context.\n"
        "3. You may discuss qualitative trends, trade offs and design options but must not supply new numbers.\n"
        "4. Use clear, concise language aimed at scientists and engineers.\n"
        "5. If information is missing, say that it is missing rather than guessing numbers."
    )


def generate_explanation(context: Dict[str, Any], question: str) -> str:
    client, err = get_openai_client()
    if err:
        return f"[AI disabled] {err}"
    if client is None:
        return "[AI disabled] Could not initialise OpenAI client."

    system_prompt = _build_system_prompt()

    context_str = json.dumps(context, indent=2)
    user_content = (
        "Here is the current context as JSON. It includes route definition, LCA impacts and "
        "performance scenario results where available.\n\n"
        f"{context_str}\n\n"
        f"User question:\n{question}\n\n"
        "Please explain the results and give qualitative eco design suggestions. "
        "Respect all numeric values as fixed observations."
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.3,
            max_tokens=800,
        )
        answer = resp.choices[0].message.content
        return answer or "No answer returned."
    except Exception as e:
        return f"[AI error] {e}"


def parse_synthesis_text_to_inventory(
    text: str,
    available_reagents: List[str],
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Use OpenAI to parse a pasted synthesis into a simple inventory structure."""
    client, err = get_openai_client()
    if err:
        return None, err
    if client is None:
        return None, "Could not initialise OpenAI client."

    system_prompt = (
        "You receive a free text lab synthesis description and a list of known reagent names.\n"
        "Your task is to propose a rough inventory for a single synthesis route.\n\n"
        "Rules:\n"
        "- Only use reagent names from the provided list. Map variants to the closest known reagent.\n"
        "- You may approximate masses if they are clearly given in grams, milligrams, millilitres etc.\n"
        "- Convert all masses to kg.\n"
        "- Do NOT invent new reagents or numeric LCA factors.\n"
        "- If a reagent is only mentioned qualitatively without amounts, either omit it or set mass_kg = 0.\n"
        "- Return a JSON object with keys: route_name, functional_unit_mass_kg, reagents (list of {name, role, mass_kg}).\n"
    )

    user_prompt = (
        "Known reagents:\n"
        + "\n".join(f"- {r}" for r in available_reagents)
        + "\n\nSynthesis text:\n"
        + text
        + "\n\nReturn only valid JSON, no explanation."
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=600,
        )
        content = resp.choices[0].message.content
        if not content:
            return None, "Empty response from model."
        content = content.strip().strip("```").strip()
        if content.lower().startswith("json"):
            content = content[4:].strip()
        data = json.loads(content)
        return data, None
    except Exception as e:
        return None, str(e)


# --------------------------------------------------------------------------------------
# SIMPLE UTILITIES
# --------------------------------------------------------------------------------------


def normalise_scores(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            continue
        col = df[c].astype(float)
        min_val = col.min(skipna=True)
        max_val = col.max(skipna=True)
        if pd.isna(min_val) or pd.isna(max_val) or max_val == min_val:
            df[c + "_norm"] = np.nan
        else:
            df[c + "_norm"] = (col - min_val) / (max_val - min_val)
    return df


def summarise_route_for_context(
    route: Optional[SynthesisRoute],
    totals: Optional[pd.Series],
) -> Dict[str, Any]:
    if route is None:
        return {}
    data = route.to_dict()
    if totals is not None:
        data["impacts_per_FU"] = {
            "GWP_kgCO2": float(totals.get("GWP_kgCO2", float("nan"))),
            "CED_MJ": float(totals.get("CED_MJ", float("nan"))),
            "Water_m3": float(totals.get("Water_m3", float("nan"))),
            "Cost_USD": float(totals.get("Cost_USD", float("nan"))),
        }
    return data


def build_v11_summary_from_routes(
    routes: Dict[str, SynthesisRoute],
    reagents_df: pd.DataFrame,
    elec_df: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise FU1 and FU2 results for the two bead routes as in V11 Table 2."""
    records: List[Dict[str, Any]] = []

    for rid in [ROUTE_ID_REF, ROUTE_ID_MOF]:
        route = routes.get(rid)
        if route is None:
            continue
        totals, contrib = calculate_route_impacts(route, reagents_df, elec_df)

        cat_values = {
            "Electricity": 0.0,
            "Polymers": 0.0,
            "MOF-specific reagents": 0.0,
            "Other chemicals": 0.0,
        }

        for _, row in contrib.iterrows():
            comp = str(row["component"])
            val = float(row["GWP_kgCO2"]) if not pd.isna(row["GWP_kgCO2"]) else 0.0
            if row["type"] == "electricity":
                cat_values["Electricity"] += val
            elif row["type"] == "reagent":
                cat = REAGENT_CATEGORY_MAP.get(comp, "Other chemicals")
                cat_values[cat] += val

        bead_name = ROUTE_LABELS.get(rid, route.name)
        gwp_total = float(totals["GWP_kgCO2"])
        cu_g = CU_REMOVED_G_PER_KG.get(rid, float("nan"))
        gwp_per_g = gwp_total / cu_g if cu_g else float("nan")

        records.append(
            {
                "route_id": rid,
                "Bead type": bead_name,
                "Electricity_kWh_per_kg": route.electricity_kWh,
                "GWP_electricity_kgCO2_per_kg": cat_values["Electricity"],
                "GWP_polymers_kgCO2_per_kg": cat_values["Polymers"],
                "GWP_MOF_reagents_kgCO2_per_kg": cat_values[
                    "MOF-specific reagents"
                ],
                "GWP_other_chemicals_kgCO2_per_kg": cat_values["Other chemicals"],
                "GWP_total_kgCO2_per_kg": gwp_total,
                "Cu_removed_g_per_kg": cu_g,
                "GWP_per_g_Cu_kgCO2_per_g": gwp_per_g,
            }
        )

    return pd.DataFrame(records)


# --------------------------------------------------------------------------------------
# STREAMLIT PAGES
# --------------------------------------------------------------------------------------


def page_home():
    st.title("MOF Sustainability and LCA Explorer")

    st.markdown(
        """
This app implements a screening, gate-to-gate life cycle assessment for a PDChNF–chitosan reference bead (Ref-Bead) and a UiO-66-NH₂ functionalised bead (U@Bead-2step-aUiO), consistent with the V11 LCA section.

Key points:

- The functional units follow V11:
  - FU1: 1 kg of dry bead at the outlet of the final freeze-dryer.
  - FU2: removal of 1 g of Cu²⁺ at about 25 °C from an initial concentration near 100 mg L⁻¹.
- Emission factors and electricity use are calibrated so that the built-in Ref-Bead and U@Bead-2step-aUiO routes reproduce the V11 screening results.
- The app retains the general tools for defining routes, running hotspot analysis and linking impacts to adsorption performance.
- A dedicated page generates the V11-style figures for use in the paper and SI.
        """
    )

    st.info(
        "All results are screening level and reflect unscaled laboratory conditions. "
        "Electricity dominates because full duty times are allocated to sub-gram batches."
    )


def page_routes(reagents_df: pd.DataFrame, elec_df: pd.DataFrame):
    st.header("Define or load synthesis routes")

    routes = get_all_routes()
    if routes:
        option_labels = [
            f"{r.name} (id: {rid})" for rid, r in routes.items()
        ]
        route_ids = list(routes.keys())
        idx_default = 0
        if "selected_route_id" in st.session_state:
            try:
                idx_default = route_ids.index(st.session_state["selected_route_id"])
            except ValueError:
                idx_default = 0
        choice = st.selectbox(
            "Select an existing route",
            options=list(range(len(route_ids))),
            format_func=lambda i: option_labels[i],
            index=idx_default,
        )
        selected_id = route_ids[choice]
        st.session_state["selected_route_id"] = selected_id
        route = routes[selected_id]

        st.subheader("Route inventory preview")
        st.write(f"**Route name:** {route.name}")
        st.write(f"**Product:** {route.product_name}")
        st.write(f"**Functional unit mass:** {route.functional_unit_mass_kg} kg")
        st.write(
            f"**Electricity:** {route.electricity_kWh} kWh of {route.electricity_mix}"
        )
        if route.notes:
            st.caption(route.notes)

        if route.reagents:
            df = pd.DataFrame(
                [
                    {"Reagent": r.name, "Role": r.role, "Mass [kg]": r.mass_kg}
                    for r in route.reagents
                ]
            )
            st.dataframe(df, use_container_width=True)
        else:
            st.info("This route currently has no reagents defined.")

    st.markdown("---")

    st.markdown(
        "You can either define a new route manually or experiment with the AI assisted parser."
    )

    col1, col2 = st.columns(2)

    with col1:
        make_new_route_from_ui(reagents_df, elec_df)

    with col2:
        st.subheader("AI assisted parsing (optional)")
        st.caption(
            "Paste a synthesis description and let the AI suggest a rough inventory. "
            "You can then recreate it manually as a proper route."
        )
        text = st.text_area(
            "Paste synthesis text",
            height=200,
            placeholder="Paste lab procedure here if you want AI to draft an inventory.",
        )
        if st.button("Parse synthesis text with AI"):
            if not text.strip():
                st.warning("Please paste some synthesis text first.")
            else:
                available_reagents = list_reagents(reagents_df)
                inv, err = parse_synthesis_text_to_inventory(text, available_reagents)
                if err:
                    st.error(err)
                elif inv is None:
                    st.error("Could not parse the text into an inventory.")
                else:
                    st.success(
                        "AI suggestion parsed. Inspect the JSON below and create a route manually if useful."
                    )
                    st.json(inv)


def page_lca(reagents_df: pd.DataFrame, elec_df: pd.DataFrame):
    st.header("LCA and hotspot analysis")

    routes = get_all_routes()
    if not routes:
        st.warning("No routes available. Please define or load routes first.")
        return

    route_ids = list(routes.keys())
    option_labels = [f"{routes[rid].name} (id: {rid})" for rid in route_ids]

    idx_default = 0
    if "selected_route_id" in st.session_state:
        try:
            idx_default = route_ids.index(st.session_state["selected_route_id"])
        except ValueError:
            idx_default = 0

    choice = st.selectbox(
        "Select a route for detailed analysis",
        options=list(range(len(route_ids))),
        format_func=lambda i: option_labels[i],
        index=idx_default,
    )
    selected_id = route_ids[choice]
    st.session_state["selected_route_id"] = selected_id
    route = routes[selected_id]

    totals, contrib = calculate_route_impacts(route, reagents_df, elec_df)

    st.subheader("Impacts per functional unit (screening, gate to gate)")
    summary_df = pd.DataFrame(
        {
            "Impact category": [
                "GWP [kg CO₂-eq]",
                "CED [MJ]",
                "Water use [m³]",
                "Cost [USD]",
            ],
            "Value per functional unit": [
                totals["GWP_kgCO2"],
                totals["CED_MJ"],
                totals["Water_m3"],
                totals["Cost_USD"],
            ],
        }
    )
    st.dataframe(summary_df, use_container_width=True)

    st.subheader("Impact breakdown by component")
    if not contrib.empty:
        contrib_plot = contrib.copy()
        contrib_plot["component_clean"] = contrib_plot["component"].astype(str).str.slice(0, 40)
        for col in ["GWP_kgCO2", "CED_MJ", "Water_m3"]:
            fig = px.bar(
                contrib_plot,
                x="component_clean",
                y=col,
                color="type",
                title=f"{col} contribution by component",
            )
            fig.update_layout(
                xaxis_title="Component",
                yaxis_title=col,
                legend_title="Type",
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Compare multiple routes")

    multi_ids = st.multiselect(
        "Select routes to compare",
        options=route_ids,
        default=[selected_id],
        format_func=lambda rid: routes[rid].name,
    )
    if multi_ids:
        selected_routes = [routes[rid] for rid in multi_ids]
        cmp_df = compare_routes_impacts(selected_routes, reagents_df, elec_df)
        st.dataframe(cmp_df, use_container_width=True)

        for col in ["GWP_kgCO2_per_FU", "CED_MJ_per_FU", "Water_m3_per_FU"]:
            fig = px.bar(
                cmp_df,
                x="route_name",
                y=col,
                title=f"{col.replace('_', ' ')} by route",
            )
            fig.update_layout(
                xaxis_title="Route",
                yaxis_title=col.replace("_", " "),
            )
            st.plotly_chart(fig, use_container_width=True)


def page_performance(
    reagents_df: pd.DataFrame,
    elec_df: pd.DataFrame,
):
    st.header("Performance and scenario analysis")

    perf_df = load_performance_data()
    if perf_df.empty:
        st.warning("Performance data not available.")
        return

    routes = get_all_routes()

    material_ids = list(perf_df["material_id"])
    material_labels = [
        f"{row.material_name} (id: {row.material_id})"
        for _, row in perf_df.iterrows()
    ]
    material_choice = st.selectbox(
        "Select a material",
        options=list(range(len(material_ids))),
        format_func=lambda i: material_labels[i],
    )
    material_id = material_ids[material_choice]
    perf_row = get_material_performance(perf_df, material_id)
    if perf_row is None:
        st.error("Selected material not found in performance data.")
        return

    route_id = str(perf_row["linked_route_id"])
    route = routes.get(route_id)
    if route is None:
        st.error(
            f"The route linked to this material ('{route_id}') is not available. "
            "Check example_routes.csv."
        )
        return

    st.write(f"**Material:** {perf_row['material_name']}")
    st.write(f"**Linked route:** {route.name} (id: {route.route_id})")
    st.write(f"**q_max:** {perf_row['q_max_mg_per_g']} mg g⁻¹ (from adsorption data)")
    st.write(
        f"**Estimated lifetime cycles:** {int(perf_row['lifetime_cycles_estimate'])} "
        f"(from regeneration tests / assumptions in example_performance.csv)."
    )
    if isinstance(perf_row.get("notes"), str) and perf_row["notes"]:
        st.caption(perf_row["notes"])

    st.markdown("---")
    st.subheader("Define a water treatment scenario")

    col1, col2, col3, col4 = st.columns(4)
    C0 = col1.number_input(
        "Initial Cu²⁺ concentration C₀ [mg L⁻¹]", min_value=0.0, value=100.0, step=5.0
    )
    Ctarget = col2.number_input(
        "Target concentration Cₜ [mg L⁻¹]",
        min_value=0.0,
        value=5.0,
        step=1.0,
    )
    volume_m3 = col3.number_input(
        "Water volume to treat [m³]", min_value=0.001, value=1.0, step=0.5
    )
    cycles_used = col4.number_input(
        "Number of reuse cycles assumed",
        min_value=1,
        value=int(perf_row["lifetime_cycles_estimate"]),
        step=1,
    )

    if st.button("Run scenario"):
        res = compute_performance_scenario(
            route=route,
            perf_row=perf_row,
            reagents_df=reagents_df,
            elec_df=elec_df,
            C0_mg_L=C0,
            C_target_mg_L=Ctarget,
            volume_m3=volume_m3,
            cycles_used=int(cycles_used),
        )

        st.subheader("Scenario results")
        st.write(
            f"Effective reuse cycles used in calculation: {res['cycles_effective']} "
            f"(lifetime estimate: {res['lifetime_cycles']})"
        )
        st.write(
            f"Estimated sorbent production per m³ of water: "
            f"{res['mass_per_m3_kg']:.4g} kg m⁻³"
        )

        impacts_per_m3 = res["impacts_per_m3"]
        impacts_total = res["impacts_total"]
        g_removed_total = res["g_removed_total"]
        gwp_per_g = res["gwp_per_g_removed"]

        table = pd.DataFrame(
            {
                "Metric": [
                    "GWP per m³ [kg CO₂-eq]",
                    "CED per m³ [MJ]",
                    "Water use per m³ [m³]",
                    "Cost per m³ [USD]",
                    "Total Cu²⁺ removed [g]",
                    "GWP per g Cu²⁺ removed [kg CO₂-eq g⁻¹]",
                ],
                "Value": [
                    impacts_per_m3["GWP_kgCO2_per_m3"],
                    impacts_per_m3["CED_MJ_per_m3"],
                    impacts_per_m3["Water_m3_per_m3"],
                    impacts_per_m3["Cost_USD_per_m3"],
                    g_removed_total,
                    gwp_per_g,
                ],
            }
        )
        st.dataframe(table, use_container_width=True)

        st.markdown("---")
        st.subheader("GWP per m³ vs reuse cycles (comparison)")

        multi_ids = st.multiselect(
            "Select materials to compare",
            options=list(perf_df["material_id"]),
            default=[material_id],
            format_func=lambda mid: perf_df[
                perf_df["material_id"] == mid
            ].iloc[0]["material_name"],
        )

        if multi_ids:
            max_cycles = int(
                max(
                    perf_df[perf_df["material_id"].isin(multi_ids)][
                        "lifetime_cycles_estimate"
                    ]
                )
            )
            max_cycles = max(max_cycles, int(cycles_used))
            cycle_range = list(range(1, max_cycles + 1))
            records: List[Dict[str, Any]] = []

            for mid in multi_ids:
                prow = get_material_performance(perf_df, mid)
                if prow is None:
                    continue
                rid = str(prow["linked_route_id"])
                r = routes.get(rid)
                if r is None:
                    continue
                for c in cycle_range:
                    scenario_res = compute_performance_scenario(
                        route=r,
                        perf_row=prow,
                        reagents_df=reagents_df,
                        elec_df=elec_df,
                        C0_mg_L=C0,
                        C_target_mg_L=Ctarget,
                        volume_m3=1.0,
                        cycles_used=c,
                    )
                    records.append(
                        {
                            "material_id": mid,
                            "material_name": prow["material_name"],
                            "cycles": c,
                            "GWP_kgCO2_per_m3": scenario_res["impacts_per_m3"][
                                "GWP_kgCO2_per_m3"
                            ],
                        }
                    )
            if records:
                df_plot = pd.DataFrame(records)
                fig = px.line(
                    df_plot,
                    x="cycles",
                    y="GWP_kgCO2_per_m3",
                    color="material_name",
                    markers=True,
                    title="GWP per m³ vs reuse cycles",
                )
                fig.update_layout(
                    xaxis_title="Reuse cycles",
                    yaxis_title="GWP per m³ [kg CO₂-eq]",
                )
                st.plotly_chart(fig, use_container_width=True)


def page_ai_assistant(
    reagents_df: pd.DataFrame,
    elec_df: pd.DataFrame,
):
    st.header("AI assistant")

    client, err = get_openai_client()
    if err or client is None:
        st.warning(
            "AI features are disabled. Configure 'openai_api_key2' in .streamlit/secrets.toml to enable them."
        )
        return

    routes = get_all_routes()
    perf_df = load_performance_data()

    st.markdown("Use this assistant to interpret your current route and scenario.")

    # Route selection
    route_ids = list(routes.keys())
    route_choice = st.selectbox(
        "Select a route",
        options=route_ids,
        format_func=lambda rid: routes[rid].name,
    )
    route = routes[route_choice]
    st.session_state["selected_route_id"] = route_choice
    totals, contrib = calculate_route_impacts(route, reagents_df, elec_df)

    # Optional performance scenario
    material_ids = list(perf_df["material_id"]) if not perf_df.empty else []
    material_context: Dict[str, Any] = {}

    if material_ids:
        mat_choice = st.selectbox(
            "Link a material (optional)",
            options=["(None)"] + material_ids,
            format_func=lambda mid: "(None)"
            if mid == "(None)"
            else perf_df[perf_df["material_id"] == mid].iloc[0]["material_name"],
        )
    else:
        mat_choice = "(None)"

    if mat_choice != "(None)" and perf_df is not None and not perf_df.empty:
        perf_row = get_material_performance(perf_df, mat_choice)
        if perf_row is not None:
            st.caption(
                f"Linked material: {perf_row['material_name']} "
                f"(q_max ≈ {perf_row['q_max_mg_per_g']} mg g⁻¹)."
            )
            C0 = st.number_input(
                "Scenario C₀ [mg L⁻¹]", min_value=0.0, value=100.0, step=5.0
            )
            Ctarget = st.number_input(
                "Scenario Cₜ [mg L⁻¹]", min_value=0.0, value=5.0, step=1.0
            )
            volume_m3 = st.number_input(
                "Scenario volume [m³]", min_value=0.001, value=1.0, step=0.5
            )
            cycles_used = st.number_input(
                "Reuse cycles for context",
                min_value=1,
                value=int(perf_row["lifetime_cycles_estimate"]),
                step=1,
            )
            scenario_res = compute_performance_scenario(
                route=route,
                perf_row=perf_row,
                reagents_df=reagents_df,
                elec_df=elec_df,
                C0_mg_L=C0,
                C_target_mg_L=Ctarget,
                volume_m3=volume_m3,
                cycles_used=int(cycles_used),
            )
            material_context = {
                "material_id": mat_choice,
                "material_name": perf_row["material_name"],
                "q_max_mg_per_g": float(perf_row["q_max_mg_per_g"]),
                "scenario": {
                    "C0_mg_L": C0,
                    "C_target_mg_L": Ctarget,
                    "volume_m3": volume_m3,
                    "cycles_used": int(cycles_used),
                },
                "scenario_results": scenario_res,
            }

    st.markdown("---")
    st.subheader("Ask a question")

    preset1 = st.button(
        "Explain why this route has high GWP and suggest eco design modifications."
    )
    preset2 = st.button(
        "Summarise the trade off between adsorption capacity and environmental impact for my scenario."
    )

    user_question = st.text_area(
        "Your question",
        height=150,
        placeholder="Ask about hotspots, design options or trade offs.",
    )

    question_to_send = None
    if preset1:
        question_to_send = (
            "Explain which reagents and process steps dominate the GWP of this route "
            "and suggest high level eco design modifications to reduce it."
        )
    elif preset2:
        question_to_send = (
            "Discuss the trade off between adsorption capacity, reuse cycles and GWP "
            "for my current scenario, compared with a lower capacity but possibly cheaper route."
        )
    elif st.button("Send question"):
        if not user_question.strip():
            st.warning("Please type a question first.")
        else:
            question_to_send = user_question.strip()

    if question_to_send:
        context = {
            "route": summarise_route_for_context(route, totals),
            "contribution_analysis": contrib.to_dict(orient="records"),
        }
        if material_context:
            context["performance"] = material_context
        answer = generate_explanation(context, question_to_send)
        st.markdown("**AI assistant response**")
        st.write(answer)


def page_paper_figures(reagents_df: pd.DataFrame, elec_df: pd.DataFrame):
    st.header("V11 LCA figures for Ref-Bead and U@Bead-2step-aUiO")

    routes = get_all_routes()
    if ROUTE_ID_REF not in routes or ROUTE_ID_MOF not in routes:
        st.error(
            "The built-in Ref-Bead and U@Bead-2step-aUiO routes were not found. "
            "Check example_routes.csv for route_ids 'ref_bead' and 'u_mof_bead'."
        )
        return

    df_v11 = build_v11_summary_from_routes(routes, reagents_df, elec_df)
    if df_v11.empty:
        st.error("Could not summarise V11 style results from the current routes.")
        return

    st.subheader("1. FU1 breakdown: GWP per kg bead by contribution")

    df_long = df_v11.melt(
        id_vars=["Bead type"],
        value_vars=[
            "GWP_electricity_kgCO2_per_kg",
            "GWP_polymers_kgCO2_per_kg",
            "GWP_MOF_reagents_kgCO2_per_kg",
            "GWP_other_chemicals_kgCO2_per_kg",
        ],
        var_name="Contribution",
        value_name="GWP_kgCO2_per_kg",
    )

    contrib_label_map = {
        "GWP_electricity_kgCO2_per_kg": "Electricity",
        "GWP_polymers_kgCO2_per_kg": "Polymers",
        "GWP_MOF_reagents_kgCO2_per_kg": "MOF-specific reagents",
        "GWP_other_chemicals_kgCO2_per_kg": "Other chemicals",
    }
    df_long["Contribution"] = df_long["Contribution"].map(contrib_label_map)

    fig1 = px.bar(
        df_long,
        x="Bead type",
        y="GWP_kgCO2_per_kg",
        color="Contribution",
        barmode="stack",
        title="GWP per kg bead (FU1) by contribution category",
    )
    fig1.update_layout(
        yaxis_title="GWP per kg bead [kg CO₂-eq kg⁻¹]",
        xaxis_title="Bead type",
    )
    fig1.update_yaxes(type="log")
    st.plotly_chart(fig1, use_container_width=True)

    st.caption(
        "This reproduces the per-kilogram screening results in V11 Table 2. "
        "Electricity dominates both beads; polymers and MOF-specific reagents are comparatively small on a log scale."
    )

    st.subheader("2. FU2 comparison: GWP per g Cu removed")

    df_fu2 = df_v11[["Bead type", "GWP_per_g_Cu_kgCO2_per_g"]].copy()
    fig2 = px.bar(
        df_fu2,
        x="Bead type",
        y="GWP_per_g_Cu_kgCO2_per_g",
        title="GWP per g Cu removed (FU2)",
    )
    fig2.update_layout(
        xaxis_title="Bead type",
        yaxis_title="GWP per g Cu removed [kg CO₂-eq g⁻¹]",
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.caption(
        "The U@Bead-2step-aUiO bead has a higher GWP per kg but also higher Cu removal, "
        "so the performance normalised GWP is only about 10–15 percent higher than the polymer bead, "
        "consistent with V11 (about 1.46×10² vs 1.62×10² kg CO₂-eq g⁻¹)."
    )

    st.subheader("3. Positioning relative to literature UiO routes and other adsorbents")

    ref_gwp = float(
        df_v11.loc[df_v11["route_id"] == ROUTE_ID_REF, "GWP_total_kgCO2_per_kg"].iloc[0]
    )
    mof_gwp = float(
        df_v11.loc[df_v11["route_id"] == ROUTE_ID_MOF, "GWP_total_kgCO2_per_kg"].iloc[0]
    )

    lit_records = [
        {
            "System": "Ref-Bead (lab gate to gate)",
            "Category": "This work – beads",
            "GWP_kgCO2_per_kg": ref_gwp,
        },
        {
            "System": "U@Bead-2step-aUiO (lab gate to gate)",
            "Category": "This work – beads",
            "GWP_kgCO2_per_kg": mof_gwp,
        },
        {
            "System": "UiO-66-NH₂ solvothermal A",
            "Category": "UiO LCAs (Luo et al.)",
            "GWP_kgCO2_per_kg": 353.0,
        },
        {
            "System": "UiO-66-NH₂ solvothermal B",
            "Category": "UiO LCAs (Luo et al.)",
            "GWP_kgCO2_per_kg": 180.0,
        },
        {
            "System": "UiO-66-NH₂ aqueous",
            "Category": "UiO LCAs (Luo et al.)",
            "GWP_kgCO2_per_kg": 43.0,
        },
        {
            "System": "UiO-66 (commercial)",
            "Category": "UiO LCAs (Dutta et al.)",
            "GWP_kgCO2_per_kg": 273.8,
        },
        {
            "System": "Biochar (mean)",
            "Category": "Meta-analysis (Arfasa and Tilahun)",
            "GWP_kgCO2_per_kg": 1.2,
        },
        {
            "System": "Biomass adsorbents (mean)",
            "Category": "Meta-analysis (Arfasa and Tilahun)",
            "GWP_kgCO2_per_kg": 2.8,
        },
        {
            "System": "MOFs (mean)",
            "Category": "Meta-analysis (Arfasa and Tilahun)",
            "GWP_kgCO2_per_kg": 25.0,
        },
        {
            "System": "Activated carbon (coal)",
            "Category": "Activated carbon (Gu et al.)",
            "GWP_kgCO2_per_kg": 18.28,
        },
        {
            "System": "Activated carbon (wood)",
            "Category": "Activated carbon (Gu et al.)",
            "GWP_kgCO2_per_kg": 8.60,
        },
    ]
    df_lit = pd.DataFrame(lit_records)

    fig3 = px.bar(
        df_lit,
        x="System",
        y="GWP_kgCO2_per_kg",
        color="Category",
        title="Cradle-to-gate GWP per kg adsorbent – this work vs literature",
    )
    fig3.update_layout(
        xaxis_title="System",
        yaxis_title="GWP per kg adsorbent [kg CO₂-eq kg⁻¹]",
    )
    fig3.update_yaxes(type="log")
    st.plotly_chart(fig3, use_container_width=True)

    st.caption(
        "This reproduces the positioning discussed in V11: literature values for UiO-66 and other adsorbents "
        "are tens to a few hundred kg CO₂-eq kg⁻¹, whereas the very large values for the beads here "
        "mainly reflect laboratory electricity allocation to sub-gram batches, not intrinsic chemistry."
    )


# --------------------------------------------------------------------------------------
# MAIN ENTRY POINT
# --------------------------------------------------------------------------------------


def main():
    st.set_page_config(
        page_title="MOF Sustainability and LCA Explorer",
        layout="wide",
    )

    reagents_df = load_reagents_table()
    elec_df = load_electricity_mix_table()

    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        options=[
            "Home",
            "Define or load synthesis routes",
            "LCA and hotspot analysis",
            "Performance and scenario analysis",
            "Paper figures (V11 LCA)",
            "AI assistant",
        ],
    )

    if page == "Home":
        page_home()
    elif page == "Define or load synthesis routes":
        page_routes(reagents_df, elec_df)
    elif page == "LCA and hotspot analysis":
        page_lca(reagents_df, elec_df)
    elif page == "Performance and scenario analysis":
        page_performance(reagents_df, elec_df)
    elif page == "Paper figures (V11 LCA)":
        page_paper_figures(reagents_df, elec_df)
    elif page == "AI assistant":
        page_ai_assistant(reagents_df, elec_df)


if __name__ == "__main__":
    main()
