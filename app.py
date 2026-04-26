"""Minimal Streamlit UI for running a single ABM simulation and viewing map results.

How components interact:
- Streamlit input widgets collect user parameters.
- The run button triggers `run_single_simulation_and_save` from the backend.
- The produced buildings CSV is merged with building shapes from GeoJSON.
- A map-like Altair geoshape view shows renovated vs non-renovated buildings.
"""

# TODO: check data collection logic, data merging, and map rendering for correctness 
from __future__ import annotations

import sys
from pathlib import Path

from matplotlib.patches import Patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scipy.stats import truncnorm
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import streamlit as st
from mesa.mesa_logging import get_module_logger

from simulation.simulation_run import repeated_simulation_runs


LOGGER = get_module_logger(__name__)

# Project defaults kept minimal and aligned with existing scripts/notebook.
BUILDINGS_JSON_PATH = "Koper_Data/buildings_info.json"
NBHD_JSON_PATH = "Koper_Data/clusters.json"
MEDIAN_INC = 14709
BEHAVIORAL_THRESHOLD: float = 0.9
RANDOM_THRESHOLD: float = 0.9
TOLERABLE_COST_PCT: float = 0.05
BUILDING_SHAPES_PATH = "Koper_Data/building_shape_filtered.geojson"


@st.cache_data(show_spinner=False)
def load_building_shapes(path: str) -> gpd.GeoDataFrame:
    """Load building polygons and normalize id column for joins."""
    gdf = gpd.read_file(path).to_crs(epsg=3857)
    return gdf

@st.cache_data(show_spinner=False)
def generate_attitude_distribution(attitude_mean: float, attitude_std: float):
    if attitude_std <= 0:
        return [attitude_mean] * 1000

    a = (0 - attitude_mean) / attitude_std
    b = (1 - attitude_mean) / attitude_std

    return truncnorm.rvs(
        a,
        b,
        loc=attitude_mean,
        scale=attitude_std,
        size=1000,
        random_state=42
    )


def prepare_map_data(
    building_shapes: gpd.GeoDataFrame,
    df_main: pd.DataFrame,
) -> gpd.GeoDataFrame:
    """Replicate notebook map logic and return merged geodataframe."""
    merged = building_shapes.merge(df_main, on="building_id", how="left")
    merged['year'] = merged['year'].astype('Int64')
    merged.fillna(0, inplace=True)
    merged.head()
    return merged


def render_renovation_map(merged: gpd.GeoDataFrame, target_year: int) -> plt.Figure:
    """Render the renovated vs non-renovated map in notebook-equivalent style."""

    # Keep duplicates for renovated
    renovated = merged[merged["year"] == target_year]
    # Renovated recently
    renovated_recently = merged[(merged["year"] >= target_year - 50) & (merged["year"] < target_year)].drop_duplicates(subset="building_id")
    print(f"Renovated by recently: {len(renovated_recently)}")
    # Drop duplicates only for not renovated
    not_renovated = merged[~(merged['building_id'].isin(renovated_recently['building_id']) | merged['building_id'].isin(renovated['building_id']))] \
        .drop_duplicates(subset="building_id")

    fig, ax = plt.subplots(figsize=(7, 7))

    # Not renovated = fixed red
    not_renovated.plot(
        ax=ax,
        color="red",
        edgecolor="black",
        linewidth=0.2,
    )

    # renovated recently = fixed orange
    renovated_recently.plot(
        ax=ax,
        color="yellow",
        edgecolor="black",
        linewidth=0.2,
    )

    # Renovated = colored by renovation_probability
    renovated.plot(
        ax=ax,
        column="renovation_probability",
        cmap="YlGn",
        edgecolor="black",
        linewidth=0.3,
        vmax=1.0,
        vmin=0.0,
        legend=True,
        legend_kwds={
            "label": "Renovation probability",
            "shrink": 0.55,   # smaller colorbar
        },
    )

    legend_handles = [
        Patch(facecolor="red", edgecolor="black", label=f"Not renovated"),
        Patch(facecolor="yellow", edgecolor="black", label=f"Renovated after {target_year - 50} and before {target_year}"),
        Patch(facecolor="green", edgecolor="black", label=f"Renovated by {target_year}")
    ]
    ax.legend(handles=legend_handles, loc="best", fontsize=5)

    ax.set_axis_off()
    ax.set_title(f"Renovated vs Non-Renovated Buildings ({target_year})")
    return fig


def main() -> None:
    """Render app layout and run simulation on demand."""
    st.set_page_config(page_title="ABM Simulation UI", layout="wide")
    st.title("Building Renovation ABM")

    # Persisted state so slider works after form submit rerun
    if "merged_map_data" not in st.session_state:
        st.session_state.merged_map_data = None
    if "year_min" not in st.session_state:
        st.session_state.year_min = None
    if "year_max" not in st.session_state:
        st.session_state.year_max = None
    if "target_year" not in st.session_state:
        st.session_state.target_year = None

    with st.form("run_simulation_form"):
        tab1, tab2 = st.tabs(["General settings", "Scenario settings"])
        with tab1:
            st.header("General settings")
            initial_year = st.number_input(
                "Initial year",
                min_value=1900,
                max_value=2200,
                value=2025,
                step=1,
            )
            number_of_years = st.number_input(
                "Number of years to simulate",
                min_value=1,
                max_value=200,
                value=11,
                step=1,
            )
        with tab2:
            st.header("Senario settings")
            col1, col2 = st.columns(2)
            with col1:
                random_threshold_1 = st.number_input(
                    label="Probability of authorities prohibiting renovation [Heritage regime 1]",
                    help="higher -> more likely to prohibit",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.9,
                    step=0.01,
                    format="%.2f",
                )
                random_threshold_2 = st.number_input(
                    label="Probability of authorities prohibiting renovation [Heritage regime 2])",
                    help="higher -> more likely to prohibit",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.9,
                    step=0.01,
                    format="%.2f",
                    disabled=True
                )
                random_threshold_3 = st.number_input(
                    label="Probability of authorities prohibiting renovation [Heritage regime 3]",
                    help="higher -> more likely to prohibit",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.9,
                    step=0.01,
                    format="%.2f",
                    disabled=True
                )
                heritage_threshold_4 = st.number_input(
                    label="Probability of authorities prohibiting renovation [Heritage regime 4]",
                    help="higher -> more likely to prohibit",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.9,
                    step=0.01,
                    format="%.2f",
                    disabled=True
                )
                behavioral_threshold = st.number_input(
                    label="Behavioral threshold",
                    help="lower -> more likely for owners to initiate renovation",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.9,
                    step=0.01,
                    format="%.2f",
                )
        with col2:
            subsidy_rate_1 = st.slider("Subsidy rate for buildings with at most 2 dwellings", 0.0, 1.0, 0.4)
            grant_cap_1 = st.number_input(
                    label="Maximum Grant Cap",
                    help="This is the upper limit of financial aid allowed per unit of measurement (square meters). The actual subsidy you receive is either the percentage of your total costs (e.g., 40%) or this fixed cap, whichever is lower.",
                    min_value=0.0,
                    value=50.0,
                    step=1.0,
                    format="%.2f",
                )
            subsidy_rate_2 = st.slider("Subsidy rate for multi-dwelling buildings", 0.0, 1.0, 0.3)
            grant_cap_2 = st.number_input(
                    label="Maximum Grant Cap",
                    help="This is the upper limit of financial aid allowed per unit of measurement (square meters). The actual subsidy you receive is either the percentage of your total costs (e.g., 40%) or this fixed cap, whichever is lower.",
                    min_value=0.0,
                    value=35.0,
                    step=1.0,
                    format="%.2f",
                )
        run_clicked = st.form_submit_button("Run simulation")

    with st.container(border=True):
        st.subheader("Attitude distribution")
        st.write("This form is used to set the attitude distribution of individual homeowners toward energy-efficient roof retrofitting. It can be used to simulate the outcomes of an advertising campaign by adjusting the average attitude and its variability, allowing you to explore how shifts in public perception might influence adoption patterns in the simulation.")
        col1, col2 = st.columns(2)
        with col1:
            attitude_mean = st.slider(
                    "Mean attitude (higher = more positive towards renovation)",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.01,
                    width=500,
                )
            attitude_std = st.slider(
                    "Attitude standard deviation",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.2,
                    step=0.01,
                    width=500
                )
        with col2:
            distribution = generate_attitude_distribution(attitude_mean, attitude_std)
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.histplot(distribution, bins=10, kde=False, ax=ax, color="red")
            st.pyplot(fig, width=500)
            plt.close(fig)
    

    with st.container(border=True, height=1000):
        st.subheader("Simulation outcomes")
        if run_clicked:
            LOGGER.info("Starting single simulation run from Streamlit UI")

            with st.spinner("Running simulation..."):
                output_df = repeated_simulation_runs(
                    number_of_years=number_of_years,
                    initial_year=initial_year,
                    random_threshold=random_threshold_1,
                    behavioral_threshold=behavioral_threshold,
                    attitude_mean=attitude_mean,
                    attitude_std=attitude_std,
                    buildings_json_path=BUILDINGS_JSON_PATH,
                    nbhd_json_path=NBHD_JSON_PATH,
                    median_annual_income=MEDIAN_INC,
                    subsidy_rate_1=subsidy_rate_1,
                    subsidy_rate_2=subsidy_rate_2,
                    subsidy_cap_1=grant_cap_1,
                    subsidy_cap_2=grant_cap_2 
                )

            building_shapes = load_building_shapes(BUILDING_SHAPES_PATH)
            merged_map_data = prepare_map_data(
                building_shapes=building_shapes,
                df_main=output_df
            )

            st.session_state.merged_map_data = merged_map_data
            st.session_state.year_min = int(initial_year)
            st.session_state.year_max = int(initial_year + number_of_years - 1)
            st.session_state.target_year = st.session_state.year_max

            st.success("Simulation complete.")

        # Always render controls/results if we already have simulation data
        if st.session_state.merged_map_data is not None:
            year_min = st.session_state.year_min
            year_max = st.session_state.year_max

            # Keep selected year in valid range
            if (
                st.session_state.target_year is None
                or st.session_state.target_year < year_min
                or st.session_state.target_year > year_max
            ):
                st.session_state.target_year = year_max

            target_year = st.slider(
                "Renovation year (map controlled by slider)",
                min_value=year_min,
                max_value=year_max,
                value=st.session_state.target_year,
                step=1,
                key="target_year_slider",
            )
            st.session_state.target_year = target_year

            st.subheader("Renovated vs Non-Renovated Buildings")
            st.pyplot(
                render_renovation_map(st.session_state.merged_map_data, target_year=target_year),
                width="content"
            )



if __name__ == "__main__":
    main()
