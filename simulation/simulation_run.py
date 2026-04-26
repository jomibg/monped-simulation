from .model import SimulationModel
from .agents import BuildingAgent
import pandas as pd

def repeated_simulation_runs(
    buildings_json_path: str,
    nbhd_json_path: str,
    number_of_years: int,
    median_annual_income: int,
    initial_year: int,
    behavioral_threshold: float = 0.9,
    random_threshold: float = 0.9,
    tolerable_cost_pct: float = 0.05,
    W_nb: float = 0.38,
    W_cost: float = 0.62,
    mu_RA: float = 0.2,
    random_attitude: bool = True,
    attitude_mean: float = 0.5,
    attitude_std: float = 1.0,
    simulations: int = 10,
    subsidy_rate_1: float = 0.4,
    subsidy_rate_2: float = 0.3,
    subsidy_cap_1: float = 35,
    subsidy_cap_2: float = 50 
) -> pd.DataFrame:
    buidling_years_data = dict()
    for _ in range(simulations):
        model = SimulationModel(
            buildings_json_path=buildings_json_path,
            nbhd_json_path=nbhd_json_path,
            number_of_years=number_of_years,
            median_annual_income=median_annual_income,
            initial_year=initial_year,
            behavioral_threshold=behavioral_threshold,
            random_threshold=random_threshold,
            tolerable_cost_pct=tolerable_cost_pct,
            W_nb=W_nb,
            W_cost=W_cost,
            mu_RA=mu_RA,
            random_attitude=random_attitude,
            attitude_mean=attitude_mean,
            attitude_std=attitude_std,
            subsidy_rate_1=subsidy_rate_1,
            subsidy_rate_2=subsidy_rate_2,
            subsidy_cap_1=subsidy_cap_1,
            subsidy_cap_2=subsidy_cap_2 
        )
        model.run_for(model.max_time)
        df_buildings = model.datacollector.get_agenttype_vars_dataframe(BuildingAgent)
        for year in range(number_of_years):
            end_year = initial_year + year
            renovated_buildings = (
            df_buildings[
                (df_buildings["renovation_year"] >= initial_year)
                & (df_buildings["renovation_year"] <= end_year)
            ]["building_id"].unique()
            )
            for building_id in renovated_buildings:
                if (building_id, end_year) not in buidling_years_data:
                    buidling_years_data[(building_id, end_year)] = 0
                buidling_years_data[(building_id, end_year)] += 1
    # Convert the dictionary to a DataFrame
    df = pd.Series(buidling_years_data).reset_index()
    df.columns = ["building_id", "year", "renovation_probability"]
    df["renovation_probability"] = df["renovation_probability"] / simulations
    return df
