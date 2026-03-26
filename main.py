from simulation.model import SimulationModel
from simulation.agents import BuildingAgent
from simulation.simulation_run import repeated_simulation_runs
import os

BUILDINGS_JSON_PATH = "Koper_Data/buildings_info.json"
NBHD_JSON_PATH = "Koper_Data/clusters.json"
INITIAL_YEAR = 2025
YEARS = 11
MEDIAN_INC = 14709
BEHAVIORAL_THRESHOLD: float = 0.9
RANDOM_THRESHOLD: float = 0.9
TOLERABLE_COST_PCT: float = 0.05
RANDOM_ATTITUDE: bool = True
ATTITUDE_MEAN: float = 0.5


if __name__ == "__main__":
    print(os.getcwd())
    df = repeated_simulation_runs(
        buildings_json_path=BUILDINGS_JSON_PATH,
        nbhd_json_path=NBHD_JSON_PATH,
        number_of_years=YEARS,
        median_annual_income=MEDIAN_INC,
        initial_year=INITIAL_YEAR
        )
    print(df.head())
    print(df.tail())
    df.to_csv("outputs/simulation_results.csv", index=False)