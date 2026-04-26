from .agents import BuildingAgent, HomeownerAgent
from mesa import Model
from mesa.datacollection import DataCollector
import networkx as nx
import json
from itertools import combinations

TIME_STEP_LENGTH = 1.0


def create_edge_list(neighborhood_data):
    """Return all within-neighborhood building pairs."""
    edge_list: list[tuple[int, int]] = []
    for neighbourhood in neighborhood_data:
        building_pairs = combinations(neighbourhood["buildings"], 2)
        for pair in building_pairs:
            edge_list.append(pair)
    return edge_list

# reporting functions for datacollector
def renovated_buildings_per_year(model):
    renovated_buildings = [building for building in model.buildings if building.renovated]
    return renovated_buildings

def number_of_renovations_per_year(model):
    return sum(1 for building in model.buildings if building.renovated and building.renovation_year == model.year)

class SimulationModel(Model):
    """
    Simulation model for building renovation decisions, incorporating social influence and economic factors.
    """
    def __init__(
        self,
        buildings_json_path: str,
        nbhd_json_path: str,
        number_of_years: int,
        median_annual_income: int,
        initial_year: int,
        behavioral_threshold: float,
        random_threshold: float,
        tolerable_cost_pct: float,
        W_nb: float = 0.38,
        W_cost: float = 0.62,
        mu_RA: float = 0.2,
        random_attitude: bool = True,
        attitude_mean: float = 0.5,
        attitude_std: float = 1.0,
        subsidy_rate_1: float = 0.4,
        subsidy_rate_2: float = 0.3,
        subsidy_cap_1: float = 35,
        subsidy_cap_2: float = 50 
    ) -> None:
        super().__init__()
        # Time parameters
        self.year: int = initial_year
        self.time_step_length: float = TIME_STEP_LENGTH
        self.max_time: float = number_of_years * (1.0 / self.time_step_length)

        # Economic parameters
        self.tolerable_cost_pct: float = tolerable_cost_pct
        self.median_annual_income: int = median_annual_income

        # Subsidies rates (SR) and capped maximum amount per m2 (SC)
        self.SC1 = subsidy_cap_1
        self.SR1 = subsidy_rate_1
        self.SC2 = subsidy_cap_2    
        self.SR2 = subsidy_rate_2

        # RA parameter
        self.mu_RA: float = mu_RA

        # PBC parameters
        self.behavioral_threshold: float = behavioral_threshold
        self.random_threshold: float = random_threshold
        self.W_nb: float = W_nb
        self.W_cost: float = W_cost

        # Attitude parameters
        self.random_attitude: bool = random_attitude
        self.attitude_mean: float = attitude_mean
        self.attitude_std: float = attitude_std

        # Data structures for agents and social network
        self.owners_id_map: dict[int, HomeownerAgent] = {}
        self.social_network: nx.Graph = nx.Graph()
        self.buildings_owners_map: dict[int | str, list[int]] = {}

        with open(buildings_json_path, "r") as f:
            loaded_building_data = json.load(f)
        
        next_homeowner_id = 0
        for building_record in loaded_building_data:
            building_agent = BuildingAgent(self, building_record)
            owner_ids_for_building: list[int] = []

            for area, price in zip(
                building_record["Net floor area of apartments"],
                building_record["Apartment prices"],
            ):
                homeowner_agent = HomeownerAgent(
                    next_homeowner_id,
                    self,
                    building_agent,
                    area,
                    price
                )
                building_agent.apartments.append(homeowner_agent)
                self.owners_id_map[next_homeowner_id] = homeowner_agent
                self.social_network.add_node(next_homeowner_id)
                owner_ids_for_building.append(next_homeowner_id)
                next_homeowner_id += 1

            for i, j in combinations(owner_ids_for_building, 2):
                self.social_network.add_edge(i, j)

            self.buildings_owners_map[building_record["Building id"]] = owner_ids_for_building

            with open(nbhd_json_path, "r") as f:
                nbhd_data = json.load(f)
            building_edge_pairs = create_edge_list(nbhd_data)

        for u_building_id, v_building_id in building_edge_pairs:
            if not (
                u_building_id in self.buildings_owners_map
                and v_building_id in self.buildings_owners_map
            ):
                continue
            for owner_u in self.buildings_owners_map[u_building_id]:
                owner_v =  self.random.choice(self.buildings_owners_map[v_building_id])
                self.social_network.add_edge(owner_u, owner_v)
        self.rewire_graph_ws(p=0.1)

        self.datacollector = DataCollector(
            agenttype_reporters={
                BuildingAgent: {
                    "building_id": lambda a: a.building_id,
                    "renovation_year": lambda a: int(a.renovation_year)
                },
            },
        )
        
    def rewire_graph_ws(self, p,):
        nodes = list(self.social_network.nodes())
        # Create a list of edges to iterate over to avoid 'dictionary changed size' errors
    
        for u, v in self.social_network.edges():
            if self.random.random() < p:
                # Find a new target 'w' that isn't 'u' and isn't already connected to 'u'
                choices = set(nodes) - {u} - set(self.social_network[u])
                if choices:
                    w = self.random.choice(list(choices))
                    self.social_network.remove_edge(u, v)
                    self.social_network.add_edge(u, w)
    
    def step(self):
        for agent_id, agent in self.owners_id_map.items():
            neighbour_nodes = list(self.social_network.neighbors(agent_id))
            if not neighbour_nodes:
                continue

            neighbour_id = self.random.choice(neighbour_nodes)
            neighbour_agent = self.owners_id_map[neighbour_id]

            op_i = agent.calculate_opinion()
            op_j = neighbour_agent.calculate_opinion()
            uncertainty_i = agent.uncertainty
            uncertainty_j = neighbour_agent.uncertainty

            opinion_overlap = min(op_j + uncertainty_j, op_i + uncertainty_i) - max(
                op_j - uncertainty_j, op_i - uncertainty_i
            )
            if opinion_overlap > uncertainty_i:
                adjustment_factor = opinion_overlap / uncertainty_j - 1.0
                updated_opinion = op_i + self.mu_RA * adjustment_factor * (op_j - op_i)
                updated_opinion = min(1.0, max(updated_opinion, -1.0))

                agent.update_attitude(updated_opinion)
                agent.uncertainty = uncertainty_i + self.mu_RA * adjustment_factor * (
                    uncertainty_j - uncertainty_i
                )

        self.agents_by_type[BuildingAgent].shuffle_do("step")

        self.datacollector.collect(self)
        if self.steps % int(1.0 / self.time_step_length) == 0:
            self.year += 1