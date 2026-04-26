from __future__ import annotations
from mesa import Agent
from scipy.stats import truncnorm

# Income category based on price to income ratio thresholds
LOW_INCOME_THRESHOLD = 8.41
MEDIUM_INCOME_THRESHOLD = 11.93
HIGH_INCOME_THRESHOLD = 15.5

# Rate of renters in the population
RENTERS_RATE = 0.3

# Required agreement percentage for renovation to occur
AGREEMENT_THRESHOLD = 0.75

def uncertainty_map(opinion, eps = 1e-3):
        return (2 * eps) / (abs(opinion) + eps)

class HomeownerAgent(Agent):
    """
    Homeowner agent with decision rule for roof renovation.
    - building: reference to the building agent
    - area: net floor area of the dwelling
    - price: dwelling price
    """

    def __init__(self, homeowner_id, model, building, area, price):
        super().__init__(model)
        self.ho_id = homeowner_id
        self.building = building
        self.net_floor_area = area
        self.dwelling_price = price
        self.income_category = self.calculate_income_category()
        self.attitude = self.calculate_attitude()
        self.pbc = self.calculate_pbc()
        self.uncertainty = uncertainty_map(self.calculate_opinion())
        self.renter= self.random.uniform(0.0, 1.0) <= RENTERS_RATE
        self.willing_to_renov = False
    

    def calculate_income_category(self):
        ratio = self.dwelling_price / self.model.median_annual_income
        if ratio < LOW_INCOME_THRESHOLD:
            return 1  # Low income
        elif LOW_INCOME_THRESHOLD <= ratio < MEDIUM_INCOME_THRESHOLD:
            return 2  # Medium income
        elif MEDIUM_INCOME_THRESHOLD <= ratio < HIGH_INCOME_THRESHOLD:
            return 3  # High income
        else:
            return 4  # Very high income
        
    def calculate_attitude(self):
        # Attitude calculation based on building age and income category, with optional randomness
       if self.model.random_attitude:
            a = (0 - self.model.attitude_mean) / self.model.attitude_std
            b = (1 - self.model.attitude_mean) / self.model.attitude_std
            attitude =  truncnorm.rvs(a, b, loc=self.model.attitude_mean, scale=self.model.attitude_std)
            return max(0.0, min(1.0, attitude))
       else:
           age_score = min(1.0, self.building.years_from_renov / 75.0)
           income_score = 0.25 * self.income_category
           random_bias = self.random.uniform(-0.1, 0.1)
           return 0.4 * age_score + 0.4 * income_score + random_bias
    
    def calculate_pbc(self):
        # Perceived behavioral control calculation based on number of co-owners and cost factors
        co_owners = self.building.num_apartments + self.building.num_business
        tolerable_cost = self.model.tolerable_cost_pct * self.dwelling_price
        estmated_cost = self.building.estimate_cost()

        cost_impact = max(0, 1 - (estmated_cost / tolerable_cost) )
        neiborhood_impact = 1/(1 + co_owners)

        return self.model.W_nb * neiborhood_impact + self.model.W_cost * cost_impact

    def calculate_opinion(self):
        # Map attitude to an opinion value between 0 and 1
        return 2 * self.attitude - 1
    
    def update_attitude(self, opinion):
        # Map opinion value back to attitude
        self.attitude = (opinion + 1) / 2

    def reevaluate_decision(self):
        # Decision rule based on attitude and perceived behavioral control
        if not self.renter:
            behavioural_intent = 0.5 * (self.attitude + self.pbc)
            if behavioural_intent >= self.model.behavioral_threshold:
                self.willing_to_renov = True
        else:
            if self.random.uniform(0.0, 1.0) <= 0.5:
                self.willing_to_renov = True

class BuildingAgent(Agent):
    """
    Building agent that contains information about the building and its owners.

    """

    def __init__(self, model, building_data):
        super().__init__(model)
        self.building_id = building_data["Building id"]
        self.heritage_status = building_data["Cultural heritage status"]
        self.num_apartments = building_data["Number of apartments"]
        self.num_business = building_data["Number of commercial spaces"]

        if building_data["Roof renovation year"] < self.model.year:
            self.years_from_renov = self.model.year - building_data["Roof renovation year"]
            self.renovation_year = building_data["Roof renovation year"]
        else:
            self.years_from_renov = self.model.year - building_data["Construction year"]
            self.renovation_year = building_data["Construction year"]
        
        self.outline_area = building_data["Area of the building outline"]
        self.subsidy_type = 0 if (self.num_apartments + self.num_business) <= 2 else 1
        self.apartments: list[HomeownerAgent] = []
        self.renovated: bool = self.years_from_renov <= 15

    def estimate_cost(self):
        # Simple cost estimation based on outline area and subsidy type, divided by number of co-owners
        total_renovation_cost = self.outline_area * 122.43
        num_owners = self.num_apartments + self.num_business

        if self.subsidy_type == 0:
            subsidised_cost = min(
                total_renovation_cost - self.outline_area * self.model.SC1,
                total_renovation_cost - total_renovation_cost * self.model.SR1,
            )
        else:
            subsidised_cost = min(
                total_renovation_cost - self.outline_area * self.model.SC2,
                total_renovation_cost - total_renovation_cost * self.model.SR2,
            )
        return subsidised_cost / num_owners
    
    def step(self):
        self.years_from_renov += self.model.time_step_length
        if not self.renovated:
            agree_count = 0
            for owner in self.apartments:
                if not owner.willing_to_renov:
                    owner.reevaluate_decision()
                if owner.willing_to_renov:
                    agree_count += 1

            flips = self.random.choices([0, 1], k=self.num_business)
            agree_count += sum(flips)

            if agree_count >= AGREEMENT_THRESHOLD * (self.num_apartments + self.num_business):
                if not self.heritage_status:
                    self.renovated = True
                    self.renovation_year = self.model.year
                elif self.random.uniform(0, 1) > self.model.random_threshold:
                    # random_threshold represents the probability that authorities allow renovation despite heritage status
                    self.renovated = True
                    self.renovation_year = self.model.year
                    for owner in self.apartments:
                        owner.attitude = min(1.0, owner.attitude + 0.25 * owner.attitude)
                else:
                    # If renovation doesn't occur due to heritage status, decrease attitude and PBC for owners
                    # this simulates how authorities' decision can negatively impact homeowners' willingness to renovate in the future
                    for owner in self.apartments:
                        owner.pbc = max(0.0, owner.pbc - 0.5 * owner.pbc)
        