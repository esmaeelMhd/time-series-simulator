import numpy as np
import pandas as pd
from pandas import DataFrame
import copy
from dataclasses import dataclass
from typing import Any

@dataclass(eq=0)    
class PhosphorusReward:
    """Define constants used in the reward calculation."""
    ideal_target_level: int = 1         # Optimal P-concentration in mg/L
    green_tax: np.float32 = 1.675       # Tax rate per Kg of P in DKK
    JSF_price: np.float32 = 0.20        # Cost per liter of JSF in DKK
    PAX_price: np.float32 = 3.54        # Cost per liter of PAX in DKK
    weight_target: np.float32 = 0.3     # Weighting factor for P-concentration deviations
    weight_tax: np.float32 = 0.4        # Weighting factor for tax penalties
    weight_action: np.float32 = 0.3     # Weighting factor for the cost of actions
    
    data: Any = None
    target: str = None
    q_column: str = None
    
    def __post_init__(self):
        self.initialized = False
        pass
    
    def calculate_reward(self, source='env'):
        """Initializes some of the variables used for calculating the reward."""
        if not self.initialized:
            self._initialize_values()
            
        """Determine the appropriate reward calculation method based on the data context."""
        if source == 'actual':
            return self._calculate_plant_data_rewards()
        else:
            return self._calculate_env_data_rewards()

    def _initialize_values(self):
        self.target_idx = self.data.columns.get_loc(self.target)
        self.q_idx = self.data.columns.get_loc(self.q_column)
        self.q_tank = copy.deepcopy(self.data[self.q_column])
        self.freq = self.data.index.to_series().diff().dropna().mode()[0]
        self.freq_min = self.freq.total_seconds() / 60
        
        self.initialized = True
        
    def _calculate_plant_data_rewards(self):
        """Calculate rewards for the whole dataset within a plant environment."""
        # Extract the target column from the dataset
        target_column = self.data.iloc[:, self.target_idx]
        # Calculate the absolute deviations from the ideal target level
        deviations = np.abs(target_column - self.ideal_target_level)
        max_deviation = deviations.max()
        # Normalize deviations to have a relative scale
        norm_deviation = deviations / max_deviation

        # Calculate the tax based on phosphorus levels and flow rates
        target_kg = target_column * self.q_tank * self.freq_min / (60 * 1000)
        taxes = self.green_tax * target_kg
        # Normalize the tax to be between 0 and 1
        norm_tax = (taxes - taxes.min()) / (taxes.max() - taxes.min())

        # Calculate costs for chemicals used based on flow rates
        JSF_L = self.data.iloc[:, 0] * self.freq_min / 60
        PAX_L = self.data.iloc[:, 2] * self.freq_min / 60
        JSF_costs = self.JSF_price * JSF_L
        PAX_costs = self.PAX_price * PAX_L
        # Normalize costs for actions
        norm_JSF = (JSF_costs - JSF_costs.min()) / (JSF_costs.max() - JSF_costs.min())
        norm_PAX = (PAX_costs - PAX_costs.min()) / (PAX_costs.max() - PAX_costs.min())
        norm_action_cost = norm_JSF + norm_PAX

        # Combine the weighted costs and penalties to form the overall reward
        real_rewards = - (self.weight_target * norm_deviation +
                          self.weight_tax * norm_tax +
                          self.weight_action * norm_action_cost)

        return np.array(real_rewards)

    def _calculate_env_data_rewards(self, targets):
        """Calculate rewards based on the latest data in a non-plant environment."""
        # Calculate phosphorus deviation and normalize it
        target_deviation = abs(targets[-1] - self.ideal_target_level)
        norm_deviation = target_deviation / self.max_deviation
        
        # Update phosphorus load and calculate tax
        target_kg = targets[-1] * self.q_ep[self.round] * self.freq_min / (60 * 1000)
        tax = self.green_tax * target_kg
        tax_min, tax_max = min(self.taxes), max(self.taxes)
        # Normalize the tax to avoid scaling issues
        norm_tax = (tax - tax_min) / (tax_max - tax_min) if tax_max != tax_min else 0

        # Calculate the costs for actions taken and normalize them
        JSF_L = self.actions[-1][0] * self.freq_min / 60
        PAX_L = self.actions[-1][2] * self.freq_min / 60
        JSF_cost = self.JSF_price * JSF_L
        PAX_cost = self.PAX_price * PAX_L
        JSF_min, JSF_max = min(self.JSF_costs), max(self.JSF_costs)
        PAX_min, PAX_max = min(self.PAX_costs), max(self.PAX_costs)
        norm_JSF = (JSF_cost - JSF_min) / (JSF_max - JSF_min) if JSF_max != JSF_min else 0
        norm_PAX = (PAX_cost - PAX_min) / (PAX_max - PAX_min) if PAX_max != PAX_min else 0
        norm_action_cost = norm_JSF + norm_PAX
        
        # Combine all normalized values to calculate the reward
        reward = - (self.weight_target * norm_deviation +
                    self.weight_tax * norm_tax +
                    self.weight_action * norm_action_cost)

        return reward