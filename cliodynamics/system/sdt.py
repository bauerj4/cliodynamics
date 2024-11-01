from cliodynamics.base import DynamicalSystem

from typing import Tuple, List

import numpy as np


class SDTModel(DynamicalSystem):
    """
    Structural-Demographic Theory (SDT) model, based on Peter Turchin's theory of population dynamics,
    resources, and elite competition. This model describes the cyclical patterns of societal stability
    and collapse due to structural and demographic pressures.

    References
    ----------
    Turchin, P. (2016). Ages of Discord: A Structural-Demographic Analysis of American History.

    Attributes
    ----------
    birth_rate : float
        Natural growth rate of the population.
    death_rate : float
        Baseline death rate for the population.
    elite_growth_rate : float
        Growth rate of elite wealth.
    resource_depletion_rate : float
        Rate of resource depletion by the population.
    resource_replenish_rate : float
        Rate of resource growth per capita.

    Methods
    -------
    system_equations(t, y)
        Defines the differential equations for the SDT model.
    """

    def __init__(
        self,
        initial_conditions: List[float],
        time_span: Tuple[float, float],
        time_points: np.ndarray,
        birth_rate: float,
        death_rate: float,
        elite_growth_rate: float,
        resource_depletion_rate: float,
        resource_replenish_rate: float,
    ):
        super().__init__(initial_conditions, time_span, time_points)
        self.birth_rate = birth_rate
        self.death_rate = death_rate
        self.elite_growth_rate = elite_growth_rate
        self.resource_depletion_rate = resource_depletion_rate
        self.resource_replenish_rate = resource_replenish_rate

    def system_equations(self, t: float, y: List[float]) -> List[float]:
        """
        Defines the differential equations for the SDT model.

        Parameters
        ----------
        t : float
            Current time in the integration.
        y : list of float
            Current values of [population, resources per capita, elite wealth].

        Returns
        -------
        list of float
            Derivatives [d_population/dt, d_resources_per_capita/dt, d_elite_wealth/dt].
        """
        population, resources_per_capita, elite_wealth = y  # Unpack variables

        # Population dynamics with carrying capacity limited by resources
        d_population_dt = (
            self.birth_rate
            * population
            * (1 - population / (resources_per_capita + 1e-6))
            - self.death_rate * population
        )

        # Economic resources affected by population and elite wealth
        d_resources_per_capita_dt = (
            self.resource_replenish_rate * resources_per_capita
            - self.resource_depletion_rate * population
            - 0.1 * elite_wealth
        )

        # Wealth of elites dependent on population and resource strain
        d_elite_wealth_dt = (
            self.elite_growth_rate * elite_wealth
            - 0.01 * population * elite_wealth / (resources_per_capita + 1e-6)
        )

        return [d_population_dt, d_resources_per_capita_dt, d_elite_wealth_dt]
