import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from typing import List, Tuple


class DynamicalSystem:
    """
    Base class for dynamical systems modeled by ordinary differential equations (ODEs).

    Attributes
    ----------
    initial_conditions : list of float
        Initial values for the system variables.
    time_span : tuple of float
        Start and end time for the integration.
    time_points : numpy.ndarray
        Array of time points where solution is evaluated.

    Methods
    -------
    system_equations(t, y)
        Defines the system's differential equations; must be implemented by subclasses.
    solve(method='RK45')
        Solves the system using the specified SciPy ODE solver.
    """

    def __init__(
        self,
        initial_conditions: List[float],
        time_span: Tuple[float, float],
        time_points: np.ndarray,
    ):
        self.initial_conditions = initial_conditions
        self.time_span = time_span
        self.time_points = time_points

    def system_equations(self, t: float, y: List[float]) -> List[float]:
        """Defines the system's differential equations. To be overridden by subclasses."""
        raise NotImplementedError("Subclasses should implement this method.")

    def solve(self, method: str = "RK45") -> solve_ivp:
        """
        Solves the system using a specified SciPy ODE solver.

        Parameters
        ----------
        method : str, optional
            The integration method to use (default is 'RK45').

        Returns
        -------
        solution : solve_ivp
            The solution to the differential equations.
        """
        solution = solve_ivp(
            self.system_equations,
            self.time_span,
            self.initial_conditions,
            t_eval=self.time_points,
            method=method,
        )
        return solution
