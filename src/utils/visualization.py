import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple
from models.delphi import DiscreteDELPHISolution


def plot_solution(solution: DiscreteDELPHISolution, figsize: Tuple[float, float] = (15.0, 7.5)) -> plt.figure:
    """
    Plot a visualization of the solution showing the change in population composition over time and the cumulative
    casualties.
    :param solution: a DiscreteDELPHISolution object
    :param figsize: a tuple of floats that specifies the dimension of the plot
    :return: a matplotlib figure object
    """
    # Initialize figure
    fig, ax = plt.subplots(ncols=2, figsize=figsize)

    # Define plot settings
    plot_settings = dict(alpha=0.7, linestyle="solid", marker="" if solution.days_per_timestep < 2 else ".")

    # Plot population breakdown
    n_timesteps = solution.susceptible.shape[-1]
    days = np.arange(n_timesteps) * solution.days_per_timestep
    ax[0].plot(
        days, solution.susceptible.sum(axis=(0, 1)) / solution.population.sum() * 100,
        label="Susceptible", color="tab:blue", **plot_settings
    )
    ax[0].plot(
        days,
        (np.minimum(solution.vaccinated * solution.vaccine_effectiveness, solution.susceptible).sum(axis=(0, 1)).cumsum()
         + solution.recovered.sum(axis=(0, 1))) / solution.population.sum() * 100,
        label="Vaccinated or recovered", color="tab:green", **plot_settings
    )
    ax[0].plot(
        days, (solution.exposed + solution.infectious).sum(axis=(0, 1)) / solution.population.sum() * 100,
        label="Exposed or infectious", color="tab:red", **plot_settings
    )
    ax[0].plot(
        days,
        (solution.hospitalized + solution.quarantined + solution.undetected).sum(axis=(0, 1)) / solution.population.sum() * 100,
        label="Hospitalized, quarantined or undetected", color="tab:orange", **plot_settings
    )
    ax[0].axhline(0, color="black", alpha=0.5, linestyle="--")
    ax[0].axhline(100, color="black", alpha=0.5, linestyle="--")
    ax[0].legend(fontsize=12)
    ax[0].set_xlabel("Days", fontsize=14)
    ax[0].set_ylabel("% of population", fontsize=14)
    ax[0].set_title("Population composition", fontsize=16)
    ax[0].set_ylim([-5, 105])

    # Make plot of cumulative deaths
    ax[1].plot(
        days, solution.deceased.sum(axis=(0, 1)),
        label="Deaths", color="black", **plot_settings
    )
    ax[1].set_xlabel("Days", fontsize=14)
    ax[1].set_ylabel("Cumulative total (k)", fontsize=14)
    ax[1].set_title("Deaths", fontsize=16)
