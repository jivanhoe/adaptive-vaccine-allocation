import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Optional


def plot_solution(
        vaccines: np.ndarray,
        cases: np.ndarray,
        unimmunized_pop: np.ndarray,
        deaths: np.ndarray,
        figsize: Tuple[float, float] = (8, 6),
        path: Optional[str] = None
) -> None:
    fig = plt.figure(figsize=figsize)

    # Plot aggregate timeseries
    periods = range(vaccines.shape[-1])
    plt.plot(periods, vaccines.sum(axis=(0, 1)).cumsum(), label="Vaccines", alpha=0.7, marker="o")
    plt.plot(periods, cases.sum(axis=(0, 1)).cumsum(), label="Cases", alpha=0.7, marker="o")
    plt.plot(periods, unimmunized_pop.sum(axis=(0, 1)), label="Unmmunized population", alpha=0.7, marker="o")
    plt.plot(periods, deaths.sum(axis=(0, 1)).cumsum(), label="Deaths", alpha=0.7, marker="o")

    # Label figure
    plt.xlabel("Timestep", fontsize=14)
    plt.ylabel("Units", fontsize=14)
    plt.legend(fontsize=12)

    if path:
        fig.savefig(path)

