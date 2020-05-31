from src.tests.data_generation.delphi import RandomDELPHIData

data_generator = RandomDELPHIData(
    n_regions=50,
    n_risk_classes=2,
    n_timesteps=100,
    days_per_timestep=1,
    vaccine_budget_pct=0.01
)
model = data_generator.generate_model()
solution, trajectories = model.solve(
    exploration_tol=0.02,
    termination_tol=1,
    n_restarts=1,
    max_iterations=2,
    output_flag=False,
    log=True
)

vaccines = solution.vaccinated
old_objective = solution.get_total_deaths()
new_vaccine, objective, bound = model.solve_benchmark(vaccines, check_warm_start_feasibility=True)

new_objective = model.simulate(new_vaccine)
print(old_objective, new_objective)

