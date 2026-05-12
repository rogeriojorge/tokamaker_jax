from tokamaker_jax import load_config, solve_from_config
from tokamaker_jax.plotting import save_equilibrium_plot

config = load_config("examples/fixed_boundary.toml")
solution = solve_from_config(config)
print(solution.stats())
save_equilibrium_plot(solution, "outputs/quickstart.png")
