"""End-to-end inverse urban cooling: 30 Adam steps on an 80x40 canyon."""
import json
import os

os.environ.setdefault("MPLBACKEND", "Agg")

import jax
import jax.numpy as jnp
import optax

from urbancool.grid import create_urban_canyon_domain
from urbancool.optimization import ComfortObjective, value_and_grad_fn
from urbancool.physics import SimConfig
from urbancool.solver import rollout_simulation

BURN, OUTER, INNER, STEPS, LR = 200, 10, 15, 30, 0.1


def main():
    domain = create_urban_canyon_domain(80, 40)
    config = SimConfig()
    objective = ComfortObjective()
    # Start from a sparse canopy (gamma ~ 0.02) so Adam must add trees
    # where the cooling benefit beats the budget/binarisation penalties.
    theta = -jnp.ones((80, 40))
    optimizer = optax.adam(LR)
    opt_state = optimizer.init(theta)

    kw = dict(burn_in_steps=BURN, outer_steps=OUTER, inner_steps=INNER)

    @jax.jit
    def step(theta, opt_state):
        (loss, aux), grads = value_and_grad_fn(theta, config, domain, **kw,
                                               objective=objective)
        updates, opt_state = optimizer.update(grads, opt_state, theta)
        return optax.apply_updates(theta, updates), opt_state, loss, aux

    history = []
    print("Step | Total Loss | Mean Ped Temp | Max Ped Wind | Canopy Fraction")
    (loss0, aux0), _ = value_and_grad_fn(theta, config, domain, **kw,
                                         objective=objective)
    init_out = rollout_simulation(theta, config, domain, **kw)
    print(f"{0:4d} | {float(loss0):.6f} | {float(aux0['ped_temp']):.6f} | "
          f"{float(aux0['ped_wind_max']):.6f} | {float(aux0['canopy_frac']):.4f}")
    for s in range(1, STEPS + 1):
        theta, opt_state, loss, aux = step(theta, opt_state)
        row = {"step": s, "loss": float(loss), "ped_temp": float(aux["ped_temp"]),
               "ped_wind": float(aux["ped_wind"]),
               "ped_wind_max": float(aux["ped_wind_max"]),
               "canopy_frac": float(aux["canopy_frac"])}
        history.append(row)
        print(f"{s:4d} | {row['loss']:.6f} | {row['ped_temp']:.6f} | "
              f"{row['ped_wind_max']:.6f} | {row['canopy_frac']:.4f}")

    final_out = rollout_simulation(theta, config, domain, **kw)
    loss_f, aux_f = objective.compute(final_out["T_mean"], final_out["speed_mean"],
                                      final_out["gamma"], domain)
    final_row = {"step": STEPS, "loss": float(loss_f),
                 "ped_temp": float(aux_f["ped_temp"]),
                 "ped_wind": float(aux_f["ped_wind"]),
                 "ped_wind_max": float(aux_f["ped_wind_max"]),
                 "canopy_frac": float(aux_f["canopy_frac"])}
    summary = {
        "config": {"nx": 80, "ny": 40, "burn_in_steps": BURN,
                   "outer_steps": OUTER, "inner_steps": INNER,
                   "opt_steps": STEPS, "learning_rate": LR,
                   "canopy_budget": objective.canopy_budget},
        "initial": {"loss": float(loss0), "ped_temp": float(aux0["ped_temp"]),
                    "ped_wind_max": float(aux0["ped_wind_max"]),
                    "canopy_frac": float(aux0["canopy_frac"])},
        "final": final_row,
        "history": history,
        "checks": {
            "loss_decreased": bool(final_row["loss"] < float(loss0)),
            "ped_temp_decreased": bool(final_row["ped_temp"] < float(aux0["ped_temp"])),
            "canopy_within_budget": bool(
                final_row["canopy_frac"] <= objective.canopy_budget + 0.05),
        },
    }
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/optimization_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    for ax, field, title, cmap, vmax in [
            (axes[0, 0], init_out["gamma"], "Canopy gamma (initial)", "Greens", 1.0),
            (axes[0, 1], final_out["gamma"], "Canopy gamma (optimized)", "Greens", 1.0),
            (axes[1, 0], init_out["T_mean"], "T_mean (initial)", "inferno", 0.2),
            (axes[1, 1], final_out["T_mean"], "T_mean (optimized)", "inferno", 0.2)]:
        im = ax.imshow(jnp.asarray(field).T, origin="lower", cmap=cmap,
                       vmin=0.0, vmax=vmax)
        ax.imshow(jnp.asarray(domain.building_mask).T, origin="lower",
                  cmap="gray", alpha=jnp.asarray(domain.building_mask).T * 0.7)
        ax.set_title(title); fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig("artifacts/urban_cooling_comparison.png", dpi=120)
    print(json.dumps(summary["checks"], indent=2))
    print("Wrote artifacts/optimization_summary.json and "
          "artifacts/urban_cooling_comparison.png")
    assert summary["checks"]["loss_decreased"], "loss did not decrease"
    assert summary["checks"]["ped_temp_decreased"], "ped temp did not decrease"
    assert summary["checks"]["canopy_within_budget"], "canopy budget violated"


if __name__ == "__main__":
    main()
