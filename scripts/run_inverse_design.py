"""End-to-end inverse urban cooling: 30 Adam steps on an 80x40 canyon.

Uses a beta continuation schedule (2 -> 4 -> 8 -> 16). Writes to
``artifacts/``: the JSON summary, the optimised design arrays (.npz),
comparison / domain-mask / history figures, and an animated GIF of the
canopy, temperature and velocity fields evolving across the Adam steps.
Initial-vs-final metrics are both evaluated at the final beta (16) so the
comparison is on a single, fixed design parameterisation.
"""
import json
import os

os.environ.setdefault("MPLBACKEND", "Agg")

import jax
import jax.numpy as jnp
import numpy as np
import optax

from urbancool.grid import create_urban_canyon_domain
from urbancool.optimization import ComfortObjective, value_and_grad_fn
from urbancool.physics import SimConfig
from urbancool.solver import rollout_simulation

BURN, OUTER, INNER, STEPS, LR = 200, 10, 15, 30, 0.1
BETA_SCHEDULE = (2.0, 4.0, 8.0, 16.0)


def _stage_steps(total, n_stages):
    base, extra = divmod(total, n_stages)
    return [base + (1 if i < extra else 0) for i in range(n_stages)]


def _frame(out):
    return {k: np.asarray(out[k]) for k in
            ("gamma", "T_mean", "speed_mean", "u_x_mean", "u_y_mean")}


def main():
    domain = create_urban_canyon_domain(80, 40)
    base_config = SimConfig()
    objective = ComfortObjective()
    theta_init = -jnp.ones((80, 40))  # sparse canopy; Adam must add trees
    theta = theta_init
    optimizer = optax.adam(LR)
    opt_state = optimizer.init(theta)
    kw = dict(burn_in_steps=BURN, outer_steps=OUTER, inner_steps=INNER)

    @jax.jit
    def forward(th, cfg):
        return rollout_simulation(th, cfg, domain, **kw)

    # Baseline on the final-beta parameterisation.
    final_config = base_config._replace(beta=BETA_SCHEDULE[-1])
    init_out = forward(theta_init, final_config)
    loss0, aux0 = objective.compute(init_out["T_mean"], init_out["speed_mean"],
                                    init_out["gamma"], domain)
    frames = [_frame(init_out)]
    print("Step | Beta | Total Loss | Mean Ped Temp | Max Ped Wind | Canopy Fraction")
    print(f"{0:4d} | {BETA_SCHEDULE[-1]:4.0f} | {float(loss0):.6f} | "
          f"{float(aux0['ped_temp']):.6f} | {float(aux0['ped_wind_max']):.6f} | "
          f"{float(aux0['canopy_frac']):.4f}  (baseline at final beta)")

    history = []
    step_no = 0
    for beta, n_steps in zip(BETA_SCHEDULE, _stage_steps(STEPS, len(BETA_SCHEDULE))):
        stage_config = base_config._replace(beta=float(beta))

        @jax.jit
        def step(th, st, _cfg=stage_config):
            (loss, aux), grads = value_and_grad_fn(th, _cfg, domain, **kw,
                                                   objective=objective)
            updates, st = optimizer.update(grads, st, th)
            return optax.apply_updates(th, updates), st, loss, aux

        for _ in range(n_steps):
            step_no += 1
            theta, opt_state, loss, aux = step(theta, opt_state)
            row = {"step": step_no, "beta": float(beta),
                   "loss": float(loss), "ped_temp": float(aux["ped_temp"]),
                   "ped_wind": float(aux["ped_wind"]),
                   "ped_wind_max": float(aux["ped_wind_max"]),
                   "canopy_frac": float(aux["canopy_frac"])}
            history.append(row)
            frames.append(_frame(forward(theta, stage_config)))
            print(f"{step_no:4d} | {beta:4.0f} | {row['loss']:.6f} | "
                  f"{row['ped_temp']:.6f} | {row['ped_wind_max']:.6f} | "
                  f"{row['canopy_frac']:.4f}")

    final_out = frames[-1]
    loss_f, aux_f = objective.compute(
        jnp.asarray(final_out["T_mean"]), jnp.asarray(final_out["speed_mean"]),
        jnp.asarray(final_out["gamma"]), domain)
    final_row = {"step": STEPS, "beta": float(BETA_SCHEDULE[-1]),
                 "loss": float(loss_f), "ped_temp": float(aux_f["ped_temp"]),
                 "ped_wind": float(aux_f["ped_wind"]),
                 "ped_wind_max": float(aux_f["ped_wind_max"]),
                 "canopy_frac": float(aux_f["canopy_frac"])}
    summary = {
        "config": {"nx": 80, "ny": 40, "burn_in_steps": BURN,
                   "outer_steps": OUTER, "inner_steps": INNER,
                   "opt_steps": STEPS, "learning_rate": LR,
                   "beta_schedule": list(BETA_SCHEDULE),
                   "canopy_budget": objective.canopy_budget},
        "initial": {"loss": float(loss0), "ped_temp": float(aux0["ped_temp"]),
                    "ped_wind": float(aux0["ped_wind"]),
                    "ped_wind_max": float(aux0["ped_wind_max"]),
                    "canopy_frac": float(aux0["canopy_frac"])},
        "final": final_row,
        "history": history,
        "checks": {
            "loss_decreased": bool(final_row["loss"] < float(loss0)),
            "ped_temp_decreased": bool(final_row["ped_temp"] < float(aux0["ped_temp"])),
            "canopy_within_budget": bool(
                final_row["canopy_frac"] <= objective.canopy_budget),
        },
    }
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/optimization_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    np.savez("artifacts/optimized_design.npz",
             theta_init=np.asarray(theta_init), theta_final=np.asarray(theta),
             gamma_init=frames[0]["gamma"], gamma_final=frames[-1]["gamma"],
             T_mean_init=frames[0]["T_mean"], T_mean_final=frames[-1]["T_mean"],
             speed_mean_init=frames[0]["speed_mean"],
             speed_mean_final=frames[-1]["speed_mean"],
             building_mask=np.asarray(domain.building_mask),
             plantable_mask=np.asarray(domain.plantable_mask),
             pedestrian_mask=np.asarray(domain.pedestrian_mask),
             solar_mask=np.asarray(domain.solar_mask))

    _write_figures(domain, frames, history, summary)
    print(json.dumps(summary["checks"], indent=2))
    print("Wrote artifacts/optimization_summary.json, optimized_design.npz, "
          "urban_cooling_comparison.png, domain_masks.png, "
          "optimization_history.png, optimization_evolution.gif")
    assert summary["checks"]["loss_decreased"], "loss did not decrease"
    assert summary["checks"]["ped_temp_decreased"], "ped temp did not decrease"
    assert summary["checks"]["canopy_within_budget"], "canopy budget violated"


def _overlay_buildings(ax, building):
    ax.imshow(building.T, origin="lower", cmap="gray", alpha=building.T * 0.7)


def _write_figures(domain, frames, history, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    building = np.asarray(domain.building_mask)
    first, last = frames[0], frames[-1]

    # Before/after comparison.
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    for ax, field, title, cmap, vmax in [
            (axes[0, 0], first["gamma"], "Canopy gamma (initial)", "Greens", 1.0),
            (axes[0, 1], last["gamma"], "Canopy gamma (optimized)", "Greens", 1.0),
            (axes[1, 0], first["T_mean"], "T_mean (initial)", "inferno", 0.2),
            (axes[1, 1], last["T_mean"], "T_mean (optimized)", "inferno", 0.2)]:
        im = ax.imshow(field.T, origin="lower", cmap=cmap, vmin=0.0, vmax=vmax)
        _overlay_buildings(ax, building)
        ax.set_title(title); fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig("artifacts/urban_cooling_comparison.png", dpi=120)
    plt.close(fig)

    # Domain masks.
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    for ax, field, title in [
            (axes[0, 0], building, "Building mask"),
            (axes[0, 1], np.asarray(domain.solar_mask), "Solar mask"),
            (axes[1, 0], np.asarray(domain.plantable_mask), "Plantable mask"),
            (axes[1, 1], np.asarray(domain.pedestrian_mask), "Pedestrian mask")]:
        ax.imshow(field.T, origin="lower", cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig("artifacts/domain_masks.png", dpi=120)
    plt.close(fig)

    # Optimization history.
    steps = [r["step"] for r in history]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    axes[0].plot(steps, [r["loss"] for r in history]); axes[0].set_title("Loss")
    axes[1].plot(steps, [r["ped_temp"] for r in history])
    axes[1].set_title("Mean pedestrian temperature")
    axes[2].plot(steps, [r["canopy_frac"] for r in history])
    axes[2].axhline(summary["config"]["canopy_budget"], color="r", ls="--",
                    label="budget")
    axes[2].set_title("Canopy fraction"); axes[2].legend()
    for ax in axes:
        ax.set_xlabel("Adam step"); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("artifacts/optimization_history.png", dpi=120)
    plt.close(fig)

    # Animated evolution: canopy gamma / temperature / velocity+streamlines.
    nx, ny = building.shape
    gx, gy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    fig, axes = plt.subplots(3, 1, figsize=(9, 7.5))

    def draw(frame_idx):
        fr = frames[frame_idx]
        beta = BETA_SCHEDULE[-1] if frame_idx == 0 else history[frame_idx - 1]["beta"]
        for ax in axes:
            ax.clear()
        axes[0].imshow(fr["gamma"].T, origin="lower", cmap="Greens",
                       vmin=0.0, vmax=1.0)
        _overlay_buildings(axes[0], building)
        axes[0].set_title(f"Adam step {frame_idx}/{STEPS} (beta={beta:g}) — canopy gamma")
        axes[1].imshow(fr["T_mean"].T, origin="lower", cmap="inferno",
                       vmin=0.0, vmax=0.2)
        _overlay_buildings(axes[1], building)
        axes[1].set_title("Mean temperature T")
        axes[2].imshow(fr["speed_mean"].T, origin="lower", cmap="viridis",
                       vmin=0.0, vmax=0.08)
        _overlay_buildings(axes[2], building)
        axes[2].streamplot(gx.T, gy.T, fr["u_x_mean"].T, fr["u_y_mean"].T,
                           color="white", density=0.7, linewidth=0.5)
        axes[2].set_title("Mean wind speed |u| + streamlines")
        for ax in axes:
            ax.set_xlim(0, nx - 1); ax.set_ylim(0, ny - 1)
        fig.tight_layout()
        return []

    anim = FuncAnimation(fig, draw, frames=len(frames), blit=False)
    anim.save("artifacts/optimization_evolution.gif",
              writer=PillowWriter(fps=3))
    plt.close(fig)


if __name__ == "__main__":
    main()
