# Technical Specification: `UrbanCool-JAX`
**Differentiable Coupled Fluid-Thermal Lattice Boltzmann Solver & Inverse Topology Optimizer**

---

## 1. Mathematical & Physical Formulation

### 1.1 Lattice Discretization (D2Q9 Fluid + D2Q5 Thermal)
To maximize computational throughput and memory efficiency during reverse-mode automatic differentiation, the primary reference engine couples a **D2Q9** lattice for momentum/mass transport with a **D2Q5** lattice for scalar thermal advection-diffusion on a uniform Cartesian grid $\mathbf{x} \in \{0, \dots, N_x-1\} \times \{0, \dots, N_y-1\}$ with lattice spacing $\Delta x = \Delta t = 1$ and lattice speed of sound $c_s = 1/\sqrt{3}$.

#### D2Q9 Discrete Velocities $\mathbf{e}_i$ and Weights $w_i$ ($i \in \{0, \dots, 8\}$)
$$\mathbf{e}_i = \begin{bmatrix} 0 & 1 & 0 & -1 & 0 & 1 & -1 & -1 & 1 \\ 0 & 0 & 1 & 0 & -1 & 1 & 1 & -1 & -1 \end{bmatrix}, \quad w_i = \begin{cases} 4/9 & i = 0 \\ 1/9 & i \in \{1,2,3,4\} \\ 1/36 & i \in \{5,6,7,8\} \end{cases}$$

#### D2Q5 Discrete Velocities $\mathbf{c}_k$ and Weights $\omega_k$ ($k \in \{0, \dots, 4\}$)
$$\mathbf{c}_k = \begin{bmatrix} 0 & 1 & 0 & -1 & 0 \\ 0 & 0 & 1 & 0 & -1 \end{bmatrix}, \quad \omega_k = \begin{cases} 1/3 & k = 0 \\ 1/6 & k \in \{1,2,3,4\} \end{cases}$$

---

### 1.2 Macroscopic Moments & Equilibrium Distributions
At each grid node $\mathbf{x}$ and time step $t$:
- **Fluid Density:** $\rho(\mathbf{x}, t) = \sum_{i=0}^{8} f_i(\mathbf{x}, t)$
- **Shifted Macroscopic Velocity (accounting for external force density $\mathbf{F}$):**
  $$\mathbf{u}(\mathbf{x}, t) = \frac{1}{\rho(\mathbf{x}, t)} \left( \sum_{i=0}^{8} \mathbf{e}_i f_i(\mathbf{x}, t) + \frac{1}{2} \mathbf{F}(\mathbf{x}, t) \right)$$
- **Normalized Temperature Perturbation:** $T(\mathbf{x}, t) = \sum_{k=0}^{4} g_k(\mathbf{x}, t)$

**Second-Order D2Q9 Equilibrium $f_i^{\text{eq}}(\rho, \mathbf{u})$:**
$$f_i^{\text{eq}}(\rho, \mathbf{u}) = w_i \rho \left[ 1 + 3 (\mathbf{e}_i \cdot \mathbf{u}) + \frac{9}{2} (\mathbf{e}_i \cdot \mathbf{u})^2 - \frac{3}{2} \Vert{}\mathbf{u}\Vert{}_2^2 \right]$$

**First-Order Advective D2Q5 Thermal Equilibrium $g_k^{\text{eq}}(T, \mathbf{u})$:**
$$g_k^{\text{eq}}(T, \mathbf{u}) = \omega_k T \left[ 1 + 3 (\mathbf{c}_k \cdot \mathbf{u}) \right]$$

---

### 1.3 Local Smagorinsky Large Eddy Simulation (LES)
Urban airflow exhibits high Reynolds numbers ($Re \sim 10^4 - 10^6$), causing standard BGK collision operators to diverge when viscosity $\nu_0 \to 0$. `UrbanCool-JAX` computes the local strain-rate tensor magnitude $\Vert{}\mathbf{S}\Vert{}$ purely locally from the second-order non-equilibrium momentum flux tensor $\Pi_{\alpha\beta}^{\text{neq}} = \sum_{i=0}^8 e_{i\alpha} e_{i\beta} (f_i - f_i^{\text{eq}})$:
$$\Vert{}\Pi^{\text{neq}}\Vert{}_F = \sqrt{ \left(\Pi_{xx}^{\text{neq}}\right)^2 + 2\left(\Pi_{xy}^{\text{neq}}\right)^2 + \left(\Pi_{yy}^{\text{neq}}\right)^2 + \epsilon_{\text{reg}} }$$
where $\epsilon_{\text{reg}} = 10^{-12}$ ensures finite derivatives at zero shear for `jax.grad`. The effective relaxation time $\tau_{\text{eff}}(\mathbf{x}, t)$ is given analytically by solving the quadratic relation:
$$\tau_{\text{eff}}(\mathbf{x}, t) = \frac{1}{2} \left( \tau_0 + \sqrt{ \tau_0^2 + \frac{18 \sqrt{2} \, C_{\text{smag}}^2 \, \Vert{}\Pi^{\text{neq}}\Vert{}_F}{\rho(\mathbf{x}, t)} } \right), \quad \text{where } \tau_0 = 3 \nu_0 + \frac{1}{2}$$
Similarly, the turbulent thermal relaxation time is coupled via the turbulent Prandtl number $Pr_t \approx 0.85$:
$$\tau_T(\mathbf{x}, t) = 3 \left( \alpha_0 + \frac{\nu_t(\mathbf{x}, t)}{Pr_t} \right) + \frac{1}{2}, \quad \text{where } \nu_t = \frac{\tau_{\text{eff}} - \tau_0}{3}$$

---

### 1.4 Continuous Porous Canopy & Solid Obstacle Parameterization
Every spatial cell $\mathbf{x}$ in the plantable design region $\Omega_{\text{design}}$ carries an unconstrained optimization parameter $\theta(\mathbf{x}) \in \mathbb{R}$.

1. **Spatial Smoothing (2D Consecutive Box/Gaussian Filter):**
   To prevent checkerboard patterns and single-pixel tree artifacts, raw parameters $\theta$ are convolved with a $3\times 3$ normalized spatial smoothing kernel $K_r$:
   $$\hat{\theta}(\mathbf{x}) = (K_r * \theta)(\mathbf{x})$$
2. **Smooth Sigmoidal Projection:**
   $$\tilde{\gamma}(\mathbf{x}) = \text{Mask}_{\text{design}}(\mathbf{x}) \cdot \sigma\left( \beta \hat{\theta}(\mathbf{x}) \right) \in [0, 1]$$
   where $\tilde{\gamma}(\mathbf{x}) = 0$ represents open air and $\tilde{\gamma}(\mathbf{x}) = 1$ represents dense tree canopy.
3. **Coupled Momentum Forcing & Implicit Canopy Drag:**
   To avoid explicit time-step instabilities when drag coefficient $C_d$ is large, the effective macroscopic velocity inside porous canopy $\tilde{\gamma}(\mathbf{x})$ and solid building mask $B(\mathbf{x}) \in \{0, 1\}$ is damped via **Implicit Brinkman-Forchheimer Velocity Correction** before evaluating equilibrium:
   $$\mathbf{u}_{\text{eff}}(\mathbf{x}, t) = \frac{1 - B(\mathbf{x})}{1 + \alpha_{\text{drag}} \tilde{\gamma}(\mathbf{x}) \left(0.1 + \Vert{}\mathbf{u}^*( \mathbf{x}, t )\Vert{}_2\right)} \mathbf{u}^*(\mathbf{x}, t)$$
   (the constant $0.1$ is the linear Brinkman part; $\Vert{}\mathbf{u}^*\Vert{}_2$ is regularised as $\sqrt{u_x^{*2} + u_y^{*2} + 10^{-12}}$). A smooth Mach limiter then caps $\Vert{}\mathbf{u}_{\text{eff}}\Vert{}_2 \le 0.20$ via $\mathbf{u} \leftarrow \mathbf{u} \cdot \min(1, 0.20/\Vert{}\mathbf{u}\Vert{}_2)$, and the macroscopic temperature is clipped to $T \in [-0.5, 2.0]$ so Boussinesq forces stay bounded under aggressive optimisation transients.
   while Boussinesq buoyancy acts along the vertical axis $\hat{\mathbf{j}}$:
   $$\mathbf{F}_{\text{buoy}}(\mathbf{x}, t) = \rho(\mathbf{x}, t) \, g_{\text{grav}} \, \beta_T \left( T(\mathbf{x}, t) - T_{\text{ref}} \right) \hat{\mathbf{j}}$$
   incorporated into the post-collision distribution via **Guo's exact forcing scheme**:
   $$S_i(\mathbf{x}, t) = \left( 1 - \frac{1}{2\tau_{\text{eff}}} \right) w_i \left[ 3(\mathbf{e}_i - \mathbf{u}) + 9(\mathbf{e}_i \cdot \mathbf{u})\mathbf{e}_i \right] \cdot \mathbf{F}_{\text{buoy}}$$
4. **Thermal Sources & Evapotranspiration Sink:**
   Solar-heated building surfaces/asphalt masks $M_{\text{solar}}(\mathbf{x})$ inject heat $Q_{\text{solar}}$, while porous tree canopies $\tilde{\gamma}(\mathbf{x})$ provide latent evaporative cooling $-\Lambda_{\text{cool}} \tilde{\gamma}(\mathbf{x}) (T(\mathbf{x}, t) - T_{\text{wet}})$:
   $$Q_{\text{net}}(\mathbf{x}, t) = \left[ Q_{\text{solar}} M_{\text{solar}}(\mathbf{x}) - \Lambda_{\text{cool}} \tilde{\gamma}(\mathbf{x}) \, \frac{\text{Softplus}\left( 5 \left( T(\mathbf{x}, t) - T_{\text{wet}} \right) \right)}{5} \right] \left(1 - B(\mathbf{x})\right)$$
   $$\tilde{g}_k(\mathbf{x}, t) = g_k(\mathbf{x}, t) - \frac{1}{\tau_T} \left( g_k - g_k^{\text{eq}}(T, \mathbf{u}_{\text{eff}}) \right) + \omega_k Q_{\text{net}}(\mathbf{x}, t)$$

---

### 1.5 Streaming & Boundary Conditions (Pure Array Roll Operations)
1. **Streaming Step:** Executed via `jnp.roll` along each lattice direction $\mathbf{e}_i$ and $\mathbf{c}_k$:
   $$f_i(\mathbf{x}, t+1) = \text{roll}\left(\tilde{f}_i(\mathbf{x}, t), \text{shift}=(e_{ix}, e_{iy}), \text{axis}=(0, 1)\right)$$
2. **Solid Building Obstacles ($B(\mathbf{x}) = 1$) and street ($y = 0$):** Full-way on-site bounce-back: the populations *streamed into* a solid node are reversed in place,
   $$f_i^{\text{bb}}(\mathbf{x}) = f_{\bar{i}}^{\text{streamed}}(\mathbf{x}), \quad \text{where } \mathbf{e}_{\bar{i}} = -\mathbf{e}_i$$
   applied branchlessly via `jnp.where(obstacle[..., None], f_streamed[..., OPP], f_streamed)`. Because this is a permutation of the streamed populations it conserves total mass exactly (overwriting with pre-streaming post-collision values $\tilde{f}_{\bar{i}}$ would discard streamed mass at every solid node and fail the conservation gate below).
3. **West Inlet ($x = 0$):** Prescribed velocity $\mathbf{u}_{\text{in}} = (u_0, 0)$ and ambient temperature $T_{\text{in}} = 0$ imposed via equilibrium boundary condition $f_i(0, y) = f_i^{\text{eq}}(1.0, \mathbf{u}_{\text{in}})$ and $g_k(0, y) = g_k^{\text{eq}}(T_{\text{in}}, \mathbf{u}_{\text{in}})$.
4. **East Outlet ($x = N_x - 1$):** Zero-gradient convective outflow $f_i(N_x-1, y) = f_i(N_x-2, y)$ and $g_k(N_x-1, y) = g_k(N_x-2, y)$.
5. **Top & Street Boundaries ($y = 0, N_y - 1$):** Free-slip specular reflection at top sky boundary ($y = N_y - 1$); no-slip bounce-back + solar thermal flux at street level ($y = 0$).

---

## 2. Differentiable Rollout & Memory Architecture

### 2.1 Two-Phase Rollout (TBPTT + Binomial Checkpointing)
Unrolling $N_{\text{total}} = 1400$ LBM steps naively stores $1400 \times (N_x \times N_y \times 14)$ floating-point tensors on the tape. `UrbanCool-JAX` splits each forward evaluation into two phases:
1. **Phase A — Non-Differentiable Burn-In ($N_{\text{burn}}$ steps):**
   Advances $(f, g)$ from initial rest to quasi-steady-state vortex shedding using `jax.lax.scan`, followed immediately by `jax.lax.stop_gradient((f, g))` so zero memory is allocated for the spin-up trajectory.
2. **Phase B — Checkpointed Differentiable Window ($N_{\text{grad}}$ steps):**
   Factors $N_{\text{grad}} = K_{\text{outer}} \times K_{\text{inner}}$ (e.g., $20 \times 20 = 400$ steps). The inner loop of $K_{\text{inner}}$ steps is wrapped with `@jax.checkpoint`, reducing peak memory from $\mathcal{O}(N_{\text{grad}})$ to $\mathcal{O}(K_{\text{outer}} + K_{\text{inner}}) = \mathcal{O}(\sqrt{N_{\text{grad}}})$.

### 2.2 Composite Inverse Design Objective $\mathcal{L}(\theta)$
Evaluated as the time-averaged macroscopic fields $(\bar{T}, \bar{\mathbf{u}})$ over the differentiable window inside the pedestrian evaluation zone $\Omega_{\text{ped}}$:
$$\mathcal{L}(\theta) = w_T \frac{1}{\vert{}\Omega_{\text{ped}}\vert{}} \sum_{\mathbf{x} \in \Omega_{\text{ped}}} \text{Softplus}_{\tau}\left( \bar{T}(\mathbf{x}) - T_{\text{target}} \right)^2 + w_u \frac{1}{\vert{}\Omega_{\text{ped}}\vert{}} \sum_{\mathbf{x} \in \Omega_{\text{ped}}} \text{Softplus}_{\tau}\left( \Vert{}\bar{\mathbf{u}}(\mathbf{x})\Vert{}_2 - u_{\max} \right)^2 + w_B \text{Softplus}_{\tau}\left( \frac{\sum_{\mathbf{x}} \tilde{\gamma}(\mathbf{x})}{\vert{}\Omega_{\text{design}}\vert{}} - B_{\text{canopy}} \right)^2 + w_{\text{bin}} \frac{1}{\vert{}\Omega_{\text{design}}\vert{}} \sum_{\mathbf{x}} \tilde{\gamma}(\mathbf{x})\left(1 - \tilde{\gamma}(\mathbf{x})\right)$$
where $\text{Softplus}_{\tau}(z) = \frac{1}{\tau} \ln(1 + \exp(\tau z))$ with sharpness $\tau = 20.0$.

---

## 3. Verification & Benchmark Suite

1. **Mass Conservation & Invariance (`tests/test_invariants.py`):**
   - In a periodic domain with obstacles and zero boundary flux, total fluid mass $\sum_{\mathbf{x}, i} f_i(\mathbf{x}, t)$ over 200 steps must be conserved to `< 1e-5` relative error (the gate asserted by the test). Measured over 500 steps: $\approx 7 \times 10^{-6}$ in `float32` and $\approx 5 \times 10^{-14}$ in `float64` — i.e. roundoff-limited, as expected since streaming, bounce-back and collision are each exactly mass-conserving operations.
2. **Coupled Thermal Plume & Canopy Cooling (`tests/test_thermal.py`):**
   - Verify that activating solar heating $M_{\text{solar}}$ strictly increases downstream temperature ($\Delta \bar{T} > 0$), and inserting porous canopy $\tilde{\gamma} = 0.8$ strictly reduces downstream pedestrian temperature and velocity magnitude compared to $\tilde{\gamma} = 0.0$.
3. **Exact vs. Finite-Difference Gradient Checks (`tests/test_gradients.py`):**
   - Enable `jax_enable_x64 = True`. Compute analytical gradients $\mathbf{g}_{\text{ad}} = \nabla_{\theta} \mathcal{L}(\theta)$ via `jax.grad` on a $32 \times 20$ grid and compare against central finite differences $g_{\text{fd}, m} = \frac{\mathcal{L}(\theta + \epsilon \mathbf{e}_m) - \mathcal{L}(\theta - \epsilon \mathbf{e}_m)}{2\epsilon}$ ($\epsilon = 10^{-5}$) at sampled design pixels. **Pass Criterion:** cosine similarity $> 0.999$ and max relative error $< 10^{-3}$.
   - Two complementary checks are asserted: (a) the full objective with 20 burn-in + 20 differentiable steps (measured max relative error $\approx 4 \times 10^{-5}$); at this point the canopy-budget term dominates the gradient, so (b) a *physics-active* check sets the budget/binarisation weights to zero, the temperature target to $0$, and burn-in to $0$ over 60 differentiable steps, so finite differences measure exactly the un-truncated through-physics gradient that `jax.grad` returns (measured max relative error $\approx 2 \times 10^{-8}$). With burn-in $> 0$, AD intentionally returns the `stop_gradient`-truncated (TBPTT) gradient and is *not* expected to match finite differences of the full forward map wherever the truncated path dominates.
4. **Domain Construction (`tests/test_domain_import.py`):**
   - Plantable and pedestrian masks are ground-rooted (no building cell in the column at or below them, so no rooftop canopy), and externally rasterised building footprints (`urbancool.osm`, plan-view mode) produce mutually disjoint masks on which rollouts stay finite.

---

## 4. Extensions

- **Beta continuation:** `optimize_canopy_layout(..., beta_schedule=(2, 4, 8, 16))` splits the Adam budget across sigmoid-sharpness stages, pushing grey canopy values toward binary placements (the demo's final design has $< 1\%$ of plantable cells with $\tilde{\gamma} \in (0.1, 0.9)$). Initial-vs-final demo metrics are both evaluated at the final $\beta$.
- **Real-world footprints:** `urbancool.osm.domain_from_osm(lat, lon, dist, nx, ny)` (optional `osmnx` extra) rasterises OpenStreetMap buildings into a plan-view domain via the same mask-derivation path as the synthetic canyon; plan-view runs should use $g_{\text{buoy}} = 0$ since buoyancy acts along $+y$.
- **Time-averaged velocity components:** rollouts return $\bar{u}_x, \bar{u}_y$ alongside $\bar{T}$ and $\Vert{}\bar{\mathbf{u}}\Vert{}$, enabling streamline visualisations (see the demo's `optimization_evolution.gif`).
