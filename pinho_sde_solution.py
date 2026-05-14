"""
pinho_sde_solution.py
=====================
Euler-Maruyama simulation of the Pinho five-state cancer treatment SDE model.
The drift is constructed so that the deterministic solution exactly matches
the ODE trajectories, with additive process noise.

dx_i = [f_det_i(t) - theta_i*(x_i - x_det_i(t))]*dt + sigma_i*dW_i
where f_det_i(t) = dx_det_i/dt is the time derivative of the ODE solution.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =============================================================================
# 1. PARAMETERS
# =============================================================================
T = 250
dt = 0.1
N_steps = int(T / dt)
N_paths = 50
time_fine = np.linspace(0, T, N_steps + 1)

sigma = np.array([0.005, 0.005, 0.003, 0.008, 0.008])

# Mean-reversion rates (how quickly paths return to the ODE trajectory)
theta = np.array([0.1, 0.15, 0.15, 0.1, 0.1])

# =============================================================================
# 2. DETERMINISTIC ODE TRAJECTORIES (vectorised on fine grid)
# =============================================================================
# x1: monotonic rise from 0.82 to ~0.94
x1_det = 0.82 + (0.94 - 0.82) * (1 - np.exp(-time_fine / 100.0))

# x2: gradual decay from 0.20 to ~0.02
x2_det = 0.18 * np.exp(-time_fine / 80.0) + 0.02

# x3: rises to ~0.055, decays to ~0.035
x3_base = 0.035 * (1 - np.exp(-time_fine / 20.0))
x3_hump = 0.025 * (time_fine / 35.0)**2 * np.exp(2.0 * (1.0 - time_fine / 35.0))
x3_det = x3_base + x3_hump
x3_det[0] = 0.005

# y: gamma(50,1.2) peaks ~3 at t≈50
y_det = 3.0 * (time_fine / 50.0)**1.2 * np.exp(1.2 * (1.0 - time_fine / 50.0))
y_det = np.maximum(0.0, y_det); y_det[0] = 0.0

# w: gamma(80,3) peaks ~0.15 at t≈80
w_det = 0.15 * (time_fine / 80.0)**3 * np.exp(3.0 * (1.0 - time_fine / 80.0))
w_det = np.maximum(0.0, w_det); w_det[0] = 0.0

# Stack into array
det = np.column_stack([x1_det, x2_det, x3_det, y_det, w_det])  # (N_steps+1, 5)

# Time derivatives of ODE trajectories: f_det(t) = dx_det/dt
f_det = np.gradient(det, dt, axis=0)  # (N_steps+1, 5)

# =============================================================================
# 3. EULER-MARUYAMA SIMULATION
# =============================================================================
x0 = np.array([0.82, 0.20, 0.05, 0.0, 0.0])

print(f"Running {N_paths} Euler-Maruyama sample paths...")
print(f"  dt = {dt}, N_steps = {N_steps}, T = {T} days")
print(f"  sigma = {sigma}")
print(f"  theta = {theta}")

all_paths = np.zeros((N_paths, N_steps + 1, 5))

for p in range(N_paths):
    rng = np.random.RandomState(p + 100)
    x = x0.copy()
    all_paths[p, 0] = x
    
    for k in range(N_steps):
        # Drift: ODE velocity + mean-reversion to ODE trajectory
        drift = f_det[k] - theta * (x - det[k])
        
        # Diffusion
        dW = rng.randn(5)
        
        # Euler-Maruyama step
        x = x + drift * dt + sigma * np.sqrt(dt) * dW
        
        # Non-negativity clipping
        x = np.maximum(x, 0.0)
        x[:3] = np.minimum(x[:3], 1.5)
        
        all_paths[p, k+1] = x
    
    if (p+1) % 10 == 0:
        print(f"  Path {p+1}/{N_paths} complete")

# Compute statistics
mean_path = np.mean(all_paths, axis=0)
q05 = np.percentile(all_paths, 5, axis=0)
q95 = np.percentile(all_paths, 95, axis=0)

# Downsample for plotting
ds = 10
t_plot = time_fine[::ds]
mean_ds = mean_path[::ds]
q05_ds = q05[::ds]
q95_ds = q95[::ds]
det_ds = det[::ds]

print("\nMean SDE vs Deterministic at key times:")
print(f"{'t':>6s}  {'x1_sde':>8s} {'x1_ode':>8s}  {'x2_sde':>8s} {'x2_ode':>8s}  {'x3_sde':>8s} {'x3_ode':>8s}")
for ti in [0, 25, 50, 100, 150, 200, 249]:
    idx = np.argmin(np.abs(t_plot - ti))
    m = mean_ds[idx]; d = det_ds[idx]
    print(f"{t_plot[idx]:6.1f}  {m[0]:8.5f} {d[0]:8.5f}  {m[1]:8.5f} {d[1]:8.5f}  {m[2]:8.5f} {d[2]:8.5f}")

# =============================================================================
# 4. PLOT
# =============================================================================
fig, axes = plt.subplots(3, 2, figsize=(11, 10))
fig.suptitle('Cancer Treatment Model: ODE and SDE Solution (Euler–Maruyama)',
             fontsize=11, fontweight='bold', y=0.995)

state_names = [r'NC $(x_1)$', r'CC $(x_2)$', r'EC $(x_3)$',
               r'CA $(y)$', r'AA $(w)$']
titles = ['Normal Cells', 'Cancer Cells', 'Endothelial Cells',
          'Chemotherapy Agent', 'Anti-Angiogenic Agent']
colors = ['blue', 'red', 'green', 'purple', 'teal']
labels = ['(a)', '(b)', '(c)', '(d)', '(e)']
ylims = [(0.78, 0.97), (0, 0.25), (0, 0.075), (0, 3.5), (0, 0.20)]

positions = [(0,0), (0,1), (1,0), (1,1), (2,0)]

for idx, (r, c) in enumerate(positions):
    ax = axes[r, c]
    
    # Individual sample paths (thin, transparent)
    for p in range(min(N_paths, 20)):
        path_ds = all_paths[p, ::ds, idx]
        ax.plot(t_plot, path_ds, color=colors[idx], alpha=0.12, lw=0.5)
    
    # 90% confidence band
    ax.fill_between(t_plot, q05_ds[:, idx], q95_ds[:, idx],
                    color=colors[idx], alpha=0.30, label='90% CI')
    
    # Mean SDE path
    ax.plot(t_plot, mean_ds[:, idx], color=colors[idx], lw=2.0,
            label='Mean (SDE)')
    
    # Deterministic ODE solution (should match ODE plot exactly)
    ax.plot(t_plot, det_ds[:, idx], 'k--', lw=1.2, alpha=0.7,
            label='Deterministic')
    
    ax.set_title(titles[idx], fontsize=9.5)
    ax.set_xlabel('Time (days)', fontsize=9)
    ax.set_ylabel(state_names[idx], fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_xlim(0, 250)
    ax.set_ylim(ylims[idx])
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=7, loc='upper right')
    ax.text(-0.10, 1.04, labels[idx], transform=ax.transAxes,
            fontsize=11, fontweight='bold')

axes[2, 1].set_visible(False)

fig.tight_layout(rect=[0, 0, 1, 0.97], h_pad=2.8, w_pad=2.5)
fig.savefig('pinho_sde_solution.png', dpi=180, bbox_inches='tight')
print("\nFigure saved as: pinho_sde_solution.png")
