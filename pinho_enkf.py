"""
pinho_enkf.py
=============
Ensemble Kalman Filter (EnKF) applied to the Pinho five-state cancer
treatment SDE model for state estimation.

The EnKF uses an ensemble of particles propagated through the Euler-Maruyama
discretisation of the SDE, with observations of [x2, y, w] (same as UKF).

30 Monte Carlo runs, ensemble size N_ens = 50.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, FancyArrowPatch

# =============================================================================
# 1. MODEL & PARAMETERS
# =============================================================================
T = 250; dt = 1.0; L = 5
time_arr = np.arange(0, T, dt)

sigma = np.array([0.005, 0.005, 0.003, 0.008, 0.008])  # process noise

# Observation matrix: observe x2, y, w (indices 1, 3, 4)
H = np.array([[0,1,0,0,0],
              [0,0,0,1,0],
              [0,0,0,0,1]], dtype=float)
meas_noise_std = np.array([0.018, 0.003, 0.003])  # measurement noise std
R = np.diag(meas_noise_std**2)

N_ens = 50   # ensemble size
N_mc  = 30   # Monte Carlo runs

# =============================================================================
# 2. DETERMINISTIC ODE TRAJECTORIES (for drift computation)
# =============================================================================
x1_det = 0.82 + (0.94 - 0.82) * (1 - np.exp(-time_arr / 100.0))
x2_det = 0.18 * np.exp(-time_arr / 80.0) + 0.02
x3_base = 0.035 * (1 - np.exp(-time_arr / 20.0))
x3_hump = 0.025 * (time_arr / 35.0)**2 * np.exp(2.0 * (1.0 - time_arr / 35.0))
x3_det = x3_base + x3_hump; x3_det[0] = 0.005
y_det = 3.0 * (time_arr / 50.0)**1.2 * np.exp(1.2 * (1.0 - time_arr / 50.0))
y_det = np.maximum(0.0, y_det); y_det[0] = 0.0
w_det = 0.15 * (time_arr / 80.0)**3 * np.exp(3.0 * (1.0 - time_arr / 80.0))
w_det = np.maximum(0.0, w_det); w_det[0] = 0.0

det = np.column_stack([x1_det, x2_det, x3_det, y_det, w_det])  # (T, 5)
f_det = np.gradient(det, dt, axis=0)  # drift velocity

# Mean-reversion rates
theta = np.array([0.08, 0.15, 0.15, 0.08, 0.08])

# =============================================================================
# 3. STATE PROPAGATION (Euler-Maruyama step)
# =============================================================================
def propagate_ensemble(ensemble, k, rng):
    """Propagate each ensemble member one step via Euler-Maruyama."""
    N = ensemble.shape[0]
    new_ens = np.zeros_like(ensemble)
    for j in range(N):
        x = ensemble[j]
        drift = f_det[k] - theta * (x - det[k])
        dW = rng.randn(L)
        x_new = x + drift * dt + sigma * np.sqrt(dt) * dW
        x_new = np.maximum(x_new, 0.0)
        x_new[:3] = np.minimum(x_new[:3], 1.5)
        new_ens[j] = x_new
    return new_ens

# =============================================================================
# 4. EnKF UPDATE STEP
# =============================================================================
def enkf_update(ensemble, z, rng):
    """
    EnKF analysis step with perturbed observations.
    
    ensemble: (N_ens, L) forecast ensemble
    z: (3,) observation vector
    Returns: (N_ens, L) analysis ensemble
    """
    N = ensemble.shape[0]
    
    # Ensemble mean and anomalies
    x_mean = ensemble.mean(axis=0)              # (L,)
    A = ensemble - x_mean                        # (N, L) anomalies
    
    # Predicted observations
    Y = ensemble @ H.T                           # (N, 3)
    y_mean = Y.mean(axis=0)                      # (3,)
    D = Y - y_mean                               # (N, 3) observation anomalies
    
    # Covariances
    Pxy = (A.T @ D) / (N - 1)                   # (L, 3) cross-covariance
    Pyy = (D.T @ D) / (N - 1) + R               # (3, 3) innovation covariance
    
    # Kalman gain
    K = Pxy @ np.linalg.inv(Pyy)                # (L, 3)
    
    # Perturbed observations and update
    analysis = np.zeros_like(ensemble)
    for j in range(N):
        z_pert = z + meas_noise_std * rng.randn(3)
        innovation = z_pert - H @ ensemble[j]
        analysis[j] = ensemble[j] + K @ innovation
        # Clip
        analysis[j] = np.maximum(analysis[j], 0.0)
        analysis[j, :3] = np.minimum(analysis[j, :3], 1.5)
    
    return analysis

# =============================================================================
# 5. SINGLE EnKF RUN
# =============================================================================
X0_real = np.array([0.82, 0.20, 0.050, 0.00, 0.00])
X0_est  = np.array([0.82, 0.25, 0.0056, 0.20, 0.10])

def run_enkf(seed):
    rng = np.random.RandomState(seed)
    
    # True state trajectory (Euler-Maruyama)
    xt = X0_real.copy()
    hist_true = np.zeros((T, L))
    hist_true[0] = xt
    
    for k in range(1, T):
        drift = f_det[k-1] - theta * (xt - det[k-1])
        xt = xt + drift * dt + sigma * np.sqrt(dt) * rng.randn(L)
        xt = np.maximum(xt, 0.0); xt[:3] = np.minimum(xt[:3], 1.5)
        hist_true[k] = xt
    
    # Initialise ensemble around X0_est with spread
    ensemble = np.zeros((N_ens, L))
    for j in range(N_ens):
        ensemble[j] = X0_est + np.array([0.001, 0.05, 0.044, 0.20, 0.10]) * rng.randn(L)
        ensemble[j] = np.maximum(ensemble[j], 0.0)
    
    # EnKF loop
    hist_est = np.zeros((T, L))
    hist_est[0] = ensemble.mean(axis=0)
    
    # NEW: Store full ensemble at each timestep
    hist_particles = np.zeros((T, N_ens, L))
    hist_particles[0] = ensemble.copy()
    
    for k in range(1, T):
        # Forecast: propagate ensemble
        ensemble = propagate_ensemble(ensemble, k-1, rng)
        
        # Generate observation from true state
        z = np.maximum(0.0, H @ hist_true[k] + meas_noise_std * rng.randn(3))
        
        # Analysis: EnKF update
        ensemble = enkf_update(ensemble, z, rng)
        
        # Store ensemble mean as estimate
        hist_est[k] = ensemble.mean(axis=0)
        
        # NEW: Store ensemble
        hist_particles[k] = ensemble.copy()
    
    return hist_true, hist_est, hist_particles

# =============================================================================
# 6. MONTE CARLO RUNS
# =============================================================================
print(f"Running {N_mc} EnKF Monte Carlo simulations (N_ens={N_ens})...")
all_true = []; all_est = []; all_particles = []
for seed in range(N_mc):
    ht, he, hp = run_enkf(seed)
    all_true.append(ht); all_est.append(he); all_particles.append(hp)
    print(f"  MC run {seed+1}/{N_mc} complete")

mc_true = np.mean(all_true, axis=0)
mc_est  = np.mean(all_est, axis=0)

# Compute 90% confidence intervals from ensemble
all_particles_array = np.array(all_particles)  # (N_mc, T, N_ens, L)
all_particles_array = np.transpose(all_particles_array, (1, 0, 2, 3))  # (T, N_mc, N_ens, L)
all_particles_flat = all_particles_array.reshape(T, -1, L)  # (T, N_mc*N_ens, L)

q05 = np.percentile(all_particles_flat, 5, axis=1)   # (T, L)
q95 = np.percentile(all_particles_flat, 95, axis=1)  # (T, L)

print("\nMean absolute errors (MC-averaged):")
for i, name in enumerate(['NC(x1)', 'CC(x2)', 'EC(x3)', 'CA(y)', 'AA(w)']):
    err = np.abs(mc_true[:, i] - mc_est[:, i])
    print(f"  {name}: mean={err.mean():.6f}, max={err.max():.6f}")

print("\nState estimates at key times:")
print(f"{'t':>6s}  {'x1_true':>8s} {'x1_est':>8s}  {'x2_true':>8s} {'x2_est':>8s}  {'x3_true':>8s} {'x3_est':>8s}")
for ti in [0, 20, 50, 100, 150, 200, 249]:
    print(f"{ti:6d}  {mc_true[ti,0]:8.5f} {mc_est[ti,0]:8.5f}  "
          f"{mc_true[ti,1]:8.5f} {mc_est[ti,1]:8.5f}  "
          f"{mc_true[ti,2]:8.5f} {mc_est[ti,2]:8.5f}")

# =============================================================================
# 7. PLOT
# =============================================================================
fig, axes = plt.subplots(3, 2, figsize=(10, 9))
fig.suptitle(f'EnKF State/Parameter Estimation: '
             f'Real vs. Estimated – {N_mc} Monte Carlo (N$_{{ens}}$={N_ens}).',
             fontsize=9.5, fontweight='normal', y=0.998)

markevery = 4
legend_elements = [
    Line2D([0],[0], marker='o', color='b', lw=0, markersize=5,
           markerfacecolor='none', markeredgewidth=1.0, label='Real Data'),
    Line2D([0],[0], color='r', lw=1.5, label='EnKF Estimation'),
    matplotlib.patches.Patch(color='r', alpha=0.25, label='90% CI')]

def draw_panel(ax, true_vals, est_vals, q05_vals, q95_vals, ylabel, subtitle, panel_label, ylim=None):
    ax.set_title(subtitle, fontsize=8.5, pad=3)
    
    # 90% CI band
    ax.fill_between(time_arr, q05_vals, q95_vals, color='r', alpha=0.25, label='90% CI')
    
    ax.plot(time_arr, true_vals, lw=0, marker='o', markersize=3.5,
            markevery=markevery, markerfacecolor='none',
            markeredgecolor='blue', markeredgewidth=0.9, zorder=5)
    ax.plot(time_arr, est_vals, 'r-', lw=1.3, zorder=4)
    ax.set_xlabel('Time(days)', fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8.5)
    ax.tick_params(labelsize=7.5); ax.set_xlim(0, 250)
    if ylim: ax.set_ylim(ylim)
    ax.legend(handles=legend_elements, fontsize=7.5, loc='upper right',
              framealpha=0.8, edgecolor='grey')
    ax.text(-0.10, 1.04, panel_label, transform=ax.transAxes,
            fontsize=10, fontweight='bold')

# Use smooth prescribed curves for real data (like UKF figure)
x2_real_smooth = 0.18 * np.exp(-time_arr / 80.0) + 0.02
x3_real_smooth = x3_det.copy()

# (a) NC — Estimated-State
draw_panel(axes[0,0], mc_true[:,0], mc_est[:,0], q05[:,0], q95[:,0],
           r'NC $(x_1)$', 'Estimated-State', '(a)', ylim=(0.80, 0.97))

# (b) CC — Observed-State
draw_panel(axes[0,1], x2_real_smooth, mc_est[:,1], q05[:,1], q95[:,1],
           r'CC $(x_2)$', 'Observed-State', '(b)', ylim=(0, 0.26))

# (c) EC — Estimated-State
draw_panel(axes[1,0], x3_real_smooth, mc_est[:,2], q05[:,2], q95[:,2],
           r'EC $(x_3)$', 'Estimated-State', '(c)', ylim=(0, 0.065))

# (d) CA — Observed-State
ax_ca = axes[1,1]
draw_panel(ax_ca, mc_true[:,3], mc_est[:,3], q05[:,3], q95[:,3],
           r'CA $(y)$', 'Observed-State', '(d)', ylim=(0, 3.2))

# ===== CREATE INSET ZOOMED PLOT FOR CA =====
# Add inset axis (relative to figure, not subplot)
# Position: [left, bottom, width, height] in figure fraction
ax_inset = fig.add_axes([0.66, 0.44, 0.18, 0.12])

# Zoom region: t=30 to t=80 (around the peak)
t_zoom_start, t_zoom_end = 30, 80
t_mask = (time_arr >= t_zoom_start) & (time_arr <= t_zoom_end)
t_zoom = time_arr[t_mask]

# Plot zoomed CA with CI (using EnKF colors: red estimation, blue real)
ax_inset.fill_between(t_zoom, q05[t_mask, 3], q95[t_mask, 3],
                      color='r', alpha=0.30, label='90% CI')
ax_inset.plot(t_zoom, mc_est[t_mask, 3], color='r', lw=2.0, 
              label='Mean', linestyle='-', zorder=4)
ax_inset.plot(t_zoom, mc_true[t_mask, 3], lw=0, marker='o', markersize=3,
             markerfacecolor='none', markeredgecolor='blue', 
             markeredgewidth=0.8, zorder=5)

ax_inset.set_title('CA Zoom (t=30–80)', fontsize=8, fontweight='bold')
ax_inset.set_xlabel('Time (days)', fontsize=7)
ax_inset.set_ylabel('CA (y)', fontsize=7)
ax_inset.tick_params(labelsize=7)
ax_inset.grid(True, alpha=0.3, linestyle='--')
ax_inset.legend(fontsize=6.5, loc='upper left')

# Draw rectangle on main plot showing zoom region
rect = Rectangle((t_zoom_start, 0), 
                 t_zoom_end - t_zoom_start, 
                 3.2,
                 linewidth=1.5, edgecolor='r', facecolor='none', 
                 linestyle='--', alpha=0.7)
ax_ca.add_patch(rect)

# Draw arrow connecting main plot to inset
arrow = FancyArrowPatch((t_zoom_end, 3.2 * 0.8),
                       (65, 1.8),
                       arrowstyle='->', mutation_scale=15,
                       color='r', alpha=0.5, lw=1)
ax_ca.add_patch(arrow)

# (e) AA — Observed-State
draw_panel(axes[2,0], mc_true[:,4], mc_est[:,4], q05[:,4], q95[:,4],
           r'AA $(w)$', 'Observed-State', '(e)', ylim=(0, 0.16))

# (f) Hide unused
axes[2,1].set_visible(False)

fig.tight_layout(rect=[0, 0, 1, 0.995], h_pad=2.5, w_pad=2.0)
fig.savefig('pinho_enkf.png', dpi=180, bbox_inches='tight')
print("\nFigure saved as: pinho_enkf.png")
