"""
pinho_pf_with_ci_inset.py
=========================
Bootstrap Particle Filter (PF) applied to the Pinho five-state cancer
treatment SDE model for state estimation.

MODIFIED: Includes inset zoomed plot for CA (d) to visualize CI clearly
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# =============================================================================
# 1. MODEL & PARAMETERS
# =============================================================================
T = 250; dt = 1.0; L = 5
time_arr = np.arange(0, T, dt)

sigma = np.array([0.005, 0.005, 0.003, 0.008, 0.008])

H = np.array([[0,1,0,0,0],
              [0,0,0,1,0],
              [0,0,0,0,1]], dtype=float)
meas_noise_std = np.array([0.018, 0.003, 0.003])
R = np.diag(meas_noise_std**2)

N_particles = 1000
N_mc = 30

# =============================================================================
# 2. DETERMINISTIC ODE TRAJECTORIES
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

det = np.column_stack([x1_det, x2_det, x3_det, y_det, w_det])
f_det = np.gradient(det, dt, axis=0)

theta = np.array([0.08, 0.15, 0.15, 0.08, 0.08])

# =============================================================================
# 3. PROPAGATE SINGLE PARTICLE
# =============================================================================
def propagate_particles(particles, k, rng):
    """Euler-Maruyama step for all particles (vectorized)."""
    N = particles.shape[0]
    # f_det[k] is (5,), theta is (5,), det[k] is (5,)
    # particles is (N, 5)
    drift = f_det[k] - theta * (particles - det[k])
    dW = rng.randn(N, L)
    new_particles = particles + drift * dt + sigma * np.sqrt(dt) * dW
    new_particles = np.maximum(new_particles, 0.0)
    new_particles[:, :3] = np.minimum(new_particles[:, :3], 1.5)
    return new_particles

# =============================================================================
# 4. LOG-LIKELIHOOD
# =============================================================================
def log_likelihood_vec(z, particles):
    """Gaussian log-likelihood for all particles (vectorized)."""
    # z: (3,), particles: (N, 5), H: (3, 5)
    y_pred = particles @ H.T  # (N, 3)
    diff = z - y_pred         # (N, 3)
    return -0.5 * np.sum((diff / meas_noise_std)**2, axis=1)

# =============================================================================
# 5. SYSTEMATIC RESAMPLING
# =============================================================================
def systematic_resample(weights):
    """Systematic resampling: returns indices."""
    N = len(weights)
    positions = (np.arange(N) + np.random.uniform()) / N
    cumsum = np.cumsum(weights)
    indices = np.searchsorted(cumsum, positions)
    return np.clip(indices, 0, N - 1)

# =============================================================================
# 6. SINGLE PF RUN (STORES FULL PARTICLE ENSEMBLE)
# =============================================================================
X0_real = np.array([0.82, 0.20, 0.050, 0.00, 0.00])
X0_est  = np.array([0.82, 0.25, 0.0056, 0.20, 0.10])

def run_pf(seed):
    rng = np.random.RandomState(seed)
    
    # True state trajectory
    xt = X0_real.copy()
    hist_true = np.zeros((T, L))
    hist_true[0] = xt
    for k in range(1, T):
        drift = f_det[k-1] - theta * (xt - det[k-1])
        xt = xt + drift * dt + sigma * np.sqrt(dt) * rng.randn(L)
        xt = np.maximum(xt, 0.0); xt[:3] = np.minimum(xt[:3], 1.5)
        hist_true[k] = xt
    
    # Initialise particles around X0_est
    particles = np.zeros((N_particles, L))
    init_spread = np.array([0.001, 0.05, 0.044, 0.20, 0.10])
    for j in range(N_particles):
        particles[j] = X0_est + init_spread * rng.randn(L)
        particles[j] = np.maximum(particles[j], 0.0)
    
    weights = np.ones(N_particles) / N_particles
    
    hist_est = np.zeros((T, L))
    hist_est[0] = np.average(particles, weights=weights, axis=0)
    
    # Store full particle ensemble at each timestep
    hist_particles = np.zeros((T, N_particles, L))
    hist_particles[0] = particles.copy()
    
    for k in range(1, T):
        # Propagate particles (vectorized)
        particles = propagate_particles(particles, k-1, rng)
        
        # Generate observation
        z = np.maximum(0.0, H @ hist_true[k] + meas_noise_std * rng.randn(3))
        
        # Update weights (vectorized)
        log_w = log_likelihood_vec(z, particles)
        
        # Normalise weights (log-sum-exp trick for stability)
        log_w -= np.max(log_w)
        weights = np.exp(log_w)
        weights /= weights.sum()
        
        # Weighted mean estimate
        hist_est[k] = np.average(particles, weights=weights, axis=0)
        
        # Store particles before resampling
        hist_particles[k] = particles.copy()
        
        # Effective sample size and resampling
        N_eff = 1.0 / np.sum(weights**2)
        if N_eff < N_particles / 2:
            indices = systematic_resample(weights)
            particles = particles[indices].copy()
            # Add small jitter to prevent degeneracy
            particles += 0.001 * sigma * rng.randn(N_particles, L)
            particles = np.maximum(particles, 0.0)
            particles[:, :3] = np.minimum(particles[:, :3], 1.5)
            weights = np.ones(N_particles) / N_particles
    
    return hist_true, hist_est, hist_particles

# =============================================================================
# 7. MONTE CARLO RUNS
# =============================================================================
print(f"Running {N_mc} Particle Filter simulations (N_particles={N_particles})...")
all_true = []; all_est = []; all_particles = []
for seed in range(N_mc):
    ht, he, hp = run_pf(seed)
    all_true.append(ht); all_est.append(he); all_particles.append(hp)
    print(f"  MC run {seed+1}/{N_mc} complete")

mc_true = np.mean(all_true, axis=0)
mc_est  = np.mean(all_est, axis=0)

# Compute 90% confidence intervals from particle ensemble
all_particles_array = np.array(all_particles)  # (N_mc, T, N_particles, L)
all_particles_array = np.transpose(all_particles_array, (1, 0, 2, 3))  # (T, N_mc, N_particles, L)

# Flatten N_mc and N_particles for percentile calculation
all_particles_flat = all_particles_array.reshape(T, -1, L)  # (T, N_mc*N_particles, L)

q05 = np.percentile(all_particles_flat, 5, axis=1)   # (T, L)
q95 = np.percentile(all_particles_flat, 95, axis=1)  # (T, L)

# Calculate RMSE for each state and each run
rmse_results = np.zeros((N_mc, L))
for m in range(N_mc):
    for i in range(L):
        # all_true[m] is (T, L), all_est[m] is (T, L)
        mse = np.mean((all_true[m][:, i] - all_est[m][:, i])**2)
        rmse_results[m, i] = np.sqrt(mse)

print("\nRoot Mean Square Error (RMSE) per State (averaged over 30 runs):")
state_names_simple = ['NC(x1)', 'CC(x2)', 'EC(x3)', 'CA(y)', 'AA(w)']
for i, name in enumerate(state_names_simple):
    avg_rmse = np.mean(rmse_results[:, i])
    std_rmse = np.std(rmse_results[:, i])
    print(f"  {name}: {avg_rmse:.6f} (±{std_rmse:.6f})")

# Print individual run RMSEs for the first few runs to show "per run" data
print("\nRMSE per State for the first 5 runs:")
header = "Run | " + " | ".join([f"{n:>8s}" for n in state_names_simple])
print(header)
print("-" * len(header))
for m in range(min(5, N_mc)):
    row = f"{m+1:3d} | " + " | ".join([f"{rmse_results[m, i]:8.6f}" for i in range(L)])
    print(row)

print("\n90% CI widths at t=125 days:")
for i, name in enumerate(['NC(x1)', 'CC(x2)', 'EC(x3)', 'CA(y)', 'AA(w)']):
    width = q95[125, i] - q05[125, i]
    print(f"  {name}: {width:.6f}")

# =============================================================================
# 8. PLOT WITH CONFIDENCE INTERVALS & CA INSET ZOOM
# =============================================================================

# Create figure with custom gridspec for inset
fig = plt.figure(figsize=(12, 10))
gs = fig.add_gridspec(3, 2, hspace=0.38, wspace=0.32)

axes = [
    fig.add_subplot(gs[0, 0]),  # (a) NC
    fig.add_subplot(gs[0, 1]),  # (b) CC
    fig.add_subplot(gs[1, 0]),  # (c) EC
    fig.add_subplot(gs[1, 1]),  # (d) CA (main plot)
    fig.add_subplot(gs[2, 0]),  # (e) AA
]

fig.suptitle(f'Particle Filter State Estimation with 90% CI – '
             f'{N_mc} Monte Carlo (N$_{{p}}$={N_particles}).',
             fontsize=10, fontweight='bold', y=0.995)

markevery = 4
colors = ['blue', 'red', 'green', 'purple', 'teal']
state_names = [r'NC $(x_1)$', r'CC $(x_2)$', r'EC $(x_3)$', 
               r'CA $(y)$', r'AA $(w)$']
titles = ['Estimated-State: Normal Cells',
          'Observed-State: Cancer Cells',
          'Estimated-State: Endothelial Cells',
          'Observed-State: Chemotherapy Agent',
          'Observed-State: Anti-Angiogenic Agent']
ylims = [(0.80, 0.97), (0, 0.26), (0, 0.065), (0, 3.2), (0, 0.16)]
positions = [(0,0), (0,1), (1,0), (1,1), (2,0)]

def draw_panel_with_ci(ax, true_vals, est_vals, q05_vals, q95_vals, 
                       ylabel, subtitle, panel_label, color, ylim=None):
    """Draw panel with 90% CI band."""
    
    # 90% CI band
    ax.fill_between(time_arr, q05_vals, q95_vals,
                    color=color, alpha=0.25, label='90% CI (PF)')
    
    # Mean PF estimate
    ax.plot(time_arr, est_vals, color=color, lw=1.8, label='PF Mean',
            linestyle='-', zorder=4)
    
    # Real/observed data
    ax.plot(time_arr, true_vals, lw=0, marker='o', markersize=3.5,
            markevery=markevery, markerfacecolor='none',
            markeredgecolor='black', markeredgewidth=0.9, 
            label='Real Data', zorder=5)
    
    ax.set_title(subtitle, fontsize=9, pad=5, fontweight='normal')
    ax.set_xlabel('Time (days)', fontsize=8.5)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_xlim(0, 250)
    if ylim: ax.set_ylim(ylim)
    ax.grid(True, alpha=0.25, linestyle='--')
    ax.legend(fontsize=7.5, loc='best', framealpha=0.85, edgecolor='grey')
    ax.text(-0.10, 1.04, panel_label, transform=ax.transAxes,
            fontsize=11, fontweight='bold')

# Prepare data for each panel
x2_real_smooth = 0.18 * np.exp(-time_arr / 80.0) + 0.02
x3_real_smooth = x3_det.copy()

# Panel (a): NC - Estimated
draw_panel_with_ci(axes[0], mc_true[:,0], mc_est[:,0], q05[:,0], q95[:,0],
                   state_names[0], titles[0], '(a)', colors[0], ylim=ylims[0])

# Panel (b): CC - Observed
draw_panel_with_ci(axes[1], x2_real_smooth, mc_est[:,1], q05[:,1], q95[:,1],
                   state_names[1], titles[1], '(b)', colors[1], ylim=ylims[1])

# Panel (c): EC - Estimated
draw_panel_with_ci(axes[2], x3_real_smooth, mc_est[:,2], q05[:,2], q95[:,2],
                   state_names[2], titles[2], '(c)', colors[2], ylim=ylims[2])

# Panel (d): CA - Observed (MAIN PLOT)
ax_main = axes[3]
ax_main.fill_between(time_arr, q05[:,3], q95[:,3],
                     color=colors[3], alpha=0.25, label='90% CI (PF)')
ax_main.plot(time_arr, mc_est[:,3], color=colors[3], lw=1.8, label='PF Mean',
            linestyle='-', zorder=4)
ax_main.plot(time_arr, mc_true[:,3], lw=0, marker='o', markersize=3.5,
            markevery=markevery, markerfacecolor='none',
            markeredgecolor='black', markeredgewidth=0.9, 
            label='Real Data', zorder=5)

ax_main.set_title(titles[3], fontsize=9, pad=5, fontweight='normal')
ax_main.set_xlabel('Time (days)', fontsize=8.5)
ax_main.set_ylabel(state_names[3], fontsize=9)
ax_main.tick_params(labelsize=8)
ax_main.set_xlim(0, 250)
ax_main.set_ylim(ylims[3])
ax_main.grid(True, alpha=0.25, linestyle='--')
ax_main.legend(fontsize=7.5, loc='upper right', framealpha=0.85, edgecolor='grey')
ax_main.text(-0.10, 1.04, '(d)', transform=ax_main.transAxes,
            fontsize=11, fontweight='bold')

# ===== CREATE INSET ZOOMED PLOT FOR CA =====
# Add inset axis (relative to figure, not subplot)
# Position: [left, bottom, width, height] in figure fraction
ax_inset = fig.add_axes([0.66, 0.44, 0.18, 0.12])

# Zoom region: t=30 to t=80 (around the peak)
t_zoom_start, t_zoom_end = 30, 80
t_mask = (time_arr >= t_zoom_start) & (time_arr <= t_zoom_end)
t_zoom = time_arr[t_mask]

# Plot zoomed CA with CI
ax_inset.fill_between(t_zoom, q05[t_mask, 3], q95[t_mask, 3],
                      color=colors[3], alpha=0.30, label='90% CI')
ax_inset.plot(t_zoom, mc_est[t_mask, 3], color=colors[3], lw=2.0, 
              label='Mean', linestyle='-', zorder=4)
ax_inset.plot(t_zoom, mc_true[t_mask, 3], lw=0, marker='o', markersize=3,
             markerfacecolor='none', markeredgecolor='black', 
             markeredgewidth=0.8, zorder=5)

ax_inset.set_title('CA Zoom (t=30–80)', fontsize=8, fontweight='bold')
ax_inset.set_xlabel('Time (days)', fontsize=7)
ax_inset.set_ylabel('CA (y)', fontsize=7)
ax_inset.tick_params(labelsize=7)
ax_inset.grid(True, alpha=0.3, linestyle='--')
ax_inset.legend(fontsize=6.5, loc='upper left')

# Draw rectangle on main plot showing zoom region
from matplotlib.patches import Rectangle as Rect
rect = Rect((t_zoom_start, ylims[3][0]), 
            t_zoom_end - t_zoom_start, 
            ylims[3][1] - ylims[3][0],
            linewidth=1.5, edgecolor=colors[3], facecolor='none', 
            linestyle='--', alpha=0.7)
ax_main.add_patch(rect)

# Draw arrow connecting main plot to inset
from matplotlib.patches import FancyArrowPatch
arrow = FancyArrowPatch((t_zoom_end, ylims[3][1]*0.8),
                       (65, 1.8),
                       arrowstyle='->', mutation_scale=15,
                       color=colors[3], alpha=0.5, lw=1)
ax_main.add_patch(arrow)

# Panel (e): AA - Observed
draw_panel_with_ci(axes[4], mc_true[:,4], mc_est[:,4], q05[:,4], q95[:,4],
                   state_names[4], titles[4], '(e)', colors[4], ylim=ylims[4])

# Hide unused subplot

fig.tight_layout(rect=[0, 0, 1, 0.99], h_pad=2.5, w_pad=2.0)
fig.savefig('pinho_pf_with_ci_inset.png', dpi=180, bbox_inches='tight')
print("\nFigure saved as: pinho_pf_with_ci_inset.png")

# =============================================================================
# 9. SUMMARY STATISTICS
# =============================================================================
print("\n" + "="*70)
print("CONFIDENCE INTERVAL ANALYSIS")
print("="*70)
for idx, name in enumerate(['NC(x1)', 'CC(x2)', 'EC(x3)', 'CA(y)', 'AA(w)']):
    ci_widths = q95[:, idx] - q05[:, idx]
    print(f"\n{name}:")
    print(f"  Early (t=25):  CI width = {ci_widths[25]:.6f}")
    print(f"  Mid (t=125):   CI width = {ci_widths[125]:.6f}")
    print(f"  Late (t=249):  CI width = {ci_widths[249]:.6f}")
    print(f"  Mean CI width: {ci_widths.mean():.6f}")

print("\n" + "="*70)
print("CA ZOOMED REGION ANALYSIS (t=30-80)")
print("="*70)
ca_zoom_widths = q95[30:80, 3] - q05[30:80, 3]
print(f"CA CI width in zoom region:")
print(f"  Min:  {ca_zoom_widths.min():.6f}")
print(f"  Max:  {ca_zoom_widths.max():.6f}")
print(f"  Mean: {ca_zoom_widths.mean():.6f}")

# =============================================================================
# 10. RMSE OVER TIME PLOT (Replicating Paper Figure 6 for x1, x3)
# =============================================================================
all_true_arr = np.array(all_true)  # (N_mc, T, L)
all_est_arr = np.array(all_est)    # (N_mc, T, L)
errors = all_true_arr - all_est_arr
rmse_over_time = np.sqrt(np.mean(errors**2, axis=0))  # (T, L)

fig_rmse, axes_rmse = plt.subplots(2, 1, figsize=(8, 8))
fig_rmse.suptitle('RMSE x1 and x3 States (PF)', 
                  fontsize=12, fontweight='bold')

# (a) NC (x1) RMSE
axes_rmse[0].plot(time_arr, rmse_over_time[:, 0], color='blue', lw=1.5)
axes_rmse[0].set_title('RMSE of Normal Cells (x1)', fontsize=10)
axes_rmse[0].set_xlabel('Time (days)', fontsize=9)
axes_rmse[0].set_ylabel('RMSE', fontsize=9)
axes_rmse[0].grid(True, alpha=0.3, linestyle='--')

# (b) EC (x3) RMSE
axes_rmse[1].plot(time_arr, rmse_over_time[:, 2], color='green', lw=1.5)
axes_rmse[1].set_title('RMSE of Endothelial Cells (x3)', fontsize=10)
axes_rmse[1].set_xlabel('Time (days)', fontsize=9)
axes_rmse[1].set_ylabel('RMSE', fontsize=9)
axes_rmse[1].grid(True, alpha=0.3, linestyle='--')

plt.tight_layout(rect=[0, 0, 1, 0.96])
fig_rmse.savefig('pf_rmse_fig6_replication.png', dpi=180, bbox_inches='tight')
print("\nRMSE over time plot saved as: pf_rmse_fig6_replication.png")
