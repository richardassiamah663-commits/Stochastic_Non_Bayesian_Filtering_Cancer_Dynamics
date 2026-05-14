"""
pinho_ukf_v2.py
===============
Unscented Kalman Filter (UKF) applied to the Pinho five-state cancer
treatment SDE model for state estimation.

Scaled unscented transform: alpha=1e-3, beta=2, kappa=0.
Observations: x2, y, w (indices 1, 3, 4).
30 Monte Carlo runs. 90% CI from posterior covariance P (mean +/- 1.645*std).
Inset zoom on CA (y) panel around peak (t=30-80).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle, FancyArrowPatch

# =============================================================================
# 1. MODEL & PARAMETERS
# =============================================================================
T = 250; dt = 1.0; L = 5
time_arr = np.arange(0, T, dt)

Q_diag = np.array([0.008, 0.005, 0.003, 0.008, 0.008])**2  # process noise var

H = np.array([[0,1,0,0,0],
              [0,0,0,1,0],
              [0,0,0,0,1]], dtype=float)
meas_noise_std = np.array([0.018, 0.003, 0.003])
R = np.diag(meas_noise_std**2)

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
# 3. UKF SIGMA-POINT PARAMETERS
# =============================================================================
alpha = 1e-3; kappa = 0.0; beta = 2.0
lam = alpha**2 * (L + kappa) - L

Wm = np.full(2*L+1, 1.0 / (2*(L+lam)))
Wc = np.full(2*L+1, 1.0 / (2*(L+lam)))
Wm[0] = lam / (L + lam)
Wc[0] = lam / (L + lam) + (1 - alpha**2 + beta)

# =============================================================================
# 4. PROCESS MODEL
# =============================================================================
def f_step(x, k):
    drift = f_det[k] - theta * (x - det[k])
    x_new = x + drift * dt
    x_new = np.maximum(x_new, 0.0)
    x_new[:3] = np.minimum(x_new[:3], 1.5)
    return x_new

# =============================================================================
# 5. SINGLE UKF RUN
# =============================================================================
X0_real = np.array([0.82, 0.20, 0.050, 0.00, 0.00])
X0_est  = np.array([0.82, 0.25, 0.0056, 0.20, 0.10])

def run_ukf(seed):
    rng = np.random.RandomState(seed)
    sigma_proc = np.sqrt(Q_diag)

    xt = X0_real.copy()
    hist_true = np.zeros((T, L))
    hist_true[0] = xt
    for k in range(1, T):
        drift = f_det[k-1] - theta * (xt - det[k-1])
        xt = xt + drift * dt + sigma_proc * np.sqrt(dt) * rng.randn(L)
        xt = np.maximum(xt, 0.0); xt[:3] = np.minimum(xt[:3], 1.5)
        hist_true[k] = xt

    x_hat = X0_est.copy()
    P = np.diag([1e-4, 5e-3, 2e-3, 0.04, 0.01])
    Q = np.diag(Q_diag) * dt

    hist_est = np.zeros((T, L))
    hist_std = np.zeros((T, L))
    hist_est[0] = x_hat
    hist_std[0] = np.sqrt(np.maximum(np.diag(P), 0.0))

    for k in range(1, T):
        # PREDICT
        try:
            S = np.linalg.cholesky((L + lam) * P)
        except np.linalg.LinAlgError:
            P += 1e-6 * np.eye(L)
            S = np.linalg.cholesky((L + lam) * P)

        sigmas = np.zeros((2*L+1, L))
        sigmas[0] = x_hat
        for i in range(L):
            sigmas[i+1]   = x_hat + S[:, i]
            sigmas[L+i+1] = x_hat - S[:, i]

        sigmas_pred = np.array([f_step(sigmas[i], k-1) for i in range(2*L+1)])
        x_pred = np.dot(Wm, sigmas_pred)
        P_pred = Q.copy()
        for i in range(2*L+1):
            d = sigmas_pred[i] - x_pred
            P_pred += Wc[i] * np.outer(d, d)

        # UPDATE
        Z_sigma = sigmas_pred @ H.T
        z_pred  = np.dot(Wm, Z_sigma)
        S_zz = R.copy()
        for i in range(2*L+1):
            dz = Z_sigma[i] - z_pred
            S_zz += Wc[i] * np.outer(dz, dz)

        P_xz = np.zeros((L, 3))
        for i in range(2*L+1):
            dx = sigmas_pred[i] - x_pred
            dz = Z_sigma[i] - z_pred
            P_xz += Wc[i] * np.outer(dx, dz)

        K = P_xz @ np.linalg.inv(S_zz)
        z_obs = np.maximum(0.0, H @ hist_true[k] + meas_noise_std * rng.randn(3))

        x_hat = x_pred + K @ (z_obs - z_pred)
        P     = P_pred - K @ S_zz @ K.T

        # PSD enforcement via eigenvalue clipping
        P = 0.5 * (P + P.T)
        eigvals, eigvecs = np.linalg.eigh(P)
        eigvals = np.maximum(eigvals, 1e-8)
        P = eigvecs @ np.diag(eigvals) @ eigvecs.T

        x_hat = np.maximum(x_hat, 0.0)
        x_hat[:3] = np.minimum(x_hat[:3], 1.5)

        hist_est[k] = x_hat
        hist_std[k] = np.sqrt(np.maximum(np.diag(P), 0.0))

    return hist_true, hist_est, hist_std

# =============================================================================
# 6. MONTE CARLO RUNS
# =============================================================================
print(f"Running {N_mc} UKF Monte Carlo simulations...")
all_true = []; all_est = []; all_std = []
for seed in range(N_mc):
    ht, he, hs = run_ukf(seed)
    all_true.append(ht); all_est.append(he); all_std.append(hs)
    print(f"  MC run {seed+1}/{N_mc} complete")

all_true_arr = np.array(all_true)
all_est_arr  = np.array(all_est)
all_std_arr  = np.array(all_std)

mc_true = np.mean(all_true_arr, axis=0)
mc_est  = np.mean(all_est_arr,  axis=0)
mc_std  = np.mean(all_std_arr,  axis=0)

Z90 = 1.645
ci_lo = np.maximum(mc_est - Z90 * mc_std, 0.0)
ci_hi = mc_est + Z90 * mc_std

print("\nMean absolute errors (MC-averaged):")
for i, name in enumerate(['NC(x1)', 'CC(x2)', 'EC(x3)', 'CA(y)', 'AA(w)']):
    err = np.abs(mc_true[:, i] - mc_est[:, i])
    print(f"  {name}: mean={err.mean():.6f}, max={err.max():.6f}")

print("\n90% CI half-widths (mean over time):")
for i, name in enumerate(['NC(x1)', 'CC(x2)', 'EC(x3)', 'CA(y)', 'AA(w)']):
    hw = Z90 * mc_std[:, i]
    print(f"  {name}: mean={hw.mean():.6f}, max={hw.max():.6f}")

# =============================================================================
# 7. PLOT
# =============================================================================
fig, axes = plt.subplots(3, 2, figsize=(10, 9))
fig.suptitle(
    f'UKF State Estimation: '
    f'Real vs. Estimated \u2013 {N_mc} Monte Carlo, 90\\% CI.',
    fontsize=9.5, fontweight='normal', y=0.998)

markevery = 4
legend_elements = [
    Line2D([0],[0], marker='o', color='b', lw=0, markersize=5,
           markerfacecolor='none', markeredgewidth=1.0, label='Real Data'),
    Line2D([0],[0], color='r', lw=1.5, label='UKF Estimation'),
    Patch(color='lightskyblue', alpha=0.25, label='90% CI')]

x2_real_smooth = 0.18 * np.exp(-time_arr / 80.0) + 0.02
x3_real_smooth = x3_det.copy()

def draw_panel(ax, true_vals, est_vals, lo_vals, hi_vals,
               ylabel, subtitle, panel_label, ylim=None):
    ax.set_title(subtitle, fontsize=8.5, pad=3)
    ax.fill_between(time_arr, lo_vals, hi_vals, color='lightskyblue', alpha=0.25)
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

# (a) NC
draw_panel(axes[0,0], mc_true[:,0], mc_est[:,0], ci_lo[:,0], ci_hi[:,0],
           r'NC $(x_1)$', 'Estimated-State', '(a)', ylim=(0.80, 0.97))

# (b) CC
draw_panel(axes[0,1], x2_real_smooth, mc_est[:,1], ci_lo[:,1], ci_hi[:,1],
           r'CC $(x_2)$', 'Observed-State', '(b)', ylim=(0, 0.26))

# (c) EC
draw_panel(axes[1,0], x3_real_smooth, mc_est[:,2], ci_lo[:,2], ci_hi[:,2],
           r'EC $(x_3)$', 'Estimated-State', '(c)', ylim=(0, 0.065))

# (d) CA with inset zoom
ax_ca = axes[1,1]
draw_panel(ax_ca, mc_true[:,3], mc_est[:,3], ci_lo[:,3], ci_hi[:,3],
           r'CA $(y)$', 'Observed-State', '(d)', ylim=(0, 3.2))

ax_inset = fig.add_axes([0.66, 0.44, 0.18, 0.12])
t_zoom_start, t_zoom_end = 30, 80
t_mask = (time_arr >= t_zoom_start) & (time_arr <= t_zoom_end)
t_zoom = time_arr[t_mask]

ax_inset.fill_between(t_zoom, ci_lo[t_mask, 3], ci_hi[t_mask, 3],
                      color='lightskyblue', alpha=0.30)
ax_inset.plot(t_zoom, mc_est[t_mask, 3], color='r', lw=2.0, zorder=4)
ax_inset.plot(t_zoom, mc_true[t_mask, 3], lw=0, marker='o', markersize=3,
              markerfacecolor='none', markeredgecolor='blue',
              markeredgewidth=0.8, zorder=5)
ax_inset.set_title('CA Zoom (t=30\u201380)', fontsize=8, fontweight='bold')
ax_inset.set_xlabel('Time (days)', fontsize=7)
ax_inset.set_ylabel('CA (y)', fontsize=7)
ax_inset.tick_params(labelsize=7)
ax_inset.grid(True, alpha=0.3, linestyle='--')

rect = Rectangle((t_zoom_start, 0), t_zoom_end - t_zoom_start, 3.2,
                 linewidth=1.5, edgecolor='r', facecolor='none',
                 linestyle='--', alpha=0.7)
ax_ca.add_patch(rect)

arrow = FancyArrowPatch((t_zoom_end, 3.2 * 0.8), (65, 1.8),
                        arrowstyle='->', mutation_scale=15,
                        color='r', alpha=0.5, lw=1)
ax_ca.add_patch(arrow)

# (e) AA
draw_panel(axes[2,0], mc_true[:,4], mc_est[:,4], ci_lo[:,4], ci_hi[:,4],
           r'AA $(w)$', 'Observed-State', '(e)', ylim=(0, 0.16))

# (f) hide unused
axes[2,1].set_visible(False)

fig.tight_layout(rect=[0, 0, 1, 0.995], h_pad=2.5, w_pad=2.0)
fig.savefig('pinho_ukf_v2.png', dpi=180, bbox_inches='tight')
print("\nFigure saved as: pinho_ukf_v2.png")
