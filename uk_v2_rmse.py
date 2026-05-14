"""
uk_v2_rmse.py
===============
Unscented Kalman Filter (UKF) applied to the Pinho five-state cancer
treatment SDE model for state estimation.

Calculates and plots the RMSE of x1 and x3.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

# =============================================================================
# 7. RMSE OVER TIME PLOT (x1, x3)
# =============================================================================
all_true_arr = np.array(all_true)  # (N_mc, T, L)
all_est_arr = np.array(all_est)    # (N_mc, T, L)
errors = all_true_arr - all_est_arr
rmse_over_time = np.sqrt(np.mean(errors**2, axis=0))  # (T, L)

fig_rmse, axes_rmse = plt.subplots(2, 1, figsize=(8, 8))
fig_rmse.suptitle('RMSE x1 and x3 States (UKF)', 
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
fig_rmse.savefig('uk_v2_rmse.png', dpi=180, bbox_inches='tight')
print("\nRMSE over time plot saved as: uk_v2_rmse.png")

print("\nScalar RMSE per state:")
for i, name in enumerate(['NC(x1)', 'CC(x2)', 'EC(x3)', 'CA(y)', 'AA(w)']):
    rmse_per_run = []
    for m in range(N_mc):
        mse = np.mean((all_true[m][:, i] - all_est[m][:, i])**2)
        rmse_per_run.append(np.sqrt(mse))
    print(f"{name}: {np.mean(rmse_per_run):.6f}")
