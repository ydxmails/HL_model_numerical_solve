"""Critical scaling tests reproducing the paper's exponents."""

import numpy as np

from hlmodel import Grid, solve_steady


def test_critical_flow_exponent():
    # At alpha = alpha_c, sigma ~ gammadot^{1/5} as gammadot -> 0 (Eq. 6).
    grid = Grid(sigma_max=12, n_per_unit=250)
    gds = np.array([3e-4, 1e-3, 3e-3])
    sig = np.array([solve_steady(0.5, gd, grid).stress for gd in gds])
    slope = np.diff(np.log(sig)) / np.diff(np.log(gds))
    # lowest-rate local slope should be close to 1/5
    assert abs(slope[0] - 0.2) < 0.05


def test_yield_stress_scaling():
    # Jammed phase: sigma_y ~ (alpha_c - alpha)^{1/2} (Eq. 5), via HB fit.
    from scipy.optimize import curve_fit
    grid = Grid(sigma_max=12, n_per_unit=300)

    def hb(gd, sy, A, n):
        return sy + A * np.power(gd, n)

    alphas = np.array([0.30, 0.36, 0.42, 0.46])
    gds = np.logspace(-2.7, -1.0, 6)
    sy = []
    for a in alphas:
        s = np.array([solve_steady(a, gd, grid, n_scan=36).stress for gd in gds])
        popt, _ = curve_fit(hb, gds, s, p0=[s[0] * 0.8, 0.7, 0.6], maxfev=20000)
        sy.append(popt[0])
    p = np.polyfit(np.log(0.5 - alphas), np.log(np.array(sy)), 1)[0]
    assert abs(p - 0.5) < 0.1
