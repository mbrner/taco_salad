import numpy as np
from sympy import symbols, Integral


def generate(n,
             purity,
             hit_point_func,
             x_lims=[-1, 1],
             sig_bkg_ratio=0.5,
             sig_eff=0.8):

    eff = 0.8
    rho = 0.9
    d_B = (hit_point * eff * rho) / (eff * rho + rho -1)
    x_B = x_C + d_B
    h_B = (eff * rho + rho -1) / (hit_point * (rho - 1))

    A_B = d_B * h_B


    x = np.random.uniform(x_lims[0], x_lims[1], n)
    y_true = np.random.uniform(size=n)
    y_true = np.array(np.random.uniform(size=n) <= sig_bkg_ratio, dtype=int)

