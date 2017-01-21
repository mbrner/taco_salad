import numpy as np
import os

from taco_salad.toppings import ConfidenceCutter, criteria


def generate(n, purity, x_lims=[-1., 1]):
    d = (-1) * (0.5 * np.sqrt(1 - purity)) / (np.sqrt(1 - purity) - np.sqrt(2))
    y = 2. / (0.5 + d)
    x_c = 0.5 - d
    a = y / (1 - (1 / x_c))
    m = -a / x_c
    r = np.random.uniform(size=n)
    y_pred = (np.sqrt(a**2 + 2 * a * x_c * m +
                      x_c**2 * m**2 + 2 * m * r) - a) / m
    y_true = np.random.randint(0, 2, size=n)
    idx = y_true == 0
    y_pred[idx] = -(y_pred[idx] - 0.5) + 0.5
    x = np.random.uniform(x_lims[0], x_lims[1], n)
    return x, y_pred, y_true


def generate_lines(eff, pur, conf_cut, n_bkg=1, n_sig=1):
    B0_num = 2 * pur * n_bkg**2 + (2 * eff * pur - 2 * eff) * n_bkg * n_sig
    B0_den = conf_cut * np.sqrt(eff) * np.sqrt(1 - pur) * np.sqrt(pur) * \
        np.sqrt(n_bkg) * np.sqrt(n_sig) + (conf_cut * pur * n_bkg)
    B0 = B0_num / B0_den

    Bx_num = conf_cut * pur * n_bkg + conf_cut * \
        np.sqrt((eff * pur - eff * pur**2) * n_bkg * n_sig)
    Bx_den = (eff * pur - eff) * n_sig + pur * n_bkg
    Bx = Bx_num / Bx_den

    Bm = -B0 / Bx
    Bb = B0

    S0_num = ((2 + 2 * np.sqrt(1 - eff)) * eff * n_sig)
    S0_den = (conf_cut - 1) * eff + (2 - conf_cut * 2) * np.sqrt(1 - eff) - \
        2 * conf_cut + 2
    S0 = S0_num / S0_den

    Sx_num = -1 + conf_cut + (conf_cut - 1) * np.sqrt(1 - eff) + eff
    Sx_den = eff
    Sx = Sx_num / Sx_den

    Sm = S0 / (1 - Sx)
    Sb = S0 * Sx / (Sx - 1)
    return Bm, Sm, Bb, Sb, Bx, Sx, B0, S0


def generate_x(n, eff, pur, conf_cut_func=lambda x: np.ones_like(x) * 0.5,
               x_lims=[-1., 1.]):
    y_true = np.random.randint(0, 2, size=n)
    x = np.random.uniform(x_lims[0], x_lims[1], n)
    conf_cut = np.array([conf_cut_func(x_i) for x_i in x])

    Bm, Sm, Bb, Sb, Bx, Sx, B0, S0 = generate_lines(eff, pur, conf_cut)

    Sl = Sm / 2. * Sx**2 + Sb * Sx

    r = np.random.uniform(size=n)
    y_pred_S = -Sb / Sm + np.sqrt((Sb / Sm)**2 + 2 * (Sl + r) / Sm)
    y_pred_B = -Bb / Bm - np.sqrt((Bb / Bm)**2 + (2 * r) / Bm)

    y_pred = np.zeros_like(y_true, dtype=float)
    idx_sig = y_true == 1
    y_pred[idx_sig] = y_pred_S[idx_sig]
    y_pred[~idx_sig] = y_pred_B[~idx_sig]
    return x, y_pred, y_true


def test_conf_cutter():
    eff = 0.90
    pur = 0.90

    def conf_cut_func(x):
        if x < -0.9:
            y = 0.5
        elif x < -0.1:
            y = (np.sin((x + 0.9) / 0.8 * 2 * np.pi)) / 3. + .5
        elif x < 0.:
            y = 0.5
        elif x < 0.5:
            x_def = [0., 0.5]
            y_def = [0.5, 0.7]
            m = (y_def[1] - y_def[0]) / (x_def[1] - x_def[0])
            b = y_def[0] - x_def[0] * m
            y = m * x + b
        else:
            x_def = [0.5, 1.]
            y_def = [0.7, 0.3]
            m = (y_def[1] - y_def[0]) / (x_def[1] - x_def[0])
            b = y_def[0] - x_def[0] * m
            y = m * x + b
        return y

    x, y_pred, y_true = generate_x(1000000, eff, pur, conf_cut_func)
    x_cut_true = np.linspace(-1, 1, 1000)
    y_cut_true = np.array([conf_cut_func(x_i) for x_i in x_cut_true])

    pur_treshold = 0.9

    pur_crit = criteria.purity_criteria(threshold=pur_treshold)

    conf_cutter = ConfidenceCutter(n_steps=100,
                                   window_size=0.1,
                                   n_bootstraps=1,
                                   criteria=pur_crit,
                                   conf_index=1,
                                   n_jobs=1)

    X = np.vstack((x, y_pred)).T

    conf_cutter.fit(X, y_true)

    x_cut_curve = np.linspace(-1., 1., 1000)
    y_cut_curve = conf_cutter.cut_opts.cut_curve(x_cut_curve)

    y_pred_conf = conf_cutter.predict(X)
    y_true_bool = np.array(y_true, dtype=bool)
    y_pred_bool = np.array(y_pred_conf, dtype=bool)
    tp = np.sum(y_true_bool[y_pred_bool])
    fp = np.sum(~y_true_bool[y_pred_bool])

    diff_cutter_truth = np.absolute(y_cut_true - y_cut_curve)
    assert all(diff_cutter_truth <= 0.07)

    assert tp / (tp + fp) - pur_treshold < 0.07

    cut_curve_file = conf_cutter.save_curve('test_curve.npz')
    conf_cutter_reloaded = ConfidenceCutter(curve_file=cut_curve_file)
    os.remove(cut_curve_file)
    y_reloaded_cutter = conf_cutter_reloaded.cut_opts.cut_curve(x_cut_curve)
    assert all(y_reloaded_cutter == y_cut_curve)

    gen_crit = criteria.general_confusion_matrix_criteria(
        'tp / (tp + fp)',
        threshold=pur_treshold)
    y_true = [0, 0, 1, 1]
    y_predict = [0, 1, 0, 1]
    pos = 0.5
    val_gen = gen_crit(y_true, y_predict, pos)
    val_pur = pur_crit(y_true, y_predict, pos)
    assert val_gen == val_pur

