#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
import logging

import numpy as np
from matplotlib import pyplot as plt

from taco_salad.toppings import ConfidenceCutter, criteria


def generate_lines(eff, pur, conf_cut, n_bkg=1, n_sig=1):
    B0_num = 2 * pur * n_bkg**2 + (2 * eff * pur - 2 * eff) * n_bkg * n_sig
    B0_den = conf_cut * np.sqrt(eff) * np.sqrt(1 - pur) * np.sqrt(pur) * \
        np.sqrt(n_bkg) *  np.sqrt(n_sig) + (conf_cut * pur * n_bkg)
    B0 = B0_num / B0_den

    Bx_num = conf_cut * pur * n_bkg + conf_cut * \
        np.sqrt((eff * pur - eff * pur**2) * n_bkg * n_sig)
    Bx_den = (eff * pur - eff) * n_sig + pur * n_bkg
    Bx = Bx_num/Bx_den

    Bm = -B0 / Bx
    Bb = B0

    S0_num = ((2 + 2 * np.sqrt(1 - eff)) * eff * n_sig)
    S0_den = (conf_cut - 1) * eff + (2 - conf_cut * 2) * np.sqrt(1 - eff) - \
        2 * conf_cut + 2
    S0 = S0_num / S0_den

    Sx_num = -1 + conf_cut + (conf_cut - 1) * np.sqrt(1 - eff) + eff
    Sx_den = eff
    Sx = Sx_num / Sx_den

    Sm = S0 / (1- Sx)
    Sb = S0 * Sx / (Sx - 1)



    return Bm, Sm, Bb, Sb, Bx, Sx, B0, S0


def generate_x(n, eff, pur, conf_cut_func=lambda x: np.ones_like(x) * 0.5,
               x_lims=[-1., 1.]):
    y_true = np.random.randint(0, 2, size=n)
    x = np.random.uniform(x_lims[0], x_lims[1], n)
    conf_cut = np.array([conf_cut_func(x_i) for x_i in x])

    Bm, Sm, Bb, Sb, Bx, Sx, B0, S0 = generate_lines(eff, pur, conf_cut)

    Sl = Sm / 2. * Sx**2 + Sb * Sx
    Bl = 0.

    r = np.random.uniform(size=n)
    y_pred_S = -Sb / Sm + np.sqrt((Sb / Sm)**2 + 2 * (Sl + r) / Sm)
    y_pred_B = -Bb / Bm - np.sqrt((Bb / Bm)**2 + (2 * r) / Bm)

    y_pred = np.zeros_like(y_true, dtype=float)
    idx_sig = y_true == 1
    y_pred[idx_sig] = y_pred_S[idx_sig]
    y_pred[~idx_sig] = y_pred_B[~idx_sig]
    return x, y_pred, y_true


def MC_CUT(x):
    if x < -0.9:
        y = 0.5
    elif x < -0.1:
        y = (np.sin((x+0.9)/0.8*2*np.pi)) / 3. + .5
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

EFFICIENCY = 0.90
PURITY = 0.90



if __name__ == '__main__':
    logging.captureWarnings(True)
    logging.basicConfig(format=('%(asctime)s|%(name)s|%(levelname)s| ' +
                        '%(message)s'), level=logging.INFO)

    logging.info('Staring Confidence Cutter Example 2')

    logging.info('Generating ToyMC')

    x, y_pred, y_true = generate_x(1000000, EFFICIENCY, PURITY, MC_CUT)

    x_cut_true = np.linspace(-1, 1, 1000)
    y_cut_true = np.array([MC_CUT(x_i) for x_i in x_cut_true])

    logging.info('Setting Up 90% Purity Criteria')
    crit = criteria.general_confusion_matrix_criteria('tp / (tp + fp)',
                                                   threshold=0.90)

    conf_cutter = ConfidenceCutter(n_steps=100,
                                   window_size=0.1,
                                   n_bootstraps=10,
                                   criteria=crit,
                                   conf_index=1,
                                   n_jobs=4)

    X = np.vstack((x, y_pred)).T

    conf_cutter.fit(X, y_true)


    x_cut_curve = np.linspace(-1., 1., 1000)
    y_cut_curve = conf_cutter.cut_opts.curve(x_cut_curve)


    plt.hexbin(y_pred, x, gridsize=50, cmap=plt.cm.viridis)
    plt.plot(y_cut_true, x_cut_true, '-', color='0.5', lw=5,
             label='ToyMC Input')
    plt.plot(y_cut_curve, x_cut_curve, '--', color='w', lw=5,
             label='Confidence Cutter')
    plt.xlabel('Classifier Score')
    plt.ylabel('Observable')
    plt.legend(loc='best', title='Purity 90%')
    plt.xlim([0.,1.])
    plt.ylim([-1, 1])
    plt.savefig('conf_cutter_proof.png', dpi=200)
    plt.clf()
    conf_cutter.save_curve('test_curve')

    y_pred_conf = conf_cutter.predict(X)
    y_true_bool = np.array(y_true, dtype=bool)
    y_pred_bool = np.array(y_pred_conf, dtype=bool)
    tp = np.sum(y_true_bool[y_pred_bool])
    fp = np.sum(~y_true_bool[y_pred_bool])
    logging.info('Achieved Purity on a Train sample:')
    logging.info(tp / (tp + fp))
    logging.info('True Positive Rate:')
    logging.info(tp / np.sum(y_true_bool))
    del conf_cutter


    conf_cutter_reloaded = ConfidenceCutter(curve_file='test_curve')
    x, y_pred, y_true = generate_x(100000, EFFICIENCY, PURITY, MC_CUT)

    conf_cutter_reloaded.conf_index = 0
    X = np.vstack((y_pred, x)).T
    y_pred_conf = conf_cutter_reloaded.predict(X)
    y_true_bool = np.array(y_true, dtype=bool)
    y_pred_bool = np.array(y_pred_conf, dtype=bool)
    tp = np.sum(y_true_bool[y_pred_bool])
    fp = np.sum(~y_true_bool[y_pred_bool])
    logging.info('Achieved Purity on a Test sample:')
    logging.info(tp / (tp + fp))
    logging.info('True Positive Rate:')
    logging.info(tp / np.sum(y_true_bool))



    pur_crit = criteria.purity_criteria(threshold=0.90)
    conf_cutter_reloaded.criteria = pur_crit
    naive_cut = conf_cutter_reloaded.__find_best_cut__(0, -1, 1, 0,
                          X, y_true, None, n_points=10)

    y_true_bool = np.array(y_true, dtype=bool)
    y_pred_bool = np.array(y_pred >= naive_cut, dtype=bool)
    tp = np.sum(y_true_bool[y_pred_bool])
    fp = np.sum(~y_true_bool[y_pred_bool])
    logging.info('Achieved Purity for naive cut:')
    logging.info(tp / (tp + fp))
    logging.info('True Positive Rate:')
    logging.info(tp / np.sum(y_true_bool))

    x_curves = np.linspace(-1, 1, 1000)
    y_cut_true = np.array([MC_CUT(x_i) for x_i in x_curves])
    y_cut_curve = conf_cutter_reloaded.cut_opts.curve(x_curves)
    y_naive_cut = np.ones_like(x_curves) * naive_cut


    plt.hexbin(y_pred, x, gridsize=50, cmap=plt.cm.plasma)
    plt.plot(y_cut_true, x_curves, '-', color='0.5', lw=5,
             label='ToyMC Input')
    plt.plot(y_cut_curve, x_curves, '--', color='w', lw=5,
             label='Confidence Cutter')

    plt.plot(y_naive_cut, x_curves, '--', color='r', lw=5,
             label='Straight Cut')
    plt.xlabel('Classifier Score')
    plt.ylabel('Observable')
    plt.legend(loc='best', title='Purity 90%')
    plt.xlim([0.,1.])
    plt.ylim([-1, 1])
    plt.savefig('conf_cutter_reloaded.png', dpi=200)


















