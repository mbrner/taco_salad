import numpy as np
from matplotlib import pyplot as plt


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


if __name__ == '__main__':
    x, y_pred, y_true = generate(100000, 0.9)
    plt.hist(y_pred, bins=101)
    plt.show()
