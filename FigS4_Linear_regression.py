### Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy.random as npr
import seaborn as sns

from Functions_Library import solver, no_mimicry
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression


stats = []
all_X = np.array([], dtype='float64')
all_Y = np.array([], dtype='float64')

### Set the figure
fig, ax = plt.subplots()

ax.set_xlim(0, 1)
ax.set_xlabel(r'predation rate $p$', fontsize=20, fontweight='bold')
ax.set_ylabel(r'proportion of male at equilibrium $\rho_*$', fontsize=20, fontweight='bold')
ax.tick_params(axis='both', which='major', labelsize=15)

### Data
for i in range(4):
    h = [2, 3, 4, 5][i]
    col = ['red', 'blue', 'orange', 'green'][i]

    random_cond = [[npr.uniform(1, 1000), npr.uniform(0.2, 0.8), 0, 0,
                    npr.uniform(0.7, 1), npr.uniform(0.1, 0.3), npr.uniform(0, 1)] for i in range(5000)]

    sol = [
        solver(no_mimicry, cond[0], cond[1], cond[2], cond[3], cond[4], cond[5], cond[6], 0.01, h, 0, 1, 1, 0, 1000, 5,
               0.8) for cond in random_cond]

    M = [item[4] for item in sol]
    F = [item[3] for item in sol]
    E = [item[0] for item in sol]

    SR = [M[i] / (F[i] + M[i]) if E[i] == 1 else -1 for i in range(5000)]

    X = np.array([cond[6] for cond in random_cond], dtype='float64')
    Y = np.array(SR, dtype='float64')

    X = X[Y != -1]
    Y = Y[Y != -1]

    all_X = np.append(all_X, X)
    all_Y = np.append(all_Y, Y)

    Xcol = X.reshape((-1, 1))

    modeleReg = LinearRegression()

    modeleReg.fit(Xcol, Y)
    freg = f_regression(Xcol, Y)

    stats.append(
        'h = {0}'.format(h) + ', df = {0}'.format(len(Y))+', F = {0}'.format(freg[0]) + ', p = {0}'.format(freg[1]) +
        ', coef = {0}'.format(modeleReg.coef_))

    sns.regplot(x=X, y=Y, color=col, ax=ax, scatter=True, scatter_kws={'alpha': 0.3}, label='h = {0}'.format(h),
                line_kws={'linewidth': 3})

print(stats)

all_Xcol = all_X.reshape((-1, 1))
all_modeleReg = LinearRegression()
print(len(all_Y))

all_modeleReg.fit(all_Xcol, all_Y)
all_freg = f_regression(all_Xcol, all_Y)

score = all_modeleReg.score(all_Xcol, all_Y)

print('all included' + ', df = {0}'.format(len(all_Y))+', F = {0}'.format(all_freg[0]) + ', p = {0}'.format(all_freg[1])
      + ', coef = {0}'.format(all_modeleReg.coef_))

ax.legend(loc='best', prop={'size': 20})
plt.show()
