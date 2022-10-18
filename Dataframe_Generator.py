import pandas as pd
import multiprocessing as mp
import os
from glob import glob
import numpy.random as npr

from Functions_Library import solver, dslm, no_mimicry, mimicry


def dataframe_generator(func=mimicry, sp2=True, N=5, comp=0.3, label=''):
    """
    :param func: function to use (no_mimicry, mimicry or dslm)
    :param sp2: True for two species, False for one species only
    :param N: number of simulations batches
    :param comp: interspecific competition value

    parameters=[(AB,SR,ab,sr,b,d,p,l1,k1,l2,k2,cw,cb,K,a,B)]
        - fixed parameters are replaced by a number
        - parameters to be drawn randomly in each batch of simulations must be indicated by rcond[] and placed
        in 'random_cond' object, specifying the minimum and maximum values of the interval
        - parameters of interest, for which different values are defined, must remain in letter form. The values for
        which one wants to run simulations must be indicated in the form: for X in [x1,x2...xn]


    :return: a csv dataframe with all parameters value, abundances, male proportions and state at the equilibrium.
    """
    if sp2 == False:
        comp = 0
        random_cond = [[npr.uniform(1, 1000), npr.uniform(0.2, 0.8), 0, 0, npr.uniform(0.7, 1), npr.uniform(0.1, 0.3),
                        npr.uniform(0.3, 0.7)] for i in range(N)]
    else:
        random_cond = [[npr.uniform(1, 1000), npr.uniform(0.2, 0.8), npr.uniform(1, 1000), npr.uniform(0.2, 0.8),
                        npr.uniform(0.7, 1), npr.uniform(0.1, 0.3), npr.uniform(0.3, 0.7)] for i in range(N)]

    parameters = [
        (rcond[0], rcond[1], rcond[2], rcond[3], rcond[4], rcond[5], 0.6, 0.02, rcond[6], 0, 1, 1, comp, 1000, a, B)
        for rcond in random_cond
        for a in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # parameter of interest 1
        for B in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]  # parameter of interest 2
        ]

    sol = [solver(func, AB, SR, ab, sr, b, d, p, l1, k1, l2, k2, cw, cb, K, a, B)
           for (AB, SR, ab, sr, b, d, p, l1, k1, l2, k2, cw, cb, K, a, B) in parameters]

    df = pd.DataFrame({'AB': [item[0] for item in parameters],
                       'SR': [item[1] for item in parameters],
                       'ab': [item[2] for item in parameters],
                       'sr': [item[3] for item in parameters],
                       'b': [item[4] for item in parameters],
                       'd': [item[5] for item in parameters],
                       'p': [item[6] for item in parameters],
                       'l1': [item[7] for item in parameters],
                       'k1': [item[8] for item in parameters],
                       'l2': [item[9] for item in parameters],
                       'k2': [item[10] for item in parameters],
                       'cw': [item[11] for item in parameters],
                       'cb': [item[12] for item in parameters],
                       'K': [item[13] for item in parameters],
                       'a': [item[14] for item in parameters],
                       'B': [item[15] for item in parameters],
                       'eq_sp1': [item[0] for item in sol],
                       'eq_sp2': [item[1] for item in sol],
                       'coexistence': [item[2] for item in sol],
                       'F': [item[3] for item in sol],
                       'M': [item[4] for item in sol],
                       'f': [item[5] for item in sol],
                       'm': [item[6] for item in sol]
                       })

    df.to_csv("./df_{0}.csv".format(label))


if __name__ == '__main__':
    dataframe_generator(func=no_mimicry, sp2=False, N=1, comp=0.3, label='one_sp_no_mimicry_aB')
