#!/usr/bin/env python
# coding: utf-8

# value of Legendre Polynomial N(Ω) at position Ω[-1, 1].

import numpy as np

def legendre(N, Ω):

    polynomial = np.zeros(2 * N)

    if N == 0:
        polynomial[0] = 1
    elif N == 1:
        polynomial[1] = Ω
    else:
        polynomial[0] = 1
        polynomial[1] = Ω
    for i in range(2, N + 1):
        polynomial[i] = (1.0 / float(i)) * ((2 * i - 1) * Ω * polynomial[i - 1] - (i - 1) *
                                   polynomial[i - 2])

    return(polynomial[N])

