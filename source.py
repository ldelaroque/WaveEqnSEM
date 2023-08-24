#!/usr/bin/env python
# coding: utf-8


import math
import numpy as np


def source(dt, pt):
    
    nt = int(2 * pt / dt)
    s = np.zeros(nt)
    t0 = pt / dt
    a_ricker = 4 / pt
    
    for it in range(0, nt):
        t = ((it + 1) - t0) * dt
        s[it] = -2 * a_ricker * t * math.exp(-(a_ricker * t) ** 2)

    return s

def sourceRicker(dt, pt, f0):

    nt = int(2 * pt / dt)
    s = np.zeros(nt)
    t0 = pt / dt
    
    for it in range(0, nt):
        t = ((it + 1) - t0) * dt
        s[it] = (1.0 - 2.0 * (math.pi * f0 * (t - t0))**2) * math.exp(-math.pi**2 * f0**2 * (t - t0)**2)

    return s
