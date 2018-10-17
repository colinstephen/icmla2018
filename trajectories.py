#!/usr/bin/python
# -*- coding: iso-8859-15 -*-

'''
About
=====
Implements functions generating trajectories for chaotic attractors and
stochastic systems.
'''


import numpy as np
from scipy.stats import zscore
from scipy.integrate import odeint
from statsmodels.tsa.arima_process import arma_generate_sample


def lorentz_attractor(
    trajectory_length, 
    delay=1000, 
    noise_factor=0.1,
    sigma=10, 
    beta=8.0/3, 
    rho=28, 
    u0=None, 
    v0=None, 
    w0=None,
    normalize=True):

    if u0 is None:
        u0 = np.random.uniform()
    if v0 is None:
        v0 = np.random.uniform()
    if w0 is None:
        w0 = np.random.uniform()
    
    def lorenz(X, t, sigma, beta, rho):
        """The Lorenz equations."""
        u, v, w = X
        up = -sigma*(u - v)
        vp = rho*u - v - u*w
        wp = -beta*w + u*v
        return up, vp, wp

    t = np.arange(delay + trajectory_length)
    f = odeint(lorenz, (u0, v0, w0), t, args=(sigma, beta, rho))
    x, y, z = f.T

    if normalize:
        x = zscore(x)
        y = zscore(y)
        z = zscore(z)

    x += np.random.randn(len(x)) * np.std(x) * noise_factor
    y += np.random.randn(len(y)) * np.std(y) * noise_factor
    z += np.random.randn(len(z)) * np.std(z) * noise_factor

    return x[delay:], y[delay:], z[delay:]


def henon_attractor(
    trajectory_length, 
    delay=1000,
    noise_factor=0.1,
    alpha=1.4,
    alpha_coeff_range=0.005,
    beta=0.3035, 
    beta_coeff_range=0.005,
    u0=None, 
    v0=None,
    normalize=True):
    
    if u0 is None:
        u0 = np.random.uniform(1,2)
    if v0 is None:
        v0 = np.random.uniform(1,2)

    alpha = np.random.uniform(low = alpha - alpha_coeff_range / 2,
                              high = alpha + alpha_coeff_range / 2)
    beta = np.random.uniform(low = beta - beta_coeff_range / 2,
                              high = beta + beta_coeff_range / 2)

    # Henon map
    def henon(u, v):
        up = 1 - alpha*u**2 + v
        vp = beta*u
        return up, vp

    x_, y_ = u0, v0
    for i in range(delay):
        x_, y_ = henon(x_, y_)

    x, y = np.empty(trajectory_length), np.empty(trajectory_length)
    x[0], x[0] = x_, y_

    for i in range(trajectory_length - 1):
        x[i+1], y[i+1] = henon(x[i], y[i])

    if normalize:
        x = zscore(x)
        y = zscore(y)

    x += np.random.randn(trajectory_length) * noise_factor
    y += np.random.randn(trajectory_length) * noise_factor

    return x, y


def chua_attractor(
    trajectory_length,
    delay=2000,
    noise_factor=0.1,
    alpha=15.6,
    alpha_coeff_range=0.005,
    beta=28.0,
    beta_coeff_range=0.005,
    c=-0.714,
    c_coeff_range=0.005,
    d=-1.143,
    d_coeff_range=0.005,
    u0=0.7, 
    v0=0, 
    w0=0,
    normalize=True):

    if u0 is None:
        u0 = np.random.uniform(0,1)
    if v0 is None:
        v0 = np.random.uniform(0,1)
    if w0 is None:
        w0 = np.random.uniform(0,1)

    alpha = np.random.uniform(low = alpha - alpha_coeff_range / 2,
                              high = alpha + alpha_coeff_range / 2)
    beta = np.random.uniform(low = beta - beta_coeff_range / 2,
                              high = beta + beta_coeff_range / 2)
    c = np.random.uniform(low = c - c_coeff_range / 2,
                              high = c + c_coeff_range / 2)
    d = np.random.uniform(low = d - d_coeff_range / 2,
                              high = d + d_coeff_range / 2)

    def chua(x, y, z, dt):
        f = c*x + 0.5*(d-c)*(abs(x+1)-abs(x-1))
        xp = x + alpha*(y - x - f)*dt
        yp = y + (x - y + z)*dt
        zp = z - beta * y *dt
        return xp, yp, zp

    x_, y_, z_ = u0, v0, w0

    dt = 1 / 1000
    for i in range(delay):
        x_, y_, z_ = chua(x_, y_, z_, dt)

    N = trajectory_length
    x = np.empty(N)
    y = np.empty(N)
    z = np.empty(N)

    x[0], y[0], z[0] = x_, y_, z_
    
    for i in range(N - 1):
        x[i+1], y[i+1], z[i+1] = chua(x[i], y[i], z[i], dt)
        
    if normalize:
        x = zscore(x)
        y = zscore(y)
        z = zscore(z)

    x += np.random.randn(N) * np.std(x) * noise_factor
    y += np.random.randn(N) * np.std(y) * noise_factor
    z += np.random.randn(N) * np.std(z) * noise_factor

    return x, y, z


def rossler_attractor(
    trajectory_length,
    delay=40000,
    noise_factor=0.1,
    a=0.2,
    b=0.2,
    c=5.7,
    u0=None,
    v0=None,
    w0=None,
    normalize=True):
    
    if u0 is None:
        u0 = np.random.random()
    if v0 is None:
        v0 = np.random.random()
    if w0 is None:
        w0 = np.random.random()
    
    def rossler(x, y, z, dt):
        xp = x + (-y - z) * dt
        yp = y + (x + a*y) * dt
        zp = z + (b + z * (x - c)) * dt
        return xp, yp, zp
    
    N = trajectory_length
    dt = 100 / delay

    x_, y_, z_ = u0, v0, w0
    for i in range(delay):
        x_, y_, z_ = rossler(x_, y_, z_, dt)

    x, y, z = np.empty(N), np.empty(N), np.empty(N)
    x[0], y[0], z[0] = x_, y_, z_

    for i in range(N-1):
        x[i+1], y[i+1], z[i+1] = rossler(x[i], y[i], z[i], dt)

    if normalize:
        x, y, z = zscore(x), zscore(y), zscore(z)
        
    x += np.random.randn(N) * np.std(x) * noise_factor
    y += np.random.randn(N) * np.std(x) * noise_factor
    z += np.random.randn(N) * np.std(x) * noise_factor

    return x, y, z


def logistic_attractor(
    trajectory_length,
    delay=1000,
    noise_factor=0.1,
    a=3.9995,
    a_coeff_range=0.001,
    u0=None,
    normalize=True,
    **kwargs):
    
    if u0 is None:
        u0 = np.random.random()
    
    a = np.random.uniform(low = a - a_coeff_range/2,
                          high = a + a_coeff_range/2)

    def logistic(x):
        return a * x * (1-x)
    
    x_ = u0
    for i in range(delay):
        x_ = logistic(x_)

    x = np.empty(trajectory_length)
    x[0] = x_

    for i in range(trajectory_length - 1):
        x[i+1] = logistic(x[i])
    
    x += np.random.randn(trajectory_length) * noise_factor

    if normalize:
        x = zscore(x)
        
    return (x,)


def arma_model(
    trajectory_length,
    delay=1000,
    ar=[1],
    ma=[1],
    sigma=0.1,
    normalize=True):
    
    x = arma_generate_sample(ar, ma, trajectory_length, burnin=delay,
                             sigma=sigma)
    if normalize:
        x = zscore(x)

    return x

