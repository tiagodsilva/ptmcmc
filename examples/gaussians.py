import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from ptmcmc.mh import random_walk
from ptmcmc.tgts import GaussianMixture2D
from ptmcmc.utils import show_kitty

logp = GaussianMixture2D(a=0.5, sigma=1e-1)

x = jnp.zeros((2,))

xs = random_walk(x, logp=logp, steps=int(1e6), sigma=0.15)

print(xs)

plt.scatter(xs[:, 0], xs[:, 1])
show_kitty()
