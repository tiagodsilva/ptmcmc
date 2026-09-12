from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from ptmcmc.mh import pt_random_walk, random_walk
from ptmcmc.tgts import Rings2D
from ptmcmc.utils import show_kitty

logp = Rings2D(r1=0.5, r2=2.0, w1=0.5, sigma=1e-2)

N = 32
D = 2
x = jnp.zeros((N, D))

xs_rw = jax.vmap(
    partial(random_walk, logp=logp, steps=int(5e5), sigma=0.15),
    in_axes=(0,),
    out_axes=0,
)(x)
xs_rw = xs_rw.reshape(-1, D)

xs_pt = pt_random_walk(
    x,
    logp=logp,
    steps=int(5e5),
    sigma=0.15,
)

_, (ax_rw, ax_pt) = plt.subplots(
    1, 2, sharex=True, sharey=True, figsize=(8, 4)
)


ax_rw.scatter(xs_rw[:, 0], xs_rw[:, 1])
ax_rw.set_title("Random Walk")

ax_pt.scatter(xs_pt[:, 0], xs_pt[:, 1])
ax_pt.set_title("Random Walk with PT")

plt.tight_layout()

plt.savefig("rings.png")
show_kitty()
