import jax
import jax.numpy as jnp


class GaussianMixture2D:
    def __init__(self, a: float, sigma: float):
        # We put the centers at [a, a], [a, -a], [-a, a], [-a, -a]
        self.centers = jnp.array([[a, a], [a, -a], [-a, a], [-a, -a]])
        self.sigma = sigma

    def __call__(self, x: jax.Array):
        logp = jax.nn.logsumexp(
            -0.5
            * jnp.sum((x[None, :] - self.centers) ** 2, axis=1)
            / self.sigma**2,
            axis=0,
        )
        return logp


class Rings2D:
    def __init__(self, r1: float, r2: float, w1: float, sigma: float):
        self.r1 = r1
        self.r2 = r2
        self.ws = jnp.array([w1, 1 - w1])

        self.sigma = sigma

    def __call__(self, x: jax.Array):
        x_norm = jnp.linalg.norm(x)
        dists = jnp.array([(x_norm - self.r1) ** 2, (x_norm - self.r2) ** 2])

        return jax.nn.logsumexp(-dists / self.sigma**2, b=self.ws, axis=0)
