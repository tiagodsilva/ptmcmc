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
