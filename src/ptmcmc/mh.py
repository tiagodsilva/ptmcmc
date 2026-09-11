import jax
import jax.numpy as jnp


def random_walk(
    x: jax.Array,
    logp: callable,
    steps: int,
    burnin: float = 0.1,
    sigma: float = 1e-2,
    seed: int = 42,
):
    key = jax.random.key(seed)

    # Random walk Metropolis Hasting
    @jax.jit
    def random_walk_step(carry: tuple[jax.Array, jax.Array, jax.Array], _):
        x_t, key, logps_t = carry

        # For the step forward, and for the MH step
        kf, km, k_next = jax.random.split(key, 3)

        x_tp1 = sigma * jax.random.normal(kf, x_t.shape) + x_t
        logps_tp1 = logp(x_tp1)

        # Since the kernel is symmetrical, \alpha = \min { 1, p(x_{t + 1}) / p(x_t)}
        log_alpha = jnp.minimum(0.0, logps_tp1 - logps_t)
        log_u = jnp.log(jax.random.uniform(km, log_alpha.shape))

        x_next = jnp.where((log_u <= log_alpha)[None], x_tp1, x_t)
        logps_next = jnp.where(log_u <= log_alpha, logps_tp1, logps_t)

        return (x_next, k_next, logps_next), x_next

    _, xs = jax.lax.scan(
        f=random_walk_step, init=(x, key, logp(x)), length=steps
    )

    return xs[int(burnin * steps) :]
