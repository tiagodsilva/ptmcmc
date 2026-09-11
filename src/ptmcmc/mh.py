from functools import partial

import jax
import jax.numpy as jnp


def random_walk_step(
    x_t: jax.Array,
    key: jax.Array,
    logps_t: jax.Array,
    beta: jax.Array,
    *,
    sigma: float,
    logp: callable,
):

    # For the step forward, and for the MH step
    kf, km, k_next = jax.random.split(key, 3)

    x_tp1 = sigma * jax.random.normal(kf, x_t.shape) + x_t
    logps_tp1 = beta * logp(x_tp1)

    # Since the kernel is symmetrical, \alpha = \min { 1, p(x_{t + 1}) / p(x_t)}
    log_alpha = jnp.minimum(0.0, logps_tp1 - logps_t)
    log_u = jnp.log(jax.random.uniform(km, log_alpha.shape))

    x_next = jnp.where((log_u <= log_alpha)[None], x_tp1, x_t)
    logps_next = jnp.where(log_u <= log_alpha, logps_tp1, logps_t)

    return x_next, k_next, logps_next


def random_walk(
    x: jax.Array,
    logp: callable,
    steps: int,
    burnin: float = 0.1,
    sigma: float = 1e-2,
    seed: int = 42,
):
    key = jax.random.key(seed)
    beta = jnp.array(1.0)

    @jax.jit
    def step(carry: tuple[jax.Array, jax.Array, jax.Array], _):
        x_t, key, logps_t = carry
        x_next, k_next, logps_next = random_walk_step(
            x_t, key, logps_t, beta, sigma=sigma, logp=logp
        )
        return (x_next, k_next, logps_next), x_next

    # Random walk Metropolis Hasting
    _, xs = jax.lax.scan(
        f=step,
        init=(x, key, logp(x)),
        length=steps,
    )

    return xs[int(burnin * steps) :]


def pt_random_walk(
    x: jax.Array,
    beta: jax.Array,
    logp: callable,
    steps: int,
    burnin: float = 0.1,
    sigma: float = 1e-2,
    alpha: float = 0.9,
    seed: int = 42,
):

    @jax.jit
    def pt_step(carry: tuple[jax.Array, jax.Array, jax.Array], _):
        x_t, key, logps_t = carry

        kf, ks = jax.random.split(key, 2)

        num_chains = len(x_t)
        # Decide whether to swap MCMC chains or to update them in parallel
        should_swap = jax.random.bernoulli(ks, 1 - alpha)

        def swap():
            # Pick a pair uniformly at random
            ki, ku, k_next = jax.random.split(kf, 3)

            i = jax.random.categorical(ki, jnp.ones(num_chains - 1))

            # Compute the probabilities for each state
            logp_i_i, logp_ip1_ip1 = logps_t[i], logps_t[i + 1]

            # p^\beta[i] -> \beta[i] log p
            logp_i_ip1 = logp_i_i * beta[i + 1] / beta[i]
            logp_ip1_i = logp_ip1_ip1 * beta[i] / beta[i + 1]

            # Compute the swapping probability
            log_alpha = jnp.minimum(
                0.0, logp_i_ip1 + logp_ip1_i - logp_i_i - logp_ip1_ip1
            )
            log_u = jnp.log(jax.random.uniform(ku, log_alpha.shape))

            x_next = jnp.where(
                (log_u <= log_alpha)[None],
                x_t.at[i].set(x_t[i + 1]).at[i + 1].set(x_t[i]),
                x_t,
            )
            logps_next = jnp.where(
                log_u <= log_alpha,
                logps_t.at[i].set(logps_t[i + 1]).at[i + 1].set(logps_t[i]),
                logps_t,
            )

            return x_next, k_next, logps_next

        step = jax.vmap(
            partial(random_walk_step, sigma=sigma, logp=logp),
            in_axes=(0, None, 0, 0),
            out_axes=(0, None, 0),
        )

        x_next, k_next, logps_next = jax.lax.cond(
            should_swap, swap, lambda: step(x_t, key, logps_t, beta)
        )

        return (x_next, k_next, logps_next), x_next

    logp_vmap = jax.vmap(logp, in_axes=(0,), out_axes=0)

    key = jax.random.key(seed)
    _, xs = jax.lax.scan(
        f=pt_step,
        init=(x, key, beta * logp_vmap(x)),
        length=steps,
    )

    # Take the samples from the target chain
    return xs[int(burnin * steps) :, -1]
