import functools

import arviz
import jax
import jax.numpy as jnp
import joblib
import numpy as np
from tqdm import tqdm


def get_min_ess(samples, method='bulk'):
    """get_min_ess
    Parameters
    ----------
    samples : np.array
        (n_chains, n_samples, n_dim)
    Returns
    -------
    """
    if samples.ndim == 1:
        samples = samples[None, :, None]
    elif samples.ndim == 2:
        samples = samples[None, ...]

    n_chains, n_samples, n_dim = samples.shape
    ess_list = joblib.Parallel(n_jobs=joblib.cpu_count(), prefer='threads')(
        joblib.delayed(arviz.ess)(samples[..., ii], relative=True, method=method)
        for ii in tqdm(range(n_dim))
    )
    ess_list = np.array(ess_list)
    return np.min(ess_list)


def make_samplers(joint_energy, sample_q_other, get_step_size=None):
    if get_step_size is None:

        @jax.jit
        def get_step_size(q_other, epsilon):
            return epsilon

    @jax.jit
    def take_leapfrog_step(q_hmc, p, q_other, epsilon):
        step_size = get_step_size(q_other, epsilon)
        p = p - 0.5 * step_size * jax.grad(joint_energy, argnums=0)(q_hmc, q_other)
        q_hmc = q_hmc + step_size * p
        p = p - 0.5 * step_size * jax.grad(joint_energy, argnums=0)(q_hmc, q_other)
        return q_hmc, p

    @jax.jit
    def mh_correction(q_hmc0, p0, q_other0, q_hmc, p, q_other, delta_U, v):
        U0 = joint_energy(q_hmc0, q_other0) + 0.5 * jnp.sum(p0 ** 2)
        U = joint_energy(q_hmc, q_other) + 0.5 * jnp.sum(p ** 2)
        accept = jnp.abs(v) <= jnp.exp(-(U - U0 - delta_U))
        q_hmc, p, q_other, v = jax.lax.cond(
            accept,
            lambda _: (q_hmc, p, q_other, v * jnp.exp(U - U0 - delta_U)),
            lambda _: (q_hmc0, p0, q_other0, v),
            None,
        )
        return q_hmc, p, q_other, v, accept

    @functools.partial(jax.jit, static_argnames="L")
    def take_multiple_leapfrog_steps(q_hmc, p, q_other, epsilon, L):
        def scan_f(carry, ii):
            q_hmc, p = carry
            q_hmc, p = take_leapfrog_step(q_hmc, p, q_other, epsilon)
            return (q_hmc, p), None

        (q_hmc, p), _ = jax.lax.scan(scan_f, (q_hmc, p), jnp.arange(L))
        return q_hmc, p

    # MALA within Gibbs
    @functools.partial(jax.jit, static_argnames="L")
    def take_multiple_mala_steps(q_hmc, q_other, key, epsilon, L):
        def scan_f(carry, ii):
            q_hmc, q_other, key = carry
            key, subkey = jax.random.split(key)
            p = jax.random.normal(subkey, shape=q_hmc.shape)
            q_hmc0, p0 = q_hmc, p
            q_hmc, p = take_multiple_leapfrog_steps(q_hmc, p, q_other, epsilon, 1)
            key, subkey = jax.random.split(key)
            q_hmc, p, q_other, _, accept = mh_correction(
                q_hmc0, p0, q_other, q_hmc, p, q_other, 0.0, jax.random.uniform(subkey)
            )
            return (q_hmc, q_other, key), accept

        (q_hmc, q_other, key), accept_list = jax.lax.scan(
            scan_f, (q_hmc, q_other, key), jnp.arange(L)
        )
        return q_hmc, key, accept_list

    @functools.partial(jax.jit, static_argnames="L")
    def mala_within_gibbs(q_hmc, q_other, key, epsilon, L):
        q_hmc, key, accept_list = take_multiple_mala_steps(
            q_hmc, q_other, key, epsilon, L
        )
        q_other, key = sample_q_other(q_hmc, key)
        return q_hmc, q_other, key, accept_list

    # HMC-within-Gibbs
    @functools.partial(jax.jit, static_argnames="L")
    def take_hmc_step(q_hmc, q_other, key, epsilon, L):
        key, subkey = jax.random.split(key)
        p = jax.random.normal(subkey, shape=q_hmc.shape)
        q_hmc0, p0 = q_hmc, p
        q_hmc, p = take_multiple_leapfrog_steps(q_hmc, p, q_other, epsilon, L)
        key, subkey = jax.random.split(key)
        q_hmc, p, q_other, _, accept = mh_correction(
            q_hmc0, p0, q_other, q_hmc, p, q_other, 0.0, jax.random.uniform(subkey)
        )
        return q_hmc, key, accept

    @functools.partial(jax.jit, static_argnames="L")
    def hmc_within_gibbs(q_hmc, q_other, key, epsilon, L):
        q_hmc, key, accept = take_hmc_step(q_hmc, q_other, key, epsilon, L)
        q_other, key = sample_q_other(q_hmc, key)
        return q_hmc, q_other, key, accept

    # MAHMC
    @functools.partial(jax.jit, static_argnames=("L", "N"))
    def take_mahmc_step(q_hmc, q_other, key, epsilon, L, N):
        key, subkey = jax.random.split(key)
        p = jax.random.normal(subkey, shape=q_hmc.shape)
        q_hmc0, q_other0, p0 = q_hmc, q_other, p

        def scan_f(carry, ii):
            q_hmc, p, q_other, delta_U, key = carry
            q_hmc, p = take_multiple_leapfrog_steps(q_hmc, p, q_other, epsilon, L)
            q_other0 = q_other
            q_other, key = sample_q_other(q_hmc, key)
            delta_U = (
                delta_U + joint_energy(q_hmc, q_other) - joint_energy(q_hmc, q_other0)
            )
            return (q_hmc, p, q_other, delta_U, key), None

        (q_hmc, p, q_other, delta_U, key), _ = jax.lax.scan(
            scan_f, (q_hmc, p, q_other, 0.0, key), jnp.arange(N - 1)
        )
        q_hmc, p = take_multiple_leapfrog_steps(q_hmc, p, q_other, epsilon, L)
        key, subkey = jax.random.split(key)
        q_hmc, p, q_other, _, accept = mh_correction(
            q_hmc0, p0, q_other0, q_hmc, p, q_other, delta_U, jax.random.uniform(subkey)
        )
        return q_hmc, q_other, key, accept

    @functools.partial(jax.jit, static_argnames=("L", "N"))
    def mahmc_within_gibbs(q_hmc, q_other, key, epsilon, L, N):
        q_hmc, q_other, key, accept = take_mahmc_step(
            q_hmc, q_other, key, epsilon, L, N
        )
        q_other, key = sample_q_other(q_hmc, key)
        return q_hmc, q_other, key, accept

    # MALA with persistent momentum
    @functools.partial(jax.jit, static_argnames="L")
    def take_multiple_mala_persistent_steps(q_hmc, p, q_other, key, epsilon, L, alpha):
        def scan_f(carry, ii):
            q_hmc, p, q_other, key = carry
            key, subkey = jax.random.split(key)
            n = jax.random.normal(subkey, shape=q_hmc.shape)
            p = alpha * p + jnp.sqrt(1 - alpha ** 2) * n
            q_hmc0, p0 = q_hmc, p
            q_hmc, p = take_multiple_leapfrog_steps(q_hmc, p, q_other, epsilon, 1)
            key, subkey = jax.random.split(key)
            q_hmc, p, q_other, _, accept = mh_correction(
                q_hmc0, -p0, q_other, q_hmc, p, q_other, 0.0, jax.random.uniform(subkey)
            )
            return (q_hmc, p, q_other, key), accept

        (q_hmc, p, q_other, key), accept_list = jax.lax.scan(
            scan_f, (q_hmc, p, q_other, key), jnp.arange(L)
        )
        return q_hmc, p, key, accept_list

    @functools.partial(jax.jit, static_argnames="L")
    def mala_persistent_within_gibbs(q_hmc, p, q_other, key, epsilon, L, alpha):
        q_hmc, p, key, accept_list = take_multiple_mala_persistent_steps(
            q_hmc, p, q_other, key, epsilon, L, alpha
        )
        q_other, key = sample_q_other(q_hmc, key)
        return q_hmc, p, q_other, key, accept_list

    # MALA with persistent momentum and non-reversible Metropolis accept/reject
    @functools.partial(jax.jit, static_argnames="L")
    def take_multiple_mala_persistent_nonreversible_steps(
        q_hmc, p, q_other, v, key, epsilon, L, alpha, delta
    ):
        def scan_f(carry, ii):
            q_hmc, p, q_other, v, key = carry
            key, subkey = jax.random.split(key)
            n = jax.random.normal(subkey, shape=q_hmc.shape)
            p = alpha * p + jnp.sqrt(1 - alpha ** 2) * n
            q_hmc0, p0 = q_hmc, p
            q_hmc, p = take_multiple_leapfrog_steps(q_hmc, p, q_other, epsilon, 1)
            q_hmc, p, q_other, v, accept = mh_correction(
                q_hmc0, -p0, q_other, q_hmc, p, q_other, 0.0, v
            )
            v = (v + 1 + delta) % 2 - 1
            return (q_hmc, p, q_other, v, key), accept

        (q_hmc, p, q_other, v, key), accept_list = jax.lax.scan(
            scan_f, (q_hmc, p, q_other, v, key), jnp.arange(L)
        )
        return q_hmc, p, v, key, accept_list

    @functools.partial(jax.jit, static_argnames="L")
    def mala_persistent_nonreversible_within_gibbs(
        q_hmc, p, q_other, v, key, epsilon, L, alpha, delta
    ):
        (
            q_hmc,
            p,
            v,
            key,
            accept_list,
        ) = take_multiple_mala_persistent_nonreversible_steps(
            q_hmc, p, q_other, v, key, epsilon, L, alpha, delta
        )
        q_other, key = sample_q_other(q_hmc, key)
        return q_hmc, p, q_other, v, key, accept_list

    return (
        mala_within_gibbs,
        hmc_within_gibbs,
        mahmc_within_gibbs,
        mala_persistent_within_gibbs,
        mala_persistent_nonreversible_within_gibbs,
    )
