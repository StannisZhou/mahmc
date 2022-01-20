# Adapted from https://github.com/StannisZhou/mixed_hmc/blob/master/scripts/simple_gmm/test_naive_mixed_hmc.py

import matplotlib.pyplot as plt
import numba
import numpy as np
from tqdm import tqdm


def get_mixture_density(x, pi, mu_list, sigma_list):
    mixture_density = np.zeros_like(x)
    for ii in range(pi.shape[0]):
        mixture_density += (
            pi[ii]
            * np.exp(-0.5 * (x - mu_list[ii]) ** 2 / sigma_list[ii] ** 2)
            / np.sqrt(2 * np.pi * sigma_list[ii] ** 2)
        )

    return mixture_density


def mahmc(x0, q0, n_samples, epsilon, L, pi, mu_list, sigma_list):
    @numba.jit(nopython=True)
    def potential(x, q):
        potential = (
            -np.log(pi[x])
            + 0.5 * np.log(2 * np.pi * sigma_list[x] ** 2)
            + 0.5 * (q - mu_list[x]) ** 2 / sigma_list[x] ** 2
        )
        return potential

    @numba.jit(nopython=True)
    def grad_potential(x, q):
        grad_potential = (q - mu_list[x]) / sigma_list[x] ** 2
        return grad_potential

    @numba.jit(nopython=True)
    def take_mahmc_step(x0, q0, epsilon, L, n_components):
        # Resample momentum
        p0 = np.random.randn()
        # Initialize q, delta_U
        x = x0
        q = q0
        p = p0
        delta_U = 0.0
        # Take L steps
        for ii in range(L):
            q, p = leapfrog_step(x=x, q=q, p=p, epsilon=epsilon)
            x, delta_U = update_discrete(
                x0=x,
                q=q,
                delta_U=delta_U,
                n_components=n_components,
            )

        # Accept or reject
        current_E = potential(x0, q0) + 0.5 * p0 ** 2
        proposed_E = potential(x, q) + 0.5 * p ** 2
        accept = np.random.rand() < np.exp(current_E + delta_U - proposed_E)
        if not accept:
            x, q = x0, q0

        return x, q, accept

    @numba.jit(nopython=True)
    def leapfrog_step(x, q, p, epsilon):
        p -= 0.5 * epsilon * grad_potential(x, q)
        q += epsilon * p
        p -= 0.5 * epsilon * grad_potential(x, q)
        return q, p

    @numba.jit(nopython=True)
    def update_discrete(x0, q, delta_U, n_components):
        x = x0
        distribution = np.ones(n_components)
        distribution[x] = 0
        distribution /= np.sum(distribution)
        proposal_for_ind = np.argmax(np.random.multinomial(1, distribution))
        x = proposal_for_ind
        delta_E = potential(x, q) - potential(x0, q)
        # Decide whether to accept or reject
        accept = np.random.exponential() > delta_E
        if accept:
            delta_U += potential(x, q) - potential(x0, q)
        else:
            x = x0

        return x, delta_U

    x, q = x0, q0
    x_samples, q_samples, accept_list = [], [], []
    for _ in tqdm(range(n_samples)):
        x, q, accept = take_mahmc_step(
            x0=x, q0=q, epsilon=epsilon, L=L, n_components=pi.shape[0]
        )
        x_samples.append(x)
        q_samples.append(q)
        accept_list.append(accept)

    x_samples = np.array(x_samples)
    q_samples = np.array(q_samples)
    accept_list = np.array(accept_list)
    return x_samples, q_samples, accept_list


pi = np.array([0.15, 0.3, 0.3, 0.25])
mu_list = np.array([-2, 0, 2, 4])
sigma_list = np.sqrt(0.1) * np.ones(pi.shape[0])


x0 = np.random.randint(4)
q0 = np.random.randn()
n_warm_up_samples = int(1e6)
n_samples = int(4e6)
epsilon = 0.3
L = 15


x_samples, q_samples, accept_list = mahmc(
    x0,
    q0,
    n_warm_up_samples + n_samples,
    epsilon,
    L,
    pi,
    mu_list,
    sigma_list,
)

print(np.mean(accept_list))

x = np.linspace(-10, 10, int(1e4))
mixture_density = get_mixture_density(x, pi, mu_list, sigma_list)
fig, ax = plt.subplots(1, 1)
ax.hist(q_samples[n_warm_up_samples:], density=True, bins=500)
ax.plot(x, mixture_density)
plt.show()
