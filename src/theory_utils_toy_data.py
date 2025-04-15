import numpy as np
from scipy import optimize
from collections import defaultdict

import jax
import jax.numpy as jnp

import os

from jax.numpy import trace
from jax.numpy.linalg import pinv
from jax.scipy.special import erfc

from .manifolds import GaussianToyManifold, GaussianToyManifoldStructered

pinv = jax.jit(pinv)
trace = jax.jit(trace)
erfc = jax.jit(erfc)


@jax.jit
def F_G_fns(eps_eff):

    erf_piece = erfc(eps_eff/jnp.sqrt(2))
    exp_piece = jnp.sqrt(2/jnp.pi) * jnp.exp(-eps_eff**2/2)

    f = erf_piece
    g = 1 + eps_eff**2 - eps_eff * exp_piece / erf_piece

    return f, g


@jax.jit
def G_matrix_fn(inputs, *args):

    (alpha, epsilon, lamb, corr, noise) = args
    kappa, eps_eff = inputs
    β = corr
    σ = noise

    # Additional definitions
    # Σ_0 = Σ_ψ
    # Σ_1 = Σ_ψ - Σ_ψ_ψbar
    # Σ_2 = Σ_ψbar + Σ_ψ - Σ_ψ_ψbar - Σ_ψ_ψbar.T

    # N = len(Σ_ψ)
    # alpha = p/N

    # Get F1 and F2
    f, g = F_G_fns(eps_eff)

    # Effective sample size
    alpha_eff = alpha*f

    # gamma and delta
    gamma = alpha_eff/(alpha_eff+kappa)**2  # alpha_eff/kappa**2 * M**2
    delta = kappa/(alpha_eff+kappa)**2  # 1/kappa * M*Ginv
    # assert jnp.allclose(delta, 1 - gamma - lamb/kappa)

    # Weights
    w_eff = 1/(1+β*σ**2)

    # E_inf
    E_inf = β*σ**2/(1+β*σ**2)*(σ != 0) + β*(σ == 0)

    # W_tilde
    W_tilde = E_inf + (1-E_inf) * kappa**2/(alpha_eff*(1+β*σ**2)/(1+(1-β)*σ**2) + kappa)**2

    return alpha_eff, f, g, gamma, delta, E_inf, W_tilde, w_eff


@jax.jit
def kappa_fn(kappa, eps, *args):

    (alpha, epsilon, lamb, corr, noise) = args
    β = corr
    σ = noise

    # Get F1 and F2
    f, g = F_G_fns(eps)

    # Effective sample size
    alpha_eff = alpha*f

    return lamb/(1+σ**2*(1-β)) + kappa/(alpha_eff + kappa)


@jax.jit
def eps_fn(kappa, eps, *args):

    (alpha, epsilon, lamb, corr, noise) = args

    # Get F1 and F2
    f, g = F_G_fns(eps)

    # Effective sample size
    alpha_eff = alpha*f
    gamma = alpha_eff / (alpha_eff + kappa)**2

    W_tilde = corr + (1-corr) * kappa**2 / (alpha_eff + kappa)**2
    eps_self = epsilon * jnp.sqrt(1 - g*gamma) / jnp.sqrt(W_tilde)
    C_self = W_tilde / (1 - g*gamma)

    return eps_self, C_self


@jax.jit
def eps_fn_old(W_tilde, g, gamma, *args):

    (alpha, epsilon, lamb, corr, noise) = args

    eps_self = epsilon * jnp.sqrt(1 - g*gamma) / jnp.sqrt(W_tilde)
    C_self = W_tilde / (1 - g*gamma)

    return eps_self, C_self


@jax.jit
def kappa_eps_fn(inputs, *args):

    kappa, eps = inputs

    # (alpha_eff, f, g,
    #  gamma, delta, E_inf, W_tilde, w_eff) = G_matrix_fn(inputs, *args)

    # kappa self
    kappa_self = kappa_fn(kappa, eps, *args)

    # C self
    #  eps_self, _ = eps_fn(W_tilde, g, gamma, *args)
    eps_self, _ = eps_fn(kappa, eps, *args)

    # Self-consistent equations
    kappa_eq = kappa - kappa_self
    eps_eq = eps - eps_self

    return [kappa_eq, eps_eq]


@jax.jit
def get_logZ_gamma(inputs, *args):

    (alpha, epsilon, lamb, corr, noise) = args
    kappa, eps = inputs
    β = corr
    σ = noise

    (alpha_eff, f, g,
     gamma, delta, E_inf, W_tilde, w_eff) = G_matrix_fn(inputs, *args)

    # _, C = eps_fn(W_tilde, g, gamma, *args)
    _, C = eps_fn(kappa, eps, *args)
    Q = kappa - lamb
    Qhat = alpha_eff / kappa
    Chat = -alpha_eff/kappa**2 * C*g

    # G_0
    G_0 = Qhat*(E_inf + (1-E_inf)*kappa/(alpha_eff*(1+β*σ**2)/(1+(1-β)*σ**2) + kappa) - C)

    # G_1
    G_1 = Qhat*C*g

    # logZ
    logZ = G_0 + G_1

    # S1 = alpha_eff/kappa*C*g*(1-gamma) + (1 - Ginv)**2 * proj**2

    # Training error
    Etr = lamb**2/kappa**2 * C*f*g / (1 + σ**2*(1-β))**2
    Etr_scaled = alpha/lamb * Etr

    # # Weight norm
    # w_mean_sq = (1 - Ginv)**2 * proj**2
    # w_sq_mean = C*g*gamma + w_mean_sq

    # # w \cdot \bar\w
    # overlap = (1 - Ginv) * proj

    return_dict = dict(f=f,
                       g=g,
                       alpha_eff=alpha_eff,
                       alpha=alpha,
                       epsilon=epsilon,
                       lamb=lamb,
                       corr=β,
                       noise=σ,
                       E_inf=E_inf,
                       kappa=kappa,
                       gamma=gamma,
                       delta=delta,
                       epsilon_eff=eps,
                       W_tilde=W_tilde,
                       C_eps=C*f*g,
                       C=C,
                       Q=Q,
                       Qhat=Qhat,
                       Chat=Chat,
                       logZ=logZ,
                       Etr=Etr,
                       Etr_scaled=Etr_scaled,
                       #    S=S1,
                       #    w_mean_sq=w_mean_sq,
                       #    w_sq_mean=w_sq_mean,
                       #    overlap=overlap
                       )

    return return_dict


kappa_eps_fn_prime = jax.jit(jax.jacfwd(kappa_eps_fn, argnums=0))


def solve_kappa_gamma(pvals, epsilon, ridge, corr=0, noise=0, debug=True):

    if isinstance(ridge, float):
        ridge = np.array([ridge]*len(pvals))
    assert len(ridge) == len(pvals)

    if corr == 1:
        corr = 9.99e-1

    fun, fprime = kappa_eps_fn, kappa_eps_fn_prime

    returns_dict = defaultdict(list)
    returns_dict["pvals_th"] = list(pvals)
    returns_dict["lamb_th"] = list(ridge)

    for i, (alpha, lamb) in enumerate(zip(pvals, ridge)):
        args = (alpha, epsilon, lamb, corr, noise)

        kappa_0 = lamb + 1
        C_0 = 1
        eps_0 = epsilon / jnp.sqrt(C_0)

        kappa, eps = optimize.root(fun=fun,
                                   jac=fprime,
                                   args=args,
                                   x0=np.array([kappa_0, eps_0]),  # When p = 0
                                   method='hybr',
                                   #  method='lm',
                                   #  method='broyden1',
                                   #  method='broyden2',
                                   #  method='anderson',
                                   #  method='linearmixing',
                                   #  method='diagbroyden',
                                   #  method='excitingmixing',
                                   #  method='krylov',
                                   #  method='df-sane',
                                   tol=1e-14).x

        returns = get_logZ_gamma([kappa, eps], *args)
        if debug:
            pass

        for key, val in returns.items():
            returns_dict[key] += [val.item()]

    for key, val in returns_dict.items():
        returns_dict[key] = np.array(val)

    return returns_dict


def linear_problem(fun, manifold, p_tr, num_trials=1, **kwargs):

    fun_name = fun.__name__

    tr_err = np.zeros(num_trials)
    gen_err = np.zeros(num_trials)

    try:
        epsilon = kwargs['epsilon']
    except Exception:
        epsilon = 0

    for i in range(num_trials):

        X_tr, y_tr, X_test, y_test = manifold.get_manifold(p_tr, manifold_seed=i)

        tr_err[i], gen_err[i] = fun(X_tr, y_tr, X_test, y_test, **kwargs)

    returns = {f'tr_err_{fun_name}': tr_err.mean(),
               f'gen_err_{fun_name}': gen_err.mean(),
               f'eff_ratio_{fun_name}': erfc(epsilon/jnp.sqrt(2*gen_err)).mean(),
               f'eff_ratio2_{fun_name}': erfc(epsilon/jnp.sqrt(2*gen_err.mean())),
               }

    return returns

def SVR_th(alpha, epsilon=0, lamb=1e-10, corr=0, noise=0):

    returns = {}

    returns |= solve_kappa_gamma([alpha], epsilon, lamb, corr, noise)

    return returns


def SVR_exp(fns, alpha, manifold, epsilon=0, lamb=1e-10, num_trials=1):

    N = manifold.N

    p_tr = int(alpha*N)

    kwargs = {'p_tr': p_tr,
              'epsilon': epsilon,
              'num_trials': num_trials}

    returns = {}
    for fn in fns:

        if fn.__name__ in ["linsvr1", "linsvr_jax"]:
            kwargs['reg'] = 1/(2*lamb)

        elif fn.__name__ == "linsvr2":
            kwargs['lamb'] = 2*lamb/p_tr

        elif "linreg" in fn.__name__:
            kwargs['lamb'] = lamb

        else:
            raise Exception(f'Specify the regularization for this algorithm: {fn.__name__}')

        returns |= linear_problem(fn, manifold, **kwargs)

    return returns


def run_experiment(fns, cfg, override=False, SVR_th=SVR_th, SVR_exp=SVR_exp, Manifold=GaussianToyManifold, suffix=""):

    manifold_kwargs = cfg.manifold_kwargs
    exp_kwargs = cfg.exp_kwargs

    lamb = exp_kwargs.lamb
    num_trials = exp_kwargs.num_trials

    alpha_list = np.arange(0.1, 3, 0.1)
    alpha_list_th = np.linspace(alpha_list[0], alpha_list[-1], 101)

    epsilon_list = np.linspace(0, 1.2, exp_kwargs.grid_size)
    corr_list = np.linspace(0, 1., exp_kwargs.grid_size)
    noise_list = [0]
    if Manifold == GaussianToyManifoldStructered:
        noise_list = np.linspace(1e-5, 1., exp_kwargs.grid_size)

    exp_params = {'alpha_list': alpha_list,
                  'alpha_list_th': alpha_list_th,
                  'epsilon_list': epsilon_list,
                  'noise_list': noise_list,
                  'corr_list': corr_list}
    exp_params |= dict(manifold_kwargs.items() | exp_kwargs.items())

    results_dir = './results'
    file_name = exp_kwargs.file_name

    os.makedirs(results_dir, exist_ok=True)
    file_name = os.path.join(results_dir, file_name)

    if os.path.isfile(file_name) and not override:
        data = np.load(file_name, allow_pickle=True)
        exp_params = data['exp_params'].tolist()
        exp_params['file_name'] = file_name

        all_results_exp = data['all_results_exp'].tolist()
        all_results_th = data['all_results_th'].tolist()

    else:
        # Generate Manifold with fixed seed but varying noise std
        manifold = Manifold(**manifold_kwargs)

        all_results_exp = None
        all_results_th = None
        for i, corr in enumerate(corr_list):
            manifold.update_corr(corr=corr)

            for j, noise in enumerate(noise_list):
                manifold.update_corr(noise=noise)

                for k, epsilon in enumerate(epsilon_list):
                    print(f"corr: {corr:.2f}, eps: {epsilon:.2f}, noise: {noise:.2f}")

                    # Theory Loop
                    for n, alpha in enumerate(alpha_list_th):

                        scores = SVR_th(alpha, epsilon=epsilon, lamb=lamb, corr=corr, noise=noise)

                        if all_results_th is None:
                            all_results_th = {key: np.zeros((len(corr_list),
                                                             len(noise_list),
                                                             len(epsilon_list),
                                                             len(alpha_list_th))) for key in scores.keys()}

                        for key, val in scores.items():
                            all_results_th[key][i, j, k, n] = val

                    # Experiment Loop
                    for n, alpha in enumerate(alpha_list):

                        scores = SVR_exp(fns, alpha, manifold, epsilon, lamb=lamb, num_trials=num_trials)

                        if all_results_exp is None:
                            all_results_exp = {key: np.zeros((len(corr_list),
                                                              len(noise_list),
                                                              len(epsilon_list),
                                                              len(alpha_list))) for key in scores.keys()}

                        for key, val in scores.items():
                            all_results_exp[key][i, j, k, n] = val

        exp_params['file_name'] = file_name
        np.savez(file_name,
                 exp_params=exp_params,
                 all_results_exp=all_results_exp,
                 all_results_th=all_results_th)

    return exp_params, all_results_exp, all_results_th


def run_simulation(cfg, override=False, SVR_th=SVR_th, suffix="", Manifold=GaussianToyManifold, verbose=True):

    manifold_kwargs = cfg.manifold_kwargs
    exp_kwargs = cfg.exp_kwargs

    lamb = exp_kwargs.lamb
    # num_trials = exp_kwargs.num_trials

    alpha_list = np.arange(0.1, 3, 0.1)
    alpha_list_th = np.linspace(alpha_list[0], alpha_list[-1], 100)

    epsilon_list = np.linspace(0, 1.2, exp_kwargs.grid_size)
    corr_list = np.linspace(0, 1, exp_kwargs.grid_size)
    noise_list = [0]
    if Manifold == GaussianToyManifoldStructered:
        noise_list = np.linspace(1e-5, 1., exp_kwargs.grid_size)

    exp_params = {'alpha_list': alpha_list,
                  'alpha_list_th': alpha_list_th,
                  'epsilon_list': epsilon_list,
                  'noise_list': noise_list,
                  'corr_list': corr_list}
    exp_params |= dict(manifold_kwargs.items() | exp_kwargs.items())

    results_dir = './results'
    file_name = exp_kwargs.file_name

    os.makedirs(results_dir, exist_ok=True)
    file_name = os.path.join(results_dir, file_name)

    if os.path.isfile(file_name) and not override:
        data = np.load(file_name, allow_pickle=True)
        exp_params = data['exp_params'].tolist()
        exp_params['file_name'] = file_name

        all_results_exp = data['all_results_exp'].tolist()
        all_results_th = data['all_results_th'].tolist()

    else:
        all_results_exp = None
        all_results_th = None

        for i, corr in enumerate(corr_list):

            for j, noise in enumerate(noise_list):

                for k, epsilon in enumerate(epsilon_list):
                    if verbose:
                        print(f"corr: {corr:.2f}, eps: {epsilon:.2f}, noise: {noise:.2f}")

                    # Theory Loop
                    for n, alpha in enumerate(alpha_list_th):

                        scores = SVR_th(alpha, epsilon=epsilon, lamb=lamb, corr=corr, noise=noise)

                        if all_results_th is None:
                            all_results_th = {key: np.zeros((len(corr_list),
                                                             len(noise_list),
                                                             len(epsilon_list),
                                                             len(alpha_list_th))) for key in scores.keys()}

                        for key, val in scores.items():
                            all_results_th[key][i, j, k, n] = val

        exp_params['file_name'] = file_name
        np.savez(file_name,
                 exp_params=exp_params,
                 all_results_exp=all_results_exp,
                 all_results_th=all_results_th)

    return exp_params, all_results_exp, all_results_th
