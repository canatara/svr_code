import jax
import jax.numpy as jnp
import numpy as np

from jax.numpy import trace
from jax.numpy.linalg import pinv
from jax.scipy.special import erfc, erfinv

from .manifolds import DataManifold
from .solvers import best_solver
from collections import defaultdict

jax.config.update("jax_enable_x64", True)

pinv = jax.jit(pinv)
trace = jax.jit(trace)
erfc = jax.jit(erfc)
erfinv = jax.jit(erfinv)
erfcinv = jax.jit(lambda x: erfinv(jnp.float64(1-x)))


@jax.jit
def G_matrix_fn(inputs, *args):

    (p, epsilon, lamb, Σ_ψ, Σ_ψ_ybar, E_0) = args
    kappa, eps = inputs
    kappa = jnp.abs(kappa)

    N = len(Σ_ψ)
    alpha = p/N

    # Get F1 and F2
    f = erfc(eps/jnp.sqrt(2))
    df = -jnp.sqrt(2/jnp.pi) * jnp.exp(-eps**2/2)
    g = 1 + eps**2 + eps*df/f
    # h = eps + df/f - f/df

    # Effective sample size
    alpha_eff = alpha*f

    # Define matrices
    Id = jnp.eye(N)
    G = Id + alpha_eff/kappa * N*Σ_ψ

    # Compute inverses
    Ginv = pinv(G)
    Σ_ψ_inv = pinv(Σ_ψ)

    # Matrix M
    M = Σ_ψ@Ginv

    # gamma and delta
    gamma = alpha_eff/kappa**2 * N*trace(M@M)
    delta = 1/kappa * trace(M@Ginv)

    # Weights
    w_eff = Σ_ψ_inv@Σ_ψ_ybar

    # E_inf
    E_inf = E_0 - trace(w_eff@w_eff.T@Σ_ψ)

    # W_tilde
    W_tilde = E_inf + trace(w_eff@w_eff.T@M@Ginv)
    C_self = W_tilde / (1 - g*gamma)

    kappa_self = lamb + trace(M)
    eps_self = epsilon * jnp.sqrt(1 - g*gamma) / jnp.sqrt(W_tilde)

    kappa_self = jnp.abs(kappa_self)

    return kappa_self, eps_self, C_self, M, Ginv, alpha_eff, f, g, gamma, delta, E_inf, W_tilde, w_eff


@jax.jit
def kappa_eps_fn(inputs, *args):

    kappa, eps = inputs
    kappa = jnp.abs(kappa)
    # eps = eps * (eps > 0)

    (kappa_self, eps_self, C_self, M, Ginv, alpha_eff, f, g,
     gamma, delta, E_inf, W_tilde, w_eff) = G_matrix_fn(inputs, *args)

    # Self-consistent equations
    kappa_eq = (kappa - kappa_self)
    eps_eq = (eps - eps_self)

    return jnp.array([kappa_eq, eps_eq])


@jax.jit
def kappa_eps_fn_prime(inputs, *args):

    kappa, eps = inputs
    kappa = jnp.abs(kappa)
    jax_jacobian = jax.jit(jax.jacfwd(kappa_eps_fn, argnums=0))
    jac = jax_jacobian(jnp.array([kappa, eps]), *args)
    return jac

    (p, epsilon, lamb, Σ_ψ, Σ_ψ_ybar, E_0) = args

    kappa, eps = inputs
    kappa = jnp.abs(kappa)

    (kappa_self, eps_self, C_self, M, Ginv, alpha_eff, f, g,
     gamma, delta, E_inf, W_tilde, w_eff) = G_matrix_fn(inputs, *args)

    # Get F1 and F2
    f = erfc(eps/jnp.sqrt(2))
    df = -jnp.sqrt(2/jnp.pi) * jnp.exp(-eps**2/2)
    g = 1 + eps**2 + eps*df/f
    h = eps + df/f - f/df

    # Optimal Epsilon and Kappa
    kappa_opt = trace(M@M@Ginv)*(C_self*g)/trace(M@M@Ginv@w_eff@w_eff.T)
    A = (kappa_opt - kappa)
    A = A*(A > 0)
    A = A * trace(M@M@Ginv@w_eff@w_eff.T) / trace(M@M) / (C_self)

    dFdκ = gamma
    dFdε = -df/f*kappa*gamma

    dGdκ = eps/kappa * gamma/(1-g*gamma) * A
    dGdε = (df*eps/f) * gamma/(1-g*gamma) * (h*eps-A)

    jac = [[1-dFdκ, -dFdε],
           [-dGdκ, 1-dGdε]]
    jac = jnp.array(jac)

    return jac


@jax.jit
def G_matrix_fn_eps_opt(inputs, *args):

    (p, epsilon, lamb, Σ_ψ, Σ_ψ_ybar, E_0) = args
    kappa, eps = inputs
    kappa = jnp.abs(kappa)

    N = len(Σ_ψ)
    alpha = p/N

    # Get F1 and F2
    f = erfc(eps/jnp.sqrt(2))
    df = -jnp.sqrt(2/jnp.pi) * jnp.exp(-eps**2/2)
    g = 1 + eps**2 + eps*df/f
    h = eps + df/f - f/df

    # Effective sample size
    alpha_eff = alpha*f

    # Define matrices
    Id = jnp.eye(N)
    G = Id + alpha_eff/kappa * N*Σ_ψ

    # Compute inverses
    Ginv = pinv(G)
    Σ_ψ_inv = pinv(Σ_ψ)

    # Matrix M
    M = Σ_ψ@Ginv

    # gamma and delta
    gamma = alpha_eff/kappa**2 * N*trace(M@M)
    delta = 1/kappa * trace(M@Ginv)

    # Weights
    w_eff = Σ_ψ_inv@Σ_ψ_ybar

    # E_inf
    E_inf = E_0 - trace(w_eff@w_eff.T@Σ_ψ)

    # W_tilde
    W_tilde = E_inf + trace(w_eff@w_eff.T@M@Ginv)
    C_self = W_tilde / (1 - g*gamma)

    kappa_self = lamb + trace(M)

    # Optimal Epsilon and Kappa
    kappa_opt = trace(M@M@Ginv)*(C_self*g)/trace(M@M@Ginv@w_eff@w_eff.T)
    A = (kappa_opt - kappa)
    A = A*(A > 0)
    A = A * trace(M@M@Ginv@w_eff@w_eff.T) / trace(M@M) / (C_self)
    eps_opt = A/h/(delta+lamb/kappa)

    kappa_self = kappa_self
    eps_self = eps_opt

    kappa_self = jnp.abs(kappa_self)

    return kappa_self, eps_self, C_self, M, Ginv, alpha_eff, f, g, gamma, delta, E_inf, W_tilde, w_eff


@jax.jit
def kappa_eps_fn_eps_opt(inputs, *args):

    kappa, eps = inputs
    kappa = jnp.abs(kappa)

    (kappa_self, eps_self, C_self, M, Ginv, alpha_eff, f, g,
     gamma, delta, E_inf, W_tilde, w_eff) = G_matrix_fn_eps_opt(inputs, *args)

    # Self-consistent equations
    kappa_eq = (kappa - kappa_self)
    eps_eq = (eps - eps_self)

    return jnp.array([kappa_eq, eps_eq])


@jax.jit
def kappa_eps_fn_eps_opt_prime(inputs, *args):

    # kappa, eps = inputs
    # kappa = jnp.abs(kappa)
    # jax_jacobian = jax.jit(jax.jacfwd(kappa_eps_fn_eps_opt, argnums=0))
    # jac = jax_jacobian(jnp.array([kappa, eps]), *args)
    # return jac

    (p, epsilon, lamb, Σ_ψ, Σ_ψ_ybar, E_0) = args

    kappa, eps = inputs
    kappa = jnp.abs(kappa)

    (kappa_self, eps_self, C_self, M, Ginv, alpha_eff, f, g,
     gamma, delta, E_inf, W_tilde, w_eff) = G_matrix_fn_eps_opt(inputs, *args)

    # Get F1 and F2
    f = erfc(eps/jnp.sqrt(2))
    df = -jnp.sqrt(2/jnp.pi) * jnp.exp(-eps**2/2)
    g = 1 + eps**2 + eps*df/f
    h = eps + df/f - f/df

    # Optimal Epsilon and Kappa
    kappa_opt = trace(M@M@Ginv)*(C_self*g)/trace(M@M@Ginv@w_eff@w_eff.T)
    A = (kappa_opt - kappa)
    A = A*(A > 0)
    A = A * trace(M@M@Ginv@w_eff@w_eff.T) / trace(M@M) / (C_self)

    dFdκ = gamma
    dFdε = -df/f*kappa*gamma

    dGdκ = eps/kappa * gamma/(1-g*gamma) * A
    dGdε = (df*eps/f) * gamma/(1-g*gamma) * (h*eps-A)

    # dGdκ = eps/kappa * gamma/(1-g*gamma) * (1-gamma) * h * eps
    # dGdε = (df*eps/f) * gamma/(1-g*gamma) * gamma * h * eps  # * ((h*eps-A/(1-gamma)) < 0)

    jac = [[1-dFdκ, -dFdε],
           [-dGdκ, 1-dGdε]]
    jac = jnp.array(jac)

    return jac


@jax.jit
def G_matrix_fn_lamb_opt(inputs, *args):

    (p, epsilon, lamb, Σ_ψ, Σ_ψ_ybar, E_0) = args
    kappa, eps = inputs
    kappa = jnp.abs(kappa)

    N = len(Σ_ψ)
    alpha = p/N

    f = 1.
    g = 1.

    # Effective sample size
    alpha_eff = alpha*f

    # Define matrices
    Id = jnp.eye(N)
    G = Id + alpha_eff/kappa * N*Σ_ψ

    # Compute inverses
    Ginv = pinv(G)
    Σ_ψ_inv = pinv(Σ_ψ)

    # Matrix M
    M = Σ_ψ@Ginv

    # gamma and delta
    gamma = alpha_eff/kappa**2 * N*trace(M@M)
    delta = 1/kappa * trace(M@Ginv)

    # Weights
    w_eff = Σ_ψ_inv@Σ_ψ_ybar

    # E_inf
    E_inf = E_0 - trace(w_eff@w_eff.T@Σ_ψ)

    # W_tilde
    W_tilde = E_inf + trace(w_eff@w_eff.T@M@Ginv)
    C_self = W_tilde / (1 - g*gamma)

    # Optimal Epsilon and Kappa
    kappa_opt = trace(M@M@Ginv)*(C_self*g)/trace(M@M@Ginv@w_eff@w_eff.T)

    kappa_self = kappa_opt
    eps_self = 0.

    kappa_self = jnp.abs(kappa_self)

    return kappa_self, eps_self, C_self, M, Ginv, alpha_eff, f, g, gamma, delta, E_inf, W_tilde, w_eff


@jax.jit
def kappa_eps_fn_lamb_opt(inputs, *args):

    kappa, eps = inputs
    kappa = jnp.abs(kappa)
    # eps = eps * (eps > 0)

    (kappa_self, eps_self, C_self, M, Ginv, alpha_eff, f, g,
     gamma, delta, E_inf, W_tilde, w_eff) = G_matrix_fn_lamb_opt(inputs, *args)

    # Self-consistent equations
    kappa_eq = (kappa - kappa_self)
    eps_eq = (eps - eps_self)

    return jnp.array([kappa_eq, eps_eq])


@jax.jit
def kappa_eps_fn_lamb_opt_prime(inputs, *args):

    # jax_jacobian = jax.jit(jax.jacfwd(kappa_eps_fn_lamb_opt, argnums=0))
    # kappa, eps = inputs
    # kappa = jnp.abs(kappa)
    # jac = jax_jacobian(jnp.array([kappa, eps]), *args)

    # jac = jac.at[0, 1].set(0)
    # jac = jac.at[1, 0].set(0)
    # jac = jac.at[1, 1].set(0)

    # return jac

    (p, epsilon, lamb, Σ_ψ, Σ_ψ_ybar, E_0) = args

    kappa, eps = inputs
    kappa = jnp.abs(kappa)

    (kappa_self, eps_self, C_self, M, Ginv, alpha_eff, f, g,
     gamma, delta, E_inf, W_tilde, w_eff) = G_matrix_fn_lamb_opt(inputs, *args)

    # Optimal Epsilon and Kappa
    # kappa_opt = trace(M@M@Ginv)*(C_self*g)/trace(M@M@Ginv@w_eff@w_eff.T)
    # A = (kappa_opt - kappa)
    # A = A*(A > 0)
    # A = A * trace(M@M@Ginv@w_eff@w_eff.T) / trace(M@M) / (C_self)

    dFdκ = gamma

    jac = [[1-dFdκ, 0],
           [0, 1]]
    jac = jnp.array(jac)

    return jac


def get_logZ_gamma(inputs, *args):

    (p, epsilon, lamb, Σ_ψ, Σ_ψ_ybar, E_0) = args
    kappa, eps = inputs
    kappa = jnp.abs(kappa)
    # eps = jnp.abs(eps)

    (kappa_self, eps_self, C_self, M, Ginv, alpha_eff, f, g,
     gamma, delta, E_inf, W_tilde, w_eff) = G_matrix_fn(inputs, *args)

    # eps = eps_self
    # kappa = kappa_self

    alpha = alpha_eff/f
    p = alpha*len(Σ_ψ)

    C = C_self
    Q = kappa - lamb
    Qhat = alpha_eff / kappa
    Chat = -alpha_eff/kappa**2 * C*g

    # G_0
    G_0 = Qhat*(E_inf + trace(w_eff@w_eff.T@M) - C)

    # G_1
    G_1 = Qhat*C*g

    # logZ
    logZ = G_0 + G_1

    # Get F1 and F2
    f = erfc(eps/jnp.sqrt(2))
    df = -jnp.sqrt(2/jnp.pi) * jnp.exp(-eps**2/2)
    g = 1 + eps**2 + eps*df/f
    h = eps + df/f - f/df

    kappa_opt = trace(M@M@Ginv)*(C_self*g)/trace(M@M@Ginv@w_eff@w_eff.T)
    A = (kappa_opt - kappa)
    # A = A*(A > 0)
    A = A * trace(M@M@Ginv@w_eff@w_eff.T) / trace(M@M) / (C_self)
    eps_opt = A/h/(delta+lamb/kappa)

    epsilon_th = eps * jnp.sqrt(C)
    lamb_eff = kappa - trace(M)

    # Training error
    Etr = lamb_eff**2/kappa**2 * C*f*g
    Etr_raw = lamb_eff**2/kappa**2 * C

    if False:
        try:
            assert jnp.allclose(eps, eps_self), \
                f"α {alpha:.1f} and ε {epsilon:.1f} | ε_eff mismatch: {eps:.2e}, {eps_self:.2e}"
        except Exception as e:
            print(e)

        try:
            assert jnp.allclose(kappa, kappa_self), \
                f'α {alpha:.1f} and ε {epsilon:.1f} | κ mismatch: {kappa:.2e}, {kappa_self:.2e}'
        except Exception as e:
            print(e)

        try:
            assert jnp.allclose(delta+lamb_eff/kappa, 1-gamma), \
                f'α {alpha:.1f} and ε {epsilon:.1f} | γ mismatch: {1-gamma:.2e}, {delta+lamb_eff/kappa:.2e}'
        except Exception as e:
            print(e)

    return_dict = dict(f=f,
                       g=g,
                       h=h,
                       alpha_eff=alpha_eff,
                       alpha=alpha,
                       p=p,
                       p_eff=alpha_eff*len(Σ_ψ),
                       epsilon=epsilon,
                       epsilon_th=epsilon_th,
                       lamb_eff=lamb_eff,
                       lamb=lamb,
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
                       Etr_raw=Etr_raw,
                       A=A,
                       eps_opt=eps_opt,
                       kappa_opt=kappa_opt,
                       #    w_mean_sq=w_mean_sq,
                       #    w_sq_mean=w_sq_mean,
                       #    overlap=overlap,
                       eps_self=eps_self,
                       kappa_self=kappa_self,
                       )

    return return_dict


def solve_kappa_gamma(pvals, epsilon, ridge, Σ_ψ, Σ_ψ_ybar, E_0, debug=True):

    if isinstance(ridge, float) or isinstance(ridge, int):
        ridge = np.array([ridge]*len(pvals))
    assert len(ridge) == len(pvals)

    if epsilon == -1:
        fun, fprime = kappa_eps_fn_eps_opt, kappa_eps_fn_eps_opt_prime
    elif epsilon == -2:
        fun, fprime = kappa_eps_fn_lamb_opt, kappa_eps_fn_lamb_opt_prime
    else:
        fun, fprime = kappa_eps_fn, kappa_eps_fn_prime
        # fun, fprime = kappa_eps_fn, kappa_eps_fn_prime
        # fprime = kappa_eps_jac

    returns_dict = defaultdict(list)
    returns_dict["pvals_th"] = list(pvals)
    returns_dict["lamb_th"] = list(ridge)

    N = len(Σ_ψ)
    # eff_dim = trace(Σ_ψ)**2/trace(Σ_ψ@Σ_ψ)

    Σ_ψ_inv = pinv(Σ_ψ)
    w_eff = Σ_ψ_inv@Σ_ψ_ybar
    E_inf = E_0 - trace(w_eff@w_eff.T@Σ_ψ)  # Irreducible Error
    E_1 = E_0 - E_inf  # Reducible error

    eps_0 = epsilon / jnp.sqrt(E_0)
    kappa_opt_0 = E_0*trace(Σ_ψ@Σ_ψ)/trace(Σ_ψ_ybar.T@Σ_ψ_ybar)

    for i, (p, lamb) in enumerate(zip(pvals, ridge)):

        kappa_0 = lamb + trace(Σ_ψ)

        if epsilon == -1:
            alpha = jnp.float64(p/N) * E_1
            eps_0 = -jnp.sqrt(2)
            if alpha >= 1.0:
                eps_0 = erfcinv(1/alpha)*jnp.sqrt(2)
                eps_0 = -eps_0
        elif epsilon == -2:
            eps_0 = 0.
            kappa_0 = kappa_opt_0

        x0 = jnp.float64([kappa_0, eps_0])  # When p = 0
        args = (p, epsilon, lamb, Σ_ψ, Σ_ψ_ybar, E_0)

        (kappa, eps, sol_stat,
         sol_kappa_res, sol_eps_res) = best_solver(fun, fprime, args, x0)

        # (kappa, eps, sol_stat,
        #  sol_kappa_res, sol_eps_res) = solvers(fun, fprime, args, x0, optimizer='scipy_root')

        returns = get_logZ_gamma([kappa, eps], *args)

        if debug:
            sol_fun = np.array([sol_kappa_res, sol_eps_res])
            if epsilon != -1 and epsilon != -2 and not np.allclose(returns['epsilon_th'], epsilon):
                print(f'{p/len(Σ_ψ)}, {returns["epsilon_th"]}, {epsilon}')
            if sum(np.abs(sol_fun) > 1e-5):
                print(f'α {p/len(Σ_ψ):.1f} and ε {epsilon:.1f} |', sol_fun)
            # if sol_stat not in [1, 3, 5]:
            #     print(sol.message)
            pass

        returns_dict['sol_stat'] += [sol_stat]
        returns_dict['sol_kappa_res'] += [sol_kappa_res]
        returns_dict['sol_eps_res'] += [sol_eps_res]
        for key, val in returns.items():
            if isinstance(val, jax.numpy.ndarray):
                val = val.item()
            elif isinstance(val, float):
                val = val
            else:
                raise Exception('The return elements must be single numbers')

            returns_dict[key] += [val]

    for key, val in returns_dict.items():
        returns_dict[key] = np.array(val)

    return returns_dict


def SVR_th(alpha, epsilon=0, lamb=1e-10, manifold=DataManifold, **kwargs):

    returns = {}

    p_tr = alpha*manifold.N
    Σ_ψ, Σ_ψ_ybar, E_0 = manifold.get_data_statistics()
    returns |= solve_kappa_gamma([p_tr], epsilon, lamb, Σ_ψ, Σ_ψ_ybar, E_0, **kwargs)

    return returns