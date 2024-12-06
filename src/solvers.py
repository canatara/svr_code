from scipy import optimize

import jax
import jax.numpy as jnp


def solvers(fun, fprime, args, x0, optimizer='scipy_root'):

    x0 = jnp.float64(x0)  # When p = 0
    args = tuple(jnp.float64(arg) for arg in args)

    if optimizer == 'scipy_root':

        sol = optimize.root(fun=fun,
                            jac=fprime,
                            args=args,
                            x0=x0,
                            method='hybr',
                            tol=1e-18,
                            options={'xtol': 1e-18,
                                     'maxfev': 50000})

        kappa, eps = sol.x
        sol_stat = sol.status
        sol_kappa_res = sol.fun[0]
        sol_eps_res = sol.fun[1]

    elif optimizer == 'scipy_least_squares':

        sol = optimize.least_squares(fun=fun,
                                     #  jac=fprime,
                                     jac='3-point',
                                     # jac='2-point',
                                         args=args,
                                         x0=x0,
                                         method='dogbox',
                                         ftol=1e-14, xtol=1e-14, gtol=1e-14,
                                         #  loss='soft_l1',
                                     )

        kappa, eps = sol.x
        sol_stat = sol.status
        sol_kappa_res = sol.fun[0]
        sol_eps_res = sol.fun[1]

    elif optimizer == 'jax':

        import optimistix as optx
        rtol = 1e-11
        atol = 1e-11
        solver = optx.Newton(rtol=rtol, atol=atol, kappa=1e-6)
        # solver = optx.Dogleg(rtol=rtol, atol=atol)
        # solver = optx.LevenbergMarquardt(rtol=rtol, atol=atol)
        # options = dict(lower=jnp.array([lamb, 0.]), upper=jnp.array([kappa_opt_0, jnp.inf]))

        @jax.jit
        def fn(inp, args):
            return fun(inp, *args)
        # sol = optx.root_find(fn, solver, x0, args=args,
        #                      max_steps=500,
        #                      throw=False,
        #                      options=options,
        #                      )
        sol = optx.least_squares(fn, solver, x0, args=args,
                                 max_steps=1000,
                                 throw=False,
                                 #  options=options,
                                 )
        kappa, eps = sol.value
        sol_stat = 5 if optx.RESULTS[sol.result] else 1
        sol_kappa_res = sol.state.f[0]
        sol_eps_res = sol.state.f[1]

    return kappa, eps, sol_stat, sol_kappa_res, sol_eps_res


def best_solver(fun, fprime, args, x0):

    optimizers = iter(['scipy_root',
                       'scipy_least_squares',
                       'jax',
                       ])

    for optimizer in optimizers:

        (kappa, eps, sol_stat,
         sol_kappa_res, sol_eps_res) = solvers(fun, fprime, args, x0, optimizer)

        if sol_stat in [1, 3]:
            return (kappa, eps, sol_stat, sol_kappa_res, sol_eps_res)

    return (jnp.nan, jnp.nan, sol_stat, sol_kappa_res, sol_eps_res)
