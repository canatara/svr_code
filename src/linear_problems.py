import numpy as np

import jax
import jax.numpy as jnp
from jaxopt import OSQP

# from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning

simplefilter("ignore", category=ConvergenceWarning)
jax.config.update("jax_enable_x64", True)


def linreg1(X_tr, y_tr, X_test, y_test, **kwargs):

    C = y_test.shape[-1]

    assert C == 1, "SVR algorithm does not allow multi-classes!"
    y_tr = y_tr.squeeze()
    y_test = y_test.squeeze()

    # Linear Regression
    linreg = LinearRegression(fit_intercept=False)
    out = linreg.fit(X_tr, y_tr)

    y_tr_pred = out.predict(X_tr)
    y_test_pred = out.predict(X_test)

    w = linreg.coef_
    assert np.allclose(X_test@w.T, y_test_pred)

    tr_err = np.mean((y_tr-y_tr_pred)**2)
    gen_err = np.mean((y_test-y_test_pred)**2)

    return tr_err, gen_err


def linreg2(X_tr, y_tr, X_test, y_test, lamb=1e-15, **kwargs):

    # Linear Regression
    linreg = Ridge(alpha=lamb, fit_intercept=False)
    out = linreg.fit(X_tr, y_tr)

    y_tr_pred = out.predict(X_tr)
    y_test_pred = out.predict(X_test)

    w = linreg.coef_
    assert np.allclose(X_test@w.T, y_test_pred)

    tr_err = np.mean((y_tr-y_tr_pred)**2)
    gen_err = np.mean((y_test-y_test_pred)**2)

    return tr_err, gen_err


def linreg3(X_tr, y_tr, X_test, y_test, lamb=1e-15, **kwargs):

    p_tr, N = X_tr.shape

    if p_tr >= N:  # Overdetermined lstsq (K_tr = feat.T @ feat)
        Id = jnp.eye(N)
        w = jnp.linalg.pinv(X_tr.T@X_tr + lamb*Id) @ X_tr.T @ y_tr
    else:  # p_tr < N  # Underdetermined lstsq (push through identity, K_tr = feat @ feat.T)
        Id = jnp.eye(p_tr)
        w = X_tr.T @ jnp.linalg.pinv(X_tr@X_tr.T + lamb*Id) @ y_tr

    y_tr_pred = X_tr @ w
    y_test_pred = X_test @ w

    tr_err = np.mean((y_tr-y_tr_pred)**2)
    gen_err = np.mean((y_test-y_test_pred)**2)

    return tr_err, gen_err


def linsvr1(X_tr, y_tr, X_test, y_test, epsilon=0, reg=1e+10, **kwargs):

    C = y_test.shape[-1]

    assert C == 1, "SVR algorithm does not allow multi-classes!"
    y_tr = y_tr.squeeze()
    y_test = y_test.squeeze()

    # SVR
    svr = LinearSVR(C=reg,
                    epsilon=epsilon,
                    loss='squared_epsilon_insensitive',
                    max_iter=30000,
                    tol=1e-15,
                    dual=True,
                    fit_intercept=False)
    out = svr.fit(X_tr, y_tr)

    y_tr_pred = out.predict(X_tr)
    y_test_pred = out.predict(X_test)

    w = svr.coef_
    assert np.allclose(X_test@w.T, y_test_pred)

    tr_err = np.mean(np.maximum(np.abs(y_tr-y_tr_pred) - epsilon, 0)**2)
    gen_err = np.mean((y_test-y_test_pred)**2)

    return tr_err, gen_err


def linsvr2(X_tr, y_tr, X_test, y_test, epsilon=0, lamb=1e-10, **kwargs):

    C = y_test.shape[-1]

    assert C == 1, "SVR algorithm does not allow multi-classes!"
    y_tr = y_tr.squeeze()
    y_test = y_test.squeeze()

    # SVR
    svr = SGDRegressor(alpha=lamb,
                       epsilon=epsilon,
                       loss='squared_epsilon_insensitive',
                       #    loss="epsilon_insensitive",
                       max_iter=100000,
                       tol=1e-17,
                       #    learning_rate='optimal',
                       #    early_stopping=True,
                       fit_intercept=False)
    out = svr.fit(X_tr, y_tr)

    y_tr_pred = out.predict(X_tr)
    y_test_pred = out.predict(X_test)

    w = svr.coef_.squeeze()
    assert np.allclose(X_test@w.T, y_test_pred)

    tr_err = np.mean(np.maximum(np.abs(y_tr-y_tr_pred) - epsilon, 0)**2)
    gen_err = np.mean((y_test-y_test_pred)**2)

    return tr_err, gen_err


def linsvr_jax(X_tr, y_tr, X_test, y_test, epsilon=0, reg=1e+10, **kwargs):

    C = y_test.shape[-1]

    assert C == 1, "SVR algorithm does not allow multi-classes!"
    y_tr = y_tr.squeeze()
    y_test = y_test.squeeze()

    # SVR
    svr = SVR_JAX(C=reg,
                  epsilon=epsilon,
                  fit_intercept=False)
    out = svr.fit(X_tr, y_tr)

    y_tr_pred = out.predict(X_tr)
    y_test_pred = out.predict(X_test)

    w = svr.coef_
    assert np.allclose(X_test@w.T, y_test_pred)

    tr_err = np.mean(np.maximum(np.abs(y_tr-y_tr_pred) - epsilon, 0)**2)
    gen_err = np.mean((y_test-y_test_pred)**2)

    return tr_err, gen_err


class SVR_JAX:

    def __init__(self,
                 C,
                 epsilon,
                 loss='squared_epsilon_insensitive',
                 max_iter=5000,
                 tol=1e-3,
                 fit_intercept=False):
        """Solves the following problem:
        min 1/2 ||w||^2 s.t.
        +Xw <= y + epsilon
        -Xw <= y - epsilon"""

        self.epsilon = epsilon
        self.max_iter = max_iter
        self.loss = loss
        self.fit_intercept = fit_intercept

        # qp = OSQP(sigma=1/C, tol=tol, maxiter=max_iter, jit=True)
        # self.qp_run = qp.run

        qp = OSQP(sigma=1/C, tol=tol, maxiter=max_iter, momentum=1.8, jit=True)
        self.qp_run = jax.jit(qp.run)

        # qp = CvxpyQP()
        # self.qp_run = qp.run

    def fit(self, X, y):

        # Solve the problem
        # 1/2*w.T@Q@w + c@w s.t. G@w <= h and A@w = b

        P, N = X.shape

        assert y.shape[0] == P

        # Constraint matrix: 2P constraints
        G = jnp.concatenate((X, -X), axis=0)
        h = jnp.concatenate((y+self.epsilon, -y+self.epsilon), axis=0).squeeze()

        Q = jnp.eye(N)
        c = jnp.zeros(N)

        A = None
        b = None

        params_obj = (Q, c)
        params_ineq = (G, h)
        params_eq = (A, b)

        if A is None or b is None:
            params_eq = None

        qp_kwargs = dict(params_obj=params_obj,
                         params_eq=params_eq,
                         params_ineq=params_ineq,
                         #  init_params=jnp.ones(N)/N,
                         )

        self.sol = self.qp_run(**qp_kwargs)
        self.coef_ = self.sol.params.primal

        return self

    def predict(self, X):

        w = self.coef_

        return X@w.T


# def linsvr3(X_tr, y_tr, X_test, y_test, epsilon=0, reg=1e+10, **kwargs):

#     C = y_test.shape[-1]

#     assert C == 1, "SVR algorithm does not allow multi-classes!"
#     y_tr = y_tr.squeeze()
#     y_test = y_test.squeeze()

#     # SVR
#     svr = SVR(kernel='linear',
#               C=reg,
#               epsilon=epsilon,
#               max_iter=10000,
#               tol=1e-15,
#               shrinking=True)
#     out = svr.fit(X_tr, y_tr)

#     y_tr_pred = out.predict(X_tr)
#     y_test_pred = out.predict(X_test)

#     w = svr.coef_.squeeze()
#     # b = svr.intercept_

#     assert np.allclose(X_test@w.T, y_test_pred)

#     tr_err = np.mean(np.maximum(np.abs(y_tr-y_tr_pred) - epsilon, 0)**2)
#     gen_err = np.mean((y_test-y_test_pred)**2)

#     return tr_err, gen_err


# def linsvr4(X_tr, y_tr, X_test, y_test, epsilon=0, reg=1e+10, **kwargs):

#     C = y_test.shape[-1]

#     assert C == 1, "SVR algorithm does not allow multi-classes!"
#     y_tr = y_tr.squeeze()
#     y_test = y_test.squeeze()

#     K_tr = X_tr@X_tr.T
#     K_test = X_test@X_tr.T

#     # SVR
#     svr = SVR(kernel='precomputed',
#               C=reg,
#               epsilon=epsilon,
#               max_iter=10000,
#               tol=1e-15,
#               shrinking=False)
#     out = svr.fit(K_tr, y_tr)

#     y_tr_pred = out.predict(K_tr)
#     y_test_pred = out.predict(K_test)

#     tr_err = np.mean(np.maximum(np.abs(y_tr-y_tr_pred) - epsilon, 0)**2)
#     gen_err = np.mean((y_test-y_test_pred)**2)

#     return tr_err, gen_err
