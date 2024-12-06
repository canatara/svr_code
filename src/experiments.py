import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

from jax.numpy import trace
from jax.numpy.linalg import pinv
from jax.scipy.special import erfc, erfinv

from .manifolds import (GaussianToyManifoldStructered, DataManifold, StickImages, GratingImages, MNISTDigits)
from . import linear_problems
from .model_utils import get_model_activations
from .theory_utils import SVR_th

jax.config.update("jax_enable_x64", True)

pinv = jax.jit(pinv)
trace = jax.jit(trace)
erfc = jax.jit(erfc)
erfinv = jax.jit(erfinv)
erfcinv = jax.jit(lambda x: erfinv(jnp.float64(1-x)))


class GratingManifold(DataManifold):

    def __init__(self, model, layer, sample_size=100, max_img_num=50, rand_proj_dim=100, random_seed=42):

        grating_manifold = GratingImages(max_img_num=max_img_num, random_seed=random_seed)
        angles, manifold = grating_manifold.create_grating_manifold(sample_size=sample_size)

        if layer is None:
            act = manifold.cpu().numpy().astype(np.float64)
            act = act.reshape(act.shape[0], act.shape[1], -1)
        else:
            act = get_model_activations(model, layer, manifold, rand_proj_dim=rand_proj_dim)

        angles = angles.cpu().numpy().astype(np.float64)

        # print(act.shape, angles.shape)

        super().__init__(act, angles)


class StickManifold(DataManifold):

    def __init__(self, model, layer, sample_size=100, max_img_num=50, rand_proj_dim=100, random_seed=42):

        stick_manifold = StickImages(max_img_num=max_img_num, random_seed=random_seed)
        angles, manifold = stick_manifold.create_stick_manifold(sample_size=sample_size)

        if layer is None:
            act = manifold.cpu().numpy().astype(np.float64)
            act = act.reshape(act.shape[0], act.shape[1], -1)
        else:
            act = get_model_activations(model, layer, manifold, rand_proj_dim=rand_proj_dim)

        angles = angles.cpu().numpy().astype(np.float64)

        # print(act.shape, angles.shape)

        super().__init__(act, angles)


class DigitManifold(DataManifold):

    def __init__(self, model, layer, digit=5, sample_size=100, max_img_num=50, rand_proj_dim=100, random_seed=42):

        digit_manifold = MNISTDigits(max_img_num=max_img_num, random_seed=random_seed)
        angles, manifold = digit_manifold.create_digit_manifold(digit=digit, sample_size=sample_size)

        if layer is None:
            act = manifold.cpu().numpy().astype(np.float64)
            act = act.reshape(act.shape[0], act.shape[1], -1)
        else:
            act = get_model_activations(model, layer, manifold, rand_proj_dim=rand_proj_dim)

        angles = angles.cpu().numpy().astype(np.float64)

        print(act.shape, angles.shape)

        super().__init__(act, angles)


class GaussianManifold(DataManifold):

    def __init__(self, P, N, corr=0.1, noise=0.1, centroid_seed=125):

        manifold = GaussianToyManifoldStructered(P, N, corr, noise, num_classes=1, centroid_seed=centroid_seed)
        _, _, act, angles = manifold.get_manifold(1)

        act = act[:, None].astype(np.float64)
        angles = angles.squeeze().astype(np.float64)

        # print(act.shape, angles.shape)

        super().__init__(act, angles)


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
            # print('lamb,eps,alpha', kwargs['lamb'], epsilon, alpha)

        elif "linreg" in fn.__name__:
            kwargs['lamb'] = lamb

        else:
            raise Exception(f'Specify the regularization for this algorithm: {fn.__name__}')

        returns |= linear_problem(fn, manifold, **kwargs)

    return returns


def run_experiment(manifold,
                   SVR_th=SVR_th,
                   SVR_exp=SVR_exp,
                   grid_size=3,
                   num_trials=5,
                   fn_list=['linsvr_jax'],
                   epsilon_list=None,
                   alpha_list=None,
                   suffix="",
                   **kwargs):

    if epsilon_list is None:
        epsilon_list = np.append([-1, -2], np.linspace(0., 0.8, grid_size))
    else:
        epsilon_list = np.array(epsilon_list)

    if alpha_list is None:
        alpha_list = np.arange(0.1, 3, 0.1)
    else:
        alpha_list = np.array(alpha_list)

    alpha_list_th = np.linspace(alpha_list[0], alpha_list[-1], 100)
    lamb = 1e-18
    fn_list = fn_list

    fns = [getattr(linear_problems, fn) for fn in fn_list]

    Σ_ψ, Σ_ψ_ybar, E_0 = manifold.get_data_statistics()
    Σ_ψ_inv = pinv(Σ_ψ)
    N = len(Σ_ψ)
    print('manifold dim', trace(Σ_ψ)**2/trace(Σ_ψ@Σ_ψ))
    print('signal to noise', trace(Σ_ψ_inv@Σ_ψ_ybar@Σ_ψ_ybar.T) / trace(Σ_ψ) * N)

    all_results_th = None
    all_results_exp = None
    for k, epsilon in enumerate(epsilon_list):
        # Theory Loop
        for n, alpha in enumerate(alpha_list_th):
            scores = SVR_th(alpha, epsilon=epsilon, lamb=lamb, manifold=manifold, **kwargs)

            if all_results_th is None:
                all_results_th = {key: np.zeros((len(epsilon_list),
                                                 len(alpha_list_th))) for key in scores.keys()}

            for key, val in scores.items():
                all_results_th[key][k, n] = val

        # Experiment Loop
        for n, alpha in enumerate(alpha_list):

            if epsilon == -1 or epsilon == -2:
                scores_th = SVR_th(alpha, epsilon=epsilon, lamb=lamb, manifold=manifold, **kwargs)
                epsilon_th = scores_th['epsilon_th'][0]
                lamb_eff = scores_th['lamb_eff'][0]

                epsilon_th = epsilon_th * (epsilon_th > 0)
                epsilon_th = np.nan_to_num(epsilon_th, 0)
                lamb_eff = np.clip(lamb_eff, 1e-15, None)

            else:
                epsilon_th = epsilon
                lamb_eff = lamb

            scores = SVR_exp(fns, alpha, manifold, epsilon_th, lamb=lamb_eff, num_trials=num_trials)

            if all_results_exp is None:
                all_results_exp = {key: np.zeros((len(epsilon_list),
                                                  len(alpha_list))) for key in scores.keys()}

            for key, val in scores.items():
                all_results_exp[key][k, n] = val

    def correct_dict(all_results_th):

        import copy
        corrected_th = copy.deepcopy(all_results_th)

        C = all_results_th['C'][0]
        Etr = all_results_th['Etr'][0]

        idx_C = np.abs(np.diff(C, prepend=0)) > 1e-1
        idx_Etr = Etr > 1e-5

        for i in range(len(C)):
            if (idx_C[i] or idx_Etr[i]) and i > 3:
                for key in corrected_th.keys():
                    corrected_th[key][0][i] = corrected_th[key][0][i-1]

        return corrected_th
    all_results_th = correct_dict(all_results_th)

    results = dict(all_results_exp=all_results_exp,
                   all_results_th=all_results_th,
                   alpha_list=alpha_list,
                   alpha_list_th=alpha_list_th,
                   epsilon_list=epsilon_list,
                   grid_size=grid_size,
                   num_trials=num_trials,
                   lamb=lamb,
                   fn_list=fn_list)

    return results


def experiment_toy(P=800,
                   N=25,
                   centroid_seed=42,
                   grid_size=4,
                   num_trials=20,
                   overwrite=False,
                   SVR_th=SVR_th,
                   fn_list=[],
                   plot_fig=False,
                   corr_list=None,
                   noise_list=None,
                   epsilon_list=None,
                   alpha_list=None):

    import os
    import os.path

    if corr_list is None:
        corr_list = np.linspace(0.1, 1, grid_size)
    if noise_list is None:
        noise_list = np.linspace(0.5, 1, grid_size)
    if epsilon_list is None:
        epsilon_list = np.array([-1, -2, 0.5, 0])
    if alpha_list is None:
        alpha_list = np.arange(0.1, 5, 0.3)

    for corr in corr_list:
        for noise in noise_list:

            filename = f'corr_{corr:.3f}_noise_{noise:.3f}_P_{P}_N_{N}'\
                f'_grid_{grid_size}_centroid_seed_{centroid_seed}.npz'
            if SVR_th.__name__ == 'SVR_th':
                filename = 'cov_th_' + filename
            if os.path.exists('./ceph/temp/'+filename) and not overwrite:
                returns = np.load('./ceph/temp/'+filename, allow_pickle=True)['returns'].tolist()

            else:
                manifold = GaussianManifold(P, N, corr=corr, noise=noise, centroid_seed=centroid_seed)
                returns = run_experiment(manifold, grid_size=grid_size, num_trials=num_trials,
                                         SVR_th=SVR_th,
                                         fn_list=fn_list,
                                         epsilon_list=epsilon_list,
                                         alpha_list=alpha_list,
                                         debug=False)

                np.savez('./ceph/temp/'+filename, returns=returns)

            if plot_fig:

                font_axis_label = 16
                alpha_list = returns['alpha_list']
                alpha_list_th = returns['alpha_list_th']
                epsilon_list = returns['epsilon_list']
                grid_size = returns['grid_size']
                num_trials = returns['num_trials']
                # lamb = returns['lamb']
                num_trials = returns['num_trials']
                fn_list = returns['fn_list']

                all_results_exp = returns['all_results_exp']
                all_results_th = returns['all_results_th']

                C = all_results_th['C']
                gen_err = all_results_exp[f'gen_err_{fn_list[0]}'] if len(fn_list) != 0 else C
                alpha_list = alpha_list if len(fn_list) != 0 else alpha_list_th
                C = np.clip(C, a_min=None, a_max=3)
                gen_err = np.clip(gen_err, a_min=None, a_max=3)
                plt.figure(figsize=(4, 3))
                for i, eps in enumerate(epsilon_list):
                    if eps == -1:
                        label = r'$\varepsilon_{opt}$'
                    elif eps == -2:
                        label = r'$\lambda_{opt}$'
                    else:
                        label = fr'$\varepsilon = {eps}$'
                    plt.loglog(alpha_list, gen_err[i], '.', color=f'C{i}', label=label)
                    plt.loglog(alpha_list_th, C[i], color=f'C{i}')
                plt.axhline(all_results_th['E_inf'][0, 0], color='k', linestyle='--')
                # plt.ylim([1e-2, 1e1])
                plt.xlabel(r'$\alpha$', fontsize=font_axis_label)
                plt.ylabel(r'$E_g$', fontsize=font_axis_label)
                plt.legend()

                print(filename)
                if filename == 'corr_0.700_noise_1.000_P_800_N_25_grid_4_centroid_seed_42.npz':
                    plt.tight_layout()
                    plt.savefig('optimal_learning_curves.pdf')

                plt.show()

    data = dict()
    for corr in corr_list:

        if data.get(corr) is None:
            data[corr] = dict()

        for noise in noise_list:

            filename = f'corr_{corr:.3f}_noise_{noise:.3f}_P_{P}_N_{N}'\
                f'_grid_{grid_size}_centroid_seed_{centroid_seed}.npz'
            if SVR_th.__name__ == 'SVR_th':
                filename = 'cov_th_' + filename

            if os.path.exists('./ceph/temp/'+filename):
                print(filename)
                returns = np.load('./ceph/temp/'+filename, allow_pickle=True)['returns'].tolist()
                all_results_th = returns.pop('all_results_th')
                all_results_exp = returns.pop('all_results_exp')

                data[corr][noise] = all_results_th | returns

            else:
                print('./ceph/temp/'+filename, + ' does not exist')
                pass

    alldata_file = f'all_data_toy_new_P_{P}_N_{N}'\
        f'_grid_{grid_size}_centroid_seed_{centroid_seed}.npz'
    if SVR_th.__name__ == 'SVR_th':
        alldata_file = 'cov_th_' + alldata_file

    np.savez('./results/' + alldata_file, data=data)

    return data


def experiment_grating(sample_size=100,
                       max_img_num=100,
                       rand_proj_dim=50,
                       random_seed=42,
                       SVR_th=SVR_th,
                       plot_fig=True):

    import os

    model_list = ['resnet18', 'resnet34', 'resnet50']
    trained_list = ['random', 'trained']
    layer_list = ['conv1', 'layer1', 'layer2',  'layer3', 'layer4']

    filename = f'grating_manifold_data_P_{sample_size}_M_{max_img_num}_N_{rand_proj_dim}_seed_{random_seed}.npz'
    if SVR_th.__name__ == 'SVR_th':
        filename = 'cov_th_' + filename
    filedir = './ceph/temp/'+filename

    if os.path.isfile(filedir):
        data = np.load(filedir, allow_pickle=True)['data'].tolist()
    else:
        os.makedirs(os.path.dirname(filedir), exist_ok=True)

        print("computing", filedir)
        data = dict()
        for trained in trained_list:
            if data.get(trained) is None:
                data[trained] = dict()
            pretrained = False if trained == 'random' else True

            for model_name in model_list:
                if data[trained].get(model_name) is None:
                    data[trained][model_name] = dict()

                import timm
                import torch

                if torch.cuda.is_available():
                    device = torch.device("cuda")
                elif torch.backends.mps.is_available():
                    device = torch.device("mps")
                else:
                    device = torch.device("cpu")

                model = timm.create_model(model_name,
                                          pretrained=pretrained,
                                          in_chans=1).to(device)

                for layer in layer_list:

                    print(model_name, trained, layer)
                    manifold = GratingManifold(model, layer,
                                               sample_size=sample_size,
                                               max_img_num=max_img_num,
                                               rand_proj_dim=rand_proj_dim,
                                               random_seed=random_seed)

                    returns = run_experiment(manifold,
                                             SVR_th=SVR_th,
                                             fn_list=[],
                                             epsilon_list=np.array([-1, -2, 0.5, 0]),
                                             alpha_list=np.arange(0.1, 3, 0.1),
                                             debug=False)

                    data[trained][model_name][layer] = returns

                    if plot_fig:

                        alpha_list = returns['alpha_list']
                        alpha_list_th = returns['alpha_list_th']
                        epsilon_list = returns['epsilon_list']
                        fn_list = returns['fn_list']

                        all_results_exp = returns['all_results_exp']
                        all_results_th = returns['all_results_th']

                        C = all_results_th['C']
                        gen_err = all_results_exp[f'gen_err_{fn_list[0]}'] if len(fn_list) != 0 else C
                        alpha_list = alpha_list if len(fn_list) != 0 else alpha_list_th
                        C = np.clip(C, a_min=None, a_max=3)
                        gen_err = np.clip(gen_err, a_min=None, a_max=3)
                        plt.figure(figsize=(4, 3))
                        for i, eps in enumerate(epsilon_list):
                            if eps == -1:
                                label = r'$\varepsilon_{opt}$'
                            elif eps == -2:
                                label = r'$\lambda_{opt}$'
                            else:
                                label = fr'$\varepsilon = {eps}$'
                            plt.loglog(alpha_list, gen_err[i], '.', color=f'C{i}', label=label)
                            plt.loglog(alpha_list_th, C[i], color=f'C{i}')
                        plt.axhline(all_results_th['E_inf'][0, 0], color='k', linestyle='--')
                        plt.xlabel(r'$\alpha$')
                        plt.ylabel(r'$E_g$')
                        plt.legend()

                        plt.show()

        np.savez(filedir,
                 data=data,
                 model_list=model_list,
                 trained_list=trained_list,
                 layer_list=layer_list,
                 )

    return data
