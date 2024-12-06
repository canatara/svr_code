import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision import datasets, transforms

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


class GaussianToyManifold:
    """
    Gaussian Manifold model with:
    - ψbar ~ N(0, I/N)
    - δ ~ N(0, I/N)
    - ψ ~ sqrt(1-c) ψbar + sqrt(c) δ
    """

    def __init__(self, P, N, corr=0, num_classes=1, m_tr=1, centroid_seed=125, **kwargs):

        key = jax.random.PRNGKey(centroid_seed)
        key_psi, key_delta, key_wbar = jax.random.split(key, 3)

        self.P = P
        self.N = N
        self.C = num_classes
        self.c = corr
        self.m_tr = m_tr

        # Target
        self.wbar = jax.random.normal(key_wbar, shape=(N, 1))
        self.wbar /= jnp.sqrt(jnp.trace(self.wbar.T@self.wbar) / N)
        assert jnp.allclose(jnp.trace(self.wbar.T@self.wbar), N)

        self.Σ_ψbar = jnp.eye(self.N) / N
        self.Σ_δ = jnp.eye(self.N) / N

        # Centroid
        self.ψbar = jax.random.multivariate_normal(key_psi, jnp.zeros(N), self.Σ_ψbar, shape=(P,))

        # Noise
        self.δ = jax.random.multivariate_normal(key_delta, jnp.zeros(N), self.Σ_δ, shape=(P,))

        # Signal
        self.ψ = jnp.sqrt(1-self.c) * self.ψbar + jnp.sqrt(self.c) * self.δ

    def get_data_statistics(self):

        Σ_ψbar = self.Σ_ψbar
        Σ_δ = self.Σ_δ
        Σ_ψ = (1-self.c) * Σ_ψbar + self.c * Σ_δ
        Σ_ψ_ψbar = jnp.sqrt(1-self.c) * Σ_ψbar

        return (Σ_ψ, Σ_ψbar, Σ_ψ_ψbar, self.wbar)

    def get_manifold(self, p_tr, m_tr=1, manifold_seed=0):

        # Centroid and noise
        ψbar = self.ψbar
        δ = self.δ

        # Signal and target
        ψ = jnp.sqrt(1-self.c)*ψbar + jnp.sqrt(self.c)*δ
        wbar = self.wbar

        X = ψ
        y = jnp.einsum('ik,kl->il', ψbar, wbar)

        np.random.seed(manifold_seed)
        idx_centroid = np.random.choice(range(self.P), size=p_tr, replace=False)

        X_tr = X[idx_centroid].reshape(-1, self.N)
        y_tr = y[idx_centroid].reshape(-1, self.C)

        X_test = X.reshape(-1, self.N)
        y_test = y.reshape(-1, self.C)

        assert len(X_tr) == p_tr*m_tr

        return (X_tr, y_tr, X_test, y_test)

    # def update_corr(self, corr, **kwargs):
    #     self.c = corr
    #     self.ψ = jnp.sqrt(1-self.c)*self.ψbar + jnp.sqrt(self.c)*self.δ

    def update_corr(self, **kwargs):

        if kwargs.get('corr'):
            self.c = kwargs.get('corr')
            self.ψ = jnp.sqrt(1-self.c)*self.ψbar + jnp.sqrt(self.c)*self.δ


class GaussianToyManifoldStructered:
    """
    Gaussian Manifold model with:
    - ψbar ~ N(0, I/N)
    - δ ~ N(0, I/N)
    - ψ ~ sqrt(1-c) ψbar + sqrt(c) δ
    """

    def __init__(self, P, N, corr=0, noise=0, num_classes=1, m_tr=1, centroid_seed=125, **kwargs):

        key = jax.random.PRNGKey(centroid_seed)
        key_psi, key_delta, key_wbar = jax.random.split(key, 3)

        self.P = P
        self.N = N
        self.C = num_classes
        self.c = np.clip(corr, 1e-5, 1-1e-5)
        self.σ = noise
        self.m_tr = m_tr
        self.centroid_seed = centroid_seed

        # Target
        self.wbar = jax.random.normal(key_wbar, shape=(N, 1))
        self.wbar /= jnp.sqrt(jnp.trace(self.wbar.T@self.wbar) / N)
        self.W = self.wbar@self.wbar.T
        assert jnp.allclose(jnp.trace(self.W), N)

        # Projection
        self.Id = jnp.eye(N)
        self.Proj = self.Id - self.W/N

        self.Σ_ψbar = jnp.eye(N)
        self.Σ_δ = (1-self.c)*self.Proj + self.c*(self.Id - self.Proj)

        # Centroid
        self.ψbar = jax.random.multivariate_normal(key_psi, jnp.zeros(self.N), self.Σ_ψbar / N, shape=(self.P,))

        # Noise
        self.δ = jax.random.multivariate_normal(key_delta, jnp.zeros(self.N), self.Σ_δ / N, shape=(self.P,))

        # Signal
        self.ψ = self.ψbar + self.σ*self.δ

    def get_data_statistics(self):

        Σ_ψbar = self.Σ_ψbar / self.N
        Σ_δ = self.Σ_δ / self.N
        Σ_ψ = Σ_ψbar + self.σ**2 * Σ_δ
        Σ_ψ_ψbar = Σ_ψbar

        return (Σ_ψ, Σ_ψbar, Σ_ψ_ψbar, self.wbar)

    def update_corr(self, **kwargs):

        if kwargs.get('corr'):

            self.c = np.clip(kwargs.get('corr'), 1e-5, 1-1e-5)
            self.Σ_δ = (1-self.c)*self.Proj + self.c*(self.Id - self.Proj)

            key = jax.random.PRNGKey(self.centroid_seed)
            _, key_delta, _ = jax.random.split(key, 3)
            self.δ = jax.random.multivariate_normal(key_delta, jnp.zeros(self.N), self.Σ_δ/self.N, shape=(self.P,))

        if kwargs.get('noise'):
            self.σ = kwargs.get('noise')
            # Signal
            self.ψ = self.ψbar + self.σ*self.δ

    def get_manifold(self, p_tr, m_tr=1, manifold_seed=0):

        # Centroid and noise
        ψbar = self.ψbar
        δ = self.δ

        # Signal and target
        ψ = ψbar + self.σ*δ
        wbar = self.wbar

        X = ψ
        y = jnp.einsum('ik,kl->il', ψbar, wbar)

        np.random.seed(manifold_seed)
        idx_centroid = np.random.choice(range(self.P), size=p_tr, replace=False)

        X_tr = X[idx_centroid].reshape(-1, self.N)
        y_tr = y[idx_centroid].reshape(-1, self.C)

        X_test = X.reshape(-1, self.N)
        y_test = y.reshape(-1, self.C)

        assert len(X_tr) == p_tr*m_tr

        return (X_tr, y_tr, X_test, y_test)


class GaussianToyManifoldEmpirical:
    """
    Gaussian Manifold model with:
    - ψbar ~ N(0, I/N)
    - δ ~ N(0, I/N)
    - ψ ~ sqrt(1-c) ψbar + sqrt(c) δ
    """

    def __init__(self, P, N, corr=0, noise=0, num_classes=1, m_tr=1, centroid_seed=125, **kwargs):

        key = jax.random.PRNGKey(centroid_seed)
        key_psi, key_delta, key_wbar = jax.random.split(key, 3)

        self.P = P
        self.N = N
        self.C = num_classes
        self.c = np.clip(corr, 1e-5, 1-1e-5)
        self.σ = noise
        self.m_tr = m_tr
        self.centroid_seed = centroid_seed

        # Target
        self.wbar = jax.random.normal(key_wbar, shape=(N, 1))
        self.wbar /= jnp.sqrt(jnp.trace(self.wbar.T@self.wbar) / N)
        self.W = self.wbar@self.wbar.T
        assert jnp.allclose(jnp.trace(self.W), N)

        # Projection
        self.Id = jnp.eye(N)
        self.Proj = self.Id - self.W/N

        self.Σ_ψbar = jnp.eye(N)
        self.Σ_δ = (1-self.c)*self.Proj + self.c*(self.Id - self.Proj)

        # Centroid
        self.ψbar = jax.random.multivariate_normal(key_psi, jnp.zeros(self.N), self.Σ_ψbar / N, shape=(self.P,))

        # Noise
        self.δ = jax.random.multivariate_normal(key_delta, jnp.zeros(self.N), self.Σ_δ / N, shape=(self.P,))

        # Signal
        self.ψ = self.ψbar + self.σ*self.δ

    def get_data_statistics(self):

        # Σ_ψbar = self.Σ_ψbar / self.N
        # Σ_δ = self.Σ_δ / self.N
        # Σ_ψ = Σ_ψbar + self.σ**2 * Σ_δ
        # Σ_ψ_ψbar = Σ_ψbar

        Σ_ψbar = self.ψbar.T@self.ψbar / self.P
        # Σ_δ = self.δ.T@self.δ / self.P
        Σ_ψ = self.ψ.T@self.ψ / self.P
        Σ_ψ_ψbar = self.ψ.T@self.ψbar / self.P

        return (Σ_ψ, Σ_ψbar, Σ_ψ_ψbar, self.wbar)

    def update_corr(self, **kwargs):

        if kwargs.get('corr'):

            self.c = np.clip(kwargs.get('corr'), 1e-5, 1-1e-5)
            self.Σ_δ = (1-self.c)*self.Proj + self.c*(self.Id - self.Proj)

            key = jax.random.PRNGKey(self.centroid_seed)
            _, key_delta, _ = jax.random.split(key, 3)
            self.δ = jax.random.multivariate_normal(key_delta, jnp.zeros(self.N), self.Σ_δ/self.N, shape=(self.P,))

        if kwargs.get('noise'):
            self.σ = kwargs.get('noise')
            # Signal
            self.ψ = self.ψbar + self.σ*self.δ

    def get_manifold(self, p_tr, m_tr=1, manifold_seed=0):

        # Centroid and noise
        ψbar = self.ψbar
        δ = self.δ

        # Signal and target
        ψ = ψbar + self.σ*δ
        wbar = self.wbar

        X = ψ
        y = jnp.einsum('ik,kl->il', ψbar, wbar)

        np.random.seed(manifold_seed)
        idx_centroid = np.random.choice(range(self.P), size=p_tr, replace=False)

        X_tr = X[idx_centroid].reshape(-1, self.N)
        y_tr = y[idx_centroid].reshape(-1, self.C)

        X_test = X.reshape(-1, self.N)
        y_test = y.reshape(-1, self.C)

        assert len(X_tr) == p_tr*m_tr

        return (X_tr, y_tr, X_test, y_test)


class DataManifold:
    """
    Gaussian Manifold model with:
    - ψbar ~ N(0, I/N)
    - δ ~ N(0, I/N)
    - ψ ~ sqrt(1-c) ψbar + sqrt(c) δ
    """

    def __init__(self, ψ, y, **kwargs):

        assert len(ψ.shape) == 3, 'The representations must have shape (latent_dim, variability_dim, N)'
        assert len(y.shape) == 1, 'The latents must be scalar'
        assert len(y) == ψ.shape[0]

        P, M, N = ψ.shape
        C = 1

        ψbar = ψ.mean(1, keepdims=True)
        δ = ψ - ψbar

        ψbar -= ψbar.mean(0, keepdims=True)

        ψbar_std = ψbar.std(0, keepdims=True)
        # δ_std = δ.std(1, keepdims=True)

        # δ /= np.sqrt(δ_std/M)

        ψ = ψbar + δ
        ψ /= ψbar_std

        # ψ -= ψbar.mean(0, keepdims=True)
        # ψ /= ψbar.std(0, keepdims=True)

        # ψ -= ψ.mean(0, keepdims=True)
        # ψ /= ψ.std(0, keepdims=True)

        ψ = ψ.reshape(P*M, N)
        # ψ -= ψ.mean(0, keepdims=True)
        # ψ /= ψ.std(0, keepdims=True)
        ψ /= np.sqrt(np.trace(ψ.T@ψ / (P*M))/N)
        ψ = ψ.reshape(P, M, N)

        y -= y.mean(0, keepdims=True)
        y /= np.sqrt(y.T@y/P)

        assert np.allclose(y.T@y/P, 1)

        self.P = P
        self.M = M
        self.N = N
        self.C = C

        self.ψ = ψ
        self.ybar = np.repeat(y[:, None], M, axis=1).reshape(P, M, C)

        self.Σ_ψ = jnp.einsum("ijk,ijl->kl", self.ψ, self.ψ) / (P*M)
        self.Σ_ψ_ybar = jnp.einsum("ijk,ijl->kl", self.ψ, self.ybar) / (P*M)
        self.E_0 = jnp.einsum("ijk,ijk", self.ybar, self.ybar) / (P*M)

        self.eigs = None
        self.weights = None

    def get_data_statistics(self):

        Σ_ψ = self.Σ_ψ
        Σ_ψ += np.eye(self.N) / (self.P*self.M)
        Σ_ψ_ybar = self.Σ_ψ_ybar
        E_0 = self.E_0

        return (Σ_ψ, Σ_ψ_ybar, E_0)

    def get_data_spectrum(self, recompute=False):

        if (self.eigs) is None or (self.weights is None) or recompute:
            P, M, N = self.ψ.shape

            ψ = self.ψ.reshape(P*M, N)
            ybar = self.ybar.reshape(P*M)

            eigs, vecs = jnp.linalg.eigh(ψ@ψ.T / (P*M))
            eigs = eigs[::-1]
            vecs = vecs[:, ::-1] * np.sqrt(P*M)
            weights = vecs.T @ ybar / (P*M)

            eigs *= (eigs > 1e-15)

            self.eigs = eigs
            self.weights = weights

            del vecs, eigs, weights

        return (self.eigs, self.weights, self.E_0)

    def get_data_spectrum_torch(self, recompute=False):

        if (self.eigs) is None or (self.weights is None) or recompute:

            P, M, N = self.ψ.shape

            ψ = self.ψ.reshape(P*M, N)
            ybar = self.ybar.reshape(P*M)

            eigs, weights = get_eig_weight_qr(ψ, ybar)

            eigs *= (eigs > 1e-15)

            self.eigs = eigs
            self.weights = weights

        return (self.eigs, self.weights, self.E_0)

    def get_manifold(self, p_tr, m_tr=1, manifold_seed=0):

        np.random.seed(manifold_seed)
        idx_centroid = np.random.choice(range(self.P), size=p_tr, replace=False)
        idx_manifold = np.random.choice(range(self.M), size=p_tr, replace=p_tr > self.M)

        X = self.ψ
        y = self.ybar

        X_tr = X[idx_centroid, idx_manifold].reshape(-1, self.N)
        y_tr = y[idx_centroid, idx_manifold].reshape(-1, self.C)

        X_test = X.reshape(-1, self.N)
        y_test = y.reshape(-1, self.C)

        assert len(X_tr) == p_tr*m_tr

        return (X_tr, y_tr, X_test, y_test)

    def update_corr(self):
        pass


def get_eig_weight_eigh(ψ, ybar):

    import torch

    ψ = torch.from_numpy(ψ).type(torch.cuda.DoubleTensor)
    ybar = torch.from_numpy(ybar).type(torch.cuda.DoubleTensor)
    P, N = ψ.shape

    G = ψ@ψ.T

    eigs, vecs = torch.linalg.eigh(G / P)
    # assert torch.allclose(G / P, vecs@torch.diag(eigs)@vecs.T)
    # print(eigs.shape, vecs.shape)

    eigs = torch.flip(eigs, dims=[0])
    vecs = torch.flip(vecs, dims=[1]) * np.sqrt(P)
    weights = vecs.T @ ybar / P

    return eigs.cpu().numpy(), weights.cpu().numpy()


def get_eig_weight_qr(ψ, ybar):

    import torch

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device("cpu")

    ψ = np.array(ψ)
    ybar = np.array(ybar)

    ψ = torch.from_numpy(ψ).to(device).type(torch.float64)
    ybar = torch.from_numpy(ybar).to(device).type(torch.float64)
    P, N = ψ.shape

    Q, R = torch.linalg.qr(ψ)
    # G = ψ@ψ.T
    # Id = torch.eye(len(Q.T)).type(torch.cuda.DoubleTensor)
    # assert torch.allclose(ψ, Q@R)
    # assert torch.allclose(G, Q@R@R.T@Q.T)
    # assert torch.allclose(Q.T@Q, Id)
    # print(Q.shape, R.shape)

    eigs, vecs = torch.linalg.eigh(R @ R.T / P)
    eigs = torch.flip(eigs, dims=[0])
    vecs = torch.flip(vecs, dims=[1])
    vecs = Q @ vecs * np.sqrt(P)
    weights = vecs.T @ ybar / P

    eig = eigs.cpu().numpy()
    weight = weights.cpu().numpy()
    ybar = ybar.cpu().numpy()
    E_inf = ybar@ybar/P-weight@weight

    if P > N:
        eig = np.append(eig, np.ones(P-N)*1e-19)
        weight = np.append(weight, np.ones(P-N)*np.sqrt(E_inf/(P-N)))

    del vecs, eigs, weights

    return eig, weight


class StickImages:

    def __init__(self, max_img_num=500, max_sample_size=500, image_size=(64, 64), random_seed=12):

        self.max_sample_size = max_sample_size
        self.max_img_num = max_img_num
        self.random_seed = random_seed
        self.image_size = image_size

        import os
        import pickle
        ds_path = f"{os.getcwd()}/datasets/sticks_{max_sample_size}_angles_{image_size}.pkl"

        if os.path.exists(ds_path):
            dataset = pickle.load(open(ds_path, 'rb'))
            angles, manifold = dataset['angles'], dataset['manifold']
        else:
            print(f'Generating images at {ds_path}')
            angles = np.linspace(0, 1, max_sample_size)*180
            manifold = []
            for angle in angles:
                manifold.append(self.generate_augmented_stick_images(angle))
            manifold = np.array(manifold)
            dataset = {'angles': angles, 'manifold': manifold}
            pickle.dump(dataset, open(ds_path, 'wb'), protocol=4)

        np.random.seed(random_seed)
        idx = np.random.choice(range(manifold.shape[1]), size=max_img_num, replace=False)
        manifold = manifold[:, idx]

        self.angles = angles
        self.manifold = manifold

    def create_stick_manifold(self, sample_size=100, if_torch=True):

        stride = self.max_sample_size//sample_size

        angles = self.angles[::stride]
        manifold = self.manifold[::stride]

        if if_torch:
            angles = torch.Tensor(angles)
            manifold = torch.Tensor(manifold)

        return angles, manifold

    def generate_stick_image(self, angle, stick_length=20, thickness=2, translate=[0, 0]):

        import cv2
        from torchvision import transforms

        # Create a blank white canvas of size 32x32
        image = np.zeros(self.image_size, dtype=np.uint8) * 255

        # The center of the image
        center_x = self.image_size[0] // 2
        center_y = self.image_size[1] // 2

        radian_angle = np.deg2rad(angle)

        # Calculate the endpoints of the stick from the center based on the angle
        half_length = stick_length // 2
        end_x1 = int(center_x + half_length * np.cos(radian_angle))
        end_y1 = int(center_y + half_length * np.sin(radian_angle))
        end_x2 = int(center_x - half_length * np.cos(radian_angle))
        end_y2 = int(center_y - half_length * np.sin(radian_angle))

        stick_color = 255  # 0 for black, 255 for white

        # Draw the stick (line) on the canvas
        cv2.line(image, (end_x1, end_y1), (end_x2, end_y2), color=stick_color, thickness=thickness)

        image = transforms.functional.to_pil_image(image)
        image = transforms.functional.pil_to_tensor(image)
        image = transforms.functional.affine(image, angle=0, translate=translate, scale=1, shear=0)
        # image = transforms.functional.to_grayscale(num_output_channels=3)

        return image

    def generate_augmented_stick_images(self, angle):

        thickness_list = [1, 2, 3]
        stick_length_list = [10, 15, 20, 25]
        trans_list = list(range(-5, 5))

        images = []
        for thickness in thickness_list:
            for stick_length in stick_length_list:
                for tx in trans_list:
                    for ty in trans_list:
                        image = self.generate_stick_image(angle,
                                                          stick_length=stick_length,
                                                          thickness=thickness,
                                                          translate=[tx, ty])
                        images.append(image)

        return np.array(images)

    def sample_digits(self):

        fig, axs = plt.subplots(2, 5, figsize=(5, 2))
        angles, manifold = self.create_stick_manifold(sample_size=10)

        for i in range(10):
            ax = axs[int(i > 4), i % 5]
            ax.imshow(manifold[i, np.random.randint(self.max_img_num), 0], cmap='gray_r')
            ax.set_axis_off()
        plt.show()

    def sample_manifold(self, digit=5, sample_size=100):

        angles, manifold = self.create_stick_manifold(sample_size=sample_size)

        fig, axs = plt.subplots(2, 5, figsize=(5, 2))
        for i in range(10):
            ax = axs[int(i > 4), i % 5]
            ax.imshow(manifold[i*sample_size//10, -1, 0], cmap='gray_r')
            ax.set_axis_off()
        plt.show()


class GratingImages:

    def __init__(self, max_img_num=500, max_sample_size=500, image_size=(64, 64), random_seed=12):

        self.max_sample_size = max_sample_size
        self.max_img_num = max_img_num
        self.random_seed = random_seed
        self.image_size = image_size

        import os
        import pickle
        ds_path = f"{os.getcwd()}/datasets/gratings_{max_sample_size}_angles_{image_size}.pkl"

        if os.path.exists(ds_path):
            dataset = pickle.load(open(ds_path, 'rb'))
            angles, manifold = dataset['angles'], dataset['manifold']
        else:
            print(f'Generating images at {ds_path}')
            angles = np.linspace(0, 1, max_sample_size)*180
            manifold = self.generate_augmented_grating_images(angles)
            manifold = np.array(manifold).swapaxes(0, 1)
            dataset = {'angles': angles, 'manifold': manifold}

            os.makedirs(os.path.dirname(ds_path), exist_ok=True)
            pickle.dump(dataset, open(ds_path, 'wb'), protocol=4)

        print(manifold.shape)

        np.random.seed(random_seed)
        idx = np.random.choice(range(manifold.shape[1]), size=max_img_num, replace=False)
        manifold = manifold[:, idx]

        self.angles = angles
        self.manifold = manifold

    def create_grating_manifold(self, sample_size=100, if_torch=True):

        stride = self.max_sample_size//sample_size

        angles = self.angles[::stride]
        manifold = self.manifold[::stride]

        if if_torch:
            angles = torch.Tensor(angles)
            manifold = torch.Tensor(manifold)

        return angles, manifold

    def create_grating(self, angle, sf, phase, wave):
        import math
        import numpy as np
        import scipy.signal as signal

        """
        :param angle: wave orientation (in degrees, [0-360])
        :param sf: spatial frequency (in pixels)
        :param phase: wave phase (in degrees, [0-360])
        :param wave: type of wave ('sqr' or 'sin')
        :return: numpy array of shape (imsize, imsize)
        """
        # Get x and y coordinates
        imsize_x, imsize_y = self.image_size
        x, y = np.meshgrid(np.arange(imsize_x), np.arange(imsize_y))

        if isinstance(angle, int) or isinstance(angle, float):
            angle = [angle]
        angle = np.array(angle)
        assert len(angle.shape) == 1

        angle = np.expand_dims(angle, axis=(1, 2))
        x = np.expand_dims(x, axis=0)
        y = np.expand_dims(y, axis=0)

        # Get the appropriate gradient
        gradient = np.sin(angle*math.pi/180)*x - np.cos(angle*math.pi/180)*y

        # Plug gradient into wave function
        inp = (2*math.pi*gradient)/sf + (phase*math.pi)/180
        if wave == 'sin':
            grating = np.sin(inp)
        elif wave == 'sqr':
            grating = signal.square(inp)
        elif wave == 'saw':
            grating = signal.sawtooth(inp)
        else:
            raise NotImplementedError
            # grating = signal.gausspulse(inp, fc=5, retquad=True, retenv=True)[1]
            # grating = signal.chirp(inp, f0=6, f1=6, t1=100, method='linear')

        grating = np.expand_dims(grating, axis=1)
        return grating

    def generate_augmented_grating_images(self, angle):

        thickness_list = np.linspace(1, 10, 20)*7
        phase_list = np.linspace(0, 1, 20)*180
        wave_list = ['sin', 'sqr', 'saw']

        images = []
        for thickness in thickness_list:
            for phase in phase_list:
                for wave in wave_list:
                    image = self.create_grating(angle,
                                                sf=thickness,
                                                phase=phase,
                                                wave=wave)
                    images.append(image)

        return np.array(images)

    def sample_digits(self):

        fig, axs = plt.subplots(2, 5, figsize=(5, 2))
        angles, manifold = self.create_grating_manifold(sample_size=10)
        print(manifold.shape)

        for i in range(10):
            ax = axs[int(i > 4), i % 5]
            ax.imshow(manifold[i, np.random.randint(self.max_img_num), 0], cmap='gray_r')
            ax.set_axis_off()
        plt.show()

    def sample_manifold(self, digit=5, sample_size=100):

        angles, manifold = self.create_grating_manifold(sample_size=sample_size)

        fig, axs = plt.subplots(2, 5, figsize=(5, 2))
        for i in range(10):
            ax = axs[int(i > 4), i % 5]
            ax.imshow(manifold[i*sample_size//10, -1, 0], cmap='gray_r')
            ax.set_axis_off()
        plt.show()


class MNISTDigits:

    def __init__(self, max_img_num=500, random_seed=12):

        self.max_img_num = max_img_num
        self.dataset_path = './datasets'

        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))
                                        ])

        self.trainset = datasets.MNIST(self.dataset_path, download=True, train=True, transform=transform)
        self.valset = datasets.MNIST(self.dataset_path, download=True, train=False, transform=transform)

        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=64, shuffle=True)
        self.valloader = torch.utils.data.DataLoader(self.valset, batch_size=64, shuffle=True)

        self.imgs = self.trainset.data
        self.labels = self.trainset.targets

        if len(self.imgs.shape) == 3:
            self.imgs = self.imgs.expand(1, *self.imgs.shape).swapaxes(0, 1)

        self.sorted_imgs = dict()
        for i in range(10):
            digit_imgs = self.imgs[self.labels == i]
            np.random.seed(random_seed*(i+1) + 6)
            idx = np.random.choice(range(len(digit_imgs)), size=max_img_num, replace=False)
            self.sorted_imgs[i] = digit_imgs[idx].to(torch.float)

    def create_digit_manifold(self, digit, sample_size=100, if_torch=True):

        imgs = self.sorted_imgs[digit]
        imgs = transforms.functional.normalize(imgs, (0.1307,), (0.3081,))

        angles = np.linspace(0, 1, sample_size)

        manifold = []
        for angle in angles:
            manifold += [transforms.functional.rotate(imgs, angle=angle*180)]
        manifold = np.array(manifold)

        if if_torch:
            angles = torch.Tensor(angles)
            manifold = torch.Tensor(manifold)

        return angles, manifold

    def sample_digits(self):

        fig, axs = plt.subplots(2, 5, figsize=(5, 2))
        for i in range(10):
            ax = axs[int(i > 4), i % 5]

            angles, manifold = self.create_digit_manifold(digit=i, sample_size=2)
            ax.imshow(manifold[0, np.random.randint(self.max_img_num), 0], cmap='gray_r')
            ax.set_axis_off()
        plt.show()

    def sample_manifold(self, digit=5, sample_size=100):

        angles, manifold = self.create_digit_manifold(digit=digit, sample_size=sample_size)

        fig, axs = plt.subplots(2, 5, figsize=(5, 2))
        for i in range(10):
            ax = axs[int(i > 4), i % 5]
            ax.imshow(manifold[i*sample_size//10, -1, 0], cmap='gray_r')
            ax.set_axis_off()
        plt.show()
