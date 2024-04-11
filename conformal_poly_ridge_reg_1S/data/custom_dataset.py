# Test your class
import numpy as np

# np.random.seed(42)


class test_sample:
    def __init__(self, N, x_min, x_max, u_N, sigma0, noise_a):
        self.N = N
        self.x_min = x_min
        self.x_max = x_max
        self.u_N = u_N
        self.sigma0 = sigma0
        self.noise_a = noise_a

    def get_noise_profile(self):
        u_Xs = np.random.uniform(self.x_min, self.x_max, size=self.u_N)
        invXs = np.random.randint(0, self.u_N, self.N).flatten()
        Xs = u_Xs[invXs]
        b = -self.noise_a * np.max(u_Xs) + self.sigma0
        u_Noise = self.noise_a * u_Xs + b
        noises = u_Noise[invXs]
        eps = np.random.normal(0, noises)

        return Xs, eps

    def get_dataset(self, func, *args):
        Xs, eps = self.get_noise_profile()
        return Xs, func(Xs, *args) + eps
