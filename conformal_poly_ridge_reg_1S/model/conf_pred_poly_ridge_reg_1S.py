"""
Full conformal prediction for ridge regression
"""

import numpy as np  # for the math
from scipy import stats


class conformal_ridge_regression_1S:
    def __init__(self, degree):
        self.nb_param = degree
        self.param = None
        self.N = None
        self.kern = None
        self.rss = None
        self.rms = None
        self.X_mean = None
        self.X_std = None

    def fit(self, dataLoader, reg_param=0):
        #
        self.N = dataLoader.get_size()

        inputs = dataLoader.get_input()  # Nx1
        outputs = dataLoader.get_output()  # Nx1

        # Augmentation
        X_aug = np.concatenate(
            tuple(
                [
                    np.power(inputs.reshape(dataLoader.get_size(), 1), i)
                    for i in range(self.nb_param + 1)
                ]
            ),
            axis=1,
        )  # N x (degree + 1)

        # Normalization
        self.X_mean = X_aug[:, 1:].mean(axis=0)
        self.X_std = X_aug[:, 1:].std(axis=0)
        X_aug[:, 1:] = (X_aug[:, 1:] - self.X_mean) / self.X_std

        y = outputs.reshape(dataLoader.get_size(), 1)

        kern = np.matmul(X_aug.transpose(), X_aug)
        diag = np.einsum("ii->i", kern)
        diag += reg_param

        self.kern = kern

        xPy = np.matmul(X_aug.transpose(), y)
        self.param = np.linalg.solve(kern, xPy)

        e = y - X_aug @ self.param
        self.rss = (e.transpose() @ e).flatten()

    def param_p_values(self):
        S = np.linalg.solve(self.kern, np.eye(self.kern.shape[0]))
        p = self.nb_param + 1
        df = self.N - p
        self.rms = self.rss / df

        tS = self.param.flatten() / (np.sqrt(self.rms * np.diagonal(S))).flatten()
        atS = np.abs(tS)

        p_values = 2 * stats.t.sf(atS, df)
        return atS, p_values

    def forward(self, dataLoader):
        inputs = dataLoader.get_input()  # Nx1
        # Augmentation
        X_aug = np.concatenate(
            tuple(
                [
                    np.power(inputs.reshape(dataLoader.get_size(), 1), i)
                    for i in range(self.nb_param + 1)
                ]
            ),
            axis=1,
        )  # N x (degree + 1)

        # Normalization
        X_aug[:, 1:] = (X_aug[:, 1:] - self.X_mean) / self.X_std

        return X_aug @ self.param

    def get_A_and_B(self, dataLoader, K, new_X, reg_param):
        Y0 = np.concatenate(
            (dataLoader.get_output(), np.array([0]).reshape(1, 1))
        )  # (N + 1) x 1

        X = np.concatenate(
            (
                dataLoader.get_input(),
                np.array([new_X]).reshape(1, dataLoader.get_cov_dim()),
            )
        )  # (N + 1) x 1

        # Augmentation
        X_aug = np.concatenate(
            tuple([np.power(X, i) for i in range(self.nb_param + 1)]), axis=1
        )  # (N + 1) x p

        # Normalization
        X_aug[:, 1:] = (X_aug[:, 1:] - self.X_mean) / self.X_std

        kern = np.matmul(X_aug.transpose(), X_aug)  # p x p
        diag = np.einsum("ii->i", kern)
        diag += reg_param

        xPy = np.matmul(X_aug.transpose(), Y0)  # p x 1
        kxpy = np.linalg.solve(kern, xPy)  # p x 1
        xkxpy = np.matmul(X_aug, kxpy)  # (N + 1) x 1
        B = xkxpy - Y0  # (N + 1) x 1

        xPe = X_aug[-1, :].reshape(-1, 1)  # p x 1
        kxpe = np.linalg.solve(kern, xPe)  # p x 1
        xkxpe = np.matmul(X_aug, kxpe)  # (N + 1) x 1

        A = xkxpe  # (N + 1) x 1
        A[-1, :] = A[-1, :] - 1

        return A, B

    def get_intervals(self, dataLoader, K, new_X, reg_param):
        A, B = self.get_A_and_B(dataLoader, K, new_X, reg_param)
        # Coefficients
        Ai = A[:-1, :]
        Bi = B[:-1, :]
        ANp1 = A[-1, :]
        BNp1 = B[-1, :]

        # Bounds
        CiLb = []
        CiUb = []

        # ids
        iDi = np.arange(Ai.shape[0]).reshape(Ai.shape)

        # Case 1 : A_i = A_{N+1} and B_i >= B_{N+1}
        mask1 = np.logical_and(Ai == ANp1, Bi >= BNp1)
        if mask1.any():
            mInf = np.ones(iDi[mask1].shape) * (-np.inf)
            pInf = np.ones(iDi[mask1].shape) * np.inf
            CiLb.append(mInf)
            CiUb.append(pInf)

        # # Case 2 : A_i = A_{N+1} and B_i < B_{N+1}
        # mask2 = np.logical_and(Ai == ANp1, Bi < BNp1)

        # Case 3 : A_i < A_{N+1}
        mask3 = Ai < ANp1
        if mask3.any():
            mInf = np.ones(iDi[mask3].shape) * (-np.inf)
            Ui1 = (Bi[mask3] - BNp1) / (ANp1 - Ai[mask3])
            CiLb.append(mInf)
            CiUb.append(Ui1)

        # Case 4 : A_i > A_{N+1}
        mask4 = Ai > ANp1
        if mask4.any():
            Ui2 = (Bi[mask4] - BNp1) / (ANp1 - Ai[mask4])
            pInf = np.ones(iDi[mask4].shape) * (np.inf)
            CiLb.append(Ui2)
            CiUb.append(pInf)

        lb = np.concatenate(tuple(CiLb))
        ub = np.concatenate(tuple(CiUb))

        return lb, ub

    def get_p_values(self, dataLoader, K, new_X, reg_param):
        lb, ub = self.get_intervals(dataLoader, K, new_X, reg_param)

        ys = np.unique(np.concatenate((lb, ub)))
        ys = np.concatenate(
            (
                np.concatenate((-np.array(np.inf).reshape(-1, 1), ys.reshape(-1, 1))),
                np.array(np.inf).reshape(-1, 1),
            )
        )
        ys = np.unique(ys.flatten()).reshape(-1, 1)

        counts = np.sum(
            np.logical_and(
                lb.reshape(1, lb.shape[0]) <= ys[:-1].reshape(ys.shape[0] - 1, 1),
                ub.reshape(1, lb.shape[0]) >= ys[1:].reshape(ys.shape[0] - 1, 1),
            ),
            axis=1,
        )
        p_values = (counts + 1) / (K + 1)
        return p_values, ys

    def conformal_region(self, dataLoader, K, new_X, reg_param, eps):
        p_values, ys = self.get_p_values(dataLoader, K, new_X, reg_param)
        y_hat_lb = ys[:-1]
        y_hat_ub = ys[1:]
        lb_is_conform = y_hat_lb[p_values > eps]
        ub_is_conform = y_hat_ub[p_values > eps]
        return lb_is_conform, ub_is_conform

    def predictive_region(self, dataLoader_train, K, dataLoader_test, reg_param, eps):
        ub = []
        lw = []
        for new_X in dataLoader_test.get_input():
            l, u = self.conformal_region(dataLoader_train, K, new_X, reg_param, eps)
            lw.append(l.flatten().min())
            ub.append(u.flatten().max())

        lws = np.array(lw)
        ubs = np.array(ub)
        return lws, ubs
