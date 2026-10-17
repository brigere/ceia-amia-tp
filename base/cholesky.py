import numpy as np
import numpy.linalg as LA
from scipy.linalg import cholesky, solve_triangular
from scipy.linalg.lapack import dtrtri

from base.bayesian import BaseBayesianClassifier


class QDA_Chol1(BaseBayesianClassifier):
    def _fit_params(self, X, y):
        self.L_invs = [
            LA.inv(cholesky(np.cov(X[:, y.flatten() == idx], bias=True), lower=True))
            for idx in range(len(self.log_a_priori))
        ]

        self.means = [
            X[:, y.flatten() == idx].mean(axis=1, keepdims=True)
            for idx in range(len(self.log_a_priori))
        ]

    def _predict_log_conditional(self, x, class_idx):
        L_inv = self.L_invs[class_idx]
        unbiased_x = x - self.means[class_idx]

        y = L_inv @ unbiased_x

        return np.log(L_inv.diagonal().prod()) - 0.5 * (y**2).sum()


class QDA_Chol2(BaseBayesianClassifier):
    def _fit_params(self, X, y):
        self.Ls = [
            cholesky(np.cov(X[:, y.flatten() == idx], bias=True), lower=True)
            for idx in range(len(self.log_a_priori))
        ]

        self.means = [
            X[:, y.flatten() == idx].mean(axis=1, keepdims=True)
            for idx in range(len(self.log_a_priori))
        ]

    def _predict_log_conditional(self, x, class_idx):
        L = self.Ls[class_idx]
        unbiased_x = x - self.means[class_idx]

        y = solve_triangular(L, unbiased_x, lower=True)

        return -np.log(L.diagonal().prod()) - 0.5 * (y**2).sum()


class QDA_Chol3(BaseBayesianClassifier):
    def _fit_params(self, X, y):
        self.L_invs = [
            dtrtri(
                cholesky(np.cov(X[:, y.flatten() == idx], bias=True), lower=True),
                lower=1,
            )[0]
            for idx in range(len(self.log_a_priori))
        ]

        self.means = [
            X[:, y.flatten() == idx].mean(axis=1, keepdims=True)
            for idx in range(len(self.log_a_priori))
        ]

    def _predict_log_conditional(self, x, class_idx):
        L_inv = self.L_invs[class_idx]
        unbiased_x = x - self.means[class_idx]

        y = L_inv @ unbiased_x

        return np.log(L_inv.diagonal().prod()) - 0.5 * (y**2).sum()


# se decide heredar de QDA_Chol 3 porque es el que esta mas optimizado para
# el calculo de la prediccion
# la implementacino sigue una mecanica similar a TensorizedQDA
class TensorizedChol(QDA_Chol3):
    def _fit_params(self, X, y):
        super()._fit_params(X, y)
        self.tensor_L_inv = np.stack(self.L_invs)
        self.tensor_means = np.stack(self.means)
        self.log_dets = np.log(np.diagonal(self.tensor_L_inv, axis1=1, axis2=2)).sum(
            axis=1
        )

    def _predict_log_conditionals(self, x):
        unbiased_x = x - self.tensor_means
        y = self.tensor_L_inv @ unbiased_x
        return self.log_dets - 0.5 * (y**2).sum(axis=(1, 2))

    def _predict_one(self, x):
        return np.argmax(self.log_a_priori + self._predict_log_conditionals(x))


class EfficientChol(TensorizedChol):
    def _predict_log_conditionals_batch(self, X):
        unbiased_X = X - self.tensor_means
        Y = self.tensor_L_inv @ unbiased_X
        log_det_term = self.log_dets[:, None]
        return log_det_term - 0.5 * (Y**2).sum(axis=1) 

    def predict(self, X):
        log_priori = self.log_a_priori[:, None]
        log_conditional = self._predict_log_conditionals_batch(X)
        log_posteriori = log_priori + log_conditional

        return np.argmax(log_posteriori, axis=0).reshape(1, -1)
