import numpy as np
import numpy.linalg as LA

from base.bayesian import BaseBayesianClassifier


class QDA(BaseBayesianClassifier):

  def _fit_params(self, X, y):
    # estimate each covariance matrix
    # lista con matriz de covarianzas invertida, k items de pxp
    self.inv_covs = [LA.inv(np.cov(X[:,y.flatten()==idx], bias=True))
                      for idx in range(len(self.log_a_priori))]
    # Q5: por que hace falta el flatten y no se puede directamente X[:,y==idx]?
    # Q6: por que se usa bias=True en vez del default bias=False?
    # lista con 
    self.means = [X[:,y.flatten()==idx].mean(axis=1, keepdims=True)
                  for idx in range(len(self.log_a_priori))]
    # Q7: que hace axis=1? por que no axis=0?

  def _predict_log_conditional(self, x, class_idx):
    # predict the log(P(x|G=class_idx)), the log of the conditional probability of x given the class
    # this should depend on the model used
    inv_cov = self.inv_covs[class_idx]
    unbiased_x =  x - self.means[class_idx]
    return 0.5*np.log(LA.det(inv_cov)) -0.5 * unbiased_x.T @ inv_cov @ unbiased_x


class TensorizedQDA(QDA):

    def _fit_params(self, X, y):
        # ask plain QDA to fit params
        super()._fit_params(X,y)

        # stack onto new dimension
        self.tensor_inv_cov = np.stack(self.inv_covs)
        self.tensor_means = np.stack(self.means)

    def _predict_log_conditionals(self,x):
        unbiased_x = x - self.tensor_means
        inner_prod = unbiased_x.transpose(0,2,1) @ self.tensor_inv_cov @ unbiased_x

        return 0.5*np.log(LA.det(self.tensor_inv_cov)) - 0.5 * inner_prod.flatten()

    def _predict_one(self, x):
        # return the class that has maximum a posteriori probability
        return np.argmax(self.log_a_priori + self._predict_log_conditionals(x))


# Implementar el modelo `FasterQDA` (se recomienda heredarlo de `TensorizedQDA`) 
# de manera de eliminar el ciclo for en el método predict.
class FasterQDA(TensorizedQDA):

    def predict(self, X_obs):
        unbiased_X = X_obs - self.tensor_means
        cuadratic = unbiased_X.transpose(0, 2, 1) @ self.tensor_inv_cov @ unbiased_X
        cuadratic_diagonal = np.diagonal(cuadratic, axis1=1, axis2=2)

        log_conditionals = 0.5 * np.log(LA.det(self.tensor_inv_cov))[:, None] - 0.5 * cuadratic_diagonal
        log_posteriori = self.log_a_priori[:, None] + log_conditionals

        return np.argmax(log_posteriori, axis=0).reshape(1, -1)


  