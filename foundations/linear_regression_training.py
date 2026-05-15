import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_derivative(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64], N: int, X: NDArray[np.float64], desired_weight: int) -> float:
        # note that N is just len(X)
        return -2 * np.dot(ground_truth - model_prediction, X[:, desired_weight]) / N

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.squeeze(np.matmul(X, weights))

    learning_rate = 0.01

    def train_model(
        self, 
        X: NDArray[np.float64], 
        Y: NDArray[np.float64], 
        num_iterations: int, 
        initial_weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:

        # you will need to call get_derivative() for each weight
        # and update each one separately based on the learning rate!
        # return np.round(your_answer, 5)
        weights = np.array(initial_weights)
        N = len(X) 
        for one_iter in range(num_iterations): 
            model_preds = self.get_model_prediction(X, weights)
            derivatives = np.zeros(3)
            for i in range(3):
                derivatives[i] = self.get_derivative(model_prediction=model_preds, ground_truth=Y, N=N, X=X, desired_weight=i)
            weights -= self.learning_rate * derivatives
        return np.round(weights, 5)