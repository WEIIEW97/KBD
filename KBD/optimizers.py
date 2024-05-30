import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from scipy.optimize import differential_evolution, least_squares, dual_annealing
from bayes_opt import BayesianOptimization
from pyswarm import pso

from .models import model


class NelderMeadOptimizer:
    def __init__(
        self,
        gt,
        est,
        focal,
        baseline,
        local_restriction_weights=1000,
        restriction_loc=1000,
        target_rate=0.02,
    ):
        self.gt = gt
        self.est = est
        self.focal = focal
        self.baseline = baseline
        self.local_restriction_weights = local_restriction_weights
        self.restriction_loc = restriction_loc
        self.target_rate = target_rate

        self.initial_params = [1.0, 0, 0]
        self.bounds = ([0, -10, -100], [10, 10, 100])

    def loss(self, params):
        k, delta, b = params
        pred = model(self.est, self.focal, self.baseline, k, delta, b)
        residuals = pred - self.gt
        mse = np.mean(residuals**2)
        local_restric = np.abs(
            (
                pred[self.gt < self.restriction_loc]
                - self.gt[self.gt < self.restriction_loc]
            )
            / self.gt[self.gt < self.restriction_loc]
        )
        return mse + self.local_restriction_weights * max(
            0, local_restric - self.target_rate
        )

    def optimize(self, initial_params, bounds):
        result = minimize(
            self.loss, initial_params, bounds=bounds, method="Nelder-Mead"
        )
        return result

    def run(self):
        result = self.optimize(self.initial_params, self.bounds)
        print("Optimization Result:", result)

        if result.success:
            optimized_params = result.x
            k, delta, b = optimized_params

            pred = model(self.est, self.focal, self.baseline, k, delta, b)

            mse = np.mean((pred - self.gt) ** 2)
            local_restric = np.mean(
                np.abs(
                    (
                        pred[self.gt < self.restriction_loc]
                        - self.gt[self.gt < self.restriction_loc]
                    )
                    / self.gt[self.gt < self.restriction_loc]
                )
            )

            print("MSE:", mse)
            print(f"Error less than {self.restriction_loc}:", local_restric)
            print("Optimized Parameters:", optimized_params)

            return k, delta, b
        else:
            print("Optimization failed.")


class TrustRegionReflectiveOptimizer:
    def __init__(
        self,
        gt,
        est,
        focal,
        baseline,
        local_restriction_weights=1000,
        restriction_loc=1000,
        target_rate=0.02,
    ):
        self.gt = gt
        self.est = est
        self.focal = focal
        self.baseline = baseline
        self.local_restriction_weights = local_restriction_weights
        self.restriction_loc = restriction_loc
        self.target_rate = target_rate

        self.initial_params = [1.0, 0, 0]
        self.bounds = ([0, -10, -100], [10, 10, 100])

    def loss(self, params):
        k, delta, b = params
        pred = model(self.est, self.focal, self.baseline, k, delta, b)
        residuals = pred - self.gt
        mse = np.mean(residuals**2)
        local_restric = np.abs(
            (
                pred[self.gt < self.restriction_loc]
                - self.gt[self.gt < self.restriction_loc]
            )
            / self.gt[self.gt < self.restriction_loc]
        )
        return np.concatenate(
            (
                mse,
                self.local_restriction_weights
                * np.maximum(0, local_restric - self.target_rate),
            )
        )

    def optimize(self, initial_params, bounds):
        result = least_squares(self.loss, initial_params, bounds=bounds)
        return result

    def run(self):
        result = self.optimize(self.initial_params, self.bounds)
        print("Optimization Result:", result)

        if result.success:
            optimized_params = result.x
            k, delta, b = optimized_params

            pred = model(self.est, self.focal, self.baseline, k, delta, b)

            mse = np.mean((pred - self.gt) ** 2)
            local_restric = np.mean(
                np.abs(
                    (
                        pred[self.gt < self.restriction_loc]
                        - self.gt[self.gt < self.restriction_loc]
                    )
                    / self.gt[self.gt < self.restriction_loc]
                )
            )

            print("MSE:", mse)
            print(f"Error less than {self.restriction_loc}:", local_restric)
            print("Optimized Parameters:", optimized_params)

            return k, delta, b
        else:
            print("Optimization failed.")


class ParticleSwarmOptimizer:
    def __init__(
        self,
        gt,
        est,
        focal,
        baseline,
        local_restriction_weights=1000,
        restriction_loc=1000,
        target_rate=0.02,
    ):
        self.gt = gt
        self.est = est
        self.focal = focal
        self.baseline = baseline
        self.local_restriction_weights = local_restriction_weights
        self.restriction_loc = restriction_loc
        self.target_rate = target_rate

        self.swarmsize = 100
        self.maxiter = 2000
        self.bounds = [(0, 2), (-10, 10), (-100, 100)]

    def loss(self, params):
        k, delta, b = params
        pred = model(self.est, self.focal, self.baseline, k, delta, b)
        residuals = pred - self.gt
        mse = np.mean(residuals**2)
        local_restric = np.abs(
            (
                pred[self.gt < self.restriction_loc]
                - self.gt[self.gt < self.restriction_loc]
            )
            / self.gt[self.gt < self.restriction_loc]
        )
        return mse + self.local_restriction_weights * max(
            0, local_restric - self.target_rate
        )

    def optimize(self, bounds):
        lb = [b[0] for b in bounds]
        ub = [b[1] for b in bounds]

        xopt, fopt = pso(
            self.loss, lb, ub, swarmsize=self.swarmsize, maxiter=self.maxiter
        )
        return xopt, fopt

    def run(self):
        optimized_params, fopt = self.optimize(bounds=self.bounds)

        k, delta, b = optimized_params
        pred = model(self.est, self.focal, self.baseline, k, delta, b)
        mse = np.mean((pred - self.gt) ** 2)
        local_restric = np.mean(
            np.abs(
                (
                    pred[self.gt < self.restriction_loc]
                    - self.gt[self.gt < self.restriction_loc]
                )
                / self.gt[self.gt < self.restriction_loc]
            )
        )

        print("MSE:", mse)
        print(f"Error less than {self.restriction_loc}:", local_restric)
        print("Optimized Parameters:", optimized_params)

        return k, delta, b


class DualAnnealingOptimizer:
    def __init__(
        self,
        gt,
        est,
        focal,
        baseline,
        local_restriction_weights=1000,
        restriction_loc=1000,
        target_rate=0.02,
    ):
        self.gt = gt
        self.est = est
        self.focal = focal
        self.baseline = baseline
        self.local_restriction_weights = local_restriction_weights
        self.restriction_loc = restriction_loc
        self.target_rate = target_rate

        self.bounds = [(0, 2), (-10, 10), (-100, 100)]

    def loss(self, params):
        k, delta, b = params
        pred = model(self.est, self.focal, self.baseline, k, delta, b)
        residuals = pred - self.gt
        mse = np.mean(residuals**2)
        local_restric = np.abs(
            (
                pred[self.gt < self.restriction_loc]
                - self.gt[self.gt < self.restriction_loc]
            )
            / self.gt[self.gt < self.restriction_loc]
        )
        return mse + self.local_restriction_weights * max(
            0, local_restric - self.target_rate
        )

    def optimize(self, bounds):
        result = dual_annealing(self.loss, bounds=bounds)
        return result

    def run(self):
        result = self.optimize(bounds=self.bounds)
        print("Optimization Result:", result)
        if result.success:
            optimized_params = result.x
            k, delta, b = optimized_params

            pred = model(self.est, self.focal, self.baseline, k, delta, b)

            mse = np.mean((pred - self.gt) ** 2)
            local_restric = np.mean(
                np.abs(
                    (
                        pred[self.gt < self.restriction_loc]
                        - self.gt[self.gt < self.restriction_loc]
                    )
                    / self.gt[self.gt < self.restriction_loc]
                )
            )

            print("MSE:", mse)
            print(f"Error less than {self.restriction_loc}:", local_restric)
            print("Optimized Parameters:", optimized_params)

            return k, delta, b
        else:
            print("Optimization failed.")
