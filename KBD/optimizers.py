import numpy as np
from bayes_opt import BayesianOptimization
from pyswarm import pso
from scipy.optimize import (
    differential_evolution,
    dual_annealing,
    least_squares,
    minimize,
)
from sklearn.linear_model import LinearRegression

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
        apply_weights=False,
        apply_l2=False,
        reg_lambda=0.001,
    ):
        self.gt = gt
        self.est = est
        self.focal = focal
        self.baseline = baseline
        self.local_restriction_weights = local_restriction_weights
        self.restriction_loc = restriction_loc
        self.target_rate = target_rate
        self.reg_lambda = reg_lambda

        self.initial_params = [1.0, 0.01, 10]
        self.bounds = None

        self.apply_weights = apply_weights
        self.apply_l2 = apply_l2

    def loss(self, params):
        k, delta, b = params
        pred = model(self.est, self.focal, self.baseline, k, delta, b)
        residuals = self.gt - pred
        mse = np.mean(residuals**2)
        if self.apply_weights:
            local_restric = np.mean(
                np.abs(
                    (
                        pred[self.gt < self.restriction_loc]
                        - self.gt[self.gt < self.restriction_loc]
                    )
                    / self.gt[self.gt < self.restriction_loc]
                )
            )
        else:
            local_restric = 0
        if self.apply_l2:
            l2_reg = self.reg_lambda * np.sum(np.square(params))
        else:
            l2_reg = 0
        return (
            mse
            + self.local_restriction_weights * max(0, local_restric - self.target_rate)
            + l2_reg
        )

    def optimize(self, initial_params, bounds):
        result = minimize(
            self.loss, initial_params, bounds=bounds, method="Nelder-Mead"
        )
        return result

    def run(self):
        result = self.optimize(self.initial_params, self.bounds)
        print("Optimization Result:")

        if result.success:
            optimized_params = result.x
            k, delta, b = optimized_params

            pred = model(self.est, self.focal, self.baseline, k, delta, b)

            mse = np.mean((pred - self.gt) ** 2)
            if self.apply_weights:
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
            if self.apply_weights:
                print(f"Error less than {self.restriction_loc}:", local_restric)
            print("Optimized Parameters:", optimized_params)

            return k, delta, b
        else:
            print("Optimization failed.")


class JointLinearSmoothingOptimizer:
    def __init__(
        self,
        gt,
        est,
        focal,
        baseline,
        disjoint_depth_range,
        compensate_dist=200,
        scaling_factor=10,
        engine="Nelder-Mead",
        local_restriction_weights=1000,
        target_rate=0.02,
        apply_global=False,
        apply_weights=False,
        apply_l2=False,
        reg_lambda=0.001,
    ):
        assert engine in (
            "Nelder-Mead",
            "Trust-Region",
        ), f"optimize engine {engine} is not supported!"
        self.gt = gt
        self.est = est
        self.focal = focal
        self.baseline = baseline
        self.local_restriction_weights = local_restriction_weights
        self.restriction_loc = disjoint_depth_range[0]
        self.target_rate = target_rate
        self.disjoint_depth_range = disjoint_depth_range
        self.compensate_dist = compensate_dist
        self.scaling_factor = scaling_factor
        self.apply_global = apply_global
        self.apply_weights = apply_weights
        self.apply_l2 = apply_l2
        self.reg_lambda = reg_lambda

        self.fb = focal * baseline
        self.engine = engine

        self.initial_params = [1.0, 0.01, 10]
        self.bounds = ([0, -10, -100], [10, 10, 100])

    def segment(self):
        # find the range to calculate KBD params within

        mask = np.where(
            (self.gt > self.disjoint_depth_range[0])
            & (self.gt < self.disjoint_depth_range[1])
        )
        if not self.apply_global:
            self.kbd_x = self.est[mask]
            self.kbd_y = self.gt[mask]
        else:
            self.kbd_x = self.est
            self.kbd_y = self.gt

        kbd_base_optimizer = None

        if self.engine == "Nelder-Mead":
            kbd_base_optimizer = NelderMeadOptimizer(
                self.kbd_y,
                self.kbd_x,
                self.focal,
                self.baseline,
                self.local_restriction_weights,
                self.restriction_loc,
                self.target_rate,
                self.apply_weights,
                self.apply_l2,
                self.reg_lambda,
            )
        elif self.engine == "Trust-Region":
            kbd_base_optimizer = TrustRegionReflectiveOptimizer(
                self.kbd_y,
                self.kbd_x,
                self.focal,
                self.baseline,
                restriction_loc=self.restriction_loc,
            )

        kbd_result = kbd_base_optimizer.run()
        return kbd_result

    def calculate_eta(self):
        lb = self.disjoint_depth_range[0]
        # lb shoud be restrictly greater than 1.000001

        eta = self.fb / [lb - 1] - self.fb / lb
        return eta

    def run(self):
        kbd_result = self.segment()
        if kbd_result is not None:
            k_, delta_, b_ = kbd_result
            x_min = np.min(self.kbd_x)
            x_max = np.max(self.kbd_x)

            y_hat_max = k_ * self.fb / (x_min + delta_) + b_
            y_hat_min = k_ * self.fb / (x_max + delta_) + b_

            x_hat_min = self.fb / y_hat_max
            x_hat_max = self.fb / y_hat_min

            pre_y = y_hat_min - self.compensate_dist
            after_y = y_hat_max + self.compensate_dist * self.scaling_factor

            pre_x = self.fb / pre_y
            after_x = self.fb / after_y

            lm1 = LinearRegression()
            x1 = np.array([pre_x, x_max])
            y1 = np.array([pre_x, x_hat_max])
            lm1.fit(x1.reshape(-1, 1), y1)

            lm2 = LinearRegression()
            x2 = np.array([x_min, after_x])
            y2 = np.array([x_hat_min, after_x])
            lm2.fit(x2.reshape(-1, 1), y2)

            self.params = (lm1, kbd_result, lm2)

            return lm1, kbd_result, lm2
        return


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

        self.initial_params = [1.0, 0.01, 10]
        self.bounds = ([0, -10, -100], [10, 10, 100])

    def loss(self, params):
        k, delta, b = params
        pred = model(self.est, self.focal, self.baseline, k, delta, b)
        residuals = pred - self.gt
        # mse = np.mean(residuals**2)

        local_restric = np.abs(
            (
                pred[self.gt < self.restriction_loc]
                - self.gt[self.gt < self.restriction_loc]
            )
            / self.gt[self.gt < self.restriction_loc]
        )
        print(local_restric)
        return np.concatenate(
            (
                residuals,
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
