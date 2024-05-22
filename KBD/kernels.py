import numpy as np


def gaussian_kernel(x, mu, sigma):
    return np.exp(-np.linalg.norm(x - mu) ** 2 / (2 * (sigma**2)))


def polynomial_kernel_n2(x, a, b, c):
    return a * x**2 + b * x + c


def sigmoid_kernel(x, y, alpha=1.0, c=0.0):
    return np.tanh(alpha * np.dot(x, y) + c)


def linear_kernel(x, y):
    return np.dot(x, y)


def laplacian_kernel(x, mu, sigma):
    return np.exp(-np.linalg.norm(x - mu, ord=1) / sigma)


def exponential_kernel(x, mu, sigma):
    return np.exp(-np.linalg.norm(x - mu) / sigma)
