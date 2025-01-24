import scipy.odr as odr
import numpy as np
import typing
import scipy.stats as stats
import itertools
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelData:
    x_data: np.ndarray
    x_error: np.ndarray
    y_data: np.ndarray
    y_error: np.ndarray


@dataclass(frozen=True)
class ExponentModelParameters:
    power: float
    y_translation: float
    coefficient: float


@dataclass(frozen=True)
class GaussianModelParameters:
    mean: float
    std_dev: float
    normalization: float


def linear_model(beta: tuple[float, float], x: float):
    free_term = beta[0]
    slope = beta[1]

    return slope*x + free_term


def exponent_model(beta: tuple[float, float, float], x: float):
    power, translation, coefficient = beta

    return coefficient*np.e**(power*x) + translation


def gaussian_model(beta: tuple[float, float, float], x: float):
    mean, deviation, normalization = beta
    return (normalization / np.sqrt(2 * np.pi * deviation**2)) * np.exp(-(x - mean)**2 / (2 * (deviation**2)))


def gaussian_sum_model(beta: list[float], x: float):
    if len(beta) % 3 != 0:
        raise ValueError(f"Incompatible parameter list, len(beta) == {len(beta)} which is not a multiple of 3")

    gaussian_params_list = [GaussianModelParameters(beta[i], beta[i+1], beta[i+2]) for i in range(0, len(beta), 3)]
    return np.array([gaussian_model([gaussian_params.mean, gaussian_params.std_dev, gaussian_params.normalization], x) for gaussian_params in gaussian_params_list]).sum(axis=0)


def vectorized_line(free_term: float, slope: float):
    return np.vectorize(lambda x: linear_model([free_term, slope], x))


def vectorized_exponent(exponent: ExponentModelParameters):
    return np.vectorize(lambda x: exponent_model((exponent.power, exponent.y_translation, exponent.coefficient), x))


def vectorized_gaussian_sum(gaussian_parameters: list[GaussianModelParameters]):
    flattened_params = list(itertools.chain.from_iterable(
        [[params.mean, params.std_dev, params.normalization] for params in gaussian_parameters]))
    return np.vectorize(lambda x: gaussian_sum_model(flattened_params, x))


def fit_exponent(model_data: ModelData, guess: ExponentModelParameters) -> tuple[ExponentModelParameters, ExponentModelParameters, float, float]:
    """
    Fits data to exponent using odr

    Returns the fit data in the form tuple(fit params, uncertianties, chi square reduced, p_value)
    """
    fit_params, uncertainties, chi, p_value = odr_fit(
        model_data, exponent_model, [guess.power, guess.y_translation, guess.coefficient])

    fitted_exponent = ExponentModelParameters(fit_params[0], fit_params[1], fit_params[2])
    exponent_uncertainties = ExponentModelParameters(uncertainties[0], uncertainties[1], uncertainties[2])

    return (fitted_exponent, exponent_uncertainties, chi, p_value)


def fit_gaussian_sum(model_data: ModelData, guess: list[GaussianModelParameters]) -> tuple[list[GaussianModelParameters], list[tuple[float, float, float]], float, float]:
    guesses = list(itertools.chain.from_iterable(
        [[gaussian_parameters.mean, gaussian_parameters.std_dev, gaussian_parameters.normalization] for gaussian_parameters in guess]))
    beta, std_dev, chi_sq, p_value = odr_fit(model_data, gaussian_sum_model, guesses)

    fit_params_list = [GaussianModelParameters(beta[i], beta[i+1], beta[i+2]) for i in range(0, len(beta), 3)]
    std_dev_list = [(std_dev[i], std_dev[i+1], std_dev[i+2]) for i in range(0, len(std_dev), 3)]

    return fit_params_list, std_dev_list, chi_sq, p_value


def odr_fit(model_data: ModelData, model_func: typing.Callable[[list[float], float], float], guess: list[float]) -> tuple[list[float], list[float], float, float]:
    model = odr.Model(model_func)
    data = odr.Data(model_data.x_data, model_data.y_data, wd=1 /
                    np.power(model_data.x_error, 2), we=1/np.power(model_data.y_error, 2))

    odr_runner = odr.ODR(data, model, beta0=guess)
    output = odr_runner.run()
    degrees_of_freedom = len(model_data.x_data) - len(guess)

    p_value = 1 - stats.chi2.cdf(output.res_var, degrees_of_freedom)

    return (output.beta, output.sd_beta, output.res_var, p_value)
