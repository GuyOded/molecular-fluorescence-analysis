import scipy
import scipy.constants
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


@dataclass(frozen=True)
class ExcitonPolaritonParameters:
    diffraction_index: float
    cavity_length: float
    exciton_energy: float
    omega: float


def linear_model(beta: tuple[float, float], x: float):
    free_term = beta[0]
    slope = beta[1]

    return slope * x + free_term


def exponent_model(beta: tuple[float, float, float], x: float):
    power, translation, coefficient = beta

    return coefficient * np.e**(power * x) + translation


def gaussian_model(beta: tuple[float, float, float], x: float):
    mean, deviation, normalization = beta
    return (normalization / np.sqrt(2 * np.pi * deviation**2)) * np.exp(-(x - mean)**2 / (2 * (deviation**2)))


def gaussian_sum_model(beta: list[float], x: float):
    if len(beta) % 3 != 0:
        raise ValueError(f"Incompatible parameter list, len(beta) == {len(beta)} which is not a multiple of 3")

    gaussian_params_list = [GaussianModelParameters(beta[i], beta[i + 1], beta[i + 2]) for i in range(0, len(beta), 3)]
    return np.array([gaussian_model([gaussian_params.mean, gaussian_params.std_dev, gaussian_params.normalization], x)
                    for gaussian_params in gaussian_params_list]).sum(axis=0)


def exciton_polariton_energy_model(beta: tuple[float, float, float, float], x: float, is_high):
    JOULE_TO_EV = 6.242E+18
    diffraction_index, cavity_length, exciton_energy, omega = beta

    photon_energy = scipy.constants.hbar * scipy.constants.c / \
        diffraction_index * np.sqrt(x**2 + (np.pi / cavity_length)**2)

    return 0.5 * (exciton_energy + photon_energy + (-1)**(not is_high) * np.sqrt(4 *
                  scipy.constants.hbar**2 * omega**2 + (exciton_energy - photon_energy)**2))


def vectorized_line(free_term: float, slope: float):
    return np.vectorize(lambda x: linear_model([free_term, slope], x))


def vectorized_exponent(exponent: ExponentModelParameters):
    return np.vectorize(lambda x: exponent_model((exponent.power, exponent.y_translation, exponent.coefficient), x))


def vectorized_gaussian_sum(gaussian_parameters: list[GaussianModelParameters]):
    flattened_params = list(itertools.chain.from_iterable(
        [[params.mean, params.std_dev, params.normalization] for params in gaussian_parameters]))
    return np.vectorize(lambda x: gaussian_sum_model(flattened_params, x))


def vectorized_exciton_polariton_high(beta: ExcitonPolaritonParameters):
    return np.vectorize(lambda x: exciton_polariton_energy_model(
        [beta.diffraction_index, beta.cavity_length, beta.exciton_energy, beta.omega], x, True))


def vectorized_exciton_polariton_low(beta: ExcitonPolaritonParameters):
    return np.vectorize(lambda x: exciton_polariton_energy_model(
        (beta.diffraction_index, beta.cavity_length, beta.exciton_energy, beta.omega), x, False))


def fit_exponent(model_data: ModelData,
                 guess: ExponentModelParameters) -> tuple[ExponentModelParameters,
                                                          ExponentModelParameters,
                                                          float,
                                                          float]:
    """
    Fits data to exponent using odr

    Returns the fit data in the form tuple(fit params, uncertianties, chi square reduced, p_value)
    """
    fit_params, uncertainties, chi, p_value = odr_fit(
        model_data, exponent_model, [guess.power, guess.y_translation, guess.coefficient])

    fitted_exponent = ExponentModelParameters(fit_params[0], fit_params[1], fit_params[2])
    exponent_uncertainties = ExponentModelParameters(uncertainties[0], uncertainties[1], uncertainties[2])

    return (fitted_exponent, exponent_uncertainties, chi, p_value)


def fit_gaussian_sum(model_data: ModelData,
                     guess: list[GaussianModelParameters]) -> tuple[list[GaussianModelParameters],
                                                                    list[tuple[float,
                                                                               float,
                                                                               float]],
                                                                    float,
                                                                    float]:
    guesses = list(itertools.chain.from_iterable(
        [[gaussian_parameters.mean, gaussian_parameters.std_dev, gaussian_parameters.normalization] for gaussian_parameters in guess]))
    beta, std_dev, chi_sq, p_value = odr_fit(model_data, gaussian_sum_model, guesses)

    fit_params_list = [GaussianModelParameters(beta[i], beta[i + 1], beta[i + 2]) for i in range(0, len(beta), 3)]
    std_dev_list = [(std_dev[i], std_dev[i + 1], std_dev[i + 2]) for i in range(0, len(std_dev), 3)]

    return fit_params_list, std_dev_list, chi_sq, p_value


def fit_exciton_polariton_energy(model_data: ModelData, guess: ExcitonPolaritonParameters,
                                 is_high: bool) -> tuple[ExcitonPolaritonParameters, list[float], float, float]:
    def model_function(beta, x): return exciton_polariton_energy_model(beta, x, is_high)

    beta, std_dev, chi_sq, p_value = odr_fit(
        model_data, model_function, [
            guess.diffraction_index, guess.cavity_length, guess.exciton_energy, guess.omega])
    fitted_params = ExcitonPolaritonParameters(beta[0], beta[1], beta[2], beta[3])

    return fitted_params, std_dev, chi_sq, p_value


def odr_fit(model_data: ModelData, model_func: typing.Callable[[
            list[float], float], float], guess: list[float]) -> tuple[list[float], list[float], float, float]:
    model = odr.Model(model_func)
    data = odr.Data(model_data.x_data, model_data.y_data, wd=1 /
                    np.power(model_data.x_error, 2), we=1 / np.power(model_data.y_error, 2))

    odr_runner = odr.ODR(data, model, beta0=guess)
    output = odr_runner.run()
    degrees_of_freedom = len(model_data.x_data) - len(guess)

    p_value = 1 - stats.chi2.cdf(output.res_var, degrees_of_freedom)

    return (output.beta, output.sd_beta, output.res_var, p_value)
