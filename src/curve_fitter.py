import scipy.odr as odr
import numpy as np
import typing
import scipy.stats as stats
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelData:
    x_data: np.ndarray
    x_error: np.ndarray
    y_data: np.ndarray
    y_error: np.ndarray


@dataclass(frozen=True)
class ExponentModel:
    power: float
    y_translation: float
    coefficient: float


def linear_model(beta: tuple[float, float], x: float):
    free_term = beta[0]
    slope = beta[1]

    return slope*x + free_term


def exponent_model(beta: tuple[float, float, float], x: float):
    power, translation, coefficient = beta

    return coefficient*np.e**(power*x) + translation


def vectorized_line(free_term: float, slope: float):
    return np.vectorize(lambda x: linear_model([free_term, slope], x))


def vectorized_exponent(exponent: ExponentModel):
    return np.vectorize(lambda x: exponent_model((exponent.power, exponent.y_translation, exponent.coefficient), x))


def fit_exponent(model_data: ModelData, guess: ExponentModel) -> tuple[ExponentModel, ExponentModel, float, float]:
    """
    Fits data to exponent using odr

    Returns the fit data in the form tuple(fit params, uncertianties, chi square reduced, p_value)
    """
    fit_params, uncertainties, chi, p_value = odr_fit(model_data, exponent_model, [guess.power, guess.y_translation, guess.coefficient])

    fitted_exponent = ExponentModel(fit_params[0], fit_params[1], fit_params[2])
    exponent_uncertainties = ExponentModel(uncertainties[0], uncertainties[1], uncertainties[2])

    return (fitted_exponent, exponent_uncertainties, chi, p_value)


def odr_fit(model_data: ModelData, model_func: typing.Callable[[list[float], float], float], guess: list[float]) -> tuple[list[float], list[float], float, float]:
    model = odr.Model(model_func)
    data = odr.Data(model_data.x_data, model_data.y_data, wd=1 /
                    np.power(model_data.x_error, 2), we=1/np.power(model_data.y_error, 2))

    odr_runner = odr.ODR(data, model, beta0=guess)
    output = odr_runner.run()
    degrees_of_freedom = len(model_data.x_data) - len(guess)

    p_value = 1 - stats.chi2.cdf(output.res_var, degrees_of_freedom)

    return (output.beta, output.sd_beta, output.res_var, p_value)
