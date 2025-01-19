from spectrometer_output_parser import SpectrometerOutput
import numpy as np
from scipy import integrate
from dataclasses import dataclass


# The uncertainty of the measured wavelength in nm
# Taken from https://www.thorlabs.com/drawings/5a9469faa287d95e-B93A8B31-D6C7-39F4-ADD5791A79AA03C7/CCS200-Manual.pdf
WAVELENGTH_UNCERTAINTY = 2
WAVELENGTH_THRESHOLD = 460


@dataclass(frozen=True)
class SpectrometerDataset:
    wavelength: np.ndarray[np.float64]
    intensity: np.ndarray[np.float64]
    wavelength_uncertainty: np.ndarray[np.float64]
    intensity_uncertainty: np.ndarray[np.float64]


def generate_spectrometer_dataset(output: SpectrometerOutput, normalize_time=True) -> SpectrometerDataset:
    wavelength = np.array(output.wavelength_to_intensity_mapping.iloc[:, 0], dtype=np.float64)
    intensity = np.copy(np.array(output.wavelength_to_intensity_mapping.iloc[:, 1], dtype=np.float64))

    if normalize_time:
        intensity /= output.sample_time

    wavelength_uncertainty = WAVELENGTH_UNCERTAINTY

    intensity_resolution_error = np.full(len(wavelength), 10**-5 / np.sqrt(3), dtype=np.float64)
    intensity_poisson_error = np.sqrt(np.abs(intensity))

    intensity_uncertainty = np.sqrt(intensity_resolution_error**2 + intensity_poisson_error**2)

    return SpectrometerDataset(wavelength, intensity, wavelength_uncertainty, intensity_uncertainty)


def integrate_spectrometer_dataset_monte_carlo(dataset: SpectrometerDataset, n_simulations=10**4, wavelength_threshold=WAVELENGTH_THRESHOLD) -> tuple[float, float]:
    """
    Uses trapezoid integration rule to calculate
    Also uses the Monte-Carlo method to find the uncertainty of the integration
    """
    filter_intensity_and_wavelength_by_wavelength_threshold(dataset, wavelength_threshold)
    measurement_points_count = len(dataset.wavelength)

    intensity_monte_carlo_sample = np.random.uniform(
        dataset.intensity - dataset.intensity_uncertainty, dataset.intensity + dataset.intensity_uncertainty, size=(n_simulations, measurement_points_count))
    wavelength_monte_carlo_sample = np.random.uniform(
        dataset.wavelength - dataset.wavelength_uncertainty, dataset.wavelength + dataset.wavelength_uncertainty, size=(n_simulations, measurement_points_count))

    integration_result = np.zeros(n_simulations, dtype=np.float64)

    for i, (wavelength, intensity) in enumerate(zip(wavelength_monte_carlo_sample, intensity_monte_carlo_sample, strict=True)):
        integration_result[i] = integrate.trapezoid(intensity, wavelength)

    return (np.mean(integration_result), np.std(integration_result))


def integrate_spectrometer_dataset(dataset: SpectrometerDataset, wavelength_threshold=WAVELENGTH_THRESHOLD) -> tuple[float, float]:
    filtered_dataset = filter_intensity_and_wavelength_by_wavelength_threshold(dataset, wavelength_threshold)

    integration_result = integrate.trapezoid(
        filtered_dataset.intensity, filtered_dataset.wavelength)

    return integration_result, 1 / np.sqrt(np.abs(integration_result))


def filter_intensity_and_wavelength_by_wavelength_threshold(dataset: SpectrometerDataset, wavelength_threshold=WAVELENGTH_THRESHOLD) -> SpectrometerDataset:
    indices_above_threshold = np.where(dataset.wavelength >= wavelength_threshold)
    if len(indices_above_threshold) == 0:
        raise ValueError(f"Threshold {wavelength_threshold} is greater than all other wavelength.")

    first_index_above_threshold = indices_above_threshold[0][0]

    return SpectrometerDataset(dataset.wavelength[first_index_above_threshold:],
                               dataset.intensity[first_index_above_threshold:],
                               dataset.wavelength_uncertainty,
                               dataset.intensity_uncertainty[first_index_above_threshold:])


def integrate_spectrometer_outputs(outputs: list[SpectrometerOutput]) -> tuple[np.ndarray, np.ndarray]:
    datasets = [generate_spectrometer_dataset(output) for output in outputs]

    result = np.array([integrate_spectrometer_dataset_monte_carlo(dataset) for dataset in datasets])
    integration_results = result[:, 0]
    uncertainties = result[:, 1]

    return (integration_results, uncertainties)
