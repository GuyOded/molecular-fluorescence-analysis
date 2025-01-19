from spectrometer_output_parser import SpectrometerOutput
import numpy as np
from scipy import integrate
from dataclasses import dataclass


# The uncertainty of the measured wavelength in nm
# Taken from https://www.thorlabs.com/drawings/5a9469faa287d95e-B93A8B31-D6C7-39F4-ADD5791A79AA03C7/CCS200-Manual.pdf
WAVELENGTH_UNCERTAINTY = 2
WAVELENGTH_THRESHOLD = 460
# We assume that background noise is measured in all wave lengths up to this value
BACKGROUND_NOISE_WAVELENGTH_LIMIT = 400


@dataclass(frozen=True)
class SpectrometerDataset:
    wavelength: np.ndarray[np.float64]
    intensity: np.ndarray[np.float64]
    wavelength_uncertainty: float
    intensity_uncertainty: np.ndarray[np.float64]


def generate_spectrometer_dataset(output: SpectrometerOutput, normalize_time=True) -> SpectrometerDataset:
    wavelength = np.array(output.wavelength_to_intensity_mapping.iloc[:, 0], dtype=np.float64)
    intensity = np.copy(np.array(output.wavelength_to_intensity_mapping.iloc[:, 1], dtype=np.float64))

    if normalize_time:
        intensity /= output.sample_time

    wavelength_uncertainty = WAVELENGTH_UNCERTAINTY

    intensity_resolution_error = np.full(len(wavelength), 10**-9 / np.sqrt(3), dtype=np.float64)
    noise_end_index = np.where(wavelength > BACKGROUND_NOISE_WAVELENGTH_LIMIT)[0][0]
    intensity_std = np.std(intensity[:noise_end_index])

    intensity_uncertainty = np.sqrt(intensity_std**2 + intensity_resolution_error**2)

    return SpectrometerDataset(wavelength, intensity, wavelength_uncertainty, intensity_uncertainty)


def average_spectrometer_dataset_integration(dataset: SpectrometerDataset, wavelength_threshold=WAVELENGTH_THRESHOLD) -> tuple[float, float]:
    """
    Calculates integral of intensity w.r.t wavelength using trapezoid rule, averaging over the uncertainty range.
    """
    filtered_dataset = filter_intensity_and_wavelength_by_wavelength_threshold(dataset, wavelength_threshold)

    intensity_upper_limit = filtered_dataset.intensity + filtered_dataset.intensity_uncertainty
    intensity_lower_limit = filtered_dataset.intensity - filtered_dataset.intensity_uncertainty

    upper_integral = integrate.trapezoid(intensity_upper_limit, filtered_dataset.wavelength)
    lower_integral = integrate.trapezoid(intensity_lower_limit, filtered_dataset.wavelength)

    return ((upper_integral + lower_integral) / 2, (upper_integral - lower_integral) / 2)


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

    result = np.array([average_spectrometer_dataset_integration(dataset) for dataset in datasets])
    integration_results = result[:, 0]
    uncertainties = result[:, 1]

    return (integration_results, uncertainties)
