from spectrometer_output_parser import SpectrometerOutput
import numpy as np
from dataclasses import dataclass


# The uncertainty of the measured wavelength in nm
# Taken from https://www.thorlabs.com/drawings/5a9469faa287d95e-B93A8B31-D6C7-39F4-ADD5791A79AA03C7/CCS200-Manual.pdf
WAVELENGTH_UNCERTAINTY = 2


@dataclass(frozen=True)
class SpectrometerDataSet:
    wavelength: np.ndarray[np.float64]
    intensity: np.ndarray[np.float64]
    wavelength_uncertainty: np.ndarray[np.float64]
    intensity_uncertainty: np.ndarray[np.float64]


def generate_spectrometer_data_set(output: SpectrometerOutput, normalize_time=True) -> SpectrometerDataSet:
    wavelength = np.array(output.wavelength_to_intensity_mapping[:, 0], dtype=np.float64)
    intensity = np.array(output.wavelength_to_intensity_mapping[:, 1], dtype=np.float64)

    if normalize_time:
        intensity /= output.sample_time

    wavelength_uncertainty = WAVELENGTH_UNCERTAINTY

    intensity_resolution_error = np.full(len(wavelength), 10**-5 / np.sqrt(3), dtype=np.float64)
    intensity_poisson_error = np.sqrt(intensity)

    intensity_uncertainty = np.sqrt(intensity_resolution_error**2 + intensity_poisson_error**2)

    return SpectrometerDataSet(wavelength, intensity, wavelength_uncertainty, intensity_uncertainty)
