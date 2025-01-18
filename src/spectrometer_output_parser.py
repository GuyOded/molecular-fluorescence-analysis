from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np


@dataclass(frozen=True)
class SpectrometerOutput:
    wavelength_to_intensity_mapping: pd.DataFrame
    sample_time: float
    output_path: Path


def parse_excel_output(output_path: Path, default_sample_time=None) -> SpectrometerOutput:
    dataframe = pd.read_excel(output_path)

    # print(output_path.stem)
    # Remove empty columns
    dataframe = dataframe.loc[:, ~dataframe.columns.str.contains("^Unnamed")]
    try:
        sample_time = np.float64(dataframe.iloc[0, 2])
    except IndexError as e:
        if default_sample_time == None:
            print("Cannot find parse sample time, please provide a default or fix data")
            raise e

        print(f"Unable to find sample time, using provided default {default_sample_time}")
        sample_time = default_sample_time

    # The code assumes the wavelength data is in the
    wavelength_to_intensity_dataframe = pd.DataFrame(
        {"wavelength": dataframe.iloc[:, 0], "intensity": dataframe.iloc[:, 1]})

    return SpectrometerOutput(wavelength_to_intensity_dataframe, sample_time, output_path)


def parse_excels_in_folder(folder_path: Path, default_sample_time=1000) -> list[SpectrometerOutput]:
    folder_path = Path(folder_path)
    if not folder_path.is_dir():
        raise ValueError(f"Path \"{folder_path}\" does not lead to a directory")

    return [parse_excel_output(out) for out in folder_path.iterdir() if Path(out).suffix == ".ods"]
