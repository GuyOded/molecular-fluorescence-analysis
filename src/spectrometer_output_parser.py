from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import os


@dataclass
class SpectrometerOutput:
    wavelength_to_intensity_mapping: pd.DataFrame
    sample_time: float
    output_path: Path


def parse_excel_output(output_path: Path, default_sample_time=None) -> SpectrometerOutput:
    dataframe = pd.read_excel(output_path)

    # Remove empty columns
    dataframe = dataframe.loc[:, ~dataframe.columns.str.contains("^Unnamed")]
    try:
        sample_time = dataframe.loc[0, 2]
    except IndexError as e:
        if default_sample_time == None:
            print("Cannot find parse sample time, please provide a default or fix data")
            raise e

        print(f"Unable to find sample time, using provided default {default_sample_time}")
        sample_time = default_sample_time

    # The code assumes the wavelength data is in the
    wavelength_to_intensity_dataframe = pd.DataFrame({"wavelength": dataframe[:, 0], "intensity": dataframe[:, 1]})

    return SpectrometerOutput(wavelength_to_intensity_dataframe, sample_time, output_path)


def parse_excels_in_folder(folder_path: Path, default_sample_time=1000) -> list[SpectrometerOutput]:
    if not Path(folder_path).is_dir():
        raise ValueError(f"Path \"{folder_path}\" does not lead to a directory")

    return [SpectrometerOutput(out, default_sample_time) for out in os.listdir(folder_path) if Path(out).suffix == ".ods"]
