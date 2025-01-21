from spectrometer_output_parser import SpectrometerOutput
import analysis_utils
import plotly.graph_objects as go


def generate_scatter_plot_from_outputs(outputs: SpectrometerOutput) -> list[go.Scatter]:
    concentration_sorted_outputs: list[SpectrometerOutput] = sorted(
        outputs, reverse=True, key=lambda output: output.output_path.stem)

    fig_data = []
    for output in concentration_sorted_outputs:
        concentration = output.output_path.stem.split("_")[-1]
        dataset = analysis_utils.generate_spectrometer_dataset(output)
        fig_data.append(go.Scatter(
            x=dataset.wavelength, y=dataset.intensity, name=concentration, mode="lines"))

    return fig_data


def generate_scatter_plots_map(outputs: SpectrometerOutput) -> list[go.Scatter]:
    """
    Creates a dictionary that maps file name (without extension) to a scatter plot from SpectrometerOutput list
    """
    sorted_spectrometer_outputs: list[SpectrometerOutput] = sorted(
        outputs, reverse=True, key=lambda output: output.output_path.stem)

    result = {}
    for output in sorted_spectrometer_outputs:
        dataset = analysis_utils.generate_spectrometer_dataset(output)
        {output.output_path.stem: go.Scatter(x=dataset.wavelength, y=dataset.intensity,
                                             name=output.output_path.stem, mode="lines")}

    return result
