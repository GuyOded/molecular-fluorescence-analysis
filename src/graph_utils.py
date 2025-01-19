from spectrometer_output_parser import SpectrometerOutput
import analysis_utils
import plotly.graph_objects as go


def get_scatter_plot_from_outputs(outputs: SpectrometerOutput):
    concentration_sorted_outputs: list[SpectrometerOutput] = sorted(outputs, reverse=True, key=lambda output: output.output_path.stem)

    fig_data = []
    for output in concentration_sorted_outputs:
        concentration = output.output_path.stem.split("_")[-1]
        dataset = analysis_utils.generate_spectrometer_dataset(output)
        fig_data.append(go.Scatter(
            x=dataset.wavelength, y=dataset.intensity, name=concentration, mode="lines"))

    return fig_data
