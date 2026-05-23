from pathlib import Path

import json

import pandas as pd

from visualization.plots import HTMPlotter


class VisualizationDashboard:
    def __init__(
        self,
        output_dir: Path,
    ) -> None:

        self.output_dir = output_dir

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.plotter = HTMPlotter(
            output_dir=output_dir,
        )

    def build(
        self,
        metrics_history: list[dict],
    ) -> None:
        dataframe = pd.DataFrame(metrics_history)
        dataframe.to_csv(
            self.output_dir / "metrics.csv",
            index=False,
        )
        self.plotter.plot_temperature_vs_anamoly(
            dataframe,
        )
        self.plotter.plot_predictive_cells(
            dataframe,
        )
        self.plotter.plot_active_columns(
            dataframe,
        )
        self.plotter.plot_sp_entropy(
            dataframe,
        )
        self.plotter.plot_sp_overlap(
            dataframe,
        )
        self.plotter.plot_dead_columns(
            dataframe,
        )
        self.plotter.plot_duty_cycles(
            dataframe,
        )

        with open(
            self.output_dir / "metrics.json",
            "w",
        ) as f:

            json.dump(
                metrics_history,
                f,
                indent=2,
            )
