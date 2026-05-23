from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


class HTMPlotter:
    def __init__(
        self,
        output_dir: Path,
    ) -> None:

        self.output_dir = output_dir

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def plot_temperature_vs_anamoly(self, dataframe: pd.DataFrame):
        fig = plt.figure(figsize=(16, 8))
        ax1 = fig.add_subplot(111)
        ax1.plot(
            dataframe["temperature"].to_numpy(),
        )
        ax1.set_xlabel("Time Step")
        ax1.set_ylabel("Temperature")
        ax2 = ax1.twinx()
        ax2.plot(
            dataframe["anomaly_score"].to_numpy(),
        )
        ax2.set_ylabel("Anomaly Score")
        plt.title("Temperature vs HTM Anomaly")
        plt.savefig(self.output_dir / "temperature_vs_anomaly.png")
        plt.close(fig)

    def plot_predictive_cells(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        fig = plt.figure(figsize=(16, 8))
        ax = fig.add_subplot(111)
        ax.plot(
            dataframe["tm_predictive_cells"].to_numpy(),
        )
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Predictive Cells")
        plt.title("Temporal Memory Predictive Cells")
        plt.savefig(self.output_dir / "predictive_cells.png")
        plt.close(fig)

    # -------------------------------------------------
    # Active Columns
    # -------------------------------------------------

    def plot_active_columns(
        self,
        dataframe: pd.DataFrame,
    ) -> None:

        fig = plt.figure(figsize=(16, 8))

        ax = fig.add_subplot(111)

        ax.plot(
            dataframe["sp_active_columns"].to_numpy(),
        )

        ax.set_xlabel("Time Step")

        ax.set_ylabel("SP Active Columns")

        plt.title("Spatial Pooler Stability")

        plt.savefig(self.output_dir / "active_columns.png")

        plt.close(fig)

    def plot_sp_entropy(
        self,
        dataframe: pd.DataFrame,
    ) -> None:

        fig = plt.figure(figsize=(16, 8))

        ax = fig.add_subplot(111)

        ax.plot(
            dataframe["sp_entropy"].to_numpy(),
        )

        ax.set_xlabel("Time Step")

        ax.set_ylabel("SP Entropy")

        plt.title("Spatial Pooler Entropy")

        plt.savefig(self.output_dir / "sp_entropy.png")

        plt.close(fig)

    def plot_sp_overlap(
        self,
        dataframe: pd.DataFrame,
    ) -> None:

        fig = plt.figure(figsize=(16, 8))

        ax = fig.add_subplot(111)

        ax.plot(
            dataframe["sp_mean_overlap"].to_numpy(),
        )

        ax.set_xlabel("Time Step")

        ax.set_ylabel("Mean SDR Overlap")

        plt.title("Spatial Pooler SDR Overlap")

        plt.savefig(self.output_dir / "sp_overlap.png")

        plt.close(fig)

    def plot_dead_columns(
        self,
        dataframe: pd.DataFrame,
    ) -> None:

        fig = plt.figure(figsize=(16, 8))

        ax = fig.add_subplot(111)

        ax.plot(
            dataframe["sp_dead_column_ratio"].to_numpy(),
        )

        ax.set_xlabel("Time Step")

        ax.set_ylabel("Dead Column Ratio")

        plt.title("Spatial Pooler Dead Columns")

        plt.savefig(self.output_dir / "dead_columns.png")

        plt.close(fig)

    def plot_duty_cycles(
        self,
        dataframe: pd.DataFrame,
    ) -> None:

        fig = plt.figure(figsize=(16, 8))

        ax = fig.add_subplot(111)

        ax.plot(
            dataframe["sp_max_duty_cycle"].to_numpy(),
        )

        ax.plot(
            dataframe["sp_mean_duty_cycle"].to_numpy(),
        )

        ax.set_xlabel("Time Step")

        ax.set_ylabel("Duty Cycle")

        plt.title("Spatial Pooler Duty Cycles")

        plt.savefig(self.output_dir / "duty_cycles.png")

        plt.close(fig)
