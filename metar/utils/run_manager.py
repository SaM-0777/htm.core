from pathlib import Path
from datetime import datetime
import json


class RunManager:
    def __init__(
        self,
        experiment_name: str,
        epochs: int,
    ) -> None:

        timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")

        self.run_name = f"{experiment_name}_epochs_{epochs}_{timestamp}"

        self.base_dir = Path("./output/runs") / self.run_name

        self.charts_dir = self.base_dir / "charts"
        self.metrics_dir = self.base_dir / "metrics"
        self.config_dir = self.base_dir / "config"

        self._create_directories()

    def _create_directories(self) -> None:
        self.charts_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.metrics_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.config_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def save_config(
        self,
        config: dict,
    ) -> None:

        config_path = self.config_dir / "config.json"

        with open(config_path, "w") as f:
            json.dump(
                config,
                f,
                indent=4,
            )
