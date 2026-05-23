# from datetime import datetime

# from model.metar import MetarRecord
# from encoders.metar_encoder import MetarEncoder

from pathlib import Path

from pipeline.metar_pipeline import (
    MetarPipeline,
)
from visualization.dashboard import (
    VisualizationDashboard,
)
from spatial_pooler.config import SP_CONFIG
from temporal_memory.config import TM_CONFIG
from utils.run_manager import RunManager

DATASET_PATH = Path("./data/metar.json")


def main(epochs: int = 1) -> None:

    pipeline = MetarPipeline()

    records = pipeline.load_dataset(DATASET_PATH)

    print()
    print(f"Loaded {len(records)} rows")
    print()

    for epoch in range(epochs):
        print(f"Epoch {epoch}")

        for index, record in enumerate(records):

            (
                columns,
                active_cells,
                predictive_cells,
                sp_metrics,
                tm_metrics,
                metrics,
                diagnostics,
            ) = pipeline.process(record)

            # print(
            #    f"[{index}] " f"{record.recordedTime} | " f"Temp={record.temperature}C"
            # )

            # print()

            # print(f"SP Active Columns: " f"{sp_metrics.active_column_count}")

            # print(f"SP Sparsity: " f"{sp_metrics.sparsity:.4f}")

            # print()

            # print(f"TM Active Cells: " f"{tm_metrics.active_cells}")

            # print(f"TM Predictive Cells: " f"{tm_metrics.predictive_cells}")

            # print(f"TM Anomaly Score: " f"{tm_metrics.anomaly_score:.4f}")

            # print()

            # print(f"Predictive Cell Sample: " f"{predictive_cells.sparse[:20]}")

            # print("-" * 80)

    run_manager = RunManager(
        experiment_name="eglc_temperature",
        epochs=epochs,
    )
    dashboard = VisualizationDashboard(run_manager.charts_dir)
    metrics_history = pipeline.metrics_collector.export()

    dashboard.build(metrics_history)

    run_manager.save_config(
        {
            "epochs": epochs,
            "airport": "EGLC",
            "sp_columns": SP_CONFIG["column_count"],
            "tm_cells_per_column": TM_CONFIG["cells_per_column"],
        }
    )

    print()
    print("Visualization export complete")
    print()


if __name__ == "__main__":
    main(5)

# """
# METAR Row
#    ↓
# Encoders
#    ↓
# SDR
#    ↓
# Spatial Pooler
#    ↓
# Temporal Memory
#    ↓
# Prediction
# """

# record = MetarRecord(
#    cityName="London City Airport Station",
#    icaoId="EGLC",
#    dataProvider="Aviation Weather",
#    name="London City Arpt, EN, GB",
#    temperature=11,
#    rawMetarCode="METAR EGLC 170450Z AUTO 26011KT 9999 BKN014 OVC020 11/08 Q1008",
#    recordedTime="2026-05-17T05:50",
#    updatedAt=datetime.utcnow(),
# )

# encoder = MetarEncoder()

# encoded_sdr = encoder.encode(record)

# print("SDR Dimensions:")
# print(encoded_sdr.dimensions)

# print()
# print("Active Bits:")
# print(encoded_sdr.sparse)
