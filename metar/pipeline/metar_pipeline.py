import json

from pathlib import Path
from datetime import datetime

from model.metar import MetarRecord

from encoders.metar_encoder import (
    MetarEncoder,
)

from spatial_pooler.manager import (
    SpatialPoolerManager,
)

from temporal_memory.manager import (
    TemporalMemoryManager,
)

from visualization.metrics_collector import (
    MetricsCollector,
    PipelineMetrics,
)


class MetarPipeline:
    def __init__(self) -> None:
        self.encoder = MetarEncoder()

        self.spatial_pooler = SpatialPoolerManager(
            input_width=self.encoder.output_width
        )

        self.temporal_memory = TemporalMemoryManager(
            column_count=self.spatial_pooler.column_count
        )

        self.metrics_collector = MetricsCollector()

    def load_dataset(self, path: Path) -> list[MetarRecord]:
        with open(path, "r") as f:
            rows = json.load(f)

        records: list[MetarRecord] = []

        rows = [row for row in rows if row["icao_id"] == "EGLC"]
        rows.sort(key=lambda row: datetime.fromisoformat(row["recorded_time"]))

        seen = set()

        for row in rows:
            ts = row["recorded_time"]

            if ts not in seen:
                seen.add(ts)
                records.append(
                    MetarRecord(
                        cityName=row["city_name"],
                        icaoId=row["icao_id"],
                        dataProvider=row["data_provider"],
                        name=row["name"],
                        temperature=row["temperature"],
                        rawMetarCode=row["raw_metar_code"],
                        recordedTime=row["recorded_time"],
                        updatedAt=row["updated_at"],
                    )
                )

        return records

    def process(self, record: MetarRecord):
        encoded_sdr = self.encoder.encode(record)
        active_columns = self.spatial_pooler.compute(
            encoded_sdr,
            learn=True,
        )
        self.temporal_memory.compute(
            active_columns,
            learn=True,
        )
        sp_metrics = self.spatial_pooler.metrics()
        tm_metrics = self.temporal_memory.metrics()
        metrics = self.spatial_pooler.metrics()
        diagnostics = self.spatial_pooler.diagnostic_metrics()
        self.metrics_collector.add(
            PipelineMetrics(
                timestamp=record.recordedTime,
                temperature=record.temperature,
                sp_active_columns=sp_metrics.active_column_count,
                sp_sparsity=sp_metrics.sparsity,
                tm_active_cells=tm_metrics.active_cells,
                tm_predictive_cells=tm_metrics.predictive_cells,
                anomaly_score=tm_metrics.anomaly_score,
                sp_entropy=diagnostics["entropy"],
                sp_mean_overlap=diagnostics["mean_overlap"],
                sp_dead_column_ratio=diagnostics["dead_column_ratio"],
                sp_max_duty_cycle=diagnostics["max_duty_cycle"],
                sp_mean_duty_cycle=diagnostics["mean_duty_cycle"],
            )
        )
        return (
            active_columns,
            self.temporal_memory.active_cells,
            self.temporal_memory.predictive_cells,
            sp_metrics,
            tm_metrics,
            metrics,
            diagnostics,
        )
