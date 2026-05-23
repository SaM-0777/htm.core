from dataclasses import dataclass, asdict


@dataclass(slots=True)
class PipelineMetrics:
    timestamp: str
    temperature: float

    sp_active_columns: int
    sp_sparsity: float

    tm_active_cells: int
    tm_predictive_cells: int

    anomaly_score: float

    sp_entropy: float

    sp_mean_overlap: float

    sp_dead_column_ratio: float

    sp_max_duty_cycle: float

    sp_mean_duty_cycle: float


class MetricsCollector:
    def __init__(self) -> None:
        self.history: list[PipelineMetrics] = []

    def add(
        self,
        metrics: PipelineMetrics,
    ) -> None:
        self.history.append(metrics)

    def export(self):
        return [asdict(metric) for metric in self.history]
