from dataclasses import dataclass


@dataclass(slots=True)
class TemporalMemoryMetrics:
    active_cells: int
    predictive_cells: int
    anomaly_score: float
