from dataclasses import dataclass


@dataclass(slots=True)
class SpatialPoolerMetrics:
    active_column_count: int
    sparsity: float
