import math
import numpy as np

from htm.bindings.sdr import SDR


class SpatialPoolerDiagnostics:
    def __init__(self, column_count: int) -> None:

        self.column_count = column_count

        self.activation_counts = np.zeros(
            column_count,
            dtype=np.int32,
        )

        self.total_steps = 0

        self.previous_active_set: set[int] | None = None

        self.overlap_history: list[float] = []

    def update(self, active_sdr: SDR) -> None:

        active_columns = active_sdr.sparse.tolist()

        for column in active_columns:
            self.activation_counts[column] += 1

        self._update_overlap(active_columns)

        self.total_steps += 1

    def _update_overlap(
        self,
        active_columns: list[int],
    ) -> None:

        current_set = set(active_columns)

        if self.previous_active_set is not None:

            intersection = len(current_set.intersection(self.previous_active_set))

            union = len(current_set.union(self.previous_active_set))

            overlap = intersection / union if union > 0 else 0.0

            self.overlap_history.append(overlap)

        self.previous_active_set = current_set

    def entropy(self) -> float:

        if self.total_steps == 0:
            return 0.0

        probabilities = self.activation_counts / self.activation_counts.sum()

        probabilities = probabilities[probabilities > 0]

        entropy = -np.sum(probabilities * np.log2(probabilities))

        return float(entropy)

    def mean_overlap(self) -> float:

        if not self.overlap_history:
            return 0.0

        return float(np.mean(self.overlap_history))

    def dead_column_ratio(self) -> float:

        dead_columns = np.sum(self.activation_counts == 0)

        return float(dead_columns / self.column_count)

    def max_duty_cycle(self) -> float:

        if self.total_steps == 0:
            return 0.0

        return float(np.max(self.activation_counts) / self.total_steps)

    def mean_duty_cycle(self) -> float:

        if self.total_steps == 0:
            return 0.0

        return float(np.mean(self.activation_counts) / self.total_steps)

    def metrics(self) -> dict:

        return {
            "entropy": self.entropy(),
            "mean_overlap": self.mean_overlap(),
            "dead_column_ratio": self.dead_column_ratio(),
            "max_duty_cycle": self.max_duty_cycle(),
            "mean_duty_cycle": self.mean_duty_cycle(),
        }
