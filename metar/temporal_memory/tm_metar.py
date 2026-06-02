from collections import Counter
from dataclasses import dataclass
from htm.bindings.algorithms import TemporalMemory
from htm.bindings.sdr import SDR
import numpy as np
from metar.spatial_pooler.sp_metar import SpatialPoolerMetarConfig


@dataclass(frozen=True)
class TemporalMemoryMetarConfig:
    activation_threshold = 12
    cells_per_column = 16
    initial_permanence = 0.21
    max_segments_per_cell = 128
    max_synapses_per_segment = 64
    min_threshold = 10
    max_new_synapse_count = 20  # prev 32
    permanence_dec = 0.1
    permanence_inc = 0.1
    seed = 42

    connected_permanence = 0.50


class TemporalMemoryMetar:
    def __init__(
        self,
        sp_config: SpatialPoolerMetarConfig,
        config: TemporalMemoryMetarConfig | None,
    ) -> None:
        self.config = config or TemporalMemoryMetarConfig()
        self.sp_config = sp_config

        self.tm = TemporalMemory(
            columnDimensions=(self.sp_config.column_count,),
            cellsPerColumn=self.config.cells_per_column,
            activationThreshold=self.config.activation_threshold,
            initialPermanence=self.config.initial_permanence,
            connectedPermanence=self.config.connected_permanence,
            minThreshold=self.config.min_threshold,
            maxNewSynapseCount=self.config.max_new_synapse_count,
            permanenceIncrement=self.config.permanence_inc,
            permanenceDecrement=self.config.permanence_dec,
            predictedSegmentDecrement=0.0,
            maxSegmentsPerCell=self.config.max_segments_per_cell,
            maxSynapsesPerSegment=self.config.max_synapses_per_segment,
        )

        self.total_cells = self.sp_config.column_count * self.config.cells_per_column
        self.active_cells = SDR(self.total_cells)
        self.predictive_cells = SDR(self.total_cells)
        self.winner_cells = SDR(self.total_cells)
        self.prev_predictive: SDR | None = None
        self.burst_rates: list[float] = []
        self.prediction_overlaps: list[float] = []
        self.active_cell_counts: list[int] = []
        self.predictive_cell_counts: list[int] = []
        self.cell_usage = Counter()

        self._print_summary()

    def compute(self, active_columns: SDR, learn: bool = True) -> SDR:
        self.tm.compute(active_columns, learn=learn)
        self.active_cells.sparse = self.tm.getActiveCells().sparse
        #self.predictive_cells.sparse = self.tm.getPredictiveCells().sparse
        self.winner_cells.sparse = self.tm.getWinnerCells().sparse

        self._update_metrics(
            active_columns,
        )

        result = SDR(self.active_cells.dimensions)
        result.setSDR(self.active_cells)

        return result

    def diagnostics(
        self,
    ):
        usage = np.zeros(self.total_cells, dtype=np.float32)

        for k, v in self.cell_usage.items():
            usage[k] = v
        total = usage.sum()

        if total > 0:
            probs = usage / total
            probs = probs[probs > 0]
            entropy = float(-np.sum(probs * np.log2(probs)))
        else:
            entropy = 0.0

        dead_cells = self.total_cells - len(self.cell_usage)

        return {
            "entropy": entropy,
            "dead_cell_ratio": dead_cells / self.total_cells,
            "mean_burst_rate": (
                float(np.mean(self.burst_rates)) if self.burst_rates else 0.0
            ),
            "mean_prediction_overlap": (
                float(np.mean(self.prediction_overlaps))
                if self.prediction_overlaps
                else 0.0
            ),
            "mean_active_cells": (
                float(np.mean(self.active_cell_counts))
                if self.active_cell_counts
                else 0.0
            ),
            "mean_predictive_cells": (
                float(np.mean(self.predictive_cell_counts))
                if self.predictive_cell_counts
                else 0.0
            ),
        }

    def reset_metrics(self) -> None:
        self.burst_rates.clear()
        self.prediction_overlaps.clear()
        self.active_cell_counts.clear()
        self.predictive_cell_counts.clear()
        self.cell_usage.clear()
        self.prev_predictive = None

    def _update_metrics(self, active_columns: SDR):
        active_count = len(self.active_cells.sparse)
        predictive_count = len(self.predictive_cells.sparse)
        self.active_cell_counts.append(active_count)
        self.predictive_cell_counts.append(predictive_count)

        for cell in self.active_cells.sparse:
            self.cell_usage[cell] += 1

        bursting_columns = active_count / self.config.cells_per_column
        active_columns_count = len(active_columns.sparse)
        burst_rate = bursting_columns / max(active_columns_count, 1)
        self.burst_rates.append(burst_rate)

        if self.prev_predictive is not None:
            predicted_columns = {
                cell // self.config.cells_per_column
                for cell in self.prev_predictive.sparse
            }
            active_columns_set = {
                cell // self.config.cells_per_column
                for cell in self.active_cells.sparse
            }
            overlap = len(set(predicted_columns) & set(active_columns_set))
            denom = max(len(active_columns_set), 1)
            self.prediction_overlaps.append(overlap / denom)

        self.prev_predictive = SDR(self.predictive_cells.dimensions)
        new_predictive = SDR(self.predictive_cells.dimensions)
        new_predictive.setSDR(self.predictive_cells)
        self.prev_predictive = new_predictive

    def _print_summary(self) -> None:
        print("TEMPORAL MEMORY SUMMARY")

        print(f"Input Columns         : {self.sp_config.column_count:,}")
        print(f"Cells Per Column      : {self.config.cells_per_column}")
        print(f"Total Cells           : {self.total_cells:,}")
        print(f"Potential Cell States : " f"{self.total_cells:,}")
