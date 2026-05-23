from htm.bindings.algorithms import TemporalMemory
from htm.bindings.sdr import SDR

from temporal_memory.config import TM_CONFIG

from temporal_memory.metrics import (
    TemporalMemoryMetrics,
)


class TemporalMemoryManager:
    def __init__(self, column_count: int) -> None:
        self.column_count = column_count
        self.cells_per_column = TM_CONFIG["cells_per_column"]
        self.tm = TemporalMemory(
            columnDimensions=(column_count,),
            cellsPerColumn=self.cells_per_column,
            activationThreshold=TM_CONFIG["activation_threshold"],
            initialPermanence=TM_CONFIG["initial_permanence"],
            connectedPermanence=TM_CONFIG["connected_permanence"],
            minThreshold=TM_CONFIG["min_threshold"],
            maxNewSynapseCount=TM_CONFIG["max_new_synapse_count"],
            permanenceIncrement=TM_CONFIG["permanence_increment"],
            permanenceDecrement=TM_CONFIG["permanence_decrement"],
            predictedSegmentDecrement=TM_CONFIG["predicted_segment_decrement"],
            maxSegmentsPerCell=TM_CONFIG["max_segments_per_cell"],
            maxSynapsesPerSegment=TM_CONFIG["max_synapses_per_segment"],
            seed=TM_CONFIG["seed"],
        )
        self.active_cells = SDR((column_count * self.cells_per_column,))
        self.predictive_cells = SDR((column_count * self.cells_per_column,))

    def compute(
        self,
        active_columns: SDR,
        learn: bool = True,
    ) -> None:
        self.tm.compute(
            active_columns,
            learn,
        )
        self.tm.activateDendrites(
            learn,
        )
        self.active_cells = self.tm.getActiveCells()
        self.predictive_cells = self.tm.getPredictiveCells()

    def metrics(self) -> TemporalMemoryMetrics:
        active_cell_count = len(self.active_cells.sparse)

        predictive_cell_count = len(self.predictive_cells.sparse)

        anomaly_score = self.tm.anomaly

        return TemporalMemoryMetrics(
            active_cells=active_cell_count,
            predictive_cells=predictive_cell_count,
            anomaly_score=anomaly_score,
        )
