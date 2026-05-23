from htm.bindings.algorithms import SpatialPooler
from htm.bindings.sdr import SDR

from spatial_pooler.config import SP_CONFIG
from spatial_pooler.metrics import (
    SpatialPoolerMetrics,
)
from spatial_pooler.diagnostics import (
    SpatialPoolerDiagnostics,
)


class SpatialPoolerManager:
    def __init__(
        self,
        input_width: int,
    ) -> None:
        self.input_width = input_width

        self.column_count = SP_CONFIG["column_count"]

        self.sp = SpatialPooler(
            inputDimensions=(input_width,),
            columnDimensions=(self.column_count,),
            potentialPct=SP_CONFIG["potential_pct"],
            potentialRadius=input_width,
            globalInhibition=SP_CONFIG["global_inhibition"],
            localAreaDensity=SP_CONFIG["local_area_density"],
            numActiveColumnsPerInhArea=SP_CONFIG["active_columns"],
            synPermInactiveDec=SP_CONFIG["syn_perm_inactive_dec"],
            synPermActiveInc=SP_CONFIG["syn_perm_active_inc"],
            synPermConnected=SP_CONFIG["syn_perm_connected"],
            boostStrength=SP_CONFIG["boost_strength"],
            seed=SP_CONFIG["seed"],
            wrapAround=SP_CONFIG["wrap_around"],
        )

        self.output_sdr = SDR((self.column_count,))

        self.diagnostics = SpatialPoolerDiagnostics(column_count=self.column_count)

    def compute(self, input_sdr: SDR, learn: bool = True) -> SDR:
        self.sp.compute(input_sdr, learn, self.output_sdr)
        self.diagnostics.update(self.output_sdr)
        return self.output_sdr

    def metrics(
        self,
    ) -> SpatialPoolerMetrics:
        active_count = len(self.output_sdr.sparse)

        sparsity = active_count / self.column_count

        return SpatialPoolerMetrics(active_column_count=active_count, sparsity=sparsity)

    def diagnostic_metrics(self) -> dict:
        return self.diagnostics.metrics()
