from __future__ import annotations

import numpy as np
from collections import Counter
from dataclasses import dataclass
from htm.bindings.sdr import SDR
from htm.bindings.algorithms import SpatialPooler


@dataclass(frozen=True)
class SpatialPoolerMetarConfig:
    column_count = 5664  # prev -> 4096
    potential_pct: float = 0.70  # prev -> 0.7, 0.3
    # potential_radius: int = 1024  # prev -> 4932, 1024, 512
    global_inhibition: bool = True
    local_density: float = 0.04
    active_columns_per_inh_area: int = int(0.02 * 5664)  # prev -> 128
    syn_perm_connected: float = 0.15
    syn_perm_active_inc: float = 0.03
    syn_perm_inactive_dec: float = 0.006
    boost_strength: float = 4.0  # prev -> 1.5, 4.0
    # duty_cycle_period: int = 1000
    # min_pct_overlap_duty_cycle: float = 0.001
    # stimulus_threshold: int = 4
    # seed: int = 42
    wrapAround = True,


class SpatialPoolerMetar:
    def __init__(
        self,
        input_size: int,
        config: SpatialPoolerMetarConfig | None = None,
    ) -> None:
        self.config = config or SpatialPoolerMetarConfig()
        self.input_size = input_size

        avg_potential_synapses = int(self.input_size * self.config.potential_pct)
        estimated_total_synapses = avg_potential_synapses * self.config.column_count
        output_sparsity = (
            self.config.active_columns_per_inh_area / self.config.column_count
        )

        print(f"Input SDR Size              : {self.input_size:,}")
        print(f"Column Count                : {self.config.column_count:,}")
        print(
            f"Active Columns              : "
            f"{self.config.active_columns_per_inh_area:,}"
        )
        print(
            f"Output Sparsity             : "
            f"{output_sparsity:.4f} "
            f"({output_sparsity * 100:.2f}%)"
        )
        # print(f"Potential Radius            : " f"{self.config.potential_radius:,}")
        print(f"Potential Percent           : " f"{self.config.potential_pct:.2f}")
        print(f"Avg Potential Synapses/Col  : " f"{avg_potential_synapses:,}")
        print(f"Estimated Total Synapses    : " f"{estimated_total_synapses:,}")
        print(f"Global Inhibition           : " f"{self.config.global_inhibition}")
        print(
            f"Active Columns/Inh Area     : "
            f"{self.config.active_columns_per_inh_area}"
        )
        print(f"Local Area Density          : " f"{self.config.local_density}")
        print(f"Syn Perm Connected          : " f"{self.config.syn_perm_connected}")
        print(f"Syn Perm Active Inc         : " f"{self.config.syn_perm_active_inc}")
        print(f"Syn Perm Inactive Dec       : " f"{self.config.syn_perm_inactive_dec}")
        print(f"Boost Strength              : " f"{self.config.boost_strength}")
        # print(f"Duty Cycle Period           : " f"{self.config.duty_cycle_period}")
        # print(
        #    f"Min Overlap Duty Cycle      : "
        #    f"{self.config.min_pct_overlap_duty_cycle}"
        # )
        # print(f"Stimulus Threshold          : " f"{self.config.stimulus_threshold}")

        self.sp = SpatialPooler(
            inputDimensions=(self.input_size,),
            columnDimensions=(self.config.column_count,),
            potentialPct=self.config.potential_pct,
            potentialRadius=self.input_size,
            globalInhibition=self.config.global_inhibition,
            localAreaDensity=self.config.local_density,
            #numActiveColumnsPerInhArea=(self.config.active_columns_per_inh_area),
            synPermInactiveDec=self.config.syn_perm_inactive_dec,
            synPermConnected=self.config.syn_perm_connected,
            synPermActiveInc=self.config.syn_perm_active_inc,
            boostStrength=self.config.boost_strength,
            # dutyCyclePeriod=(self.config.duty_cycle_period),
            # minPctOverlapDutyCycle=(self.config.min_pct_overlap_duty_cycle),
            # stimulusThreshold=(self.config.stimulus_threshold),
            # seed=self.config.seed,
            wrapAround=True,
        )

        self.output_sdr = SDR(self.config.column_count)
        self.column_usage = Counter()
        self.active_counts: list[int] = []
        self.temporal_overlaps: list[float] = []
        self.prev_output: SDR | None = None

    @property
    def output_size(self) -> int:
        return self.config.column_count

    @property
    def output_dimensions(
        self,
    ) -> tuple[int]:
        return (self.config.column_count,)

    def compute(self, input_sdr: SDR, learn: bool = True):
        self.sp.compute(input_sdr, learn, self.output_sdr)
        result = SDR(self.output_sdr.dimensions)
        result.setSDR(self.output_sdr)
        self.update_metrices(result)

        return result

    def update_metrices(self, output_sdr: SDR):
        active_count = len(output_sdr.sparse)
        self.active_counts.append(active_count)

        for c in output_sdr.sparse:
            self.column_usage[c] += 1

        if self.prev_output is not None:
            overlap = len(set(self.prev_output.sparse) & set(self.output_sdr.sparse))
            overlap /= max(
                active_count,
                1,
            )
            self.temporal_overlaps.append(overlap)

        self.prev_output = output_sdr

    def diagnostics(
        self,
    ):
        usage = np.zeros(
            self.config.column_count,
            dtype=np.float32,
        )

        for k, v in self.column_usage.items():
            usage[k] = v

        total = np.sum(usage)

        if total > 0:
            probs = usage / total
            probs = probs[probs > 0]

            entropy = float(-np.sum(probs * np.log2(probs)))

        else:
            entropy = 0

        dead_columns = self.config.column_count - len(self.column_usage)
        dead_ratio = dead_columns / self.config.column_count

        mean_overlap = (
            float(np.mean(self.temporal_overlaps)) if self.temporal_overlaps else 0
        )
        mean_active = float(np.mean(self.active_counts)) if self.active_counts else 0
        max_usage = float(np.max(usage)) if total > 0 else 0.0

        return {
            "entropy": entropy,
            "dead_column_ratio": dead_ratio,
            "mean_temporal_overlap": mean_overlap,
            "mean_active_columns": mean_active,
            "max_column_usage": max_usage,
        }

    def reset_metrics(
        self,
    ) -> None:
        self.column_usage.clear()
        self.active_counts.clear()
        self.temporal_overlaps.clear()
        self.previous_output = None
