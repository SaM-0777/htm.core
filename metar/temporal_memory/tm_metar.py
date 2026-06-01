from dataclasses import dataclass


@dataclass(frozen=True)
class TemporalMemoryMetarConfig:
    activation_threshold = 12
    cells_per_column = 16
    initial_perm = 0.21
    max_cegments_per_cell = 128
    max_synapses_per_segment = 64
    min_threshold = 10
    new_synapse_count = 32
    permanence_dec = 0.1
    permanence_inc = 0.1
    seed = 42
    
    initial_permanence= 0.21
    connected_permanence = 0.50
    max_new_synapse_count = 20


class TemporalMemoryMetar:
    def __init__(self, config: TemporalMemoryMetarConfig | None) -> None:
        self.config = config or TemporalMemoryMetarConfig()
        
