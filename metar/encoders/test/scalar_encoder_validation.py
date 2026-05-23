# metar/validation/scalar_encoder_validation.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

from htm.bindings.sdr import SDR

# ============================================================
# VALIDATION CONFIG
# ============================================================


@dataclass(frozen=True)
class ValidationConfig:
    sample_count: int = 720
    random_samples: int = 5000

    output_dir: Path = Path("./output/scalar_encoder_validation")

    pca_components: int = 2

    tsne_perplexity: int = 30
    tsne_random_state: int = 42

    umap_neighbors: int = 20
    umap_min_dist: float = 0.1
    umap_random_state: int = 42


# ============================================================
# VALIDATION SUITE
# ============================================================


class ScalarEncoderValidationSuite:
    """
    Validation framework for HTM ScalarEncoder topology analysis.

    This validates:
    - overlap continuity
    - SDR manifold geometry
    - topology smoothness
    - semantic continuity
    - sparsity stability
    - bit utilization
    """

    def __init__(
        self,
        encoder,
        variable_name: str,
        value_range: tuple[float, float],
        config: ValidationConfig | None = None,
    ):
        self.encoder = encoder
        self.variable_name = variable_name
        self.min_value = value_range[0]
        self.max_value = value_range[1]

        self.config = config or ValidationConfig()

        self.output_dir = self.config.output_dir / variable_name.lower()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.values = np.linspace(
            self.min_value,
            self.max_value,
            self.config.sample_count,
        )

        self.sdrs = self._generate_sdrs()

    # ========================================================
    # SDR GENERATION
    # ========================================================

    def _generate_sdrs(self) -> np.ndarray:
        dense_vectors = []

        for value in self.values:
            dense = self.encoder.encode_dense(float(value))
            dense_vectors.append(dense.astype(np.float32))

        return np.array(dense_vectors)

    # ========================================================
    # OVERLAP
    # ========================================================

    @staticmethod
    def normalized_overlap(a: np.ndarray, b: np.ndarray) -> float:
        intersection = np.sum(np.logical_and(a, b))
        active = max(np.sum(a), 1)

        return float(intersection / active)

    # ========================================================
    # JACCARD
    # ========================================================

    @staticmethod
    def jaccard_similarity(a: np.ndarray, b: np.ndarray) -> float:
        intersection = np.sum(np.logical_and(a, b))
        union = np.sum(np.logical_or(a, b))

        if union == 0:
            return 0.0

        return float(intersection / union)

    # ========================================================
    # 1. OVERLAP DECAY CURVE
    # ========================================================

    def plot_overlap_decay(self):
        anchor = self.sdrs[0]

        overlaps = []

        for sdr in self.sdrs:
            overlaps.append(self.normalized_overlap(anchor, sdr))

        plt.figure(figsize=(12, 6))

        plt.plot(self.values, overlaps)

        plt.title(f"{self.variable_name} Overlap Decay")

        plt.xlabel(self.variable_name)
        plt.ylabel("Normalized Overlap")

        plt.grid(True)

        plt.tight_layout()

        plt.savefig(self.output_dir / "overlap_decay.png")

        plt.close()

    # ========================================================
    # 2. SIMILARITY MATRIX
    # ========================================================

    def plot_similarity_matrix(self):
        n = len(self.sdrs)

        matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                matrix[i, j] = self.normalized_overlap(
                    self.sdrs[i],
                    self.sdrs[j],
                )

        plt.figure(figsize=(12, 10))

        plt.imshow(
            matrix,
            cmap="viridis",
            interpolation="nearest",
        )

        plt.colorbar(label="Similarity")

        plt.title(f"{self.variable_name} Similarity Matrix")

        plt.xlabel("Sample Index")
        plt.ylabel("Sample Index")

        plt.tight_layout()

        plt.savefig(self.output_dir / "similarity_matrix.png")

        plt.close()

    # ========================================================
    # 3. PCA PROJECTION
    # ========================================================

    def plot_pca_projection(self):
        pca = PCA(n_components=2)

        reduced = pca.fit_transform(self.sdrs)

        plt.figure(figsize=(12, 10))

        scatter = plt.scatter(
            reduced[:, 0],
            reduced[:, 1],
            c=self.values,
            cmap="viridis",
            s=25,
        )

        plt.colorbar(
            scatter,
            label=self.variable_name,
        )

        plt.title(f"{self.variable_name} PCA Projection")

        plt.tight_layout()

        plt.savefig(self.output_dir / "pca_projection.png")

        plt.close()

    # ========================================================
    # 4. TSNE
    # ========================================================

    def plot_tsne_projection(self):
        tsne = TSNE(
            n_components=2,
            perplexity=self.config.tsne_perplexity,
            random_state=self.config.tsne_random_state,
            init="pca",
        )

        reduced = np.asarray(tsne.fit_transform(self.sdrs))

        plt.figure(figsize=(12, 10))

        scatter = plt.scatter(
            reduced[:, 0],
            reduced[:, 1],
            c=self.values,
            cmap="viridis",
            s=25,
        )

        plt.colorbar(
            scatter,
            label=self.variable_name,
        )

        plt.title(f"{self.variable_name} t-SNE Projection")

        plt.tight_layout()

        plt.savefig(self.output_dir / "tsne_projection.png")

        plt.close()

    # ========================================================
    # 5. UMAP
    # ========================================================

    def plot_umap_projection(self):
        reducer = umap.UMAP(
            n_neighbors=self.config.umap_neighbors,
            min_dist=self.config.umap_min_dist,
            random_state=self.config.umap_random_state,
        )

        reduced = np.asarray(reducer.fit_transform(self.sdrs))

        plt.figure(figsize=(12, 10))

        scatter = plt.scatter(
            reduced[:, 0],
            reduced[:, 1],
            c=self.values,
            cmap="viridis",
            s=25,
        )

        plt.colorbar(
            scatter,
            label=self.variable_name,
        )

        plt.title(f"{self.variable_name} UMAP Projection")

        plt.tight_layout()

        plt.savefig(self.output_dir / "umap_projection.png")

        plt.close()

    # ========================================================
    # 6. ACTIVE BIT DISTRIBUTION
    # ========================================================

    def plot_active_bit_distribution(self):
        active_counts = np.sum(self.sdrs, axis=1)

        plt.figure(figsize=(12, 6))

        plt.hist(
            active_counts,
            bins=30,
        )

        plt.title(f"{self.variable_name} Active Bit Distribution")

        plt.xlabel("Active Bits")
        plt.ylabel("Frequency")

        plt.tight_layout()

        plt.savefig(self.output_dir / "active_bit_distribution.png")

        plt.close()

    # ========================================================
    # 7. BIT UTILIZATION
    # ========================================================

    def plot_bit_utilization(self):
        utilization = np.sum(self.sdrs, axis=0)

        plt.figure(figsize=(16, 6))

        plt.bar(
            np.arange(len(utilization)),
            utilization,
            width=1.0,
        )

        plt.title(f"{self.variable_name} Bit Utilization")

        plt.xlabel("Bit Index")
        plt.ylabel("Activation Count")

        plt.tight_layout()

        plt.savefig(self.output_dir / "bit_utilization.png")

        plt.close()

    # ========================================================
    # 8. JACCARD DISTRIBUTION
    # ========================================================

    def plot_jaccard_distribution(self):
        similarities = []

        rng = np.random.default_rng(42)

        for _ in range(self.config.random_samples):
            i = rng.integers(0, len(self.sdrs))
            j = rng.integers(0, len(self.sdrs))

            sim = self.jaccard_similarity(
                self.sdrs[i],
                self.sdrs[j],
            )

            similarities.append(sim)

        plt.figure(figsize=(12, 6))

        plt.hist(
            similarities,
            bins=50,
        )

        plt.title(f"{self.variable_name} Jaccard Distribution")

        plt.xlabel("Jaccard Similarity")
        plt.ylabel("Frequency")

        plt.tight_layout()

        plt.savefig(self.output_dir / "jaccard_distribution.png")

        plt.close()

    # ========================================================
    # 9. DISTANCE vs OVERLAP
    # ========================================================

    def plot_distance_vs_overlap(self):
        distances = []
        overlaps = []

        for i in range(len(self.values)):
            for j in range(i + 1, len(self.values)):
                physical_distance = abs(self.values[i] - self.values[j])

                overlap = self.normalized_overlap(
                    self.sdrs[i],
                    self.sdrs[j],
                )

                distances.append(physical_distance)
                overlaps.append(overlap)

        plt.figure(figsize=(12, 6))

        plt.scatter(
            distances,
            overlaps,
            s=4,
            alpha=0.3,
        )

        plt.title(f"{self.variable_name} Distance vs Overlap")

        plt.xlabel("Physical Distance")
        plt.ylabel("SDR Overlap")

        plt.tight_layout()

        plt.savefig(self.output_dir / "distance_vs_overlap.png")

        plt.close()

    # ========================================================
    # 10. TEMPORAL TRAJECTORY
    # ========================================================

    def plot_temporal_trajectory(self):
        trajectory = []

        for i in range(1, len(self.sdrs)):
            overlap = self.normalized_overlap(
                self.sdrs[i - 1],
                self.sdrs[i],
            )

            trajectory.append(overlap)

        plt.figure(figsize=(12, 6))

        plt.plot(
            trajectory,
        )

        plt.title(f"{self.variable_name} Temporal Trajectory")

        plt.xlabel("Step")
        plt.ylabel("Sequential Overlap")

        plt.grid(True)

        plt.tight_layout()

        plt.savefig(self.output_dir / "temporal_trajectory.png")

        plt.close()

    # ========================================================
    # RUN ALL
    # ========================================================

    def run_all(self):
        print(f"Running validation for {self.variable_name}")

        self.plot_overlap_decay()
        self.plot_similarity_matrix()
        self.plot_pca_projection()
        self.plot_tsne_projection()
        self.plot_umap_projection()
        self.plot_active_bit_distribution()
        self.plot_bit_utilization()
        self.plot_jaccard_distribution()
        self.plot_distance_vs_overlap()
        self.plot_temporal_trajectory()

        print(f"Validation complete: {self.output_dir}")


# ============================================================
# SINGLE VARIABLE VALIDATION
# ============================================================


def validate_single_encoder(
    *,
    encoder,
    variable_name: str,
    value_range: tuple[float, float],
):
    """
    Validate a single scalar encoder independently.

    Example:
        validate_single_encoder(
            encoder=pressure_encoder,
            variable_name="pressure",
            value_range=(870, 1085),
        )
    """

    suite = ScalarEncoderValidationSuite(
        encoder=encoder,
        variable_name=variable_name,
        value_range=value_range,
    )

    suite.run_all()
