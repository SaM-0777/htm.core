from __future__ import annotations

from pathlib import Path
import itertools

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

from encoders.cloud_height_encoder import CloudHeightEncoder


class CloudHeightEncoderValidator:
    """
    Comprehensive validation for CloudHeightEncoder with all reasonable brackets.
    """

    def __init__(self):
        self.encoder = CloudHeightEncoder()
        self.output_dir = Path("./output/cloud_height_validation")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Dense sampling in critical low altitude ranges
        self.heights = (
            list(range(0, 1000, 50))  # Extremely low
            + list(range(1000, 3000, 100))  # Very Low
            + list(range(3000, 6000, 200))  # Medium-Low
            + list(range(6000, 12000, 500))  # Medium to High
            + list(range(12000, 30000, 2000))  # High to Very High
        )

        self.sdrs = np.array(
            [self.encoder.encode(h).dense.astype(np.float32) for h in self.heights]
        )

    @staticmethod
    def overlap(a: np.ndarray, b: np.ndarray) -> float:
        intersection = np.sum(np.logical_and(a, b))
        active = max(np.sum(a), 1)
        return float(intersection / active)

    # =====================================================
    # COMPREHENSIVE SEMANTIC TESTS (All Reasonable Brackets)
    # =====================================================

    def semantic_tests(self):
        print()
        print("=" * 90)
        print("CLOUD HEIGHT SEMANTIC OVERLAP TESTS - ALL BRACKETS")
        print("=" * 90)
        print()

        test_pairs = [
            # Within same bracket
            (200, 400),
            (800, 1200),
            (2500, 2800),
            (4500, 5200),
            (9000, 11000),
            (18000, 22000),
            # Adjacent brackets
            (600, 1400),
            (1800, 2500),
            (3500, 4800),
            (6500, 8500),
            (13000, 17000),
            # Distant brackets
            (300, 4500),
            (800, 9000),
            (1200, 15000),
            (500, 20000),
            (100, 25000),
        ]

        for a, b in test_pairs:
            overlap_val = self.overlap(
                self.encoder.encode(a).dense, self.encoder.encode(b).dense
            )
            print(f"{a:>5} ft <-> {b:>5} ft | Overlap = {overlap_val:.4f}")

        print()
        print("-" * 60)
        print("KEY EXPECTATIONS:")
        print("• Very close heights     → High overlap (> 0.7)")
        print("• Adjacent brackets      → Medium-High overlap (0.4 - 0.7)")
        print("• Distant brackets       → Low overlap (< 0.3)")
        print("=" * 90)
        print()

    # =====================================================
    # LOW CLOUD SENSITIVITY (Critical for Aviation)
    # =====================================================

    def low_cloud_sensitivity_tests(self):
        print()
        print("=" * 80)
        print("LOW CLOUD SENSITIVITY TESTS (Most Critical Range)")
        print("=" * 80)
        print()

        critical_tests = [
            (100, 200),
            (200, 400),
            (400, 700),
            (700, 1200),
            (1200, 1800),
            (1800, 2500),
            (2500, 4000),
            (4000, 6000),
            (100, 5000),
            (500, 12000),
        ]

        for a, b in critical_tests:
            ov = self.overlap(
                self.encoder.encode(a).dense, self.encoder.encode(b).dense
            )
            print(f"{a:>5} ft <-> {b:>5} ft | Overlap = {ov:.4f}")

        print()

    # =====================================================
    # REST OF THE PLOTS (kept similar to your original)
    # =====================================================

    def plot_similarity_matrix(self):
        n = len(self.sdrs)
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                matrix[i, j] = self.overlap(self.sdrs[i], self.sdrs[j])

        plt.figure(figsize=(12, 10))
        plt.imshow(matrix, cmap="viridis", interpolation="nearest")
        plt.colorbar(label="Normalized Overlap")
        plt.title("Cloud Height Similarity Matrix")
        plt.tight_layout()
        plt.savefig(self.output_dir / "similarity_matrix.png")
        plt.close()

    def plot_overlap_decay(self):
        anchors = [300, 1000, 3000, 8000, 20000]
        plt.figure(figsize=(13, 7))

        for anchor in anchors:
            anchor_sdr = self.encoder.encode(anchor).dense
            overlaps = [
                self.overlap(anchor_sdr, self.encoder.encode(h).dense)
                for h in self.heights
            ]
            plt.plot(self.heights, overlaps, label=f"{anchor} ft")

        plt.title("Cloud Height Overlap Decay Curves")
        plt.xlabel("Cloud Height (feet)")
        plt.ylabel("Normalized Overlap")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.output_dir / "overlap_decay.png")
        plt.close()

    def plot_pca(self):
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(self.sdrs)

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            reduced[:, 0], reduced[:, 1], c=self.heights, cmap="viridis", s=30
        )
        plt.colorbar(scatter, label="Height (ft)")
        plt.title("Cloud Height PCA Projection")
        plt.tight_layout()
        plt.savefig(self.output_dir / "pca_projection.png")
        plt.close()

    def plot_tsne(self):
        tsne = TSNE(n_components=2, perplexity=15, random_state=42, init="pca")
        reduced = tsne.fit_transform(self.sdrs)

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            reduced[:, 0], reduced[:, 1], c=self.heights, cmap="viridis", s=30
        )
        plt.colorbar(scatter, label="Height (ft)")
        plt.title("Cloud Height t-SNE Projection")
        plt.tight_layout()
        plt.savefig(self.output_dir / "tsne_projection.png")
        plt.close()

    def plot_umap(self):
        reducer = umap.UMAP(n_neighbors=12, min_dist=0.1, random_state=42)
        reduced = reducer.fit_transform(self.sdrs)

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            reduced[:, 0], reduced[:, 1], c=self.heights, cmap="viridis", s=30
        )
        plt.colorbar(scatter, label="Height (ft)")
        plt.title("Cloud Height UMAP Projection")
        plt.tight_layout()
        plt.savefig(self.output_dir / "umap_projection.png")
        plt.close()

    def print_active_bits(self):
        print()
        print("=" * 80)
        print("ACTIVE BIT STABILITY")
        print("=" * 80)
        print()
        for h in [100, 800, 2500, 5200, 11000, 25000]:
            sdr = self.encoder.encode(h)
            print(f"{h:>6} ft  →  {len(sdr.sparse)} active bits")
        print()

    def run_all(self):
        print("=" * 90)
        print("CLOUD HEIGHT ENCODER VALIDATION STARTED")
        print("=" * 90)

        self.plot_similarity_matrix()
        self.plot_overlap_decay()
        self.plot_pca()
        self.plot_tsne()
        self.plot_umap()
        self.print_active_bits()
        self.low_cloud_sensitivity_tests()
        self.semantic_tests()

        print("=" * 90)
        print("CLOUD HEIGHT VALIDATION COMPLETE")
        print("=" * 90)


if __name__ == "__main__":
    validator = CloudHeightEncoderValidator()
    validator.run_all()
