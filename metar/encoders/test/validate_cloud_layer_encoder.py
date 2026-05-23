from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

from encoders.cloud_layer_encoder import CloudLayerEncoder


class CloudLayerEncoderValidator:
    """
    Comprehensive validation for CloudLayerEncoder (512 bits).
    """

    def __init__(self):
        self.encoder = CloudLayerEncoder()
        self.output_dir = Path("./output/cloud_layer_validation")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Test cases: (coverage, height_feet, cloud_type)
        self.test_layers = [
            ("CLR", 18000, "CI"),
            ("FEW", 12000, "AC"),
            ("SCT", 6500, "SC"),
            ("BKN", 4200, "ST"),
            ("BKN", 2800, "NS"),
            ("OVC", 1200, "TCU"),
            ("OVC", 800, "CB"),
            ("BKN", 3500, "CB"),
            ("FEW", 25000, "CI"),
            ("OVC", 500, "NS"),
        ]

        self.sdrs = np.array(
            [
                self.encoder.encode(cov, hgt, ctype).dense.astype(np.float32)
                for cov, hgt, ctype in self.test_layers
            ]
        )

    @staticmethod
    def overlap(a: np.ndarray, b: np.ndarray) -> float:
        intersection = np.sum(np.logical_and(a, b))
        active = max(np.sum(a), 1)
        return float(intersection / active)

    # =====================================================
    # SEMANTIC TESTS
    # =====================================================

    def semantic_tests(self):
        print()
        print("=" * 90)
        print("CLOUD LAYER SEMANTIC OVERLAP TESTS")
        print("=" * 90)
        print()

        tests = [
            ("CLR", 18000, "CI", "BKN", 4200, "ST"),
            ("FEW", 12000, "AC", "SCT", 6500, "SC"),
            ("BKN", 2800, "NS", "OVC", 800, "CB"),
            ("CLR", 25000, "CI", "OVC", 500, "CB"),
            ("BKN", 3500, "CB", "FEW", 18000, "CI"),
            ("SCT", 6500, "SC", "BKN", 2800, "NS"),
        ]

        for cov1, h1, t1, cov2, h2, t2 in tests:
            ov = self.encoder.overlap((cov1, h1, t1), (cov2, h2, t2))
            print(
                f"{cov1:>4}@{h1:>5}ft/{t1:<3}  <->  {cov2:>4}@{h2:>5}ft/{t2:<3}  |  Overlap = {ov:.4f}"
            )

        print()

    # =====================================================
    # VISUALIZATIONS
    # =====================================================

    def plot_similarity_matrix(self):
        n = len(self.sdrs)
        matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                matrix[i, j] = self.overlap(self.sdrs[i], self.sdrs[j])

        plt.figure(figsize=(11, 9))
        plt.imshow(matrix, cmap="viridis", interpolation="nearest")
        plt.colorbar(label="Normalized Overlap")
        plt.title("Cloud Layer Similarity Matrix")
        plt.tight_layout()
        plt.savefig(self.output_dir / "similarity_matrix.png")
        plt.close()

    def plot_pca(self):
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(self.sdrs)

        plt.figure(figsize=(10, 8))
        for i, (cov, hgt, ctype) in enumerate(self.test_layers):
            plt.scatter(reduced[i, 0], reduced[i, 1], s=120)
            label = f"{cov}@{hgt}/{ctype}"
            plt.text(reduced[i, 0], reduced[i, 1], label, fontsize=9)
        plt.title("Cloud Layer PCA Projection")
        plt.tight_layout()
        plt.savefig(self.output_dir / "pca_projection.png")
        plt.close()

    def print_active_bits(self):
        print()
        print("=" * 80)
        print("ACTIVE BIT COUNTS")
        print("=" * 80)
        print()
        for cov, hgt, ctype in self.test_layers:
            sdr = self.encoder.encode(cov, hgt, ctype)
            print(
                f"{cov:>4} @ {hgt:>5}ft / {ctype:<3}  →  {len(sdr.sparse):>3} active bits"
            )
        print()

    def run_all(self):
        print("=" * 90)
        print("CLOUD LAYER ENCODER VALIDATION STARTED")
        print("=" * 90)

        self.plot_similarity_matrix()
        self.plot_pca()
        self.print_active_bits()
        self.semantic_tests()

        print("=" * 90)
        print("CLOUD LAYER VALIDATION COMPLETE")
        print(f"Plots saved to: {self.output_dir}")
        print("=" * 90)


if __name__ == "__main__":
    validator = CloudLayerEncoderValidator()
    validator.run_all()
