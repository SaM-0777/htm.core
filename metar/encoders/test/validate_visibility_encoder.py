from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

from encoders.visibility_encoder import VisibilityEncoder


class VisibilityEncoderValidator:
    """
    Validation suite for VisibilityEncoder (numeric input only).
    """

    def __init__(self):
        self.encoder = VisibilityEncoder()
        self.output_dir = Path("./output/visibility_validation")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Test values - all numeric in meters
        self.vis_values = [
            50,
            150,
            300,
            600,
            900,
            1200,
            1800,
            2500,
            3500,
            5000,
            7000,
            9999,
            12000,
            18000,
            20000,
        ]

        self.sdrs = np.array(
            [self.encoder.encode(v).dense.astype(np.float32) for v in self.vis_values]
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
        print("=" * 85)
        print("VISIBILITY SEMANTIC OVERLAP TESTS")
        print("=" * 85)
        print()

        tests = [
            (100, 300),
            (500, 900),
            (1200, 1800),
            (2500, 3500),
            (5000, 7000),
            (9999, 12000),
            (50, 2000),
            (800, 9999),
            (200, 15000),
            (300, 8000),
        ]

        for a, b in tests:
            ov = self.encoder.overlap(a, b)
            print(f"{a:>6} m <-> {b:>6} m | Overlap = {ov:.4f}")

        print()

    # =====================================================
    # PLOTS
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
        plt.title("Visibility Similarity Matrix")
        plt.tight_layout()
        plt.savefig(self.output_dir / "similarity_matrix.png")
        plt.close()

    def plot_overlap_decay(self):
        anchors = [200, 1000, 3000, 7000, 15000]

        plt.figure(figsize=(12, 7))
        for anchor in anchors:
            anchor_sdr = self.encoder.encode(anchor).dense
            overlaps = [self.overlap(anchor_sdr, sdr) for sdr in self.sdrs]
            plt.plot(self.vis_values, overlaps, marker="o", label=f"Anchor {anchor}m")

        plt.title("Visibility Overlap Decay")
        plt.xlabel("Visibility (meters)")
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
            reduced[:, 0], reduced[:, 1], c=self.vis_values, cmap="viridis", s=60
        )
        plt.colorbar(scatter, label="Visibility (m)")
        plt.title("Visibility PCA Projection")
        plt.tight_layout()
        plt.savefig(self.output_dir / "pca_projection.png")
        plt.close()

    def plot_tsne(self):
        tsne = TSNE(n_components=2, perplexity=8, random_state=42, init="pca")
        reduced = tsne.fit_transform(self.sdrs)

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            reduced[:, 0], reduced[:, 1], c=self.vis_values, cmap="viridis", s=60
        )
        plt.colorbar(scatter, label="Visibility (m)")
        plt.title("Visibility t-SNE Projection")
        plt.tight_layout()
        plt.savefig(self.output_dir / "tsne_projection.png")
        plt.close()

    def plot_umap(self):
        reducer = umap.UMAP(n_neighbors=10, min_dist=0.1, random_state=42)
        reduced = np.array(reducer.fit_transform(self.sdrs))

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            reduced[:, 0], reduced[:, 1], c=self.vis_values, cmap="viridis", s=60
        )
        plt.colorbar(scatter, label="Visibility (m)")
        plt.title("Visibility UMAP Projection")
        plt.tight_layout()
        plt.savefig(self.output_dir / "umap_projection.png")
        plt.close()

    def print_active_bits(self):
        print()
        print("=" * 80)
        print("ACTIVE BIT COUNTS")
        print("=" * 80)
        print()
        for v in self.vis_values:
            sdr = self.encoder.encode(v)
            print(f"{v:>6} m  →  {len(sdr.sparse):>2} active bits")
        print()

    def run_all(self):
        print("=" * 85)
        print("VISIBILITY ENCODER VALIDATION STARTED")
        print("=" * 85)

        self.plot_similarity_matrix()
        self.plot_overlap_decay()
        self.plot_pca()
        self.plot_tsne()
        self.plot_umap()
        self.print_active_bits()
        self.semantic_tests()

        print("=" * 85)
        print("VISIBILITY VALIDATION COMPLETE")
        print(f"Plots saved to: {self.output_dir}")
        print("=" * 85)


if __name__ == "__main__":
    validator = VisibilityEncoderValidator()
    validator.run_all()
