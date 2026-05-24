# metar/encoders/test/validate_wind_gust_encoder.py

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

from encoders.wind_gust_encoder import (
    WindGustEncoder,
)


class WindGustEncoderValidator:
    """
    Rigorous validation suite for WindGustEncoder.

    Validates:
    - semantic overlap
    - atmospheric continuity
    - gust manifold geometry
    - missing gust behavior
    - SDR stability
    """

    def __init__(self):

        self.encoder = WindGustEncoder()

        self.output_dir = Path("./output/wind_gust_validation")

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # =====================================================
        # VALIDATION DOMAIN
        # =====================================================

        self.gusts = list(range(0, 101, 2))

        self.sdrs = np.array(
            [self.encoder.encode(g).dense.astype(np.float32) for g in self.gusts]
        )

    # =====================================================
    # OVERLAP
    # =====================================================

    @staticmethod
    def overlap(
        a: np.ndarray,
        b: np.ndarray,
    ) -> float:

        active = max(
            np.sum(a),
            1,
        )

        intersection = np.sum(np.logical_and(a, b))

        return float(intersection / active)

    # =====================================================
    # ACTIVE BIT TESTS
    # =====================================================

    def print_active_bits(
        self,
    ):

        print()
        print("=" * 80)
        print("ACTIVE BIT TESTS")
        print("=" * 80)
        print()

        samples = [
            None,
            0,
            5,
            15,
            30,
            60,
            100,
        ]

        for gust in samples:

            sdr = self.encoder.encode(gust)

            print(f"Gust={str(gust):>4} kt | " f"Active Bits=" f"{len(sdr.sparse)}")

        print()

    # =====================================================
    # SEMANTIC OVERLAP TESTS
    # =====================================================

    def print_overlap_tests(
        self,
    ):

        print()
        print("=" * 80)
        print("SEMANTIC OVERLAP TESTS")
        print("=" * 80)
        print()

        tests = [
            (5, 7),
            (10, 12),
            (15, 20),
            (25, 35),
            (40, 60),
            (60, 90),
            (None, 20),
            (None, None),
        ]

        for a, b in tests:

            overlap = self.encoder.overlap(
                a,
                b,
            )

            print(f"{str(a):>5} kt <-> " f"{str(b):>5} kt | " f"Overlap={overlap:.4f}")

        print()

    # =====================================================
    # OVERLAP DECAY CURVE
    # =====================================================

    def plot_overlap_decay(
        self,
    ):

        anchors = [
            5,
            15,
            30,
            60,
        ]

        plt.figure(figsize=(12, 7))

        for anchor in anchors:

            anchor_sdr = self.encoder.encode(anchor).dense

            overlaps = []

            for gust in self.gusts:

                sdr = self.encoder.encode(gust).dense

                overlaps.append(
                    self.overlap(
                        anchor_sdr,
                        sdr,
                    )
                )

            plt.plot(
                self.gusts,
                overlaps,
                label=f"{anchor} kt",
            )

        plt.title("Wind Gust Overlap Decay")

        plt.xlabel("Gust Speed (kt)")

        plt.ylabel("Normalized Overlap")

        plt.grid(True)

        plt.legend()

        plt.tight_layout()

        plt.savefig(self.output_dir / "overlap_decay.png")

        plt.close()

    # =====================================================
    # SIMILARITY MATRIX
    # =====================================================

    def plot_similarity_matrix(
        self,
    ):

        n = len(self.sdrs)

        matrix = np.zeros(
            (n, n),
        )

        for i in range(n):
            for j in range(n):

                matrix[i, j] = self.overlap(
                    self.sdrs[i],
                    self.sdrs[j],
                )

        plt.figure(figsize=(12, 10))

        sns.heatmap(
            matrix,
            cmap="viridis",
        )

        plt.title("Wind Gust Similarity Matrix")

        plt.tight_layout()

        plt.savefig(self.output_dir / "similarity_matrix.png")

        plt.close()

    # =====================================================
    # PCA
    # =====================================================

    def plot_pca(
        self,
    ):

        pca = PCA(
            n_components=2,
        )

        reduced = pca.fit_transform(self.sdrs)

        plt.figure(figsize=(10, 8))

        scatter = plt.scatter(
            reduced[:, 0],
            reduced[:, 1],
            c=self.gusts,
            cmap="viridis",
            s=40,
        )

        plt.colorbar(scatter, label="Gust Speed (kt)")

        plt.title("Wind Gust PCA Projection")

        plt.tight_layout()

        plt.savefig(self.output_dir / "pca_projection.png")

        plt.close()

    # =====================================================
    # TSNE
    # =====================================================

    def plot_tsne(
        self,
    ):

        tsne = TSNE(
            n_components=2,
            perplexity=10,
            random_state=42,
            init="pca",
        )

        reduced = tsne.fit_transform(self.sdrs)

        plt.figure(figsize=(10, 8))

        scatter = plt.scatter(
            reduced[:, 0],
            reduced[:, 1],
            c=self.gusts,
            cmap="viridis",
            s=40,
        )

        plt.colorbar(scatter, label="Gust Speed (kt)")

        plt.title("Wind Gust t-SNE Projection")

        plt.tight_layout()

        plt.savefig(self.output_dir / "tsne_projection.png")

        plt.close()

    # =====================================================
    # UMAP
    # =====================================================

    def plot_umap(
        self,
    ):

        reducer = umap.UMAP(
            n_neighbors=10,
            min_dist=0.05,
            random_state=42,
        )

        reduced = np.array(reducer.fit_transform(self.sdrs))

        plt.figure(figsize=(10, 8))

        scatter = plt.scatter(
            reduced[:, 0],
            reduced[:, 1],
            c=self.gusts,
            cmap="viridis",
            s=40,
        )

        plt.colorbar(scatter, label="Gust Speed (kt)")

        plt.title("Wind Gust UMAP Projection")

        plt.tight_layout()

        plt.savefig(self.output_dir / "umap_projection.png")

        plt.close()

    # =====================================================
    # ACTIVE BIT DISTRIBUTION
    # =====================================================

    def plot_active_bits(
        self,
    ):

        counts = [len(self.encoder.encode(g).sparse) for g in self.gusts]

        plt.figure(figsize=(10, 6))

        plt.plot(
            self.gusts,
            counts,
        )

        plt.title("Active Bit Stability")

        plt.xlabel("Gust Speed (kt)")

        plt.ylabel("Active Bits")

        plt.grid(True)

        plt.tight_layout()

        plt.savefig(self.output_dir / "active_bits.png")

        plt.close()

    # =====================================================
    # TEMPORAL TRAJECTORY
    # =====================================================

    def plot_temporal_trajectory(
        self,
    ):

        trajectory = list(range(0, 60, 2)) + list(range(60, 0, -2))

        sdrs = np.array(
            [self.encoder.encode(g).dense.astype(np.float32) for g in trajectory]
        )

        pca = PCA(
            n_components=2,
        )

        reduced = pca.fit_transform(sdrs)

        plt.figure(figsize=(10, 8))

        plt.plot(
            reduced[:, 0],
            reduced[:, 1],
            marker="o",
        )

        plt.title("Temporal Gust Evolution")

        plt.tight_layout()

        plt.savefig(self.output_dir / "temporal_trajectory.png")

        plt.close()

    # =====================================================
    # RUN ALL
    # =====================================================

    def run_all(
        self,
    ):

        self.print_active_bits()

        self.print_overlap_tests()

        self.plot_overlap_decay()

        self.plot_similarity_matrix()

        self.plot_pca()

        self.plot_tsne()

        self.plot_umap()

        self.plot_active_bits()

        self.plot_temporal_trajectory()

        print()
        print("=" * 80)
        print("WIND GUST VALIDATION COMPLETE")
        print(f"Plots saved to: " f"{self.output_dir}")
        print("=" * 80)
        print()


def main():

    validator = WindGustEncoderValidator()

    validator.run_all()


if __name__ == "__main__":
    main()
