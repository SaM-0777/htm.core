from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap


class WindDirectionValidation:
    """
    Specialized validation suite for cyclic topology.

    Validates:
    - circular continuity
    - rotational overlap geometry
    - manifold continuity
    - cyclic wraparound
    """

    def __init__(
        self,
        encoder,
    ):

        self.encoder = encoder

        self.output_dir = Path("./output/wind_direction_validation2")

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.angles = np.arange(
            0,
            360,
            1,
        )

        self.sdrs = np.array([encoder.encode_dense(a) for a in self.angles])

    # =====================================================
    # OVERLAP
    # =====================================================

    @staticmethod
    def overlap(
        a: np.ndarray,
        b: np.ndarray,
    ) -> float:

        intersection = np.sum(np.logical_and(a, b))

        active = max(np.sum(a), 1)

        return float(intersection / active)

    # =====================================================
    # ANGULAR DISTANCE
    # =====================================================

    @staticmethod
    def angular_distance(
        a: float,
        b: float,
    ) -> float:

        diff = abs(a - b) % 360

        return min(
            diff,
            360 - diff,
        )

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

        plt.imshow(
            matrix,
            cmap="viridis",
            interpolation="nearest",
        )

        plt.colorbar()

        plt.title("Wind Direction Similarity Matrix")

        plt.tight_layout()

        plt.savefig(self.output_dir / "similarity_matrix.png")

        plt.close()

    # =====================================================
    # OVERLAP DECAY
    # =====================================================

    def plot_overlap_decay(
        self,
    ):

        anchor = self.sdrs[0]

        overlaps = []

        for sdr in self.sdrs:

            overlaps.append(
                self.overlap(
                    anchor,
                    sdr,
                )
            )

        plt.figure(figsize=(12, 6))

        plt.plot(
            self.angles,
            overlaps,
        )

        plt.title("Angular Overlap Decay")

        plt.xlabel("Angle Difference")

        plt.ylabel("Normalized Overlap")

        plt.grid(True)

        plt.tight_layout()

        plt.savefig(self.output_dir / "overlap_decay.png")

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

        plt.figure(figsize=(10, 10))

        scatter = plt.scatter(
            reduced[:, 0],
            reduced[:, 1],
            c=self.angles,
            cmap="hsv",
            s=20,
        )

        plt.colorbar(
            scatter,
            label="Angle",
        )

        plt.title("Wind Direction PCA")

        plt.axis("equal")

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
            perplexity=30,
            random_state=42,
            init="pca",
        )

        reduced = tsne.fit_transform(self.sdrs)

        plt.figure(figsize=(10, 10))

        scatter = plt.scatter(
            reduced[:, 0],
            reduced[:, 1],
            c=self.angles,
            cmap="hsv",
            s=20,
        )

        plt.colorbar(
            scatter,
            label="Angle",
        )

        plt.title("Wind Direction t-SNE")

        plt.axis("equal")

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
            n_neighbors=20,
            min_dist=0.1,
            random_state=42,
        )

        reduced = np.asarray(reducer.fit_transform(self.sdrs))

        plt.figure(figsize=(10, 10))

        scatter = plt.scatter(
            reduced[:, 0],
            reduced[:, 1],
            c=self.angles,
            cmap="hsv",
            s=20,
        )

        plt.colorbar(
            scatter,
            label="Angle",
        )

        plt.title("Wind Direction UMAP")

        plt.axis("equal")

        plt.tight_layout()

        plt.savefig(self.output_dir / "umap_projection.png")

        plt.close()

    # =====================================================
    # WRAPAROUND TESTS
    # =====================================================

    def wraparound_tests(
        self,
    ):

        tests = [
            (359, 0),
            (358, 1),
            (350, 10),
            (90, 270),
            (45, 225),
        ]

        print()
        print("=" * 80)
        print("WRAPAROUND TESTS")
        print("=" * 80)
        print()

        for a, b in tests:

            overlap = self.encoder.overlap(
                a,
                b,
            )

            distance = self.angular_distance(
                a,
                b,
            )

            print(
                f"{a}° vs {b}° | "
                f"Angular Distance={distance:.1f} | "
                f"Overlap={overlap:.4f}"
            )

        print()

    # =====================================================
    # RUN
    # =====================================================

    def run_all(
        self,
    ):

        self.plot_similarity_matrix()

        self.plot_overlap_decay()

        self.plot_pca()

        self.plot_tsne()

        self.plot_umap()

        self.wraparound_tests()

        print()
        print("=" * 80)
        print("WIND DIRECTION VALIDATION COMPLETE")
        print("=" * 80)
        print()
