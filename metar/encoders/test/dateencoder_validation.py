from __future__ import annotations

from datetime import datetime
from datetime import timedelta
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import umap

from htm.bindings.sdr import SDR
from htm.bindings.encoders import DateEncoder
from htm.bindings.encoders import DateEncoderParameters

# =========================================================
# Output Directory
# =========================================================

OUTPUT_DIR = Path("./output/encoder_validation")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =========================================================
# IMPORTANT
# =========================================================
#
# VERIFY YOUR INSTALLED API FIRST:
#
# from htm.bindings.encoders import DateEncoder
#
# help(DateEncoder)
#
# =========================================================


class DateEncoderTester:

    def __init__(self) -> None:

        # ---------------------------------------------
        # VERIFY THESE ARGUMENTS
        # AGAINST YOUR INSTALLED htm.core VERSION
        # ---------------------------------------------

        params = DateEncoderParameters()

        #print(dir(params))
        #help(DateEncoderParameters)

        params.timeOfDay_width = 128
        params.timeOfDay_radius = 0.5
        
        params.season_width = 256
        params.season_radius = 14.0
        
        params.dayOfWeek_width = 0
        params.weekend_width = 0
        params.holiday_width = 0
        params.custom_width = 0

        self.encoder = DateEncoder(params)

        self.output_width = self.encoder.dimensions[0]

    # =====================================================
    # Encode datetime
    # =====================================================

    def encode(
        self,
        dt: datetime,
    ) -> SDR:

        sdr = SDR(self.output_width)

        self.encoder.encode(
            dt,
            sdr,
        )

        return sdr

    # =====================================================
    # Generate dataset
    # =====================================================

    def generate_dataset(
        self,
        start: datetime,
        hours: int,
    ) -> tuple[list[datetime], list[SDR]]:

        timestamps = []

        sdrs = []

        current = start

        for _ in range(hours):

            timestamps.append(current)

            sdrs.append(self.encode(current))

            current += timedelta(hours=1)

        return timestamps, sdrs

    # =====================================================
    # SDR → Dense Matrix
    # =====================================================

    def dense_matrix(
        self,
        sdrs: list[SDR],
    ) -> np.ndarray:

        matrix = np.zeros(
            (
                len(sdrs),
                self.output_width,
            ),
            dtype=np.uint8,
        )

        for i, sdr in enumerate(sdrs):

            matrix[i, sdr.sparse] = 1

        return matrix

    # =====================================================
    # Similarity Matrix
    # =====================================================

    def similarity_matrix(
        self,
        sdrs: list[SDR],
    ) -> np.ndarray:

        n = len(sdrs)

        matrix = np.zeros(
            (n, n),
            dtype=np.float32,
        )

        for i in range(n):

            a = set(sdrs[i].sparse.tolist())

            for j in range(n):

                b = set(sdrs[j].sparse.tolist())

                intersection = len(a.intersection(b))

                union = len(a.union(b))

                similarity = intersection / union if union > 0 else 0.0

                matrix[i, j] = similarity

        return matrix

    # =====================================================
    # Plot Heatmap
    # =====================================================

    def plot_heatmap(
        self,
        similarity: np.ndarray,
    ) -> None:

        fig = plt.figure(figsize=(14, 12))

        ax = fig.add_subplot(111)

        image = ax.imshow(similarity)

        plt.colorbar(image)

        ax.set_title("DateEncoder Similarity Matrix")

        plt.savefig(OUTPUT_DIR / "similarity_heatmap.png")

        plt.close(fig)

    # =====================================================
    # Plot Overlap Decay
    # =====================================================

    def plot_overlap_decay(
        self,
        similarity: np.ndarray,
    ) -> None:

        reference = similarity[0]

        fig = plt.figure(figsize=(14, 8))

        ax = fig.add_subplot(111)

        ax.plot(reference)

        ax.set_title("Temporal Similarity Decay")

        ax.set_xlabel("Temporal Distance")

        ax.set_ylabel("Similarity")

        plt.savefig(OUTPUT_DIR / "overlap_decay.png")

        plt.close(fig)

    # =====================================================
    # PCA Projection
    # =====================================================

    def plot_pca(
        self,
        matrix: np.ndarray,
    ) -> None:

        pca = PCA(
            n_components=2,
        )

        projected = pca.fit_transform(matrix)
        projected = np.asarray(projected)

        fig = plt.figure(figsize=(14, 10))

        ax = fig.add_subplot(111)

        scatter = ax.scatter(
            projected[:, 0],
            projected[:, 1],
            c=np.arange(len(projected)),
        )

        plt.colorbar(scatter)

        ax.set_title("DateEncoder PCA Projection")

        plt.savefig(OUTPUT_DIR / "pca_projection.png")

        plt.close(fig)

    # =====================================================
    # t-SNE Projection
    # =====================================================

    def plot_tsne(
        self,
        matrix: np.ndarray,
    ) -> None:

        tsne = TSNE(
            n_components=2,
            perplexity=30,
            random_state=42,
            init="pca",
        )

        projected = tsne.fit_transform(matrix)
        projected = np.asarray(projected)

        fig = plt.figure(figsize=(14, 10))

        ax = fig.add_subplot(111)

        scatter = ax.scatter(
            projected[:, 0],
            projected[:, 1],
            c=np.arange(len(projected)),
        )

        plt.colorbar(scatter)

        ax.set_title("DateEncoder t-SNE Projection")

        plt.savefig(OUTPUT_DIR / "tsne_projection.png")

        plt.close(fig)

    # =====================================================
    # UMAP Projection
    # =====================================================

    def plot_umap(
        self,
        matrix: np.ndarray,
    ) -> None:

        reducer = umap.UMAP(
            n_components=2,
            random_state=42,
        )

        projected = reducer.fit_transform(matrix)
        projected = np.asarray(projected)

        fig = plt.figure(figsize=(14, 10))

        ax = fig.add_subplot(111)

        scatter = ax.scatter(
            projected[:, 0],
            projected[:, 1],
            c=np.arange(len(projected)),
        )

        plt.colorbar(scatter)

        ax.set_title("DateEncoder UMAP Projection")

        plt.savefig(OUTPUT_DIR / "umap_projection.png")

        plt.close(fig)

    # =====================================================
    # Midnight Wraparound Test
    # =====================================================

    def test_midnight_overlap(
        self,
    ) -> None:

        a = self.encode(
            datetime(
                2026,
                5,
                13,
                23,
                50,
            )
        )

        b = self.encode(
            datetime(
                2026,
                5,
                14,
                0,
                10,
            )
        )

        overlap = len(
            np.intersect1d(
                a.sparse,
                b.sparse,
            )
        )

        print()
        print("=" * 80)
        print("MIDNIGHT TEST")
        print("=" * 80)
        print()
        print(f"Overlap: {overlap}")
        print()

    # =====================================================
    # Year Wraparound Test
    # =====================================================

    def test_year_overlap(
        self,
    ) -> None:

        a = self.encode(
            datetime(
                2026,
                12,
                31,
                12,
                0,
            )
        )

        b = self.encode(
            datetime(
                2027,
                1,
                1,
                12,
                0,
            )
        )

        overlap = len(
            np.intersect1d(
                a.sparse,
                b.sparse,
            )
        )

        print()
        print("=" * 80)
        print("YEAR WRAPAROUND TEST")
        print("=" * 80)
        print()
        print(f"Overlap: {overlap}")
        print()


# =========================================================
# MAIN
# =========================================================


def main() -> None:

    tester = DateEncoderTester()

    timestamps, sdrs = tester.generate_dataset(
        start=datetime(
            2026,
            1,
            1,
            0,
            0,
        ),
        hours=24 * 30,
    )

    matrix = tester.dense_matrix(sdrs)

    similarity = tester.similarity_matrix(sdrs)

    tester.plot_heatmap(similarity)

    tester.plot_overlap_decay(similarity)

    tester.plot_pca(matrix)

    tester.plot_tsne(matrix)

    tester.plot_umap(matrix)

    tester.test_midnight_overlap()

    tester.test_year_overlap()

    print()
    print("=" * 80)
    print("DATEENCODER VALIDATION COMPLETE")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
