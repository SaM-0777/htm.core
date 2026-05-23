from __future__ import annotations

from pathlib import Path
import itertools

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

from encoders.coverage_encoder import (
    CoverageEncoder,
    COVERAGE_MAP,
)


class CoverageEncoderValidator:

    def __init__(self):
        self.encoder = CoverageEncoder()
        self.categories = list(COVERAGE_MAP.keys())
        self.output_dir = Path("./output/coverage_encoder_validation")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.sdrs = np.array(
            [self.encoder.encode(c).dense.astype(np.float32) for c in self.categories]
        )

    @staticmethod
    def overlap(a: np.ndarray, b: np.ndarray) -> float:
        intersection = np.sum(np.logical_and(a, b))
        active = max(np.sum(a), 1)
        return float(intersection / active)

    # =====================================================
    # SEMANTIC TESTS - ALL POSSIBLE PAIRS
    # =====================================================

    def semantic_tests(self):
        print()
        print("=" * 80)
        print("SEMANTIC OVERLAP TESTS - ALL PAIRS")
        print("=" * 80)
        print()

        # Test ALL possible combinations
        for a, b in itertools.combinations_with_replacement(self.categories, 2):
            overlap_val = self.overlap(
                self.encoder.encode(a).dense, self.encoder.encode(b).dense
            )
            print(f"{a:<5} <-> {b:<5}  Overlap = {overlap_val:.4f}")

        print()

    # =====================================================
    # Rest of your methods (unchanged)
    # =====================================================

    def plot_similarity_matrix(self):
        n = len(self.sdrs)
        matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                matrix[i, j] = self.overlap(self.sdrs[i], self.sdrs[j])

        plt.figure(figsize=(10, 8))
        plt.imshow(matrix, cmap="viridis", interpolation="nearest")
        plt.xticks(np.arange(n), self.categories, rotation=45)
        plt.yticks(np.arange(n), self.categories)
        plt.colorbar(label="Overlap")
        plt.title("Coverage Similarity Matrix (All Pairs)")
        plt.tight_layout()
        plt.savefig(self.output_dir / "similarity_matrix.png")
        plt.close()

    def plot_overlap_decay(self):
        anchor = self.sdrs[0]
        overlaps = [self.overlap(anchor, sdr) for sdr in self.sdrs]

        plt.figure(figsize=(10, 5))
        plt.plot(self.categories, overlaps, marker="o")
        plt.title("Coverage Overlap Decay (from CLR)")
        plt.xlabel("Coverage")
        plt.ylabel("Normalized Overlap")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.output_dir / "overlap_decay.png")
        plt.close()

    def plot_pca(self):
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(self.sdrs)

        plt.figure(figsize=(9, 7))
        for i, label in enumerate(self.categories):
            plt.scatter(reduced[i, 0], reduced[i, 1], s=120)
            plt.text(reduced[i, 0], reduced[i, 1], label, fontsize=12)
        plt.title("Coverage PCA Projection")
        plt.tight_layout()
        plt.savefig(self.output_dir / "pca_projection.png")
        plt.close()

    def plot_tsne(self):
        tsne = TSNE(n_components=2, perplexity=3, random_state=42, init="pca")
        reduced = tsne.fit_transform(self.sdrs)

        plt.figure(figsize=(9, 7))
        for i, label in enumerate(self.categories):
            plt.scatter(reduced[i, 0], reduced[i, 1], s=120)
            plt.text(reduced[i, 0], reduced[i, 1], label, fontsize=12)
        plt.title("Coverage t-SNE Projection")
        plt.tight_layout()
        plt.savefig(self.output_dir / "tsne_projection.png")
        plt.close()

    def plot_umap(self):
        reducer = umap.UMAP(n_neighbors=3, min_dist=0.1, random_state=42)
        reduced = np.array(reducer.fit_transform(self.sdrs))

        plt.figure(figsize=(9, 7))
        for i, label in enumerate(self.categories):
            plt.scatter(reduced[i, 0], reduced[i, 1], s=120)
            plt.text(reduced[i, 0], reduced[i, 1], label, fontsize=12)
        plt.title("Coverage UMAP Projection")
        plt.tight_layout()
        plt.savefig(self.output_dir / "umap_projection.png")
        plt.close()

    def print_active_bits(self):
        print()
        print("=" * 80)
        print("ACTIVE BIT COUNTS")
        print("=" * 80)
        print()
        for category in self.categories:
            sdr = self.encoder.encode(category)
            print(f"{category:<5}  Active Bits = {len(sdr.sparse)}")
        print()

    def run_all(self):
        print("=" * 80)
        print("RUNNING COVERAGE ENCODER VALIDATION")
        print("=" * 80)
        print()

        self.plot_similarity_matrix()
        self.plot_overlap_decay()
        self.plot_pca()
        self.plot_tsne()
        self.plot_umap()
        self.print_active_bits()
        self.semantic_tests()

        print("=" * 80)
        print("COVERAGE VALIDATION COMPLETE")
        print("=" * 80)


if __name__ == "__main__":
    validator = CoverageEncoderValidator()
    validator.run_all()
