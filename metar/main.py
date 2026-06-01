from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from metar.encoders.metar_encoder import MetarEncoder
from metar.spatial_pooler.sp_metar import SpatialPoolerMetar

dataset_path = Path("metar/data/metar_parsed_p.json")

output_dir = Path(f"metar/output/sp/run_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}")
output_dir.mkdir(parents=True, exist_ok=True)


# Plots
def plot_active_columns(
    active_counts: list[int],
):
    plt.figure(figsize=(12, 6))
    plt.plot(active_counts)

    plt.title("Active Columns Over Time")

    plt.xlabel("Sample")
    plt.ylabel("Active Columns")

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(output_dir / "active_columns.png")

    plt.close()


def plot_temporal_overlap(
    overlaps: list[float],
):
    plt.figure(figsize=(12, 6))
    plt.plot(overlaps)

    plt.title("Temporal SDR Overlap")

    plt.xlabel("Time")
    plt.ylabel("Normalized Overlap")

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(output_dir / "temporal_overlap.png")

    plt.close()


def plot_column_utilization(
    column_usage,
    column_count: int,
):
    usage = np.zeros(column_count)

    for k, v in column_usage.items():
        usage[k] = v

    plt.figure(figsize=(12, 6))
    plt.hist(
        usage,
        bins=60,
    )

    plt.title("Column Utilization")

    plt.xlabel("Activation Count")
    plt.ylabel("Columns")

    plt.tight_layout()
    plt.savefig(output_dir / "column_utilization.png")

    plt.close()


def plot_pca(
    output_sdrs: list[np.ndarray],
):
    sdrs = np.array(output_sdrs)
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(sdrs)

    plt.figure(figsize=(10, 8))
    plt.scatter(
        reduced[:, 0],
        reduced[:, 1],
        s=5,
    )

    plt.title("SP Output PCA Projection")

    plt.tight_layout()
    plt.savefig(output_dir / "pca_projection.png")

    plt.close()


def plot_tsne(
    output_sdrs: list[np.ndarray],
):

    sdrs = np.array(output_sdrs)
    tsne = TSNE(
        n_components=2,
        perplexity=20,
        random_state=42,
        init="pca",
    )
    reduced = tsne.fit_transform(sdrs)

    plt.figure(figsize=(10, 8))
    plt.scatter(
        reduced[:, 0],
        reduced[:, 1],
        s=5,
    )

    plt.title("SP Output t-SNE Projection")
    plt.tight_layout()
    plt.savefig(output_dir / "tsne_projection.png")

    plt.close()


def plot_umap(
    output_sdrs: list[np.ndarray],
):
    import umap

    sdrs = np.array(output_sdrs)
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        random_state=42,
    )

    reduced = np.array(reducer.fit_transform(sdrs))

    plt.figure(figsize=(10, 8))
    plt.scatter(
        reduced[:, 0],
        reduced[:, 1],
        s=5,
    )

    plt.title("SP Output UMAP Projection")
    plt.tight_layout()
    plt.savefig(output_dir / "umap_projection.png")

    plt.close()


def main(epochs=5):
    metar_encoder = MetarEncoder()
    sp_metar = SpatialPoolerMetar(metar_encoder.output_size)

    output_sdrs: list[np.ndarray] = []
    input_sdrs: list[np.ndarray] = []

    with open(dataset_path, "r", encoding="utf-8") as f:
        data_list = json.load(f)

    data_list.sort(key=lambda x: x["recorded_time"])

    print(f"Loaded {len(data_list)} objects")

    for epoch in range(epochs):
        for i, record in enumerate(data_list):
            try:
                recorded_time = datetime.fromisoformat(record["recorded_time"])

                metar_sdr = metar_encoder.encode(
                    recorded_time=recorded_time,
                    pressure_hpa=record.get("pressure_hpa"),
                    dew_point_c=record.get("dew_point_c"),
                    temperature_c=record.get("temperature_c"),
                    visibility_m=record.get("visibility_m"),
                    wind_direction_deg=record.get("wind_direction_deg"),
                    wind_speed_kt=record.get("wind_speed_kt"),
                    is_wind_variable=record.get("wind_direction_deg") is None,
                    wind_gust_kt=record.get("wind_gust_kt"),
                    cloud_layers=record.get("clouds", []),
                )

                output_sdr = sp_metar.compute(
                    metar_sdr,
                )
                input_sdrs.append(metar_sdr.dense.astype(np.float32))
                output_sdrs.append(output_sdr.dense.astype(np.float32))

            except Exception as e:
                print(f"Epoch {epoch} Record {i+1:4d} | ERROR: {e}")

    plot_active_columns(sp_metar.active_counts)
    plot_temporal_overlap(sp_metar.temporal_overlaps)
    plot_column_utilization(
        sp_metar.column_usage,
        sp_metar.output_size,
    )
    plot_pca(output_sdrs)
    plot_tsne(output_sdrs)
    plot_umap(output_sdrs)

    diagnostics = sp_metar.diagnostics()

    for k, v in diagnostics.items():
        print(f"{k:<30}: " f"{v:.6f}")


if __name__ == "__main__":
    main(10)
