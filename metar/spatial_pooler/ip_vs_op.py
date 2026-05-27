import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap

from htm.bindings.sdr import SDR
from htm.bindings.algorithms import SpatialPooler

from metar.encoders.temperature_encoder import TemperatureEncoder

OUTPUT_DIR = "semantic_clustering_plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)


encoder = TemperatureEncoder()
sp = SpatialPooler(
    inputDimensions=(encoder.output_size,),
    columnDimensions=(256,),
    potentialPct=0.5,
    potentialRadius=encoder.output_size,
    globalInhibition=True,
    localAreaDensity=0,
    numActiveColumnsPerInhArea=12,
    synPermInactiveDec=0.008,
    synPermActiveInc=0.05,
    synPermConnected=0.1,
    boostStrength=0.0,
    seed=42,
    wrapAround=False,
)

training_temperatures = np.linspace(-20, 50, 400)
for temp in training_temperatures:

    dense = encoder.encode_dense(temp)

    input_sdr = SDR((encoder.output_size,))
    input_sdr.dense = dense

    active_columns = SDR(sp.getColumnDimensions())

    sp.compute(input_sdr, True, active_columns)


temperatures = np.linspace(-20, 50, 120)
encoder_vectors = []
sp_vectors = []

for temp in temperatures:
    dense = encoder.encode_dense(temp)
    encoder_vectors.append(dense.copy())

    input_sdr = SDR((encoder.output_size,))
    input_sdr.dense = dense

    active_columns = SDR(sp.getColumnDimensions())
    sp.compute(input_sdr, False, active_columns)
    sp_dense = np.zeros(sp.getNumColumns(), dtype=np.int8)
    sp_dense[active_columns.sparse] = 1
    sp_vectors.append(sp_dense)


encoder_vectors = np.array(encoder_vectors)
sp_vectors = np.array(sp_vectors)


def create_projection_plot(
    encoder_proj,
    sp_proj,
    title,
    filename,
):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    axes[0].set_title("Raw Encoder SDRs", fontsize=16)
    axes[0].set_xlabel("Component 1")
    axes[0].set_ylabel("Component 2")

    scatter2 = axes[1].scatter(
        sp_proj[:, 0], sp_proj[:, 1], c=temperatures, cmap="viridis", s=40
    )

    axes[1].set_title("Spatial Pooler Output SDRs", fontsize=16)
    axes[1].set_xlabel("Component 1")
    axes[1].set_ylabel("Component 2")

    cbar = fig.colorbar(
        scatter2, ax=axes, orientation="horizontal", fraction=0.05, pad=0.12
    )
    cbar.set_label("Temperature (°C)", fontsize=12)

    plt.suptitle(title, fontsize=20)
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, filename)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


pca_encoder = PCA(n_components=2)
pca_sp = PCA(n_components=2)
encoder_pca = pca_encoder.fit_transform(encoder_vectors)
sp_pca = pca_sp.fit_transform(sp_vectors)

create_projection_plot(
    encoder_pca,
    sp_pca,
    "PCA Projection — Encoder vs Spatial Pooler",
    "pca_projection.png",
)


tsne_encoder = TSNE(n_components=2, perplexity=20, random_state=42, init="pca")
tsne_sp = TSNE(n_components=2, perplexity=20, random_state=42, init="pca")
encoder_tsne = tsne_encoder.fit_transform(encoder_vectors)
sp_tsne = tsne_sp.fit_transform(sp_vectors)
create_projection_plot(
    encoder_tsne,
    sp_tsne,
    "t-SNE Projection — Encoder vs Spatial Pooler",
    "tsne_projection.png",
)


umap_encoder = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
umap_sp = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
encoder_umap = umap_encoder.fit_transform(encoder_vectors)
sp_umap = umap_sp.fit_transform(sp_vectors)
create_projection_plot(
    encoder_umap,
    sp_umap,
    "UMAP Projection — Encoder vs Spatial Pooler",
    "umap_projection.png",
)

print("\nAll plots saved successfully.")
print(f"Output directory -> {OUTPUT_DIR}")
