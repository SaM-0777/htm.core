from collections import Counter
import csv
import datetime
import os
import numpy as np
import random
import math

from pathlib import Path
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from htm.bindings.sdr import SDR, Metrics
from htm.encoders.rdse import RDSE, RDSE_Parameters
from htm.encoders.date import DateEncoder
from htm.bindings.algorithms import SpatialPooler
from htm.bindings.algorithms import TemporalMemory
from htm.algorithms.anomaly_likelihood import AnomalyLikelihood
from htm.bindings.algorithms import Predictor

_EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
_INPUT_FILE_PATH = os.path.join(_EXAMPLE_DIR, "gymdata.csv")

output_dir = Path(
    f"example/hotgym/output/run_{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
)
output_dir.mkdir(parents=True, exist_ok=True)


## Plots
def plot_active_columns(active_counts: list[int]):
    plt.figure(figsize=(12, 6))
    plt.plot(active_counts)

    plt.title("Active Column Over Time")

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
    sp_output_sdrs: list[np.ndarray],
):
    sdrs = np.array(sp_output_sdrs)
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
    sp_output_sdrs: list[np.ndarray],
):

    sdrs = np.array(sp_output_sdrs)
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
    sp_output_sdrs: list[np.ndarray],
):
    import umap

    sdrs = np.array(sp_output_sdrs)
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


input_sdrs: list[np.ndarray] = []
sp_output_sdrs: list[np.ndarray] = []
column_usage = Counter()
active_counts: list[int] = []
temporal_overlaps: list[float] = []
prev_output: SDR | None = None
# TM
cells_per_column = 13
total_cells = 1638 * cells_per_column
active_cells = SDR(total_cells)
predictive_cells = SDR(total_cells)
winner_cells = SDR(total_cells)
prev_predictive: SDR | None = None
burst_rates: list[float] = []
prediction_overlaps: list[float] = []
active_cell_counts: list[int] = []
predictive_cell_counts: list[int] = []
cell_usage = Counter()


def update_metrices(output_sdr: SDR):
    global prev_output

    active_count = len(output_sdr.sparse)
    active_counts.append(active_count)

    for c in output_sdr.sparse:
        column_usage[c] += 1

    if prev_output is not None:
        overlap = len(set(prev_output.sparse) & set(output_sdr.sparse))
        overlap /= max(active_count, 1)
        temporal_overlaps.append(overlap)

    prev_output = output_sdr
    sp_output_sdrs.append(output_sdr.dense.astype(np.float32))


def tm_update_metrics(active_columns: SDR):
    global prev_predictive

    active_count = len(active_cells.sparse)
    predictive_count = len(predictive_cells.sparse)
    active_cell_counts.append(active_count)
    predictive_cell_counts.append(predictive_count)

    for cell in active_cells.sparse:
        cell_usage[cell] += 1

    bursting_coulmns = len(active_cells.sparse) / cells_per_column
    active_columns_count = len(active_columns.sparse)
    burst_rate = bursting_coulmns / max(active_columns_count, 1)
    burst_rates.append(burst_rate)

    if prev_predictive is not None:
        predicted_columns = {
            cell // cells_per_column for cell in prev_predictive.sparse
        }
        active_columns_set = {
            cell // default_parameters["tm"]["cellsPerColumn"]
            for cell in active_cells.sparse
        }
        print(
            {
                "predicted_columns": predicted_columns,
                "active_columns_set": active_columns_set,
                "cells_per_column": cells_per_column,
                "prev_predictive.sparse": prev_predictive.sparse,
                "active_cells.sparse": active_cells.sparse,
            }
        )
        overlap = len(predicted_columns & active_columns_set)
        denom = max(len(active_columns_set), 1)
        prediction_overlaps.append(overlap / denom)

    print("CURRENT PREDICTIVE:", len(predictive_cells.sparse))
    print(predictive_cells.sparse[:20])
    prev_predictive = SDR(predictive_cells.dimensions)
    new_predicitve = SDR(predictive_cells.dimensions)
    new_predicitve.setSDR(predictive_cells)
    print("COPIED PREDICTIVE:", len(new_predicitve.sparse))
    print(new_predicitve.sparse[:20])
    prev_predictive = new_predicitve


def tm_dignostics():
    usage = np.zeros(total_cells, dtype=np.float32)

    for k, v in cell_usage.items():
        usage[k] = v
    total = usage.sum()

    if total > 0:
        probs = usage / total
        probs = probs[probs > 0]
        entropy = float(-np.sum(probs * np.log2(probs)))
    else:
        entropy = 0.0

    dead_cells = total_cells - len(cell_usage)

    return {
        "entropy": entropy,
        "dead_cell_ratio": dead_cells / total_cells,
        "mean_burst_rate": (float(np.mean(burst_rates)) if burst_rates else 0.0),
        "mean_prediction_overlap": (
            float(np.mean(prediction_overlaps)) if prediction_overlaps else 0.0
        ),
        "mean_active_cells": (
            float(np.mean(active_cell_counts)) if active_cell_counts else 0.0
        ),
        "mean_predictive_cells": (
            float(np.mean(predictive_cell_counts)) if predictive_cell_counts else 0.0
        ),
    }


default_parameters = {
    # there are 2 (3) encoders: "value" (RDSE) & "time" (DateTime weekend, timeOfDay)
    "enc": {
        "value": {"resolution": 0.88, "size": 700, "sparsity": 0.02},
        "time": {"timeOfDay": (30, 1), "weekend": 21},
    },
    "predictor": {"sdrc_alpha": 0.1},
    "sp": {
        "boostStrength": 3.0,
        "columnCount": 1638,
        "localAreaDensity": 0.04395604395604396,
        "potentialPct": 0.85,
        "synPermActiveInc": 0.04,
        "synPermConnected": 0.13999999999999999,
        "synPermInactiveDec": 0.006,
    },
    "tm": {
        "activationThreshold": 17,
        "cellsPerColumn": 13,
        "initialPerm": 0.21,
        "maxSegmentsPerCell": 128,
        "maxSynapsesPerSegment": 64,
        "minThreshold": 10,
        "newSynapseCount": 32,
        "permanenceDec": 0.1,
        "permanenceInc": 0.1,
    },
    "anomaly": {"period": 1000},
}


def main(parameters=default_parameters, argv=None, verbose=True):
    if verbose:
        import pprint

        print("Parameters:")
        pprint.pprint(parameters, indent=4)
        print("")

    # Read the input file.
    records = []
    with open(_INPUT_FILE_PATH, "r") as fin:
        reader = csv.reader(fin)
        headers = next(reader)
        next(reader)
        next(reader)
        for record in reader:
            records.append(record)

    # Make the Encoders.  These will convert input data into binary representations.
    dateEncoder = DateEncoder(
        timeOfDay=parameters["enc"]["time"]["timeOfDay"],
        weekend=parameters["enc"]["time"]["weekend"],
    )

    scalarEncoderParams = RDSE_Parameters()
    scalarEncoderParams.size = parameters["enc"]["value"]["size"]
    scalarEncoderParams.sparsity = parameters["enc"]["value"]["sparsity"]
    scalarEncoderParams.resolution = parameters["enc"]["value"]["resolution"]
    scalarEncoder = RDSE(scalarEncoderParams)
    encodingWidth = dateEncoder.size + scalarEncoder.size
    enc_info = Metrics([encodingWidth], 999999999)

    # Make the HTM.  SpatialPooler & TemporalMemory & associated tools.
    spParams = parameters["sp"]
    sp = SpatialPooler(
        inputDimensions=(encodingWidth,),
        columnDimensions=(spParams["columnCount"],),
        potentialPct=spParams["potentialPct"],
        potentialRadius=encodingWidth,
        globalInhibition=True,
        localAreaDensity=spParams["localAreaDensity"],
        synPermInactiveDec=spParams["synPermInactiveDec"],
        synPermActiveInc=spParams["synPermActiveInc"],
        synPermConnected=spParams["synPermConnected"],
        boostStrength=spParams["boostStrength"],
        wrapAround=True,
    )
    sp_info = Metrics(sp.getColumnDimensions(), 999999999)

    tmParams = parameters["tm"]
    tm = TemporalMemory(
        columnDimensions=(spParams["columnCount"],),
        cellsPerColumn=tmParams["cellsPerColumn"],
        activationThreshold=tmParams["activationThreshold"],
        initialPermanence=tmParams["initialPerm"],
        connectedPermanence=spParams["synPermConnected"],
        minThreshold=tmParams["minThreshold"],
        maxNewSynapseCount=tmParams["newSynapseCount"],
        permanenceIncrement=tmParams["permanenceInc"],
        permanenceDecrement=tmParams["permanenceDec"],
        predictedSegmentDecrement=0.0,
        maxSegmentsPerCell=tmParams["maxSegmentsPerCell"],
        maxSynapsesPerSegment=tmParams["maxSynapsesPerSegment"],
    )
    tm_info = Metrics([tm.numberOfCells()], 999999999)

    anomaly_history = AnomalyLikelihood(parameters["anomaly"]["period"])

    predictor = Predictor(steps=[1, 5], alpha=parameters["predictor"]["sdrc_alpha"])
    predictor_resolution = 1

    # Iterate through every datum in the dataset, record the inputs & outputs.
    inputs = []
    anomaly = []
    anomalyProb = []
    predictions = {1: [], 5: []}
    for count, record in enumerate(records):

        # Convert date string into Python date object.
        dateString = datetime.datetime.strptime(record[0], "%m/%d/%y %H:%M")
        # Convert data value string into float.
        consumption = float(record[1])
        inputs.append(consumption)

        # Call the encoders to create bit representations for each value.  These are SDR objects.
        dateBits = dateEncoder.encode(dateString)
        consumptionBits = scalarEncoder.encode(consumption)

        # Concatenate all these encodings into one large encoding for Spatial Pooling.
        encoding = SDR(encodingWidth).concatenate([consumptionBits, dateBits])
        enc_info.addData(encoding)

        # Create an SDR to represent active columns, This will be populated by the
        # compute method below. It must have the same dimensions as the Spatial Pooler.
        activeColumns = SDR(sp.getColumnDimensions())

        # Execute Spatial Pooling algorithm over input space.
        sp.compute(encoding, True, activeColumns)
        sp_info.addData(activeColumns)
        # update metrices
        sp_result = SDR(activeColumns.dimensions)
        sp_result.setSDR(activeColumns)
        update_metrices(sp_result)

        # Execute Temporal Memory algorithm over active mini-columns.
        tm_output = tm.compute(activeColumns, learn=True)
        tm_info.addData(tm.getActiveCells().flatten())
        active_cells.sparse = tm.getActiveCells().sparse
        #predictive_cells.sparse = tm.getPredictiveCells().sparse
        winner_cells.sparse = tm.getWinnerCells().sparse
        tm_update_metrics(active_columns=activeColumns)
        print(
            "active=",
            len(active_cells.sparse),
            "predictive=",
            len(predictive_cells.sparse),
            "winner=",
            len(winner_cells.sparse),
        )
        # tm_outputs.append(tm_output.dense.astype(np.float32))
        print(tm_dignostics())

        # Predict what will happen, and then train the predictor based on what just happened.
        pdf = predictor.infer(tm.getActiveCells())
        for n in (1, 5):
            if pdf[n]:
                predictions[n].append(np.argmax(pdf[n]) * predictor_resolution)
            else:
                predictions[n].append(float("nan"))

        anomaly.append(tm.anomaly)
        anomalyProb.append(anomaly_history.compute(tm.anomaly))

        predictor.learn(
            count, tm.getActiveCells(), int(consumption / predictor_resolution)
        )

    # Print information & statistics about the state of the HTM.
    print("Encoded Input", enc_info)
    print("")
    print("Spatial Pooler Mini-Columns", sp_info)
    print(str(sp))
    print("")
    print("Temporal Memory Cells", tm_info)
    print(str(tm))
    print("")

    # Plot SP plots
    print("Plotting SP Plots", flush=True)
    plot_active_columns(active_counts)
    plot_temporal_overlap(temporal_overlaps)
    plot_column_utilization(column_usage, sp.getColumnDimensions()[0])
    plot_pca(sp_output_sdrs)
    plot_tsne(sp_output_sdrs)
    plot_umap(sp_output_sdrs)

    # Shift the predictions so that they are aligned with the input they predict.
    for n_steps, pred_list in predictions.items():
        for x in range(n_steps):
            pred_list.insert(0, float("nan"))
            pred_list.pop()

    # Calculate the predictive accuracy, Root-Mean-Squared
    accuracy = {1: 0, 5: 0}
    accuracy_samples = {1: 0, 5: 0}

    for idx, inp in enumerate(inputs):
        for (
            n
        ) in predictions:  # For each [N]umber of time steps ahead which was predicted.
            val = predictions[n][idx]
            if not math.isnan(val):
                accuracy[n] += (inp - val) ** 2
                accuracy_samples[n] += 1
    for n in sorted(predictions):
        accuracy[n] = (accuracy[n] / accuracy_samples[n]) ** 0.5
        print("Predictive Error (RMS)", n, "steps ahead:", accuracy[n])

    # Show info about the anomaly (mean & std)
    print("Anomaly Mean", np.mean(anomaly))
    print("Anomaly Std ", np.std(anomaly))

    # Plot the Predictions and Anomalies.
    if verbose:
        try:
            import matplotlib.pyplot as plt
        except:
            print("WARNING: failed to import matplotlib, plots cannot be shown.")
            return -accuracy[5]

        plt.subplot(2, 1, 1)
        plt.title("Predictions")
        plt.xlabel("Time")
        plt.ylabel("Power Consumption")
        plt.plot(
            np.arange(len(inputs)),
            inputs,
            "red",
            np.arange(len(inputs)),
            predictions[1],
            "blue",
            np.arange(len(inputs)),
            predictions[5],
            "green",
        )
        plt.legend(
            labels=(
                "Input",
                "1 Step Prediction, Shifted 1 step",
                "5 Step Prediction, Shifted 5 steps",
            )
        )

        plt.subplot(2, 1, 2)
        plt.title("Anomaly Score")
        plt.xlabel("Time")
        plt.ylabel("Power Consumption")
        inputs = np.array(inputs) / max(inputs)
        plt.plot(
            np.arange(len(inputs)),
            inputs,
            "black",
            np.arange(len(inputs)),
            anomaly,
            "blue",
            np.arange(len(inputs)),
            anomalyProb,
            "red",
        )
        plt.legend(labels=("Input", "Instantaneous Anomaly", "Anomaly Likelihood"))
        plt.show()

    return -accuracy[5]


if __name__ == "__main__":
    main()
