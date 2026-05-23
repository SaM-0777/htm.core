import os
import numpy as np

from htm.bindings.sdr import SDR, Metrics
from htm.bindings.algorithms import TemporalMemory, Predictor

from karr.utils.export_import import save_brain, load_brain
from karr.encoders.alphabet import AlphabetEncoder


CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

def char_to_idx(char):
    try:
        return CHARS.index(char)
    except ValueError:
        return -1


def idx_to_char(idx):
    if 0 <= idx < len(CHARS):
        return CHARS[idx]
    return "?"

def predict_next_char(char, encoder, tm, predictor):
    if char not in CHARS:
        return "?", 0.0
    tm.reset()
    input_sdr = SDR((encoder.getWidth(),))
    encoder.encode(char, input_sdr)
    tm.compute(input_sdr, learn=False)
    pdf = predictor.infer(tm.getActiveCells())
    if pdf[1]:
        pred_index = np.argmax(pdf[1])
        confidence = pdf[1][pred_index]
        return idx_to_char(pred_index), confidence
    return "?", 0.0

def train_sequence(
    sequence, encoder, tm, predictor, step_count, iterations=10, verbose=True
):
    print(f"--- Training on: {sequence[:10]}... ({iterations} iters) ---")
    for iteration in range(iterations):
        tm.reset()
        for char in sequence:
            input_sdr = SDR((encoder.getWidth(),))
            encoder.encode(char, input_sdr)
            tm.compute(input_sdr, learn=True)
            char_idx = char_to_idx(char)
            predictor.learn(step_count, tm.getActiveCells(), char_idx)
            step_count += 1
    return step_count


default_parameters = {
    "enc": {"w": 21, "n": 1024},
    "predictor": {"sdrc_alpha": 0.1},
    "tm": {
        "columnDimensions": (1024,),
        "cellsPerColumn": 32,
        "activationThreshold": 13,
        "initialPerm": 0.21,
        "minThreshold": 10,
        "newSynapseCount": 21,
        "permanenceInc": 0.1,
        "permanenceDec": 0.1,
        "maxSegmentsPerCell": 128,
        "maxSynapsesPerSegment": 64,
    },
}


def main(parameters=default_parameters):
    brain_file = "alphabet_brain.pkl"

    print("Initializing New Smart Brain...")
    enc_params = parameters["enc"]
    encoder = AlphabetEncoder(w=enc_params["w"], n=enc_params["n"])
    enc_info = Metrics([encoder.getWidth()], 999999)

    tm_params = parameters["tm"]
    tm = TemporalMemory(
        columnDimensions=tm_params["columnDimensions"],
        cellsPerColumn=tm_params["cellsPerColumn"],
        activationThreshold=tm_params["activationThreshold"],
        initialPermanence=tm_params["initialPerm"],
        connectedPermanence=0.5,
        minThreshold=tm_params["minThreshold"],
        maxNewSynapseCount=tm_params["newSynapseCount"],
        permanenceIncrement=tm_params["permanenceInc"],
        permanenceDecrement=tm_params["permanenceDec"],
        predictedSegmentDecrement=0.001,
        maxSegmentsPerCell=tm_params["maxSegmentsPerCell"],
        maxSynapsesPerSegment=tm_params["maxSynapsesPerSegment"],
    )
    tm_info = Metrics([tm.numberOfCells()], 999999999)

    predictor = Predictor(steps=[1], alpha=parameters["predictor"]["sdrc_alpha"])
    step_count = 0

    # 1. Visualize Similarity (Proof it works)
    print("\n--- Similarity Check ---")
    sdr_A = SDR((1024,))
    sdr_a = SDR((1024,))
    sdr_B = SDR((1024,))

    encoder.encode("A", sdr_A)
    encoder.encode("a", sdr_a)
    encoder.encode("B", sdr_B)

    overlap_Aa = sdr_A.getOverlap(sdr_a)
    overlap_AB = sdr_A.getOverlap(sdr_B)

    print(f"Bits for 'A': {sdr_A.getSum()}")
    print(f"Bits for 'a': {sdr_a.getSum()}")
    print(f"Overlap 'A' vs 'a': {overlap_Aa} (High overlap = Encoder recognizes similarity)")
    print(f"Overlap 'A' vs 'B': {overlap_AB} (Low overlap = Encoder sees difference)\n")

    step_count = train_sequence(CHARS, encoder, tm, predictor, step_count, iterations=15)

    save_brain(encoder, tm, predictor, step_count, brain_file)

    print("\n--- Non-Interactive Prediction Log ---")

    test_chars = ['A', 'a', 'M', 'm', 'Y', 'y', 's', 'h', 'Q', 'q', 'o']
    for char in test_chars:
        pred_char, conf = predict_next_char(char, encoder, tm, predictor)
        
        context_str = "Unknown"
        if pred_char != "?":
            if char.isupper() and pred_char.isupper(): context_str = "Maintained Uppercase"
            elif char.islower() and pred_char.islower(): context_str = "Maintained Lowercase"
            elif char.isupper() != pred_char.isupper(): context_str = "Mixed Case (Possible Confusion)"

        print(f"Input: '{char}' -> Predicted: '{pred_char}' (Confidence: {conf:.2f})")



if __name__ == "__main__":
    main()