import numpy as np
import random
import sys
import pickle
import re
import os

# HTM.Core Imports
from htm.bindings.sdr import SDR
from htm.bindings.algorithms import SpatialPooler, TemporalMemory, Classifier
from htm.bindings.encoders import ScalarEncoder, ScalarEncoderParameters

from karr.brain.l1.brain_l1 import BrainL1

# -------------------------------------------------------------------------
# SYMBOL ENCODER
# -------------------------------------------------------------------------
class SymbolEncoder:
    def __init__(self):
        self.encoder_params = ScalarEncoderParameters()
        self.encoder_params.activeBits = 25
        self.encoder_params.size = 1200
        self.encoder_params.minimum = 32
        self.encoder_params.maximum = 126
        self.encoder_params.periodic = False
        self.encoder_params.clipInput = True

        self.encoder = ScalarEncoder(self.encoder_params)
        self.width = self.encoder_params.size

    def encode(self, char):
        output_sdr = SDR(self.width)
        val = max(32, min(ord(char), 126))
        self.encoder.encode(val, output_sdr)
        return output_sdr

    def decode(self, class_probs):
        # Helper to map prediction back to char
        best_idx = np.argmax(class_probs)
        confidence = class_probs[best_idx]
        return chr(int(best_idx)), confidence


# -------------------------------------------------------------------------
# LAYER 1: PRIMARY CORTEX (Character & Transition Recognition)
# -------------------------------------------------------------------------
#class BrainL1:
#    def __init__(self, load_path=None):
#        print(" [Brain] Building L1 (Primary Visual Cortex)...")
#        self.column_count = 2048
#        self.cells_per_col = 32
#        self.input_size = 1200  # Matches Retina

#        # Output size = Total neurons (Columns * Depth)
#        self.output_size = self.column_count * self.cells_per_col

#        # Initialize algorithms (Default state)
#        self.sp = SpatialPooler(
#            inputDimensions=(self.input_size,),
#            columnDimensions=(self.column_count,),
#            potentialPct=0.85,
#            globalInhibition=True,
#            numActiveColumnsPerInhArea=45,
#            localAreaDensity=0,
#            synPermActiveInc=0.04,
#            synPermInactiveDec=0.006,
#            boostStrength=1.5,
#        )

#        self.tm = TemporalMemory(
#            columnDimensions=(self.column_count,),
#            cellsPerColumn=self.cells_per_col,
#            activationThreshold=12,
#            initialPermanence=0.22,
#            connectedPermanence=0.5,
#            minThreshold=9,
#            maxNewSynapseCount=30,
#        )

#        # LOAD PRE-TRAINED STATE IF AVAILABLE
#        if load_path and os.path.exists(load_path):
#            print("model exists...")
#            self.load_state(load_path)

#    def load_state(self, filename):
#        print(f" [L1] Loading pre-trained memory from {filename}...")
#        try:
#            with open(filename, "rb") as f:
#                data = pickle.load(f)
#                # Overwrite the empty algorithms with the trained ones
#                # The dictionary keys match those saved in bio_brain_l1.py
#                if "sp" in data:
#                    self.sp = data["sp"]
#                if "tm" in data:
#                    self.tm = data["tm"]
#                print(" [L1] Memory integration successful.")
#        except Exception as e:
#            print(
#                f" [L1] Warning: Failed to load {filename}. Starting L1 fresh. Error: {e}"
#            )

#    def compute(self, input_sdr, learn=True):
#        # 1. SP (Input -> Active Columns)
#        active_columns = SDR(self.column_count)
#        self.sp.compute(input_sdr, learn, active_columns)

#        # 2. TM (Active Columns -> Active Cells)
#        self.tm.compute(active_columns, learn=learn)

#        # Return the Active Cells (The full neural state)
#        # This is what L2 will "see"
#        return (self.tm.anomaly, self.tm.getActiveCells())

#    def reset(self):
#        self.tm.reset()


# -------------------------------------------------------------------------
# LAYER 2: ASSOCIATION CORTEX (Word & Pattern Recognition)
# -------------------------------------------------------------------------
class BrainL2:
    def __init__(self, l1_output_size):
        print(" [Brain] Building L2 (Association Cortex)...")

        # L2 is roughly the same size as L1, but it connects to L1's massive output
        self.input_size = l1_output_size
        self.column_count = 2048
        self.cells_per_col = 32

        # SPATIAL POOLER
        # Input: The 65,536 neurons of L1
        # Task: Cluster similar firing patterns of L1
        self.sp = SpatialPooler(
            inputDimensions=(self.input_size,),
            columnDimensions=(self.column_count,),
            potentialPct=0.15,  # Lower connectivity because input is huge
            globalInhibition=True,
            numActiveColumnsPerInhArea=40,
            localAreaDensity=0,
            synPermActiveInc=0.05,
            synPermInactiveDec=0.005,
            boostStrength=1.0,
        )

        # TEMPORAL MEMORY
        # Task: Learn sequences of "chunks" (Word transitions)
        self.tm = TemporalMemory(
            columnDimensions=(self.column_count,),
            cellsPerColumn=self.cells_per_col,
            activationThreshold=13,
            initialPermanence=0.21,
            connectedPermanence=0.5,
            minThreshold=10,
            maxNewSynapseCount=32,
        )

        # Classifier for L2 to see if it predicts WORDS (not implemented fully here yet)
        # We will use anomaly to judge L2 learning
        self.classifier = Classifier()

    def compute(self, l1_active_cells, learn=True):
        # 1. SP (L1 Active Cells -> L2 Active Columns)
        active_columns = SDR(self.column_count)
        self.sp.compute(l1_active_cells, learn, active_columns)

        # 2. TM (L2 Sequence learning)
        self.tm.compute(active_columns, learn=learn)

        return (self.tm.anomaly, self.tm.getActiveCells())

    def reset(self):
        self.tm.reset()


# -------------------------------------------------------------------------
# THE HIERARCHY CONTROLLER
# -------------------------------------------------------------------------
class DeepBrain:
    def __init__(self, l1_model_path="brain_l1.pkl"):
        self.retina = SymbolEncoder()
        # Initialize L1, attempting to load the pre-trained infant brain
        self.l1 = BrainL1(load_path=l1_model_path)
        # Initialize L2 fresh (Childhood start)
        self.l2 = BrainL2(l1_output_size=self.l1.output_size)
        self.step_count = 0

    def process_char(self, char, learn=True):
        self.step_count += 1

        # 1. Retina Encode
        sdr_input = self.retina.encode(char)

        # 2. L1 Compute (Feedforward)
        # L1 sees the character
        # Note: If L1 is pre-trained, we might choose to turn off L1 learning
        # to 'freeze' the alphabet knowledge, or keep it on to refine it.
        # For now, we keep it on (plasticity).
        (_, l1_active_cells) = self.l1.compute(sdr_input,)

        # 3. L2 Compute (Feedforward)
        # L2 sees the neural state of L1
        l2_anomaly = self.l2.compute(l1_active_cells, learn=learn)

        return l2_anomaly

    def reset(self):
        self.l1.reset()
        self.l2.reset()

    def save(self, filename="deep_brain.pkl"):
        """Saves the entire brain state."""
        print(f"Saving deep brain state (Step: {self.step_count}) to {filename}...")
        data = {
            "l1_sp": self.l1.sp,
            "l1_tm": self.l1.tm,
            "l2_sp": self.l2.sp,
            "l2_tm": self.l2.tm,
            "step_count": self.step_count,
        }
        with open(filename, "wb") as f:
            pickle.dump(data, f)
        print("Brain saved.")

    def load(self, filename="deep_brain.pkl"):
        if not os.path.exists(filename):
            print(
                f"No deep brain checkpoint found at {filename}. Starting with current state."
            )
            return

        print(f"Loading deep brain from {filename}...")
        with open(filename, "rb") as f:
            data = pickle.load(f)
            self.l1.sp = data["l1_sp"]
            self.l1.tm = data["l1_tm"]
            self.l2.sp = data["l2_sp"]
            self.l2.tm = data["l2_tm"]
            self.step_count = data.get("step_count", 0)
        print("Deep brain loaded.")


# -------------------------------------------------------------------------
# DATA CORPUS HANDLER
# -------------------------------------------------------------------------
class FileCorpus:
    """Reads a text file, cleans it, and yields characters."""

    def __init__(self, filepath):
        self.filepath = filepath
        if not os.path.exists(filepath):
            # Create a dummy file if it doesn't exist for demonstration
            print(f"File {filepath} not found. Creating dummy training data.")
            with open(filepath, "w") as f:
                f.write("the cat sat on the mat. the dog ran. " * 100)

    def stream_chars(self, limit=None):
        """Yields clean characters one by one."""
        count = 0
        with open(self.filepath, "r", encoding="utf-8") as f:
            while True:
                chunk = f.read(1024)  # Read in chunks
                if not chunk:
                    break

                # Cleaning: Allow a-z, A-Z, space, and period.
                # Remove numbers and other symbols to keep Retina happy.
                cleaned = re.sub(r"[^a-zA-Z \.]", " ", chunk)

                # Compress multiple spaces
                cleaned = re.sub(r"\s+", " ", cleaned)

                for char in cleaned:
                    yield char
                    count += 1
                    if limit and count >= limit:
                        return


# -------------------------------------------------------------------------
# RUNNERS
# -------------------------------------------------------------------------
def run_large_scale_training():
    """Simulates training on a book/corpus."""
    # This will check for 'brain_l1.pkl' automatically
    brain = DeepBrain(l1_model_path="brain_l1.pkl")

    # Check if we have a deep brain checkpoint to resume
    if os.path.exists("deep_brain.pkl"):
        brain.load("deep_brain.pkl")

    # 1. Setup Data
    corpus_file = "dataset/training_corpus.txt"
    corpus = FileCorpus(corpus_file)

    print(f"\n=== TRAINING ON CORPUS: {corpus_file} ===")
    print("Stream reading text... (Press Ctrl+C to stop and save)")

    total_anomaly = 0
    window = 100

    try:
        # Train on 10,000 characters (Adjust as needed)
        # Since this is a generator, you can increase limit or remove it for full file
        for i, char in enumerate(corpus.stream_chars(limit=10000)):

            # Reset brain on sentence endings (periods)
            if char == ".":
                brain.reset()
                continue

            (anomaly, _) = brain.process_char(char, learn=True)
            total_anomaly += anomaly

            # Status update every 100 chars
            if i % window == 0 and i > 0:
                avg = total_anomaly / window
                bar = "#" * int((1.0 - avg) * 20)
                sys.stdout.write(
                    f"\rStep {i}: [{bar:<20}] Anom: {avg:.2f} | Input: '{char}'"
                )
                sys.stdout.flush()
                total_anomaly = 0

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")

    print(f"\n\nTraining complete (Steps: {brain.step_count}).")
    brain.save("deep_brain.pkl")

    # TEST
    print("\n=== TEST AFTER CORPUS ===")
    test_phrase = "the cat meows "  # Common phrase in our dummy corpus
    brain.reset()
    for char in test_phrase:
        anom = brain.process_char(char, learn=False)
        print(f"Char: '{char}' | L2 Surprise: {anom:.2f}")


#if __name__ == "__main__":
#    run_large_scale_training()
