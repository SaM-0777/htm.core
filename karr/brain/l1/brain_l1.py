import numpy as np
import random
import sys
import pickle
import re
import os
from htm.bindings.sdr import SDR
from htm.bindings.algorithms import SpatialPooler, TemporalMemory
from htm.bindings.encoders import ScalarEncoder, ScalarEncoderParameters

from encoder import SymbolEncoder


# L1 Layer (Simplified for scratch training)
class BrainL1:
    def __init__(self, model_path=None):
        self.column_count = 2048
        self.cells_per_col = 32
        self.input_size = 1200  # Matches encoder

        # Output size = Total neurons (Columns * Depth)
        self.output_size = self.column_count * self.cells_per_col

        self.sp = SpatialPooler(
            inputDimensions=(self.input_size,),
            columnDimensions=(self.column_count,),
            potentialPct=0.85,
            globalInhibition=True,
            numActiveColumnsPerInhArea=45,
            localAreaDensity=0,
            synPermActiveInc=0.04,
            synPermInactiveDec=0.006,
            boostStrength=1.5,
        )
        self.tm = TemporalMemory(
            columnDimensions=(self.column_count,),
            cellsPerColumn=self.cells_per_col,
            activationThreshold=12,
            initialPermanence=0.22,
            connectedPermanence=0.5,
            minThreshold=9,
            maxNewSynapseCount=30,
        )

        # LOAD PRE-TRAINED STATE IF AVAILABLE
        if model_path and os.path.exists(model_path):
            print("model exists...")
            self.load_state(model_path)

    def compute(self, input_sdr, learn_sp=True, learn_tm=False):
        active_columns = SDR(self.column_count)
        self.sp.compute(input_sdr, learn_sp, active_columns)
        self.tm.compute(active_columns, learn=learn_tm)
        return (self.tm.anomaly, self.tm.getActiveCells())

    def reset(self):
        self.tm.reset()

    def save_state(self, filename="trained_l1.pkl"):
        print(f"Saving L1 model to {filename}...")
        data = {"sp": self.sp, "tm": self.tm}
        with open(filename, "wb") as f:
            pickle.dump(data, f)

    def load_state(self, filename):
        print(f" [L1] Loading pre-trained memory from {filename}...")
        try:
            with open(filename, "rb") as f:
                data = pickle.load(f)

                if "sp" in data:
                    self.sp = data["sp"]
                if "tm" in data:
                    self.tm = data["tm"]

                print(" [L1] Memory integration successful.")
        except Exception as e:
            print(
                f" [L1] Warning: Failed to load {filename}. Starting L1 fresh. Error: {e}"
            )


# Corpus Handler (Loads and shuffles for random training)
class Corpus:
    def __init__(self, filepath="dataset/l1_training_corpus.txt"):
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        # Clean to only allowed chars/symbols
        # self.chars = list(re.sub(r"[^a-zA-Z0-9 \.,\"\'\?!;:()-]", "", text))
        self.chars = list(text)
        print(f"Loaded {len(self.chars)} characters for training.")

    def get_shuffled_stream(self):
        shuffled = self.chars[:]
        random.shuffle(shuffled)
        for char in shuffled:
            yield char


# Main Training Function
def train_l1_scratch(
    epochs=5, model_path=None, corpus_file="dataset/simple_lowercase.txt"
):
    encoder = SymbolEncoder()
    l1 = BrainL1(model_path)
    corpus = Corpus(corpus_file)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        total_anomaly = 0
        count = 0
        for char in corpus.chars:
            sdr = encoder.encode(char)
            # Phase 1: Learn SP only (stable reps)
            (anomaly, _) = l1.compute(
                sdr, learn_sp=True, learn_tm=True
            )  # TM after epoch 1
            total_anomaly += anomaly
            count += 1
            # Reset TM every 5 chars to break sequences
            # if count % 5 == 0:
            #    l1.reset()
            # if count % 100 == 0:
            #    print(
            #        f"Step {count}: Avg Anomaly: {total_anomaly / count:.2f} | Char: '{char}'"
            #    )

        avg_anomaly = total_anomaly / count if count > 0 else 0
        print(f"Epoch {epoch + 1} Average Anomaly: {avg_anomaly:.2f}")
        if avg_anomaly < 0.05:
            print("Converged early.")
            break

    l1.save_state("trained_l1.pkl")
    print("L1 training complete. Use 'trained_l1.pkl' in your main code.")

    # predict some of the symbols at random and record the anomaly
    sdr = encoder.encode("f")
    (anomaly, _) = l1.compute(sdr, learn_sp=False, learn_tm=False)  # pure prediction
    print(f"Prediction for f anomaly {anomaly}")


class BrainL2:
    def __init__(self, input_size, model_path=None) -> None:
        self.column_count = 4096  # Larger for higher abstraction
        self.cells_per_col = 64

        self.input_size = input_size
        self.output_size = self.column_count * self.cells_per_col

        self.sp = SpatialPooler(
            inputDimensions=(input_size,),
            columnDimensions=(self.column_count,),
            potentialPct=0.8,
            globalInhibition=True,
            numActiveColumnsPerInhArea=60,
            localAreaDensity=0,
            synPermActiveInc=0.03,
            synPermInactiveDec=0.005,
            boostStrength=2.0,
        )
        self.tm = TemporalMemory(
            columnDimensions=(self.column_count,),
            cellsPerColumn=self.cells_per_col,
            activationThreshold=15,
            initialPermanence=0.25,
            connectedPermanence=0.5,
            minThreshold=10,
            maxNewSynapseCount=40,
        )
        if model_path and os.path.exists(model_path):
            self.load_state(model_path)

    def compute(self, input_sdr, learn_sp=True, learn_tm=True):
        active_columns = SDR(self.column_count)
        self.sp.compute(input_sdr, learn_sp, active_columns)
        self.tm.compute(active_columns, learn=learn_tm)
        return (self.tm.anomaly, self.tm.getActiveCells())

    def reset(self):
        self.tm.reset()

    def save_state(self, filename="trained_l2.pkl"):
        print(f"Saving L2 model to {filename}...")
        data = {"sp": self.sp, "tm": self.tm}
        with open(filename, "wb") as f:
            pickle.dump(data, f)

    def load_state(self, filename):
        print(f"[L2] Loading from {filename}...")
        with open(filename, "rb") as f:
            data = pickle.load(f)
            self.sp = data["sp"]
            self.tm = data["tm"]


class WordCorpus:
    def __init__(
        self, filepath="dataset/word_phrases.txt"
    ):  # e.g., lines like "a for apple\nb for ball"
        with open(filepath, "r", encoding="utf-8") as f:
            self.phrases = [line.strip() for line in f.readlines() if line.strip()]
        print(f"Loaded {len(self.phrases)} phrases.")

    def get_shuffled_stream(self):
        shuffled = self.phrases[:]
        random.shuffle(shuffled)
        for phrase in shuffled:
            yield phrase


def train_l2(
    epochs=10,
    l1_path="trained_l1.pkl",
    l2_path=None,
    corpus_file="dataset/word_phrases.txt",
):
    encoder = SymbolEncoder()
    l1 = BrainL1(l1_path)  # Load pre-trained L1
    l2 = BrainL2(input_size=l1.output_size, model_path=l2_path)
    corpus = WordCorpus(corpus_file)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        total_anomaly = 0
        count = 0
        for phrase in corpus.phrases:
            l1.reset()  # Reset L1 per phrase
            l2.reset()
            phrase_anomaly = 0  # Track per phrase for logging
            step_count = 0

            for char in phrase:
                char_sdr = encoder.encode(char)
                (_, active_cells) = l1.compute(
                    char_sdr, learn_sp=False, learn_tm=False
                )  # Freeze L1

                (anomaly, _) = l2.compute(active_cells, learn_sp=True, learn_tm=True)

                phrase_anomaly += anomaly
                step_count += 1
                total_anomaly += anomaly  # Global tracking
                count += 1  # Now count steps across all phrases

            avg_phrase_anomaly = phrase_anomaly / step_count if step_count > 0 else 0
            # Union L1 SDRs for the whole phrase (or process sequentially)
            # union_sdr = SDR(l1.output_size)
            # for sdr in phrase_sdrs:
            #    union_sdr.union(sdr)  # Or feed sequentially to L2 for temporal learning

            ## L2 learns on the union/sequence
            # (anomaly, _) = l2.compute(union_sdr, learn_sp=True, learn_tm=True)
            # total_anomaly += anomaly
            # count += 1

            if count % 10 == 0:
                print(
                    f"Step {count}: Avg Anomaly: {total_anomaly / count:.2f} | Phrase: '{phrase}'"
                )

        avg_anomaly = total_anomaly / count
        print(f"Epoch {epoch + 1} Average Anomaly: {avg_anomaly:.2f}")
        if avg_anomaly < 0.1:  # Tighter threshold for L2
            break

    l2.save_state("trained_l2.pkl")
    print("L2 training complete.")


class BrainL3:
    def __init__(self, input_size, model_path=None) -> None:
        self.column_count = 8192  # Even larger for sentences
        self.cells_per_col = 128

        self.input_size = input_size
        self.output_size = self.column_count * self.cells_per_col

        self.sp = SpatialPooler(
            inputDimensions=(input_size,),
            columnDimensions=(self.column_count,),
            potentialPct=0.75,
            globalInhibition=True,
            numActiveColumnsPerInhArea=80,
            localAreaDensity=0,
            synPermActiveInc=0.025,
            synPermInactiveDec=0.004,
            boostStrength=2.5,
        )
        self.tm = TemporalMemory(
            columnDimensions=(self.column_count,),
            cellsPerColumn=self.cells_per_col,
            activationThreshold=18,
            initialPermanence=0.28,
            connectedPermanence=0.5,
            minThreshold=12,
            maxNewSynapseCount=50,
        )

        if model_path and os.path.exists(model_path):
            self.load_state(model_path)

    def compute(self, input_sdr, learn_sp=True, learn_tm=True):
        active_columns = SDR(self.column_count)
        self.sp.compute(input_sdr, learn_sp, active_columns)
        self.tm.compute(active_columns, learn=learn_tm)
        return (self.tm.anomaly, self.tm.getActiveCells())

    def reset(self):
        self.tm.reset()

    def save_state(self, filename="trained_l3.pkl"):
        print(f"Saving L3 model to {filename}...")
        data = {"sp": self.sp, "tm": self.tm}
        with open(filename, "wb") as f:
            pickle.dump(data, f)

    def load_state(self, filename):
        print(f"[L3] Loading from {filename}...")
        with open(filename, "rb") as f:
            data = pickle.load(f)
            self.sp = data["sp"]
            self.tm = data["tm"]


class SentenceCorpus:
    def __init__(self, filepath="dataset/simple_sentences.txt"):
        with open(filepath, "r", encoding="utf-8") as f:
            self.sentences = [line.strip() for line in f.readlines() if line.strip()]
        print(f"Loaded {len(self.sentences)} sentences.")

    def get_shuffled_stream(self, repeats):
        expanded = []
        for sentence in self.sentences:
            expanded.extend(
                [sentence] * repeats
            )

        random.shuffle(expanded)
        for sentence in expanded:
            yield sentence

def train_l3(
    epochs=50,
    l1_path="trained_l1.pkl",
    l2_path="trained_l2.pkl",
    l3_path=None,
    corpus_file="dataset/simple_sentences.txt",
):
    encoder = SymbolEncoder()
    l1 = BrainL1(l1_path)
    l2 = BrainL2(input_size=l1.output_size, model_path=l2_path)
    l3 = BrainL3(
        input_size=l2.output_size, model_path=l3_path
    )  # Adjust if using unions
    corpus = SentenceCorpus(corpus_file)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        total_anomaly = 0
        count = 0

        for sentence in corpus.sentences:
            words = re.findall(r"\w+|[^\w\s]", sentence)
            l1.reset()
            l2.reset()
            l3.reset()  # Reset per sentence
            sentence_anomaly = 0
            step_count = 0

            for word in words:
                l1.reset()  # Optional: Reset L1 per word for isolation
                word_anomaly = 0
                char_count = 0
                l2_cells = SDR(l2.column_count * l2.cells_per_col)

                # First pass: Inference to check anomaly
                for char in word:
                    char_sdr = encoder.encode(char)
                    (_, l1_cells) = l1.compute(char_sdr, learn_sp=False, learn_tm=False)
                    (l2_anom, l2_cells) = l2.compute(
                        l1_cells, learn_sp=False, learn_tm=False
                    )
                    word_anomaly += l2_anom
                    char_count += 1

                avg_word_anom = word_anomaly / char_count if char_count > 0 else 0
                l3_input_cells = SDR(l3.input_size)

                if avg_word_anom > 0.5:
                    print(
                        f"High anomaly for '{word}' ({avg_word_anom:.2f}) - Learning in L2."
                    )
                    l1.reset()  # Reset for clean re-feed
                    l2.reset()
                    l2_cells_post = SDR(
                        l2.column_count * l2.cells_per_col
                    )  # Dummy init
                    for char in word:
                        char_sdr = encoder.encode(char)
                        (_, l1_cells) = l1.compute(
                            char_sdr, learn_sp=False, learn_tm=False
                        )  # Still freeze L1
                        (_, l2_cells_post) = l2.compute(
                            l1_cells, learn_sp=True, learn_tm=True
                        )  # Learn in L2

                    l1.reset()
                    l2.reset()
                    post_learn_anom = 0
                    post_char_count = 0

                    for char in word:
                        char_sdr = encoder.encode(char)
                        (_, l1_cells) = l1.compute(
                            char_sdr, learn_sp=False, learn_tm=False
                        )
                        (l2_anom_post, l2_cells_post_temp) = l2.compute(
                            l1_cells, learn_sp=False, learn_tm=False
                        )  # Temp to not overwrite
                        post_learn_anom += l2_anom_post
                        post_char_count += 1
                    avg_post_learn = (
                        post_learn_anom / post_char_count if post_char_count > 0 else 0
                    )
                    print(f"Post-learning anomaly for '{word}': {avg_post_learn:.2f}")
                    l3_input_cells = l2_cells_post
                
                else:
                    l1.reset()
                    l2.reset()
                    for char in word:
                        char_sdr = encoder.encode(char)
                        (_, l1_cells) = l1.compute(char_sdr, learn_sp=False, learn_tm=False)
                        (_, l2_cells) = l2.compute(l1_cells, learn_sp=True, learn_tm=True)  # Mild learn even for low anomaly
                    l3_input_cells = l2_cells  # Use updated post-reinforcement cells
            
                (l3_anom, _) = l3.compute(l3_input_cells, learn_sp=True, learn_tm=True)
                sentence_anomaly += l3_anom
                step_count += 1
                total_anomaly += l3_anom
                count += 1
            
            avg_sentence_anom = sentence_anomaly / step_count if step_count > 0 else 0
            if count % 50 == 0:  # Log every 50 steps
                print(f"Step {count}: Avg Anomaly (this sentence): {avg_sentence_anom:.2f} | Sentence: '{sentence}'")
        
        avg_anomaly = total_anomaly / count if count > 0 else 0
        print(f"Epoch {epoch + 1} Average Anomaly: {avg_anomaly:.2f}")
        if avg_anomaly < 0.20:
            break
    
    l3.save_state("trained_l3.pkl")
    l2.save_state("trained_l2_updated.pkl")  # Save updated L2 after adaptive learning
    print("L3 training complete. L2 updated with new words.")


if __name__ == "__main__":
    # train_l1_scratch(
    #    epochs=100,
    #    model_path="trained_l1.pkl",
    #    corpus_file="dataset/simple_uppercase.txt",
    # )

    train_l3(
        epochs=50,
        l1_path="trained_l1.pkl",
        l2_path="trained_l2.pkl",
        l3_path=None,
        corpus_file="dataset/simple_sentences.txt",
    )
