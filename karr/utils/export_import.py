import os
import pickle

def save_brain(encoder, tm, predictor, step_count, filename="brain.pkl"):
    print(f"Saving brain to {filename}...")
    with open(filename, "wb") as f:
        brain_data = {
            "encoder": encoder,
            "tm": tm,
            "predictor": predictor,
            "step_count": step_count,
        }
        pickle.dump(brain_data, f)


def load_brain(filename="brain.pkl"):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Could not find brain file: {filename}")
    print(f"Loading brain from {filename}...")
    with open(filename, "rb") as f:
        brain_data = pickle.load(f)
    return (
        brain_data["encoder"],
        brain_data["tm"],
        brain_data["predictor"],
        brain_data.get("step_count", 0),
    )
