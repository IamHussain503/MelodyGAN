import os
import json
import numpy as np
from tqdm import tqdm
from transformers import T5Tokenizer, T5EncoderModel

# Paths and Parameters
CAPTIONS_FILE = "captions.txt"  # Path to captions file
MELODIES_DIR = "/root/MelodyGAN/melodies"  # Directory containing melody files in .mid format
OUTPUT_JSON = "harmonynet_dataset.json"
SAMPLE_RATE = 16000  # Target sample rate for audio processing

# Emotion Encoder
class EmotionExtractor:
    def __init__(self, model_name="t5-base"):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.encoder = T5EncoderModel.from_pretrained(model_name)

    def extract_emotion(self, text):
        """
        Extract emotion embedding from text description.
        Args:
            text (str): Input text.
        Returns:
            numpy.ndarray: Emotion embedding vector.
        """
        tokens = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            embedding = self.encoder(**tokens).last_hidden_state.mean(dim=1).squeeze()
        return embedding.numpy()

# Context Generator (Dummy Example)
def generate_context():
    """
    Generate random context features (e.g., emotion intensity, environment size, echo level).
    Returns:
        list: Random context vector.
    """
    return np.random.rand(3).tolist()

# Match Melody Files
def find_matching_melody_file(index):
    """
    Match melody files based on their index.
    Args:
        index (int): Line index from captions.txt.
    Returns:
        str: Path to the matching melody file.
    """
    melody_name = f"melody{index + 1}.mid"  # Assuming melody files follow the pattern melody1.mid, melody2.mid, etc.
    melody_path = os.path.join(MELODIES_DIR, melody_name)
    if os.path.exists(melody_path):
        return melody_path
    else:
        raise FileNotFoundError(f"Melody file not found: {melody_path}")

# Dataset Preparation
def prepare_harmonynet_dataset():
    """
    Prepare the HarmonyNet++ dataset by combining captions, melodies, and contexts.
    """
    # Initialize emotion extractor
    emotion_extractor = EmotionExtractor()

    # Read captions
    with open(CAPTIONS_FILE, "r", encoding="utf-8") as f:
        captions = [line.strip() for line in f.readlines()]

    # Dataset preparation
    harmonynet_data = []

    for idx, caption in tqdm(enumerate(captions), total=len(captions)):
        try:
            # Match melody file
            melody_file = find_matching_melody_file(idx)

            # Extract emotion embedding
            emotion_embedding = emotion_extractor.extract_emotion(caption)

            # Generate random context
            context = generate_context()

            # Append to dataset
            harmonynet_data.append({
                "caption": caption,
                "emotion_embedding": emotion_embedding.tolist(),
                "context": context,
                "melody_path": melody_file  # Save path to melody file for reference
            })

        except FileNotFoundError as e:
            print(f"Skipping sample due to missing melody file: {e}")

    # Save as JSON
    with open(OUTPUT_JSON, "w") as f:
        json.dump(harmonynet_data, f, indent=4)
    print(f"HarmonyNet++ dataset saved to {OUTPUT_JSON}")

# Main Execution
if __name__ == "__main__":
    prepare_harmonynet_dataset()