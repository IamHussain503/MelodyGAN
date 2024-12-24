import os
import json
import numpy as np
from datasets import load_dataset
from transformers import T5Tokenizer, T5EncoderModel
from tqdm import tqdm

# Paths and Parameters
HUGGINGFACE_DATASET = "AudioSubnet/ttm_validation_dataset_10sec"
MELODIES_DIR = "./melodies"
OUTPUT_JSON = "harmonynet_dataset.json"

# Emotion Extractor
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

# Match Melody Files
def load_melody_file(melody_name):
    """
    Load melody file from the ./melodies directory.
    Args:
        melody_name (str): Name of the melody file.
    Returns:
        numpy.ndarray: Melody data (e.g., MIDI representation).
    """
    melody_path = os.path.join(MELODIES_DIR, melody_name)
    if os.path.exists(melody_path):
        return np.load(melody_path, allow_pickle=True)  # Assuming melody files are stored in NumPy format
    else:
        raise FileNotFoundError(f"Melody file not found: {melody_path}")

# Dataset Preparation
def prepare_harmonynet_dataset():
    """
    Prepare the HarmonyNet++ dataset by combining captions, melodies, and WAV file identifiers.
    """
    # Load Hugging Face dataset
    dataset = load_dataset(HUGGINGFACE_DATASET)

    # Check available fields in the dataset
    print("Dataset Features:", dataset["train"].features)

    # Initialize emotion extractor
    emotion_extractor = EmotionExtractor()

    # Dataset preparation
    harmonynet_data = []

    for sample in tqdm(dataset["train"]):  # Replace 'train' with the relevant split if needed
        # Check and replace the 'text' key with the actual caption field name
        caption = sample.get("text", None)  # Replace 'text' with the correct field name if different
        if caption is None:
            print(f"Skipping sample due to missing caption: {sample}")
            continue

        audio_path = sample["audio"]["path"]
        wav_file_name = os.path.basename(audio_path)  # WAV file identifier
        melody_name = wav_file_name.replace(".wav", ".npy")  # Corresponding melody file

        try:
            # Extract emotion embedding
            emotion_embedding = emotion_extractor.extract_emotion(caption)
            melody = load_melody_file(melody_name)

            # Append to dataset
            harmonynet_data.append({
                "wav_file_name": wav_file_name,  # WAV file identifier for reference
                "caption": caption,
                "emotion_embedding": emotion_embedding.tolist(),
                "melody": melody.tolist()
            })

        except FileNotFoundError as e:
            print(f"Skipping sample due to missing file: {e}")

    # Save as JSON
    with open(OUTPUT_JSON, "w") as f:
        json.dump(harmonynet_data, f, indent=4)
    print(f"HarmonyNet++ dataset saved to {OUTPUT_JSON}")

# Main Execution
if __name__ == "__main__":
    prepare_harmonynet_dataset()
