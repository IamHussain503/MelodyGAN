import json
import os

def validate_dataset(json_file):
    with open(json_file, "r") as f:
        dataset = json.load(f)

    for idx, sample in enumerate(dataset):
        if not os.path.exists(sample["midi_path"]):
            print(f"MIDI file missing: {sample['midi_path']} (Index {idx})")
        if not isinstance(sample["emotion_embedding"], list):
            print(f"Invalid emotion embedding: {sample['emotion_embedding']} (Index {idx})")
        if not isinstance(sample["context"], list):
            print(f"Invalid context: {sample['context']} (Index {idx})")
    print("Dataset validation complete!")

# Validate the dataset
validate_dataset("harmonynet_dataset.json")
