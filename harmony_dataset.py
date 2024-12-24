import torch
from torch.utils.data import Dataset
import pretty_midi
import json

class HarmonyNetDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        """
        Dataset class for HarmonyNet++.
        
        Args:
            data (list): Pre-split dataset as a list of dictionaries.
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # Extract inputs
        emotion_embedding = torch.tensor(sample["emotion_embedding"], dtype=torch.float32)
        context = torch.tensor(sample["context"], dtype=torch.float32)
        melody_path = sample["midi_path"]
        melody = self._process_midi(melody_path)

        return emotion_embedding, context, melody

    def _process_midi(self, midi_path):
        """Convert MIDI file into a numeric representation (e.g., pitches, durations)."""
        try:
            midi_data = pretty_midi.PrettyMIDI(midi_path)
            melody = []
            for instrument in midi_data.instruments:
                for note in instrument.notes:
                    melody.append([note.pitch, note.start, note.end - note.start])
            if len(melody) == 0:  # Check if melody is empty
                raise ValueError(f"No valid notes found in {midi_path}")
            return torch.tensor(melody, dtype=torch.float32)  # Variable-length tensor
        except Exception as e:
            print(f"Error processing MIDI file {midi_path}: {e}")
            return torch.empty(0, 3)  # Return empty tensor to be filtered later





