import torch
from torch.utils.data import Dataset
import pretty_midi
import json

class HarmonyNetDataset(Dataset):
    def __init__(self, json_file):
        with open(json_file, "r") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # Extract inputs
        caption = sample["caption"]
        emotion_embedding = torch.tensor(sample["emotion_embedding"], dtype=torch.float32)
        context = torch.tensor(sample["context"], dtype=torch.float32)

        # Convert MIDI file to numeric representation
        midi_path = sample["melody_path"]
        melody = self._process_midi(midi_path)

        return emotion_embedding, context, melody

    def _process_midi(self, midi_path):
        """Convert MIDI file into a numeric representation (e.g., pitches, durations)."""
        midi_data = pretty_midi.PrettyMIDI(midi_path)
        melody = []
        for instrument in midi_data.instruments:
            for note in instrument.notes:
                melody.append([note.pitch, note.start, note.end - note.start])
        return torch.tensor(melody, dtype=torch.float32)


