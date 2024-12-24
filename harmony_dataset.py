import torch
from torch.utils.data import Dataset
import pretty_midi
import json
import random

class HarmonyNetDataset(torch.utils.data.Dataset):
    def __init__(self, data, transpose_range=2):
        """
        Dataset class for HarmonyNet++ with data augmentation.

        Args:
            data (list): Pre-split dataset as a list of dictionaries.
            transpose_range (int): Number of semitones to transpose melodies for augmentation.
        """
        self.data = data
        self.transpose_range = transpose_range

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # Extract inputs
        emotion_embedding = torch.tensor(sample["emotion_embedding"], dtype=torch.float32)
        context = torch.tensor(sample["context"], dtype=torch.float32)
        melody_path = sample["midi_path"]
        melody = self._process_midi(melody_path)

        # Apply augmentation
        if melody is not None and self.transpose_range > 0:
            melody = self._augment_melody(melody)

        return emotion_embedding, context, melody

    def _augment_melody(self, melody):
        """Apply random transposition to melody."""
        transpose = random.randint(-self.transpose_range, self.transpose_range)
        melody[:, 0] = melody[:, 0] + transpose / 127.0  # Adjust pitch
        melody[:, 0] = torch.clamp(melody[:, 0], 0, 1)  # Clamp to valid range
        return melody


    def _process_midi(self, midi_path):
        """Convert MIDI file into a normalized representation (e.g., pitches, durations)."""
        try:
            midi_data = pretty_midi.PrettyMIDI(midi_path)
            melody = []
            for instrument in midi_data.instruments:
                for note in instrument.notes:
                    pitch = note.pitch / 127.0  # Normalize pitch to [0, 1]
                    start_time = note.start
                    duration = note.end - note.start
                    melody.append([pitch, start_time, duration])
            if len(melody) == 0:  # Check if melody is empty
                raise ValueError(f"No valid notes found in {midi_path}")
            return torch.tensor(melody, dtype=torch.float32)  # Variable-length tensor
        except Exception as e:
            print(f"Error processing MIDI file {midi_path}: {e}")
            return None






