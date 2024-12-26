import torch
from torch.utils.data import Dataset
import pretty_midi
import random


class HarmonyNetDataset(Dataset):
    def __init__(self, data, transpose_range=2):
        """
        Dataset class for HarmonyNet++ with data normalization and augmentation.

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
        melody_path = sample["melody_path"]
        melody = self._process_midi(melody_path)

        # Apply augmentation
        if melody.size(0) > 0 and self.transpose_range > 0:  # Skip augmentation for empty melodies
            melody = self._augment_melody(melody)

        return emotion_embedding, context, melody

    def _process_midi(self, midi_path):
        """
        Process MIDI file into a normalized representation.
        Returns a tensor of shape [num_notes, 3] with columns:
        [normalized pitch, start time, duration].
        """
        try:
            midi_data = pretty_midi.PrettyMIDI(midi_path)
            melody = []
            for instrument in midi_data.instruments:
                for note in instrument.notes:
                    pitch = note.pitch / 127.0  # Normalize pitch to [0, 1]
                    start_time = note.start
                    duration = note.end - note.start
                    melody.append([pitch, start_time, duration])

            if len(melody) == 0:
                print(f"No valid notes found in {midi_path}. Returning empty tensor.")
                return torch.zeros((0, 3), dtype=torch.float32)  # Return empty tensor if no valid notes

            return torch.tensor(melody, dtype=torch.float32)

        except Exception as e:
            print(f"Error processing MIDI file {midi_path}: {e}")
            return torch.zeros((0, 3), dtype=torch.float32)  # Return empty tensor on error

    def _augment_melody(self, melody):
        """
        Apply random transposition to melody.
        Ensures valid pitch values after transposition.

        Args:
            melody (torch.Tensor): Melody tensor of shape [num_notes, 3].

        Returns:
            torch.Tensor: Augmented melody.
        """
        transpose = random.randint(-self.transpose_range, self.transpose_range)
        melody[:, 0] = melody[:, 0] + transpose / 127.0  # Adjust pitch
        melody[:, 0] = torch.clamp(melody[:, 0], 0, 1)  # Clamp pitch to [0, 1]
        return melody
