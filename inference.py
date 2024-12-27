import torch
import os
import pretty_midi
from melody_gan import TransformerMelodyGenerator
from transformers import T5Tokenizer, T5EncoderModel

# Load the checkpoint
def load_checkpoint(checkpoint_path, model, projection, optimizer):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    projection.load_state_dict(checkpoint["projection_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(f"Checkpoint loaded from: {checkpoint_path}")

# Generate a melody in MIDI format
def generate_midi(model, projection, text_caption, output_midi_file="generated_melody.mid", device="cuda"):
    """
    Generate a melody from text caption with auto-generated context.

    Args:
        model (torch.nn.Module): TransformerMelodyGenerator model.
        projection (torch.nn.Linear): Projection layer for emotion embeddings.
        text_caption (str): Text input for emotion and context extraction.
        output_midi_file (str): Path to save the generated MIDI file.
        device (str): Device to run the model on.

    Returns:
        None
    """
    # Load T5 for emotion and context extraction
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    encoder = T5EncoderModel.from_pretrained("t5-base").to(device)
    encoder.eval()

    # Extract emotion embedding
    tokens = tokenizer(text_caption, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    with torch.no_grad():
        embedding = encoder(**tokens).last_hidden_state.mean(dim=1)  # Shape: [1, 768]

    # Project emotion embedding
    with torch.no_grad():
        emotion_embedding_projected = projection(embedding)  # Shape: [1, 4]

    # Auto-generate context from T5 embedding
    context_tensor = embedding[:, :3]  # Use first 3 dimensions of the embedding for context
    context_tensor = torch.sigmoid(context_tensor)  # Normalize values to [0, 1]

    # Concatenate inputs
    inputs = torch.cat([emotion_embedding_projected, context_tensor], dim=1)  # Shape: [1, 7]

    # Generate melody
    model.eval()
    with torch.no_grad():
        target_length = 200  # Example length for generated melody
        outputs = model(inputs, target_length)  # Shape: [1, target_length, 3]

    # Convert output to MIDI
    melody_to_midi(outputs.squeeze(0).cpu().numpy(), output_midi_file)
    print(f"Generated MIDI saved to: {output_midi_file}")

# Convert generated melody to MIDI format
def melody_to_midi(melody, output_midi_file):
    """
    Convert generated melody to a MIDI file.

    Args:
        melody (numpy.ndarray): Generated melody as a [sequence_length, 3] array.
        output_midi_file (str): Path to save the MIDI file.

    Returns:
        None
    """
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano

    start_time = 0
    for note_info in melody:
        # Clamp pitch to MIDI range [0, 127]
        pitch = max(0, min(int(note_info[0] * 127), 127))  # Denormalize and clamp pitch
        start_time += max(note_info[1], 0.0)  # Ensure non-negative start time increment
        duration = max(note_info[2], 0.1)  # Ensure minimum duration
        end_time = start_time + duration

        # Create MIDI note
        note = pretty_midi.Note(
            velocity=100,  # Fixed velocity
            pitch=pitch,
            start=start_time,
            end=end_time
        )
        instrument.notes.append(note)

    midi.instruments.append(instrument)
    midi.write(output_midi_file)
    print(f"MIDI file saved to: {output_midi_file}")


# Convert MIDI to WAV
def midi_to_wav(midi_file, wav_file):
    """
    Convert a MIDI file to WAV format using FluidSynth.

    Args:
        midi_file (str): Path to the input MIDI file.
        wav_file (str): Path to save the output WAV file.

    Returns:
        None
    """
    os.system(f"fluidsynth -ni soundfont.sf2 {midi_file} -F {wav_file}")
    print(f"WAV file saved to: {wav_file}")

# Main inference pipeline
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize Transformer model
    melody_generator = TransformerMelodyGenerator(
        input_dim=7,
        hidden_dim=512,
        num_heads=8,
        num_layers=4,
        output_dim=128
    ).to(device)
    projection = torch.nn.Linear(768, 4).to(device)
    optimizer = torch.optim.AdamW(
        list(melody_generator.parameters()) + list(projection.parameters()),
        lr=1e-4,
        weight_decay=1e-5
    )

    # Load checkpoint
    checkpoint_path = "checkpoints/checkpoint_epoch_16.pt"
    load_checkpoint(checkpoint_path, melody_generator, projection, optimizer)

    # Example input for inference
    text_caption = "A beautiful music piece with a happy and uplifting rock beat."

    # Generate melody and save as MIDI
    output_midi_file = "generated_melody.mid"
    generate_midi(melody_generator, projection, text_caption, output_midi_file, device)

    # Convert MIDI to WAV
    midi_to_wav(output_midi_file, "generated_melody.wav")
