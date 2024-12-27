from harmony_dataset import HarmonyNetDataset
import torch
from melody_gan import TransformerMelodyGenerator
from torch.nn.utils.rnn import pad_sequence
import json
import random
from torch.utils.data import Subset
import os
import time
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from harmony_dataset import HarmonyNetDataset
from torch.cuda.amp import autocast, GradScaler
import pretty_midi


def collate_fn(batch):
    """
    Custom collate function to handle variable-length melodies.
    Pads all melodies in the batch to the same length.
    """
    batch = [sample for sample in batch if sample[2] is not None and sample[2].size(0) > 0]

    if len(batch) == 0:
        raise ValueError("All samples in the batch have invalid or empty melodies.")

    emotion_embeddings, contexts, melodies = zip(*batch)

    emotion_embeddings = torch.stack(emotion_embeddings)
    contexts = torch.stack(contexts)
    melodies_padded = pad_sequence(melodies, batch_first=True, padding_value=0.0)

    return emotion_embeddings, contexts, melodies_padded

class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.counter = 0
        self.stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def save_checkpoint(model, projection, optimizer, epoch, loss, save_dir="checkpoints"):
    """
    Save model and optimizer state as a checkpoint.
    """
    os.makedirs(save_dir, exist_ok=True)
    if epoch % 10 == 0:
        checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "projection_state_dict": projection.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss
        }, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")


def load_checkpoint(checkpoint_path, model, projection, optimizer):
    """
    Load model and optimizer state from a checkpoint.
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    projection.load_state_dict(checkpoint["projection_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(f"Checkpoint loaded from: {checkpoint_path}")




# Training loop with optimizations
def train_model(dataloader, model, optimizer, device, projection, scaler, epoch):
    """
    Training loop for HarmonyNet++ with batch-wise loss logging and mixed precision.

    Args:
        dataloader (DataLoader): Training data loader.
        model (torch.nn.Module): MelodyGAN model.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        device (torch.device): Device to run the training.
        projection (torch.nn.Linear): Projection layer for emotion embeddings.
        scaler (torch.cuda.amp.GradScaler): Gradient scaler for mixed precision.
        epoch (int): Current epoch.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    total_loss = 0

    for batch_idx, (emotion_embeddings, contexts, melodies) in enumerate(dataloader, start=1):
        # Move tensors to device
        emotion_embeddings = emotion_embeddings.to(device)
        contexts = contexts.to(device)
        melodies = melodies.to(device)

        # Project emotion embeddings
        emotion_embeddings = projection(emotion_embeddings)  # Shape: [batch_size, 4]

        # Concatenate emotion embeddings and contexts
        inputs = torch.cat([emotion_embeddings, contexts], dim=1)  # Shape: [batch_size, 7]

        # Forward pass with mixed precision
        with autocast():
            target_length = melodies.size(1)
            outputs = model(inputs, target_length)  # Shape: [batch_size, target_length, 3]
            loss = torch.nn.SmoothL1Loss()(outputs, melodies)

        # Backpropagation
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Accumulate total loss
        total_loss += loss.item()

        # Log batch loss
        # print(f"Epoch {epoch}, Batch {batch_idx}/{len(dataloader)}, Batch Loss: {loss.item():.4f}")

    # Return average loss for the epoch
    return total_loss / len(dataloader)

# Validation loop
def validate_model(dataloader, model, device, projection):
    """
    Validation loop for HarmonyNet++.

    Args:
        dataloader (DataLoader): Validation data loader.
        model (torch.nn.Module): MelodyGAN model.
        device (torch.device): Device to run the validation.
        projection (torch.nn.Linear): Projection layer for emotion embeddings.

    Returns:
        float: Average validation loss.
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for emotion_embeddings, contexts, melodies in dataloader:
            # Move tensors to device
            emotion_embeddings = emotion_embeddings.to(device)
            contexts = contexts.to(device)
            melodies = melodies.to(device)

            # Project emotion embeddings
            emotion_embeddings = projection(emotion_embeddings)  # Shape: [batch_size, 4]

            # Concatenate emotion embeddings and contexts
            inputs = torch.cat([emotion_embeddings, contexts], dim=1)  # Shape: [batch_size, 7]

            # Forward pass with dynamic sequence length
            target_length = melodies.size(1)
            outputs = model(inputs, target_length)  # Shape: [batch_size, target_length, 3]

            # Compute loss
            loss = torch.nn.SmoothL1Loss()(outputs, melodies)
            total_loss += loss.item()

    return total_loss / len(dataloader)

def process_melody_file(melody_path):
    """
    Process a MIDI file into a list of normalized notes.
    Args:
        melody_path (str): Path to the melody file.
    Returns:
        list: Melody data as a list of [pitch, start_time, duration].
    """
    try:
        midi_data = pretty_midi.PrettyMIDI(melody_path)
        melody = []
        for instrument in midi_data.instruments:
            for note in instrument.notes:
                pitch = note.pitch / 127.0  # Normalize pitch to [0, 1]
                start_time = note.start
                duration = note.end - note.start
                melody.append([pitch, start_time, duration])
        return melody
    except Exception as e:
        print(f"Error processing melody file {melody_path}: {e}")
        return []



def split_dataset(json_file, train_ratio=0.8):
    """
    Split the dataset into training and validation subsets.
    
    Args:
        json_file (str): Path to the JSON dataset file.
        train_ratio (float): Proportion of data to use for training.
    
    Returns:
        tuple: Training and validation datasets as lists.
    """
    # Load the dataset
    with open(json_file, "r") as f:
        dataset = json.load(f)

    # Shuffle the data
    random.shuffle(dataset)

    # Compute split index
    split_idx = int(len(dataset) * train_ratio)

    # Split into training and validation
    train_data = dataset[:split_idx]
    val_data = dataset[split_idx:]
    return train_data, val_data


from torch.optim.lr_scheduler import StepLR

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Split dataset
    train_data, val_data = split_dataset("harmonynet_dataset.json")

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
    scaler = GradScaler()
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
    early_stopping = EarlyStopping(patience=10)

for max_length in [50, 100, 150]:  # Gradually increase melody length
    # Filter training data
    train_subset_data = [
        sample for sample in train_data
        if "melody_path" in sample and len(process_melody_file(sample["melody_path"])) <= max_length
    ]
    train_subset = HarmonyNetDataset(train_subset_data)

    # Filter validation data
    val_subset_data = [
        sample for sample in val_data
        if "melody_path" in sample and len(process_melody_file(sample["melody_path"])) <= max_length
    ]
    val_subset = HarmonyNetDataset(val_subset_data)

    # Initialize DataLoaders
    train_dataloader = DataLoader(
        train_subset,
        batch_size=256,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4
    )
    val_dataloader = DataLoader(
        val_subset,
        batch_size=256,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
    )

    # Train and validate for each curriculum step
    for epoch in range(1, 10):  # Shorter training for each step
        train_loss = train_model(train_dataloader, melody_generator, optimizer, device, projection, scaler, epoch)
        val_loss = validate_model(val_dataloader, melody_generator, device, projection)

        print(f"Max Length {max_length}, Epoch {epoch}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        scheduler.step()
        early_stopping(val_loss)

        if early_stopping.stop:
            print("Early stopping triggered!")
            break

        save_checkpoint(melody_generator, projection, optimizer, epoch, train_loss)

