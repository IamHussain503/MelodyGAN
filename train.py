from harmony_dataset import HarmonyNetDataset
import torch
from melody_gan import MelodyGAN
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    """
    Custom collate function to handle variable-length melodies.
    Pads all melodies in the batch to the same length.
    
    Args:
        batch: List of tuples (emotion_embedding, context, melody).
    
    Returns:
        Tuple of padded emotion_embeddings, contexts, and melodies.
    """
    emotion_embeddings, contexts, melodies = zip(*batch)

    # Convert to tensors
    emotion_embeddings = torch.stack(emotion_embeddings)  # Shape: [batch_size, embedding_dim]
    contexts = torch.stack(contexts)  # Shape: [batch_size, context_dim]

    # Pad melodies to the same length
    melodies_padded = pad_sequence(melodies, batch_first=True, padding_value=0.0)

    return emotion_embeddings, contexts, melodies_padded


def train_model(dataloader, model, optimizer, device, projection):
    """
    Training loop for HarmonyNet++.

    Args:
        dataloader (DataLoader): Training data loader.
        model (torch.nn.Module): MelodyGAN model.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        device (torch.device): Device to run the training.
        projection (torch.nn.Linear): Projection layer for emotion embeddings.

    Returns:
        float: Average training loss.
    """
    model.train()
    total_loss = 0

    for emotion_embeddings, contexts, melodies in dataloader:
        # Project emotion embeddings
        emotion_embeddings = projection(emotion_embeddings.to(device))  # Shape: [batch_size, 4]

        # Concatenate emotion embeddings and contexts
        contexts = contexts.to(device)
        inputs = torch.cat([emotion_embeddings, contexts], dim=1)  # Shape: [batch_size, 7]

        # Forward pass
        melodies = melodies.to(device)
        outputs = model(inputs)

        # Compute loss
        loss = torch.nn.MSELoss()(outputs, melodies)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)




if __name__ == "__main__":
    # Dataset and DataLoader
    dataset = HarmonyNetDataset("harmonynet_dataset.json")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn
    )

    # Model and Optimizer
    melody_gan = MelodyGAN(input_dim=7, hidden_dim=256, output_dim=128).to(device)
    projection = torch.nn.Linear(768, 4).to(device)  # Reduce emotion_embeddings to 4 dimensions
    optimizer = torch.optim.Adam(list(melody_gan.parameters()) + list(projection.parameters()), lr=1e-4)

    # Training Loop
    for epoch in range(20):  # Example epochs
        avg_loss = train_model(dataloader, melody_gan, optimizer, device, projection)
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")
