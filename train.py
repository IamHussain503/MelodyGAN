from harmony_dataset import HarmonyNetDataset
import torch
from melody_gan import MelodyGAN


def train_model(dataloader, model, optimizer, device):
    model.train()
    total_loss = 0

    for emotion_embeddings, contexts, melodies in dataloader:
        inputs = torch.cat([emotion_embeddings, contexts], dim=1).to(device)
        targets = melodies.to(device)

        # Forward pass
        outputs = model(inputs)

        # Compute loss
        loss = torch.nn.MSELoss()(outputs, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load dataset
    dataset = HarmonyNetDataset("harmonynet_dataset.json")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    melody_gan = MelodyGAN(input_dim=7, hidden_dim=256, output_dim=128).to(device)
    optimizer = torch.optim.Adam(melody_gan.parameters(), lr=1e-4)


    # Train model
    for epoch in range(20):  # Example epochs
        avg_loss = train_model(dataloader, melody_gan, optimizer, device)
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")
