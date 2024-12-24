import torch
import torch.nn.functional as F

class MelodyGAN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Initialize MelodyGAN model.

        Args:
            input_dim (int): Dimension of input features (emotion embeddings + context).
            hidden_dim (int): Hidden layer dimension.
            output_dim (int): Output feature dimension (number of notes per timestep).
        """
        super(MelodyGAN, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, output_dim * 3)  # Output for pitch, start time, and duration

    def forward(self, x, target_length):
        """
        Forward pass for MelodyGAN.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_dim].
            target_length (int): Desired sequence length for the output.

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, target_length, 3].
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.view(x.size(0), -1, 3)  # Reshape to [batch_size, sequence_length, 3]

        # Adjust output to match target length
        if x.size(1) < target_length:
            # Pad output to match target length
            padding = target_length - x.size(1)
            x = F.pad(x, (0, 0, 0, padding))  # Pad along the sequence length dimension
        elif x.size(1) > target_length:
            # Truncate output to match target length
            x = x[:, :target_length, :]

        return x





