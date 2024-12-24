import torch
import torch.nn.functional as F


class MelodyGAN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Initialize MelodyGAN model with increased capacity and dropout.

        Args:
            input_dim (int): Dimension of input features.
            hidden_dim (int): Hidden layer dimension.
            output_dim (int): Output feature dimension.
        """
        super(MelodyGAN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.fc3 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim * 3)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x, target_length):
        """
        Forward pass for MelodyGAN.
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_dim].
            target_length (int): Desired sequence length for the output.

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, target_length, 3].
        """
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        x = x.view(x.size(0), -1, 3)

        if x.size(1) < target_length:
            padding = target_length - x.size(1)
            x = F.pad(x, (0, 0, 0, padding))  # Pad along the sequence length dimension
        elif x.size(1) > target_length:
            x = x[:, :target_length, :]  # Truncate to match the target sequence length

        return x








