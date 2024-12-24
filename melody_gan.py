import torch
class MelodyGAN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MelodyGAN, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, output_dim * 3)  # Output for pitch, start time, and duration

    def forward(self, x, sequence_length):
        """
        Forward pass for MelodyGAN.
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_dim].
            sequence_length (int): Desired sequence length for the output.
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, sequence_length, 3].
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.view(x.size(0), -1, 3)  # Reshape to [batch_size, fixed_length, 3]
        return x[:, :sequence_length, :]  # Truncate to match target sequence length


