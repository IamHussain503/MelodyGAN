import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class TransformerMelodyGenerator(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, output_dim):
        """
        Transformer-based Melody Generator.

        Args:
            input_dim (int): Input feature dimension (e.g., 7).
            hidden_dim (int): Hidden layer dimension.
            num_heads (int): Number of attention heads.
            num_layers (int): Number of Transformer layers.
            output_dim (int): Output feature dimension (e.g., pitch, start time, duration).
        """
        super(TransformerMelodyGenerator, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)  # Input embedding
        self.positional_encoding = nn.Parameter(torch.zeros(1, 512, hidden_dim))  # Positional encoding
        encoder_layer = TransformerEncoderLayer(hidden_dim, num_heads, dim_feedforward=hidden_dim * 4, dropout=0.3)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(hidden_dim, output_dim * 3)  # Final linear layer

    def forward(self, x, target_length):
        """
        Forward pass for Transformer Melody Generator.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_dim].
            target_length (int): Desired sequence length for the output.

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, target_length, 3].
        """
        # Embed input and add positional encoding
        x = self.embedding(x) + self.positional_encoding[:, :x.size(1), :]
        x = self.transformer_encoder(x)
        x = self.fc(x)
        x = x.view(x.size(0), -1, 3)  # Output shape: [batch_size, sequence_length, 3]
        return x[:, :target_length, :]
