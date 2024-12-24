import torch
class MelodyGAN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MelodyGAN, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, output_dim * 3)  # Output for pitch, start time, and duration

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x.view(x.size(0), -1, 3)  # Reshape to [batch_size, sequence_length, 3]

