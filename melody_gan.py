import torch.nn as nn
import torch.nn.functional as F

class MelodyGAN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MelodyGAN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, output_dim * 3)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x, target_length):
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        x = x.view(x.size(0), -1, 3)

        if x.size(1) < target_length:
            padding = target_length - x.size(1)
            x = F.pad(x, (0, 0, 0, padding))  # Pad along the sequence length dimension
        elif x.size(1) > target_length:
            x = x[:, :target_length, :]
        return x
