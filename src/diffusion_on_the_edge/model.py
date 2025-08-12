import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class SimpleScoreNet(nn.Module):
    """
    Simple fully connected network with 'layer_count' hidden layers and relu activation function.
    Used in modeling the score function
    """
    def __init__(self, input_dimesion, output_dimension, layer_count = 1, hidden_dim=512):
        super().__init__()

        # Construct M hidden layers with ReLU
        hidden_layers = [
            layer
            for _ in range(layer_count)
            for layer in (nn.Linear(hidden_dim if _ > 0 else input_dimesion, hidden_dim), nn.ReLU())
        ]
        self.net = nn.Sequential(
            *hidden_layers,
            nn.Linear(hidden_dim, output_dimension)
        )

    def forward(self, x, t):
        inp = torch.cat([x, t], dim=-1)
        return self.net(inp)

def train_scorenet(model: SimpleScoreNet, dataloader: torch.utils.data.DataLoader, epochs=10, lr=1e-4, device='cuda'):
    """
    Train the SimpleScoreNet using MSE between predicted and true score values.
    Assumes dataloader yields (x, t, true_score) batches.
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        model.train()

        for x, t, true_score in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            x = x.to(device)
            t = t.to(device)
            true_score = true_score.to(device)

            t_expanded = t.expand_as(x) if t.shape != x.shape else t
            score_pred = model(x, t_expanded)

            loss = F.mse_loss(score_pred, true_score)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * x.size(0)

        avg_loss = epoch_loss
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.6f}")

    return {
        'model': model,
        'losses': losses
    }