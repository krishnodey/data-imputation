import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class Generator(nn.Module):
    #Generate completed samples while preserving observed values.
    def __init__(self, data_dim, hidden_dim):
        #Initialize the generator network.
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(data_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, data_dim),
            nn.Sigmoid()
        )

    def forward(self, x_bar, m, z):
        #Generate missing values and combine them with observed values.
        inputs = torch.cat([x_bar, m, z], dim=1)
        gen_data = self.net(inputs)
        #Observed values pass through unchanged; the generator fills only missing entries.
        return m * x_bar + (1 - m) * gen_data


class Discriminator(nn.Module):
    #Predict the observation mask from completed samples and hint data.
    def __init__(self, data_dim, hidden_dim):
        """Initialize the discriminator network."""
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(data_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, data_dim),
            nn.Sigmoid()
        )

    def forward(self, x_tilde, h):
        #Estimate observation probabilities for each data value.
        inputs = torch.cat([x_tilde, h], dim=1)
        return self.net(inputs)


def sample_z(batch_size, data_dim):
    #Sample uniform noise for GAIN inputs.
    return torch.rand(batch_size, data_dim)


def sample_hint_matrix(m, p_hint):
    #Create a hint matrix by partially revealing the observation mask.
    batch_size, data_dim = m.shape
    u = torch.rand(batch_size, data_dim)
    hint_mask = (u < p_hint).float()
    # Hint 0.5 for revealed hints to keep the discriminator uncertain about those entries.
    return hint_mask * m + 0.5 * (1 - hint_mask)


def set_random_seed(seed=42):
    """Set random seeds for reproducible GAIN training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_gain(data, mask, epochs=1000, batch_size=64, alpha=100, p_hint=0.9, lr=0.001, seed=42):
    #Train GAIN generator and discriminator models.
    set_random_seed(seed)
    data_tensor = torch.tensor(data, dtype=torch.float32)
    mask_tensor = torch.tensor(mask, dtype=torch.float32)
    
    dataset = TensorDataset(data_tensor, mask_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    data_dim = data.shape[1]
    hidden_dim = data_dim * 2
    
    generator = Generator(data_dim, hidden_dim)
    discriminator = Discriminator(data_dim, hidden_dim)
    
    g_optimizer = optim.Adam(generator.parameters(), lr=lr)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=lr)

    print("Starting GAIN Training...")
    for epoch in range(epochs):
        for batch_x, batch_m in dataloader:
            bs = batch_x.size(0)
            
            # Train Discriminator
            z = sample_z(bs, data_dim)
            h = sample_hint_matrix(batch_m, p_hint)
            
            x_hat = generator(batch_x, batch_m, z)
            x_tilde = batch_m * batch_x + (1 - batch_m) * x_hat
            
            d_prob = discriminator(x_tilde.detach(), h)
            # Discriminator loss predicts the observation mask from completed samples.
            d_loss = -torch.mean(
                batch_m * torch.log(d_prob + 1e-8) + 
                (1 - batch_m) * torch.log(1 - d_prob + 1e-8)
            )
            
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()
            
            # Train Generator
            z = sample_z(bs, data_dim)
            h = sample_hint_matrix(batch_m, p_hint)
            
            x_hat = generator(batch_x, batch_m, z)
            x_tilde = batch_m * batch_x + (1 - batch_m) * x_hat
            
            d_prob = discriminator(x_tilde, h)
            g_loss_adversarial = -torch.mean((1 - batch_m) * torch.log(d_prob + 1e-8))
            # Reconstruction on observed cells prevents the generator from overwriting known values.
            g_loss_mse = torch.sum(batch_m * (batch_x - x_hat) ** 2) / torch.sum(batch_m)
            g_loss = g_loss_adversarial + alpha * g_loss_mse
            
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()
            
        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")
            
    return generator
