import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class Generator(nn.Module):
    def __init__(self, data_dim, hidden_dim):
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
        inputs = torch.cat([x_bar, m, z], dim=1)
        gen_data = self.net(inputs)
        return m * x_bar + (1 - m) * gen_data


class Discriminator(nn.Module):
    def __init__(self, data_dim, hidden_dim):
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
        inputs = torch.cat([x_tilde, h], dim=1)
        return self.net(inputs)


def sample_z(batch_size, data_dim):
    return torch.rand(batch_size, data_dim)


def sample_hint_matrix(m, p_hint):
    batch_size, data_dim = m.shape
    u = torch.rand(batch_size, data_dim)
    hint_mask = (u < p_hint).float()
    return hint_mask * m + 0.5 * (1 - hint_mask)


def train_gain(data, mask, epochs=1000, batch_size=64, alpha=100, p_hint=0.9, lr=0.001):
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
            g_loss_mse = torch.sum(batch_m * (batch_x - x_hat) ** 2) / torch.sum(batch_m)
            g_loss = g_loss_adversarial + alpha * g_loss_mse
            
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()
            
        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")
            
    return generator
