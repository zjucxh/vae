import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class ResNetBlock(nn.Module):
    def __init__(self, dim):
        super(ResNetBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            #nn.Linear(dim, dim)
        )

    def forward(self, x):
        return x + self.block(x)

# Define the VAE model
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.res_depth = 10
        
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            *[ResNetBlock(hidden_dim) for _ in range(self.res_depth)],
            nn.Linear(hidden_dim, latent_dim * 2)  # Two sets of outputs for mean and variance
        )
        
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            *[ResNetBlock(hidden_dim) for _ in range(self.res_depth)],
            nn.Linear(hidden_dim, input_dim)
        )
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        # Encode
        enc = self.encoder(x)
        mu, logvar = enc[:,:self.latent_dim], enc[:,self.latent_dim:]
        
        # Reparameterization trick
        z = self.reparameterize(mu, logvar)
        # Decode
        recon_x = self.decoder(z)
        
        return recon_x, mu, logvar

# Define the loss function for the VAE
def vae_loss(recon_x, x, mu, logvar):
    # Reconstruction loss
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')

    # KL divergence loss
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + kl_divergence

# Example usage
input_dim = 3  # Dimension of input points
hidden_dim = 256  # Dimension of hidden layers
latent_dim = 32  # Dimension of the latent space

# Initialize the VAE model
vae_model = VAE(input_dim, hidden_dim, latent_dim)

# Define optimizer
optimizer = torch.optim.Adam(vae_model.parameters(), lr=0.001)


if __name__=='__main__':
    # Example usage
    input_dim = 3  # Dimension of input points
    hidden_dim = 256  # Dimension of hidden layers
    latent_dim = 1024 # Dimension of the latent space
    
    # Forward vae model
    batch_size = 8
    num_sampled_points = 289

    x = torch.randn(batch_size,num_sampled_points, input_dim)
    x = torch.reshape(x, (batch_size, -1)).to(device='cuda')
   
    # Initialize the VAE model
    vae_model = VAE(x.shape[-1], hidden_dim, latent_dim).to(device='cuda')

    recon_x, mu, logvar = vae_model(x)
    print(f' x: {x.shape}, recon_x: {recon_x.shape}, mu: {mu.shape}, logvar: {logvar.shape}')
    loss = vae_loss(recon_x,x, mu, logvar)
    print(f'loss: {loss}')