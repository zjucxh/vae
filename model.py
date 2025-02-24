import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.models import resnet18
from pytorch3d.loss import chamfer_distance
from data import Cloth_in_Wind
from torch.utils.data import DataLoader, Dataset

class ResNetBlock(nn.Module):
    def __init__(self, dim):
        super(ResNetBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            # weigth normalization
            #nn.utils.parametrizations.weight_norm(nn.Linear(dim, dim)),
            nn.LeakyReLU(),
            #nn.Dropout(0.5)
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        return x + self.block(x)

# Define the VAE model
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        
        self.input_dim = input_dim # num of sampled points * 3
        self.hidden_dim = hidden_dim # num of hidden units
        self.latent_dim = latent_dim # num of latent variables
        self.res_depth = 50# num of resnet blocks
        
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            *[ResNetBlock(hidden_dim) for _ in range(self.res_depth)],
            nn.Linear(hidden_dim, latent_dim * 2)  # Two sets of outputs for mean and variance
        )
        # Modify encoder to resnet18
        #self.encoder = resnet18(pretrained=False)
        
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            *[ResNetBlock(hidden_dim) for _ in range(self.res_depth)],
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
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
        sdist = self.decoder(z)
        
        return sdist , mu, logvar

# Define the loss function for the VAE
def vae_loss(recon_x, x, mu, logvar):
    # Reconstruction loss
    #sdist_loss = loss_l1(recon_x, x)
    sdist_loss = F.mse_loss(recon_x, x, reduction='sum')

    # KL divergence loss
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return sdist_loss + kl_divergence

# loss_l1
def loss_l1(pred_distance, gt_distance, clamp=0.1):
    l1_loss = nn.L1Loss()
    pred_distance = torch.clamp(pred_distance, -clamp, clamp)
    gt_distance = torch.clamp(gt_distance, -clamp, clamp)
    loss = l1_loss(pred_distance, gt_distance)
    return loss


if __name__=='__main__':
    # load data
    dataset = Cloth_in_Wind()
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    input_dim = 289 * 3
    hidden_dim = 256
    latent_dim = 1024 
    vae_model = VAE(input_dim, hidden_dim, latent_dim).to(device='cuda')
    # Initialize the VAE model
    for i, data in enumerate(dataloader):
        x, sdist, noise, nsdist = data
        x = x.view(x.size(0), -1).to('cuda')
        y, mu, var = vae_model(x)
        print('y shape : {}'.format(y.shape))
    

    sdist , mu, logvar = vae_model(x)
    print(' sdist shape : {}'.format(sdist.shape))
    print(' mu shape : {}'.format(mu.shape))
    print(' logvar shape : {}'.format(logvar.shape))
    print('Done')