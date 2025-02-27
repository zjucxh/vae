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

# Define Gated Recurrent Unit
class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=4, output_dim=289*3):
        super(GRU, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim 
        self.num_layers = num_layers
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to('cuda')
        #print(f' h0 shape : {h0.shape}')
        out, _ = self.gru(x, h0)
        out = self.fc(out[:,-1,:])
        return out

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
    input_dim = 289 * 3
    batch_size = 8
    seq_length = 3
    dataloader = DataLoader(dataset, batch_size=batch_size*(seq_length+1), shuffle=False)
    num_epoches = 5000
    

    # Load GRU model
    gru = GRU(input_dim=input_dim, hidden_dim=256, num_layers=4, output_dim=input_dim).to('cuda')
    print(f' gru model : {gru}')
    optimizer = torch.optim.Adam(gru.parameters(), lr=1.0e-5)
    critrion = nn.MSELoss()
    # Train the model
    for epoch in range(num_epoches):
        running_loss = 0.0
        for i, data in enumerate(dataloader):
            # reshape data to batch_size, seq_length, input_dim
            #print(f' data shape : {data.shape}')
            data = data.reshape(-1, seq_length+1, input_dim).to('cuda')
            x = data[:,:seq_length,:]
            y = data[:,-1,:]
            #print(f' y shape  :{y.shape}')
            optimizer.zero_grad()
            output = gru(x)
            #print(f' output shape : {output.shape}')
            loss = critrion(output, y)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
        print(f'[{epoch + 1}], loss: {running_loss / 10}')
    print('Done')