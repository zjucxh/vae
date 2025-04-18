import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from data import CMU_simulation 
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
        #out = self.fc(out[:,-1,:])
        out = self.fc(out)
        return out

# TODO Define Nerual Level Set Model
class NLS(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=4, output_dim=1):
        super(NLS, self).__init__()
        r"""
        Args:
            input_dim: int, the input dimension of NLS
            hidden_dim: int, the number of hidden units in the GRU
            num_layers: int, the number of layers in the GRU
            output_dim: int, the output dimension of NLS
        """
        # The input dimension of NLS is (B, S, D),
        # where B is the batch size, S is the sequence length, and D is the input dimension.
        # The hidden dimension is the number of hidden units in the GRU.
        # The output dimension is 1 as the output of NLS is signed distance(scalar) for each vertex.
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, vertices, poses):
        # The input of NLS is the concatenation of vertices and poses.
        # vertices is the vertex sequence with shape (B, S, V*3),
        # where B is the batch size, S is the sequence length, and V is the number of vertices.
        # The poses is the pose sequence with shape (B, S, pose_dim), pose_dim is 181
        x = torch.cat([vertices, poses], dim=-1)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to('cuda')
        out, _ = self.gru(x, h0)
        out = self.fc(out)
        return out
    
# TODO define Forcefield Model 
class ForceField(nn.Module):
    def __init__(self, input_dim, num_layers=6, output_dim=3):
        r"""
        Attention layer based on the paper "Attention is all you need"
        Args:
            input_dim: int, the input dimension of the model containing SMPL parameters and vertex sequence
            num_layers: int, the number of layers in the model
            output_dim: int, the output dimension of the model containing the force field consistent with the vertex sequence
        """
        super(ForceField, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        #self.attention = nn.MultiheadAttention(input_dim, num_heads=8, batch_first=True)
        self.transformer = nn.TransformerEncoderLayer(input_dim, num_heads=8, dim_feedforward=1024, batch_first=True) # transformer layer
        self.encoder = nn.TransformerEncoder(self.transformer, num_layers=num_layers) # transformer encoder based on paper "Attention is all you need"
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # The input of the model is the concatenation of SMPL parameters and vertex sequence.
        # x is the input with shape (B, S, D), where B is the batch size, S is the sequence length, and D is the input dimension.
        out = self.encoder(x)
        out = self.fc(out)
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

def loss_l2(pred, gt):
    l2_loss = nn.MSELoss()
    loss = l2_loss(pred, gt)
    return loss

# batched laplacian loss
def loss_laplacian(laplacian_matrix:torch.Tensor, pred_vertices:torch.Tensor, gt_vertices:torch.Tensor, ratio=0.1):
    laplacian_loss = nn.MSELoss()
    # assert pred_vertices shape == gt_vertices shape
    assert pred_vertices.shape == gt_vertices.shape # shape : batch_size, seq_length, num_vertices*3
    # reshape pred_vertices and gt_vertices to batch_size*seq_length, num_vertices*3
    pred_vertices = pred_vertices.view(-1, pred_vertices.shape[2])
    gt_vertices = gt_vertices.view(-1, gt_vertices.shape[2])
    #print(f' gt_vertices : {gt_vertices}')
    #print(f' pred_vertices : {pred_vertices}')
    #print(f' laplacian_matrix : {laplacian_matrix}')
    laplacian_pred = torch.matmul(pred_vertices, laplacian_matrix)
    laplacian_gt = torch.matmul(gt_vertices, laplacian_matrix)
    #print(f' laplacian pred : {laplacian_pred}')
    #print(f' laplacian gt : {laplacian_gt}')
    laplacian_loss = laplacian_loss(laplacian_pred,laplacian_gt)
    #print(f' laplsaian loss : {laplacian_loss}')
    vertex_loss = loss_l2(pred_vertices, gt_vertices)
    return ratio * laplacian_loss + (1-ratio) * vertex_loss


if __name__=='__main__':
    # load data
    dataset = CMU_simulation('/home/cxh/tmp/CMU_mini_dataset')
    template_vertices = dataset.template_vertices.to(device='cuda')
    template_faces = dataset.template_faces
    laplacian_matrix = dataset.laplacian_matrix.to(device='cuda')

    input_dim = 181
    output_dim = 2590*3
    batch_size = 8
    seq_length = 130
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    num_epoches = 15000
    

    # Load GRU model
    gru = GRU(input_dim=input_dim, hidden_dim=512, num_layers=8, output_dim=output_dim).to('cuda')
    print(f' gru model : {gru}')
    optimizer = torch.optim.Adam(gru.parameters(), lr=1.0e-5)
    critrion = loss_laplacian
    # Train the model
    for epoch in range(num_epoches):
        running_loss = 0.0
        for i, data in enumerate(dataloader):
            # reshape data to batch_size, seq_length, input_dim
            gender, poses, vertex_seq = data
            #print(f' ----vertex_seq shape : {vertex_seq.shape}')
            vertex_seq = vertex_seq.to(device='cuda')
            #vertex_seq = vertex_seq.view(-1, seq_length, 7770).to(device='cuda')
            poses = poses.to(device='cuda')
            # reshape beta from batch_size, 16 to batch_size, sequence_length, 16
            optimizer.zero_grad()
            output = gru(poses)
            # reshape output as vertex_seq
            output = output.view_as(vertex_seq)
            #print(f' out shape : {output.shape}')
            #print(f' ground truth shape : {vertex_seq.shape}')
            #loss_lap = loss_laplacian(laplacian_matrix, output, vertex_seq)
            #print(f' laplacian loss : {loss_lap}')
            loss = critrion(laplacian_matrix, output, vertex_seq)
            running_loss += loss.item()
            #print(f' loss : {loss }')
            #print(f' i : {i} loss : {loss.item()}')
            if i % 10 == 9:
                print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 10}')
                running_loss = 0.0
            
            loss.backward()
            optimizer.step()

        # save model for every 100 epoches
        #if epoch % 100 == 99:
        #   torch.save(gru.state_dict(), '/home/cxh/tmp/checkpoint/gru_{0:0>3}.pth'.format((epoch-99)//100))
        
    print('Done')