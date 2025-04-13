import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from data import CMU_simulation 
from torch.utils.data import DataLoader, Dataset
from utils import signed_distance
from loss import loss_signed_distance
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

        # The GRU module take input_dim dimension as input, the 181 is the pose parameters, the 3*n is the initial guess of the vertices
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True )
        self.fc = nn.Linear(hidden_dim, output_dim) 

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
        #self.h0 = torch.normal(0, 0.1, (self.num_layers, x.size(0), self.hidden_dim)).to('cuda')
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to('cuda')
        # set h0 with normal distribution with u = 0, sigma = 0.1
        #ho = self.h0
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
            input_dim: int, the input dimension of NLS (B, S, D) where D  is concated with t, pose and vertex
            hidden_dim: int, the number of hidden units in the GRU
            num_layers: int, the number of layers in the GRU
            output_dim: int, the output dimension of NLS signed distance of the vertex
        """
        # The input dimension of NLS is (B, S, D),
        # where B is the batch size, S is the sequence length, and D is the input dimension.
        # The hidden dimension is the number of hidden units in the GRU.
        # The output dimension is 1 as the output of NLS is signed distance(scalar) for each vertex.
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        #self.force_field = ForceField(input_dim-1, num_layers=6, output_dim=3).to(device='cuda') # exclude time dimension
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, t, poses, vertices):
        # The input of NLS is the concatenation of time, vertices and poses.
        # vertices is the vertex sequence with shape (B, S, V*3), poses shpae (B, S, 181), temporal shape (B, S, 1)
        # where B is the batch size, S is the sequence length, and V is the number of vertices.
        # The poses is the pose sequence with shape (B, S, pose_dim), pose_dim is 181
        x0 = torch.cat([t, poses,vertices], dim=-1).to(device='cuda')  # input of NLS
        #x1 = torch.cat([poses, vertices], dim=-1).to(device='cuda') # input of force field
        h0 = torch.zeros(self.num_layers, x0.size(0), self.hidden_dim).to('cuda')
        out, _ = self.gru(x0, h0) 
        sdist = self.fc(out)# output of signed distance
        # feed the force field to the model
        #ff = self.force_field(x1)
        return sdist
    
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
        self.transformer = nn.TransformerEncoderLayer(input_dim, nhead=8, dim_feedforward=1024, batch_first=True) # transformer layer
        self.encoder = nn.TransformerEncoder(self.transformer, num_layers=num_layers) # transformer encoder based on paper "Attention is all you need"
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, poses, vertices):
        # The input of the model is the concatenation of SMPL parameters and vertex sequence.
        # x is the input with shape (B, S, D), where B is the batch size, S is the sequence length, and D is the input dimension.
        x = torch.cat([poses, vertices], dim=-1).to(device='cuda')  # input of force field
        out = self.encoder(x)
        out = self.fc(out)
        return out
    




# Define NLS loss
# Wip: solve backward problem
def loss_nls(model, gt_sdf,t, poses, sampled_vertex):
    t.requires_grad_(True)
    sampled_vertex.requires_grad_(True)
    poses.requires_grad_(True)
    
    # compute the signed distance loss
    pred_sdf = model(t, poses, sampled_vertex)
    sdf_loss = loss_signed_distance(pred_sdf, gt_sdf)

    # compute level set constraint loss 
    
    # overall loss
    overall_loss = sdf_loss 
    
    return overall_loss



if __name__=='__main__':
    # load data
    dataset = CMU_simulation('/home/cxh/tmp/CMU_mini_dataset')
    template_vertices = dataset.template_vertices.to(device='cuda')
    template_faces = dataset.template_faces
    laplacian_matrix = dataset.laplacian_matrix.to(device='cuda')

    input_dim = 181 + 3 + 1 # pose_dim + vertex_dim + time_dim
    output_dim = 3
    batch_size = 8
    seq_length = 130
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    num_epoches = 15000
    sampling_size = 512
    num_obj_vertex = 2590
    
    # sample random vertex from the vertex sequence,
    #torch.manual_seed(0)

    # Load Force field model
    #force_field = ForceField(input_dim-1, num_layers=6, output_dim=3).to(device='cuda') # exclude time dimension
    # Load NLS model
    nls = NLS(input_dim, hidden_dim=256, num_layers=4, output_dim=1).to(device='cuda')

    optimizer = torch.optim.Adam(nls.parameters(), lr=1.0e-5)
    #critrion = loss_laplacian
    # Train the model
    for epoch in range(num_epoches):
        running_loss = 0.0
        for i, data in enumerate(dataloader):
            # zero grad optimizer
            optimizer.zero_grad()
            # reshape data to batch_size, seq_length, input_dim
            gender, poses, vertex_seq = data # vertex shape (B, S, num_v, 3)
            
            rand_idx = torch.randint(0, num_obj_vertex, size=(sampling_size,)) # sample random vertex from the vertex sequence

            # generate time data shaped with (B, S, 1) in linear space
            #t = torch.ones(poses.shape[0], seq_length, 1).to(device='cuda')
            t = torch.linspace(0, 1, seq_length).repeat(poses.shape[0], 1).unsqueeze(-1).to(device='cuda')
            poses = poses.to(device='cuda')
            gt_sdf = torch.zeros(poses.shape[0], seq_length, 1).to(device='cuda') 
            # compute ground truth signed distance function
            # obtain sampled vertex from rand_idx
            #sdf_vertex = vertex_seq[:, :,rand_idx]

            # compute loss for each sampling
            running_loss = 0.0
            for j in rand_idx:
                sampled_vertex = vertex_seq[:, :,j].to(device='cuda')
                loss = loss_nls(nls, gt_sdf, t, poses, sampled_vertex)
                running_loss += loss

            # backward
            running_loss.backward()
            optimizer.step()
            print(f' epoch : {epoch}, loss : {running_loss.item()}')
            # save model every 1000 epochs  
            #if epoch % 1000 == 999:
            #    torch.save(nls.state_dict(), '/home/cxh/tmp/checkpoint/nls/nls_{0:0>3}.pth'.format((epoch-999)//1000))
    print('Done')