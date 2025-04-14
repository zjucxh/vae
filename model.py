import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from data import CMU_simulation 
from torch.utils.data import DataLoader, Dataset
#from utils import signed_distance
from loss import loss_signed_distance, loss_l2

# Define Gated Recurrent Unit
class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=4, output_dim=289*3):
        super(GRU, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim 
        self.num_layers = num_layers
        self.gru = torch.nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
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
            input_dim: int, the input dimension of NLS (B, S, D) where D is concatenated with t, pose, and vertex
            hidden_dim: int, the number of hidden units in the transformer
            num_layers: int, the number of transformer encoder layers
            output_dim: int, the output dimension of NLS signed distance of the vertex
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        # Transformer encoder
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=5, dim_feedforward=hidden_dim, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, t, poses, vertices):
        # The input of NLS is the concatenation of time, vertices, and poses.
        # vertices: (B, S, V*3), poses: (B, S, 181), t: (B, S, 1)
        x = torch.cat([t, poses, vertices], dim=-1).to(device='cuda')  # input of NLS
        out = self.transformer_encoder(x)
        sdist = self.fc(out)  # output of signed distance
        return sdist
    
# Forcefield Model 
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
def loss_nls(model, force_field, gt_sdf,t, poses, sampled_vertex, t_c, poses_c, sampled_vertex_c, poses_f, sampled_vertex_f):
    #t.requires_grad_(True)
    #sampled_vertex.requires_grad_(True)
    #poses.requires_grad_(True)
    
    # compute the signed distance loss
    #with torch.backends.cudnn.flags(enabled=False):
    pred_sdf = model(t, poses, sampled_vertex)
    #print(' predicted sdf shape : {0}'.format(pred_sdf.shape))
    #print(' gt sdf shape : {0}'.format(gt_sdf.shape))
    # unsqueeze gt_sdf to (B, S, 1)
    gt_sdf = gt_sdf.unsqueeze(-1)
    sdf_loss = loss_signed_distance(pred_sdf, gt_sdf)

    # Force field
    #poses_f.requires_grad_(True)
    #sampled_vertex_f.requires_grad_(True)
    #ff = force_field(poses_f, sampled_vertex_f)

    # level set constraints
    #t_c.requires_grad_(True)
    #poses_c.requires_grad_(True)
    #sampled_vertex_c.requires_grad_(True)
    #pred_sdf_c = model(t_c, poses_c, sampled_vertex_c)
    # compute the gradient of pred_sdf_c with respect to sampled_vertex_c and t_c
    #pred_sdf_c_grad = torch.autograd.grad(pred_sdf_c, [t_c, sampled_vertex_c], grad_outputs=torch.ones_like(pred_sdf_c), create_graph=True)
    #grad_t = pred_sdf_c_grad[0]
    #grad_v = pred_sdf_c_grad[1]
    #print(' grad_t shape : {0}'.format(grad_t.shape))
    ##print(' grad_v shape : {0}'.format(grad_v.shape))
    ##print(' ff shape : {0}'.format(ff.shape))
    # l2 loss 
    #constraint_loss = loss_l2(grad_t, ff * grad_v)
    #print(' ff shape : {0}'.format(ff.shape))
    #print(' sampled_vertex_c shape {0}'.format(sampled_vertex_c.shape))

    # overall loss
    overall_loss = sdf_loss # + 0.4 * constraint_loss 
    
    return overall_loss



if __name__=='__main__':
    # load data
    dataset = CMU_simulation('assets/data_noised.npz')
    template_vertices = dataset.template_vertices.to(device='cuda')
    template_faces = dataset.template_faces
    #laplacian_matrix = dataset.laplacian_matrix.to(device='cuda')

    input_dim = 181 + 3 + 1 # pose_dim + vertex_dim + time_dim
    output_dim = 3
    batch_size = 64
    seq_length = 130
    
    num_epoches = 15000
    sampling_size = 512
    num_obj_vertex = 2590
    
    # sample random vertex from the vertex sequence,
    #torch.manual_seed(0)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Load Force field model
    force_field = ForceField(input_dim=input_dim-1, num_layers=16, output_dim=3).to(device='cuda') # exclude time dimension

    # Load NLS model
    nls = NLS(input_dim, hidden_dim=256, num_layers=24, output_dim=1).to(device='cuda')

    optimizer = torch.optim.Adam(nls.parameters(), lr=1.0e-5)
    #critrion = loss_laplacian
    # Train the model
    for epoch in range(num_epoches):
        running_loss = 0.0
        for i, data in enumerate(dataloader):
            # zero grad optimizer
            optimizer.zero_grad()
            # reshape data to batch_size, seq_length, input_dim
            gender, poses, vertex_seq, noised_vertex_seq, signed_dist = data # vertex shape (B, S, num_v, 3)
            #print(' signed distance shape : {0}'.format(signed_dist.shape))
            rand_idx = torch.randint(0, num_obj_vertex, size=(sampling_size,)) # sample random vertex from the vertex sequence

            # generate time data shaped with (B, S, 1) in linear space
            #t = torch.ones(poses.shape[0], seq_length, 1).to(device='cuda')
            t = torch.linspace(0, 1, seq_length).repeat(poses.shape[0], 1).unsqueeze(-1).to(device='cuda')
            t_c = t.to(device='cuda') # time data for level set constraints
            poses = poses.to(device='cuda')
            poses_c = poses.to(device='cuda') # pose data for level set constraints
            poses_f = poses.to(device='cuda')
            
            #gt_sdf = torch.zeros(poses.shape[0], seq_length, 1).to(device='cuda') 
            # compute ground truth signed distance function
            # obtain sampled vertex from rand_idx
            #sdf_vertex = vertex_seq[:, :,rand_idx]

            # compute loss for each sampling
            running_loss = 0.0
            for j in rand_idx:
                sampled_vertex = noised_vertex_seq[:, :,j].to(device='cuda')
                sampled_vertex_c = noised_vertex_seq[:,:,j].to(device='cuda') # vertex data for level set constraints
                sampled_vertex_f = noised_vertex_seq[:,:,j].to(device='cuda') # vertex data for force field

                # unsqueeze signed distance to (B, S, num_sample, 1)
                #signed_distance = signed_distance.unsqueeze(-1)
                gt_sdf = signed_dist[:, :, j].to(device='cuda')
                # print gt_sdf shape
                #print(' gt sdf shape : {0}'.format(gt_sdf.shape))
                loss = loss_nls(nls, force_field, gt_sdf, t, poses, sampled_vertex, t_c, poses_c, sampled_vertex_c, poses_f, sampled_vertex_f)
                # print loss
                print(' epoch : {0}, loss : {1}'.format(epoch, loss.item()))
                loss.backward()

                # backward
                optimizer.step()
            # save model every 1000 epochs  
            #if epoch % 1000 == 999:
            #    torch.save(nls.state_dict(), '/home/cxh/tmp/checkpoint/nls/nls_{0:0>3}.pth'.format((epoch-999)//1000))
    print('Done')