import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.functional import jacobian
from data import CMU_simulation 
from torch.utils.data import DataLoader, Dataset

# Construct the neural level set model
class NeuralLevelSet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=4, output_dim=1):
        super(NeuralLevelSet, self).__init__()
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # The GRU module take input_dim dimension as input, the 181 is the pose parameters, the 3*n is the initial guess of the vertices
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True )
        self.fc = nn.Linear(hidden_dim, output_dim) 

        # construct velocity field via nesnet 
        self.velocity_field = nn.Sequential(
            nn.Linear(3, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 3)
        )

    def forward(self, x, theta):
        input = torch.cat((x, theta), dim=2)
        h0 = torch.zeros(self.num_layers, input.size(0), self.hidden_dim).to('cuda')
        print('input size : {0}'.format(input.size(0)))
        out, _ = self.gru(input, h0)
        out = self.fc(out)
        return out


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

def loss_l2(pred_distance, gt_distance, clamp=0.1):
    l2_loss = nn.MSELoss()
    #pred_distance = torch.clamp(pred_distance, -clamp, clamp)
    #gt_distance = torch.clamp(gt_distance, -clamp, clamp)
    loss = l2_loss(pred_distance, gt_distance)
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

    input_dim = 181 + 6
    output_dim = 1
    batch_size = 8
    seq_length = 130
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    num_epoches = 15000
    
    # generate random vertex 
    initial_vertex_guess = torch.randn(6).to(device='cuda')

    # Load GRU model
    neural_levelset = NeuralLevelSet(input_dim, hidden_dim=256, num_layers=8, output_dim=initial_vertex_guess.shape[0]//3).to(device='cuda')
    print('Neural Level Set Model : {0}'.format(neural_levelset))
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
            print('poses : {0}'.format(poses.shape)) # shape : batch_size, seq_length, 181
            # concatenate poses with initial vertex guess
            x = initial_vertex_guess.repeat(poses.shape[0], seq_length, 1)
            x.requires_grad = True
            poses.requires_grad = True
            output = neural_levelset(x, poses)

            # compute grad
            loss = output.pow(2).mean()
            y_xt = torch.autograd.grad(loss, [x, poses], create_graph=True)
            print('y_x : {0}'.format(y_xt[0].shape))
            print('y_t : {0}'.format(y_xt[1].shape))

            print('output : {0}'.format(output.shape))
            # padded temporal difference
                
            #output.pow(2).mean().backward()
            #jacobian_matrix = jacobian(neural_levelset, inputs)
            #print('jacobian_matrix : {0}'.format(jacobian_matrix.shape))

        # save model for every 100 epoches
        #if epoch % 100 == 99:
        #   torch.save(gru.state_dict(), '/home/cxh/tmp/checkpoint/gru_{0:0>3}.pth'.format((epoch-99)//100))
        
    print('Done')