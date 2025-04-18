import torch
import torch.nn as nn
#from torch.autograd import Variable
from data import CMU_simulation 
from torch.utils.data import DataLoader, Dataset
#from utils import signed_distance
from loss import loss_signed_distance, loss_l2

class PredictionTransformer(nn.Module):
    def __init__(self, smpl_param_dim, transformer_hidden_dim, num_transformer_layers, num_heads, output_dim=7770):
        """
        Initializes the SDF prediction network using a Transformer.

        Args:
            smpl_param_dim (int): Dimensionality of the SMPL pose parameters.
            transformer_hidden_dim (int): Hidden dimension of the Transformer.
            num_transformer_layers (int): Number of Transformer layers.
            num_heads (int): Number of attention heads in the Transformer.
            output_dim (int): Dimensionality of the output (SDF value). Defaults to 1.
        """
        super(PredictionTransformer, self).__init__()

        self.smpl_param_dim = smpl_param_dim
        self.transformer_hidden_dim = transformer_hidden_dim
        self.num_transformer_layers = num_transformer_layers
        self.num_heads = num_heads
        self.transformer_input_dim = 512 # Output dimension of the SMPL processor

        # Process SMPL parameters with a simple MLP
        self.smpl_processor = nn.Sequential(
            nn.Linear(smpl_param_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.transformer_input_dim),  # Reduced dimensionality before combining
            nn.ReLU()
        )

        # Linear layer to project the input to the transformer's hidden dimension
        self.input_projection = nn.Linear(self.transformer_input_dim, transformer_hidden_dim)


        # Transformer layer
        self.transformer = nn.Transformer(
            d_model=transformer_hidden_dim,
            nhead=num_heads,
            num_encoder_layers=num_transformer_layers,
            num_decoder_layers=num_transformer_layers, # We'll use it as an encoder-only transformer
            batch_first=True,  # Important: (batch, seq_len, features)
        )

        # Output layer: Predict SDF
        self.predictor = nn.Sequential(
            nn.Linear(transformer_hidden_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim),  # Predicts a single SDF value
        )

    def forward(self, smpl_params):
        """
        Predicts SDF values for a sequence of vertices, conditioned on SMPL pose parameters, using a Transformer.

        Args:
            smpl_params (torch.Tensor): SMPL pose parameters.
                Shape: [batch_size, sequence_length, smpl_param_dim]
            

        Returns:
            torch.Tensor: Predicted vertex sequences.
                Shape: [batch_size, sequence_length, output_dim]
        """
        batch_size, seq_len, _ = smpl_params.shape
        #print(' vertex sequence length : {0}'.format(vertex_sequence.shape))

        # 1. Process SMPL parameters
        smpl_features = self.smpl_processor(smpl_params) # [B, S, 128]

        # 3. Project input to the transformer's hidden dimension
        transformer_input = self.input_projection(smpl_features) # [B, S, transformer_hidden_dim]


        # 4. Use the transformer in encoder-only mode
        transformer_output = self.transformer.encoder(transformer_input)  # [B, S, transformer_hidden_dim]

        # 5. Predict SDF values
        pred_vertex_seq = self.predictor(transformer_output)  # [B, S, output_dim]
        # Reshape pred_vertex_seq to [B, S, num_vertices, 3]

        #print('pred_vertex_seq shape : {0}'.format(pred_vertex_seq.shape))
        pred_vertex_seq = pred_vertex_seq.view(batch_size, seq_len, -1, 3)  # [B, S, num_vertices, 3]

        return  pred_vertex_seq



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

# Define Nerual Level Set Model
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
    
# Define Forcefield Model 
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


if __name__=='__main__':
    # load data
    dataset = CMU_simulation('assets/data_noised.npz')
    template_vertices = dataset.template_vertices.to(device='cuda')
    template_faces = dataset.template_faces
    #laplacian_matrix = dataset.laplacian_matrix.to(device='cuda')

    input_dim = 181  # pose_dim 
    output_dim = 7770 # num_vertices * 3
    batch_size = 8
    seq_length = 130
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    num_epoches = 15000
    num_obj_vertex = 2590
    
    # Load Transformer model
    predictor = PredictionTransformer(smpl_param_dim=181, transformer_hidden_dim=512, num_transformer_layers=8, num_heads=8, output_dim=7770).to(device='cuda')

    optimizer = torch.optim.Adam(predictor.parameters(), lr=1.0e-5)
    #critrion = loss_laplacian
    # Train the model
    for epoch in range(num_epoches):
        running_loss = 0.0
        for i, data in enumerate(dataloader):
            # zero grad optimizer
            optimizer.zero_grad()
            # reshape data to batch_size, seq_length, input_dim
            gender, poses, vertex_seq, noised_vertex_seq, signed_distance = data # vertex shape (B, S, num_v, 3)
            poses = poses.to(device='cuda')
            vertex_seq = vertex_seq.to(device='cuda')
            # print poses and vertex_seq shape
            #print(f' poses shape : {poses.shape}, vertex_seq shape : {vertex_seq.shape}')

            pred_vertex_seq = predictor(poses) # [B, S, num_vertices, 3]
            # l2 loss
            loss = loss_l2(pred_vertex_seq, vertex_seq)

            # compute loss for each sampling
            

            # backward
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            #print('i : {0}, loss : {1}'.format(i, loss.item()))
            if i % 10 == 9:
                print(f' epoch : {epoch}, loss : {running_loss/10.0}')
            # save model every 1000 epochs  
            #if epoch % 1000 == 999:
            #    torch.save(nls.state_dict(), '/home/cxh/tmp/checkpoint/nls/nls_{0:0>3}.pth'.format((epoch-999)//1000))

    print('Done')