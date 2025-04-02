import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer
# import positional encoding from torch
if __name__=='__main__':
    encoder_layer = TransformerEncoderLayer(d_model=184, nhead=8, dim_feedforward=1024, batch_first=True).to(device='cuda')
    tf_encoder = TransformerEncoder(encoder_layer, num_layers=6).to(device='cuda')
    x = torch.randn(8, 130, 184).to(device='cuda') # batch_size, seq_length, input_dim
    # embedding for x


    y = tf_encoder(x)
    print(f' y shape : {y.shape}')

    print('Done')