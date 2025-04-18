import torch
import torch.nn as nn
from data import CMU_simulation
from model import PredictionTransformer, loss_l2
from torch.utils.tensorboard import SummaryWriter

class Trainer:
    def __init__(self, model, optimizer, criterion, device):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.writer = SummaryWriter(log_dir='runs/transformer')

    def train(self, train_loader, val_loader, num_epochs):
        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            n_iter = 0
            for i, data in enumerate(train_loader):
                gender, poses, vertex_seq, _, _ = data
                poses = poses.to(self.device)  # Transformer input
                vertex_seq = vertex_seq.to(self.device)  # Ground truth output

                self.optimizer.zero_grad()
                outputs = self.model(poses)  # Forward pass
                loss = self.criterion(outputs, vertex_seq)  # Compute loss
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                n_iter += 1

            print('Epoch : {0}, Loss : {1}'.format(epoch, running_loss / n_iter))
            self.writer.add_scalars('Loss', {'Train Transformer': running_loss / n_iter},
                                    epoch * len(train_loader) + i)
            running_loss = 0.0
            #self.validate(val_loader, epoch * len(train_loader) + i)
            if epoch % 100 == 99:
                self.save('/home/cxh/tmp/checkpoint/transformer/transformer_{0:0>4}.pth'.format((epoch-99)//100))
        self.writer.flush()
        self.writer.close()

    def validate(self, val_loader, n_iter):
        self.model.eval()
        with torch.no_grad():
            data = next(iter(val_loader))
            gender, poses, vertex_seq,_,_ = data
            poses = poses.to(self.device)
            vertex_seq = vertex_seq.to(self.device)
            outputs = self.model(poses)
            eval_loss = self.criterion(outputs, vertex_seq)
            self.writer.add_scalars('Loss', {'Validation Transformer': eval_loss.item()},
                                    n_iter)
        self.model.train()

    def save(self, model_path):
        torch.save(self.model.state_dict(), model_path)

    def load(self, model_path):
        self.model.load_state_dict(torch.load(model_path))


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = 181  # Dimension of SMPL pose parameters
    transformer_hidden_dim = 512
    num_transformer_layers = 8
    num_heads = 8
    output_dim = 2590 * 3  # Dimension of the output points
    lr = 1.0e-5 # Learning rate
    num_epochs = 40000

    # Initialize the Transformer model
    transformer_model = PredictionTransformer(input_dim, transformer_hidden_dim, num_transformer_layers, num_heads, output_dim).to(device)

    # Define optimizer
    optimizer = torch.optim.Adam(transformer_model.parameters(), lr=lr)

    # Define loss function
    criterion = loss_l2

    # Load the dataset
    dataset = CMU_simulation()
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

    # Initialize the trainer
    trainer = Trainer(transformer_model, optimizer, criterion, device)

    # Train the model
    trainer.train(train_loader, val_loader, num_epochs)