import torch
import torch.nn as nn
from data import CMU_simulation
from model import GRU, loss_laplacian
from torch.utils.tensorboard import SummaryWriter

class Trainer:
    def __init__(self, model, optimizer, criterion, laplacian_matrix, device):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.laplacian_matrix = laplacian_matrix.to(device)
        self.writer = SummaryWriter(log_dir='runs/gru')

    def train(self, train_loader, val_loader, num_epochs):
        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            for i, data in enumerate(train_loader):
                gender, poses, vertex_seq = data
                inputs = poses.to(self.device)
                vertex_seq = vertex_seq.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                outputs = outputs.view_as(vertex_seq)
                loss = self.criterion(self.laplacian_matrix, outputs, vertex_seq)
                loss.backward()
                
                self.optimizer.step()
                running_loss += loss.item()
                if i % 10 == 9:
                    print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 10}')
                    self.writer.add_scalars('Loss', {'Train GRU': running_loss / 10}, 
                                            epoch * len(train_loader) + i)
                    running_loss = 0.0
                    self.validate(val_loader, epoch * len(train_loader) + i)
            if epoch % 100 == 99:
                self.save(f'~/tmp/checkpoint/gru/gru_{(epoch-99)//100}.pth')
        self.writer.flush()
        self.writer.close()

    def validate(self, val_loader, n_iter):
        self.model.eval()
        with torch.no_grad():
            data = next(iter(val_loader))
            gender, poses, vertex_seq = data
            inputs = poses.to(self.device)
            vertex_seq = vertex_seq.to(self.device)
            outputs = self.model(inputs)
            outputs = outputs.view_as(vertex_seq)
            eval_loss = self.criterion(self.laplacian_matrix, outputs, vertex_seq)
            self.writer.add_scalars('Loss', {'Validation GRU': eval_loss.item()}, 
                                   n_iter)
        self.model.train()
            
    def save(self, model_path):
        torch.save(self.model.state_dict(), model_path)
    def load(self, model_path):
        self.model.load_state_dict(torch.load(model_path))
    

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = 181  # Dimension of input points
    hidden_dim = 256  # Dimension of hidden layers
    output_dim = 2590 * 3  # Dimension of the output points
    latent_dim = 1024  # Dimension of the latent space

    # Initialize the GRU model
    gru_model = GRU(input_dim, hidden_dim, num_layers=8, output_dim=output_dim).to(device)

    # Define optimizer
    optimizer = torch.optim.Adam(gru_model.parameters(), lr=1.0e-5)

    # Define loss function
    criterion = loss_laplacian
    
    # Load the dataset
    dataset = CMU_simulation()
    laplacian_matrix = dataset.laplacian_matrix.to(device)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False)
    
    # Initialize the trainer
    trainer = Trainer(gru_model, optimizer, criterion,laplacian_matrix, device)

    

    # Train the model
    trainer.train(train_loader, val_loader, num_epochs=40000)

    # Test the model
    #trainer.test(test_loader)