import torch
import torch.nn as nn
from data import Cloth_in_Wind
from model import VAE, vae_loss
from torch.utils.tensorboard import SummaryWriter
class Trainer:
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.writer = SummaryWriter(log_dir='runs/vae')
    def train(self, train_loader, val_loader, num_epochs):
        # Load the model
        #self.load('/home/cxh/mnt/cxh/Documents/assets/chckpoints/vae_99.pth')
        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            for i, data in enumerate(train_loader):
                vertices = data
                inputs = vertices.reshape(vertices.shape[0],-1)
                inputs= inputs.to(self.device)
                self.optimizer.zero_grad()
                outputs, mu, logvar = self.model(inputs)
                # reconstruction loss
                loss = self.criterion(outputs, inputs, mu, logvar)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                if i % 10 == 9:
                    print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 10}')
                    self.writer.add_scalars('Loss', {'Train VAE': running_loss / 10}, 
                                       epoch * len(train_loader) + i)
                    running_loss = 0.0
                    self.validate(val_loader, epoch * len(train_loader)+i)
            if epoch % 100 == 99:
                self.save(f'/home/cxh/mnt/cxh/Documents/assets/checkpoints/vae/vae_{(epoch-99)//100}.pth')
        self.writer.flush()
        self.writer.close()
            

    def validate(self, val_loader, n_iter):
        self.model.eval()
        with torch.no_grad():
            data = next(iter(val_loader))
            vertices = data
            inputs = vertices.reshape(vertices.shape[0], -1)
            inputs = inputs.to(self.device)
            outputs, mu, logvar = self.model(inputs)
            eval_loss = self.criterion(outputs, inputs, mu, logvar)
            self.writer.add_scalars('Loss', {'Validation VAE': eval_loss.item()}, 
                                   n_iter)
            
    def save(self, model_path):
        torch.save(self.model.state_dict(), model_path)
    def load(self, model_path):
        self.model.load_state_dict(torch.load(model_path))
    

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = 867# Dimension of input points
    hidden_dim = 256  # Dimension of hidden layers
    latent_dim = 1024 # Dimension of the latent space

    # Initialize the VAE model
    vae_model = VAE(input_dim, hidden_dim, latent_dim).to(device)

    # Define optimizer
    optimizer = torch.optim.Adam(vae_model.parameters(), lr=1.0e-5)

    # Define loss function
    criterion = vae_loss

    # Initialize the trainer
    trainer = Trainer(vae_model, optimizer, criterion, device)

    # Load the dataset
    dataset = Cloth_in_Wind()
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False)

    # Train the model
    trainer.train(train_loader, val_loader, num_epochs=40000)

    # Test the model
    #trainer.test(test_loader)