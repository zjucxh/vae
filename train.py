import torch
import torch.nn as nn
from data import Cloth_in_Wind
from model import VAE, vae_loss


class Trainer:
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train(self, train_loader, val_loader, num_epochs):
        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            for i, data in enumerate(train_loader):
                inputs, labels = data
                inputs = inputs.reshape(inputs.shape[0],-1)
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs, mu, logvar = self.model(inputs)
                # reconstruction loss
                loss = self.criterion(outputs, inputs, mu, logvar)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                if i % 10 == 9:
                    print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 10}')
                    running_loss = 0.0
            #self.validate(val_loader)

    # TODO re-implement the validate method
    def validate(self, val_loader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in val_loader:
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f'Accuracy: {100 * correct / total}')
    
    # TODO re-implement the test method
    def test(self, test_loader):
        self.model.eval()
        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                print(predicted)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = 1734  # Dimension of input points
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
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False)

    # Train the model
    trainer.train(train_loader, val_loader, num_epochs=1000)

    # Test the model
    #trainer.test(test_loader)