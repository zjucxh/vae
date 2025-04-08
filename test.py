import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer
# import positional encoding from torch

# Defone a 4 layered MLP model 
class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = torch.nn.Linear(hidden_dim, output_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, x,y,t):
        # Concatenate x, y, t
        input = torch.cat([x, y, t], dim=-1)
        # Pass through the MLP
        input = self.relu(self.fc1(input))
        input = self.relu(self.fc2(input))
        input = self.relu(self.fc3(input))
        output = self.fc4(input)
        return output 
    
# Define a L2 loss function
def l2_loss(model, x_ic,y_ic, t_ic,x_bc, y_bc, t_bc, x_res, y_res, t_res):
    # Initial condition loss
    output_ic = model(x_ic, y_ic, t_ic)
    loss_ic = torch.mean((output_ic)**2)  # u(x,y,0)=0
    # particial equation loss
    x_bc.requires_grad_(True)
    y_bc.requires_grad_(True)
    t_bc.requires_grad_(True)
    output_bc = model(x_bc, y_bc, t_bc)
    #print(f' output_bc : {output_bc}')
    # partial derivative of pred_bc with respect to input_bc
    dx_bc, dy_bc, dt_bc = torch.autograd.grad(output_bc, [x_bc, y_bc, t_bc],
                                               grad_outputs=torch.ones_like(output_bc),
                                               create_graph=True)
    loss_bc = torch.mean((2*x_bc*dx_bc + 2*y_bc*dy_bc - dt_bc)**2)  

    # residual loss
    output_res = model(x_res, y_res, t_res)
    loss_res = torch.mean((output_res)**2)  # u(x,y,t)=0
    
    return loss_ic + loss_bc + loss_res

# generate data
def generate_data():
    # Initial condition (t=0)
    batch_size = 10
    x_ic = torch.randn(batch_size,8)
    y_ic = torch.randn(batch_size,8)
    t_ic = torch.randn(batch_size,8)
    # particial equation 
    x_bc = torch.randn(batch_size, 8)*2 + 1.0
    y_bc = torch.randn(batch_size, 8)*3 + 2.0
    t_bc = x_bc **2 + y_bc **2

    # Residual data
    x_res = torch.randn(batch_size, 8)*0.2 + 1.0
    y_res = torch.randn(batch_size, 8)*2.0 - 1.0
    t_res = x_res **2 + y_res **2

    return x_ic, y_ic, t_ic, x_bc, y_bc, t_bc, x_res, y_res, t_res

# Define a function to train the model
def train(model, optimizer, criterion, x_ic, y_ic, t_ic, x_bc, y_bc, t_bc, x_res, y_res, t_res):
    model.train()
    optimizer.zero_grad()
    loss = criterion(model, x_ic,y_ic,t_ic,x_bc,y_bc,t_bc,x_res,y_res,t_res)
    loss.backward()
    optimizer.step()
    return loss.item()

# Define a function to evaluate the model
def eval(model, x_ic, y_ic, t_ic, x_bc, y_bc, t_bc, x_res, y_res, t_res):
    model.eval()
    with torch.no_grad():
        output = model(x_res, y_res, t_res)
        loss = torch.mean((output)**2)  

    return loss.item()


if __name__=='__main__':

    # train model
    input_dim = 24
    hidden_dim = 14
    output_dim = 1
    model = MLP(input_dim, hidden_dim, output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = l2_loss
    epochs = 10000
    for epoch in range(epochs):
        x_ic, y_ic, t_ic, x_bc, y_bc, t_bc, x_res, y_res, t_res = generate_data()
        optimizer.zero_grad()
        loss = train(model, optimizer, criterion, x_ic, y_ic, t_ic, x_bc, y_bc, t_bc, x_res, y_res, t_res)
        if epoch % 100 == 99:
            print(f"Epoch {epoch}, Loss: {loss:.4e}")
            # evaluate model
            x_ic, y_ic, t_ic, x_bc, y_bc, t_bc, x_res, y_res, t_res = generate_data()
            loss = eval(model, x_ic, y_ic, t_ic, x_bc, y_bc, t_bc, x_res, y_res, t_res)
            # print
            print(f"Eval Loss: {loss:.4e}")


    print('Done')