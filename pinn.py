import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 定义 PINN 网络结构
class PINN(nn.Module):
    def __init__(self, layers):
        super(PINN, self).__init__()
        self.linear_layers = nn.ModuleList()
        for i in range(len(layers)-1):
            self.linear_layers.append(nn.Linear(layers[i], layers[i+1]))
            if i < len(layers)-2:
                self.linear_layers.append(nn.Tanh())  # 使用 Tanh 激活函数

    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)  # 拼接空间坐标 x 和时间坐标 t
        for layer in self.linear_layers:
            inputs = layer(inputs)
        return inputs  # 输出预测的 u(x, t)

# 定义损失函数计算
def compute_loss(model, x_ic, t_ic, u_ic, x_bc, t_bc, x_res, t_res, nu):
    # 初始条件损失
    u_pred_ic = model(x_ic, t_ic)
    loss_ic = torch.mean((u_pred_ic - u_ic)**2)
    
    # 边界条件损失
    u_pred_bc = model(x_bc, t_bc)
    loss_bc = torch.mean(u_pred_bc**2)  # u(-1,t)=u(1,t)=0
    
    # PDE 残差损失
    x_res.requires_grad_(True)
    t_res.requires_grad_(True)
    u_res = model(x_res, t_res)
    
    # 计算一阶导数
    grad_u = torch.autograd.grad(u_res, [x_res, t_res], 
                               grad_outputs=torch.ones_like(u_res),
                               create_graph=True)
    u_x, u_t = grad_u[0], grad_u[1]
    
    # 计算二阶导数 (u_xx)
    grad_u_x = torch.autograd.grad(u_x, x_res, 
                                 grad_outputs=torch.ones_like(u_x),
                                 create_graph=True)
    u_xx = grad_u_x[0]
    
    # Burgers 方程残差
    residual = u_t + u_res * u_x - nu * u_xx
    loss_res = torch.mean(residual**2)
    
    # 总损失
    total_loss = loss_ic + loss_bc + loss_res
    return total_loss

# 生成训练数据
def generate_data(num_ic=100, num_bc=100, num_res=1000):
    # 初始条件 (t=0)
    x_ic = torch.FloatTensor(num_ic, 1).uniform_(-1, 1)
    t_ic = torch.zeros(num_ic, 1)
    u_ic = -torch.sin(np.pi * x_ic)
    
    # 边界条件 (x=-1 和 x=1)
    x_bc = torch.FloatTensor(num_bc, 1).uniform_(-1, 1)
    x_bc = torch.cat([-torch.ones(num_bc//2, 1), torch.ones(num_bc//2, 1)])
    t_bc = torch.FloatTensor(num_bc, 1).uniform_(0, 1)
    
    # 残差采样点 (内部点)
    x_res = torch.FloatTensor(num_res, 1).uniform_(-1, 1)
    t_res = torch.FloatTensor(num_res, 1).uniform_(0, 1)
    
    return x_ic, t_ic, u_ic, x_bc, t_bc, x_res, t_res

# 训练过程
def train_pinn():
    # 超参数
    layers = [2, 20, 20, 20, 1]  # 输入维度 2 (x, t), 输出维度 1 (u)
    nu = 0.01/np.pi  # 粘性系数
    epochs = 10000
    lr = 1e-3
    
    # 初始化模型和优化器
    model = PINN(layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # 生成数据
    x_ic, t_ic, u_ic, x_bc, t_bc, x_res, t_res = generate_data()
    
    # 训练循环
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = compute_loss(model, x_ic, t_ic, u_ic, x_bc, t_bc, x_res, t_res, nu)
        loss.backward()
        optimizer.step()
        
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4e}")
    
    return model

# 训练并可视化结果
if __name__ == "__main__":
    #model = train_pinn()
    
    # 测试数据
    x_test = torch.linspace(-1, 1, 100).view(-1, 1)
    t_test = torch.linspace(0, 1, 100).view(-1, 1)
    X, T = torch.meshgrid(x_test.squeeze(), t_test.squeeze())
    x_flat = X.reshape(-1, 1)
    t_flat = T.reshape(-1, 1)
    print('x_flat : {0}'.format(x_flat))
    print('t_flat : {0}'.format(t_flat)) 
    # 预测
    #with torch.no_grad():
    #    u_pred = model(x_flat, t_flat).numpy().reshape(100, 100)
    
    ## 可视化
    #plt.figure(figsize=(10, 6))
    #plt.contourf(X.numpy(), T.numpy(), u_pred, levels=50, cmap='jet')
    #plt.colorbar()
    #plt.xlabel('x')
    #plt.ylabel('t')
    #plt.title('PINN Solution for Burgers Equation')
    #plt.show()