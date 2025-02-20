import numpy as np
# load dataset with torch
import torch
from torch.utils.data import DataLoader, Dataset
import trimesh

class Cloth_in_Wind(Dataset):
    def __init__(self, datapath:str='/home/cxh/mnt/cxh/Documents/dataset/cloth_in_wind'):
        self.datapath = datapath
    def __len__(self):

        return 600 # Number of obj files in the dataset

    def __getitem__(self, idx):
        # Load obj file
        mesh = trimesh.load_mesh(self.datapath + '/cloth_seq%04d.obj' % (idx+1))
        vertices = mesh.vertices
        faces = mesh.faces

        return torch.tensor(vertices, dtype=torch.float32)
    


if __name__=='__main__':
    dataset = Cloth_in_Wind()
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    for i, data in enumerate(dataloader):
        print(f'Batch {i}, data shape: {data.shape}')