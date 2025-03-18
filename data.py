import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import trimesh
from utils import laplacian_matrix, mesh_normals

class CMU_simulation(Dataset):
    def __init__(self, datapath:str='/home/cxh/tmp/CMU_mini_dataset'):
        super().__init__()
        self.datapath = datapath
        self.sequence_length = 130 # Number of obj in each sequence
        self.initial_mesh  = trimesh.load_mesh('assets/template_align.obj')
        # compute laplacian matrix via trimesh
        #self.trimesh_laplacian_matrix= np.array(trimesh.smoothing.laplacian_calculation(self.initial_mesh).todense()[-1])
        #print(f' laplacian : {self.trimesh_laplacian_matrix}')
        self.template_faces = torch.tensor(self.initial_mesh.faces, dtype=torch.int64)
        self.template_vertices = torch.tensor(self.initial_mesh.vertices, dtype=torch.float32)
        #print(f' template_vertices : {self.template_vertices.shape}')
        # walk through all the sequences
        self.npz_files = []
        self.npz_indices = []
        for root, dirs, files in os.walk(self.datapath):
            for file in files:
                if file.endswith('.npz'):
                    self.npz_files.append(os.path.join(root, file))
                    self.npz_indices.append(int(file.split('.')[0].split('_')[0]))
        #print(f' npz_files: {self.npz_files}')
        #print(f' npz_indices: {self.npz_indices}') 
        self.laplacian_matrix = laplacian_matrix(self.template_faces,normalize=True)
        print(f' laplacian matirx shape: {(self.laplacian_matrix.shape)}')
        print(f' sum of laplacian matirx: {(self.laplacian_matrix)}')
        self.dataset_length = len(self.npz_files)
        self.data = []
        self.normals = []
        self.gender = 0
        # load all data sequences
        for i in range(self.dataset_length):
            seq = np.load(self.npz_files[i])
            self.data.append(seq)
        #print(f' dataset_length: {self.dataset_length}')
        
        # Compute vertex normals given vertex_seq and template faces
        #self.compute_vertex_normals()


    def __len__(self):
        return self.dataset_length
    
    def __getitem__(self, index):
        
        gender = 0
        data = self.data[index]
        if data['gender'] == 'female':
            gender = 0 # female
        else:
            gender = 1 # male
        betas = data['betas']
        poses = data['poses']
        vertex_seq = torch.tensor(data['vertex_seq'],dtype=torch.float32)
        vertex_seq = vertex_seq - self.template_vertices
        betas = np.repeat(betas[np.newaxis, :], self.sequence_length, axis=0) 
        # concat betas with poses
        poses = np.concatenate((betas, poses), axis=1)
        # return the sequence
        poses = torch.tensor(poses, dtype=torch.float32)
        return gender, poses, vertex_seq
    
    def compute_vertex_normals(self):
        """
        Compute the vertex normals from the vertex sequence and faces
        Returns:
            np.ndarray of shape (sequence_length, num_vertices, 3) with dtype=np.float32
        """
        normals = []
        for i, data in enumerate(self.data):
            seq = torch.tensor(data['vertex_seq'],dtype=torch.float32)
            # reshape vertex_seq to (sequence_length, num_vertices, 3)
            #print(f' vertex seq shape : {seq.shape}')
            for j, vertex in enumerate(seq):
                vertex_normals = mesh_normals(vertex, self.template_faces)
                normals.append(vertex_normals)
                print(f' vertex_normals : {vertex_normals.shape}')
            self.normals = torch.stack(normals)
            print(f' self.normals : {self.normals.shape}')
        

if __name__=='__main__':
    cmu_simulation_dataset = CMU_simulation()
    template_vertices = cmu_simulation_dataset.template_vertices
    template_faces = cmu_simulation_dataset.template_faces
    # Dataloader
    dataloader = DataLoader(cmu_simulation_dataset, batch_size=4, shuffle=True)
    for i, data in enumerate(dataloader):
        gender, poses, vertex_seq = data
        print('poses : {0}'.format(poses.shape))
        print('vertex_seq : {0}'.format(vertex_seq.shape))
        break