import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import trimesh
from utils import laplacian_matrix, mesh_normals, signed_distance, boundary_edges, boundary_vertices

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

        self.laplacian_matrix = laplacian_matrix(self.template_faces,normalize=True)
        self.dataset_length = len(self.npz_files)
        self.data = []
        self.normals = []
        self.gender = 0
        # load all data sequences
        for i in range(self.dataset_length):
            seq = np.load(self.npz_files[i])
            self.data.append(seq)
        
        # Compute vertex normals given vertex_seq and template faces
        #self.compute_vertex_normals()
        # print keys in data
        print(f' keys in data : {self.data[0].keys()}')
        # generate noise from vertex sequence
    def add_noise(self):
        noised_signed_distance = []
        for data in self.data:
            vertex_seq = data['vertex_seq']
            # add noise to vertex_seq
            noise = np.random.normal(0, 0.5, vertex_seq.shape)
            print(f' noise shape: {noise.shape}')
            noised_vertex_seq = vertex_seq + noise
            print(f' noised vertex seq shape : {noised_vertex_seq.shape}')
            # compute signed distance
            seq_sdist = np.zeros((self.sequence_length, noised_vertex_seq.shape[1]))
            for i in range(self.sequence_length):
                # compose Trimesh
                mesh = trimesh.Trimesh(vertices=vertex_seq[i], faces=self.template_faces)
                #print(' noised_vertex_seq[i] : ', noised_vertex_seq[i])
                sdist = signed_distance(mesh, noised_vertex_seq[i]) # array with length noised num_vertices
                #print(f' sdist : {sdist.shape}')
                seq_sdist[i] = sdist
            noised_signed_distance.append(seq_sdist)   
            # seq_sdist shape
            print(f' seq_sdist shape : {seq_sdist.shape}')
            # write noised veetex sequence and noised signed distance
            np.savez(f'assets/noised_{self.npz_files[i]}', vertex_seq=noised_vertex_seq, signed_distance=noised_signed_distance)


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
    template_faces = cmu_simulation_dataset.template_faces.numpy()

    lgender = []
    lposes = []
    lvertex_seq = []
    lnoised_vertex_seq = []
    lsdist_seq = []
    for i, (gender, poses, vertex_seq )in enumerate(cmu_simulation_dataset):
        poses = poses.numpy()
        vertex_seq = vertex_seq.numpy()

        print('gender : {0}\n poses: {1} \n vertex seq : {2}\n i : {3}'.format( gender, poses.shape, vertex_seq.shape, i))
        # append
        lgender.append(gender)     
        lposes.append(poses)
        lvertex_seq.append(vertex_seq)   
        # generate noised data from vertex sequnces
        noised_vertex_seq = vertex_seq + np.random.normal(0, 0.5, vertex_seq.shape)

        print(' noised vertex seq shape : ', noised_vertex_seq.shape)
        # compute signed distance for each mesh
        # compose trimesh
        sdist_seq = np.zeros((cmu_simulation_dataset.sequence_length, noised_vertex_seq.shape[1]))
        for j, mesh_vertex in enumerate(vertex_seq):
            mesh = trimesh.Trimesh(vertices=mesh_vertex, faces=template_faces)
            # compute signed distance
            sdist = signed_distance(mesh, noised_vertex_seq[j])
            sdist_seq[j] = sdist
        

    #Save to npz file
    #np.savez(file='assets/data.npz', gender=lgender, poses=lposes, vertex_seq=lvertex_seq, template_faces = cmu_simulation_dataset.template_faces\
    #         , template_vertices = cmu_simulation_dataset.template_vertices)

    # Load data 
    #data = np.load('assets/data.npz')
    #print(f' data keys : {data.keys()}')

    