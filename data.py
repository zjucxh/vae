import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import trimesh
from utils import laplacian_matrix, mesh_normals, signed_distance, boundary_edges, boundary_vertices

class CMU_simulation(Dataset):
    def __init__(self, datapath: str = 'assets/data_noised.npz'):
        """
        Initialize the CMU_simulation dataset class.

        Args:
            datapath (str): Path to the data_noised.npz file.
        """
        super().__init__()
        self.datapath = datapath
        

        # Load the data from the .npz file
        data = np.load(self.datapath)
        self.gender = data['gender']
        self.poses = data['poses']
        self.vertex_seq = torch.tensor(data['vertex_seq'], dtype=torch.float32)
        self.noised_vertex_seq = torch.tensor(data['noised_vertex_seq'], dtype=torch.float32)
        self.signed_distance = torch.tensor(data['signed_distance'], dtype=torch.float32) # Signed distances of noised vertex sequences
        self.template_faces = torch.tensor(data['template_faces'], dtype=torch.int64)
        self.template_vertices = torch.tensor(data['template_vertices'], dtype=torch.float32)
        self.sequence_length = self.vertex_seq.shape[1] # Number of objects in each sequence
        self.dataset_length = self.vertex_seq.shape[0] # Dataset length is determined by the number of sequences
        
        # Compute the Laplacian matrix for the template mesh
        #self.laplacian_matrix = laplacian_matrix(self.template_faces, normalize=True)

        
        

    def __len__(self):
        """
        Return the number of sequences in the dataset.

        Returns:
            int: Number of sequences.
        """
        return self.dataset_length

    def __getitem__(self, index):
        """
        Retrieve a single sequence from the dataset.

        Args:
            index (int): Index of the sequence to retrieve.

        Returns:
            tuple: (gender, poses, vertex_seq, noised_vertex_seq, signed_distance)
        """
        gender = self.gender[index]
        poses = torch.tensor(self.poses[index], dtype=torch.float32)
        vertex_seq = self.vertex_seq[index] + self.template_vertices
        noised_vertex_seq = self.noised_vertex_seq[index] + self.template_vertices
        signed_distance = self.signed_distance[index]

        return gender, poses, vertex_seq, noised_vertex_seq, signed_distance

def generate_noised_data():
    # Load data 
    data = np.load('assets/data.npz')
    
    gender = data['gender']
    poses = data['poses']
    vertex_seq = data['vertex_seq']
    template_faces = data['template_faces']
    template_vertices = data['template_vertices']

    # Generate noised vertex sequence from vertex_seq with mean 0.0 and std 0.5
    noise = np.random.normal(0, 0.5, vertex_seq.shape)
    noised_vertex_seq = vertex_seq + noise

    # print data shape
    print('gender dtype : {0}'.format(gender.dtype))
    print('poses dtype : {0}'.format(poses.dtype))
    print('vertex_seq dtype : {0}'.format(vertex_seq.dtype))
    print('template_faces dtype : {0}'.format(template_faces.dtype))
    print('template_vertices dtype : {0}'.format(template_vertices.dtype))

    num_sequences = vertex_seq.shape[0]
    sequence_length = vertex_seq.shape[1]
    num_vertices = vertex_seq.shape[2]

    # From vertex_seq and template_faces compute signed distance given vertex_seq
    # Initialize the signed distance array with zero
    sdist = np.zeros((num_sequences, sequence_length, num_vertices))
    for i in range(num_sequences):
        for j in range(sequence_length):
            # compose Trimesh
            mesh = trimesh.Trimesh(vertices=vertex_seq[i][j], faces=template_faces)
            points = noised_vertex_seq[i][j]
            # compute signed distance
            sd = signed_distance(mesh, points, eps=1.0e-4)
            sdist[i][j] = sd
            print(' seq {0}, frame {1}, sdist shape : {2}'.format(i, j, sdist[i][j]))

    #Save to npz file
    np.savez(file='assets/data_noised.npz', gender=gender, poses=poses, vertex_seq=vertex_seq, template_faces = template_faces\
             , template_vertices = template_vertices, noised_vertex_seq=noised_vertex_seq, signed_distance=sdist)

if __name__ == '__main__':
    # Initialize cmu simulation dataset
    cmu_dataset = CMU_simulation()
    _, _, _, vertex_seq,_ = cmu_dataset[44]
    template_faces = cmu_dataset.template_faces
    # compose vertex sequence and template faces
    for j in range(cmu_dataset.sequence_length):

        mesh = trimesh.Trimesh(vertices=vertex_seq[j], faces=cmu_dataset.template_faces)
        # write obj file to assets
        mesh.export('assets/seq/vertex_seq_{0}.obj'.format(j))


    print(' Done')
    