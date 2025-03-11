import os
import numpy as np
# load dataset with torch
import torch
from torch.utils.data import DataLoader, Dataset
import trimesh

class Cloth_in_Wind(Dataset):
    def __init__(self, datapath:str='/home/cxh/mnt/cxh/Documents/dataset/cloth_in_wind'):
        self.datapath = datapath
        self.data_length = 600
        self.wind_velocity = np.ones(3,dtype=np.float32)
        # Load obj file via trimesh and store to list
        print(f' loading data sequences')
        self.mesh = [trimesh.load_mesh(os.path.join(self.datapath, 'cloth_seq%04d.obj' % (i+1))) for i in range(self.data_length)]
        initial_mesh = self.mesh[0]
        print('Done')


    def signed_distance(self, mesh:trimesh.Trimesh, points:np.ndarray,eps:float=1e-5):
        """
        Compute the signed distance between a mesh and a set of points.
        Args:
            mesh: trimesh.Trimesh object
            points: np.ndarray of shape (N, 3)
            eps: float, small value to judge if a point is very close to the interface
        Returns:
            np.ndarray of shape (N,) with dtype=np.float32
        """
        # Compute the boundary edges 
        boundary_edges = self.boundary_edges(mesh)
        vertices = np.array(mesh.vertices,dtype=np.float32)
        boundary_vertices = vertices[np.unique(boundary_edges.flatten())]
        boundary_distance = np.zeros(points.shape[0],dtype=np.float32)
        mesh_distance = np.zeros(points.shape[0],dtype=np.float32)
        mesh_distance = np.abs(trimesh.proximity.signed_distance(mesh, points))
        signed_distance = np.zeros(points.shape[0],dtype=np.float32)
        # Compute the distance between each point and the boundary vertices
        for i, point in enumerate(points):
            boundary_distance[i] = np.min(np.linalg.norm(boundary_vertices - point,axis=1))
            if boundary_distance[i] > eps and mesh_distance[i] > eps:  # the point is outside the mesh
                signed_distance[i] = boundary_distance[i]
            elif boundary_distance[i] < eps:  # the point is on the boundary
                signed_distance[i] = 0
            else:  # the point is inside the mesh
                signed_distance[i] = -boundary_distance[i]
        return signed_distance

    def boundary_edges(self, mesh: trimesh.Trimesh):
        """
        Calculate edges not shared by faces in a mesh. If the mesh is watertight, the result should be empty.
        For non-watertight meshes, the result should be the boundary edges.
        Args:
            mesh: trimesh.Trimesh object
        Returns:
            np.ndarray of shape (N, 2) with dtype=np.int32, representing the non-shared edges
        """
        #edges_unique = mesh.edges_unique
        edges_sorted = mesh.edges_sorted
        # Find the edges that are duplicated
        edges, counts = np.unique(edges_sorted, axis=0, return_counts=True)
        non_shared_edges = edges[counts == 1]
        return non_shared_edges
    
    def boundary_vertices(self, mesh: trimesh.Trimesh):
        """
        Calculate vertices not shared by faces in a mesh. If the mesh is watertight, the result should be empty.
        For non-watertight meshes, the result should be the boundary vertices.
        Args:
            mesh: trimesh.Trimesh object
        Returns:
            np.ndarray of shape (N, 3) with dtype=np.float32, representing the non-shared vertices
        """
        boundary_edges = self.boundary_edges(mesh)
        vertices = np.array(mesh.vertices,dtype=np.float32)
        boundary_vertices = vertices[np.unique(boundary_edges.flatten())]
        return boundary_vertices

    
    def __len__(self):

        return self.data_length # Number of obj files in the dataset

    def __getitem__(self, idx):
        mesh = self.mesh[idx]
        #triangle_center = mesh.triangles_center
        vertices = mesh.vertices 
        #boundary_vertices = self.boundary_vertices(mesh)
        # faces = mesh.faces
        #mean = np.mean(vertices, axis=0)
        #print(f' mean : {mean}')
        # Append noise to vertices
        #noise = self.add_noise(boundary_vertices)
        #vertices = torch.tensor(vertices, dtype=torch.float32)
        #append_vertices = torch.cat((torch.tensor(vertices, dtype=torch.float32), torch.tensor(noised_vertices, dtype=torch.float32)), dim=0)
        # Get signed distance field from mesh
        #sdist = self.signed_distance(mesh, vertices)
        #nsdist = self.signed_distance(mesh, noise)
        return torch.tensor(vertices ,dtype=torch.float32) 

    def add_noise(self, vertices):
        noise = np.random.normal(0, 0.1, vertices.shape)
        return vertices + noise
    
class CMU_simulation(Dataset):
    def __init__(self, datapath:str='/home/cxh/mnt/cxh/Documents/dataset/CMU_mini_dataset'):
        super().__init__()
        self.datapath = datapath
        self.sequence_length = 130 # Number of obj in each sequence
        self.initial_mesh  = trimesh.load_mesh('assets/template_align.obj')
        self.template_faces = np.array(self.initial_mesh.faces, dtype=np.int64)
        self.template_vertices = np.array(self.initial_mesh.vertices, dtype=np.float32)
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
        self.dataset_length = len(self.npz_files)
        #print(f' dataset_length: {self.dataset_length}')
    def __len__(self):
        return self.dataset_length
    
    def __getitem__(self, index):
        # Load the sequence
        npz_file = self.npz_files[index]
        data = np.load(npz_file)
        gender = 0
        if data['gender'] == 'female':
            gender = 0 # female
        else:
            gender = 1 # male
        betas = data['betas']
        poses = data['poses']
        vertex_seq = data['vertex_seq'] - self.template_vertices
        #print(f' .............................')
        #print(f' vertex seq : {vertex_seq}')
        #vertex_seq = vertex_seq / 40.0
        #print(f' vertex_seq : {vertex_seq.shape}')
        # reshape beta from (,16) to (sequence_length, 16)
        betas = np.repeat(betas[np.newaxis, :], self.sequence_length, axis=0) 
        # concat betas with poses
        poses = np.concatenate((betas, poses), axis=1)
        # print
        #print('gender : {0}'.format(gender))
        #print('beta : {0}'.format(beta.shape))
        #print('poses : {0}'.format(poses.shape))
        #print('vertex_seq : {0}'.format(vertex_seq.shape))
        # return the sequence
        return gender, torch.tensor(poses,dtype=torch.float32), torch.tensor(vertex_seq, dtype=torch.float32)

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