import numpy as np
# load dataset with torch
import torch
from torch.utils.data import DataLoader, Dataset
import trimesh

class Cloth_in_Wind(Dataset):
    def __init__(self, datapath:str='/home/cxh/mnt/cxh/Documents/dataset/cloth_in_wind'):
        self.datapath = datapath

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
        boundary_vertices = vertices[boundary_edges.flatten()]
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

    
    def __len__(self):

        return 600 # Number of obj files in the dataset

    def __getitem__(self, idx):
        # Load obj file
        mesh = trimesh.load_mesh(self.datapath + '/cloth_seq%04d.obj' % (idx+1))
        #triangle_center = mesh.triangles_center
        vertices = mesh.vertices - np.array([0,1,1])
        # faces = mesh.faces
        #mean = np.mean(vertices, axis=0)
        #print(f' mean : {mean}')
        # Append noise to vertices
        noised_vertices = self.add_noise(vertices)
        #vertices = torch.tensor(vertices, dtype=torch.float32)
        append_vertices = np.append(vertices, noised_vertices, axis=0)
        #append_vertices = torch.cat((torch.tensor(vertices, dtype=torch.float32), torch.tensor(noised_vertices, dtype=torch.float32)), dim=0)
        # Get signed distance field from mesh
        sdist = self.signed_distance(mesh, append_vertices)

        return torch.tensor(append_vertices,dtype=torch.float32), torch.tensor(sdist,dtype=torch.float32)
    
    def add_noise(self, vertices):
        noise = np.random.normal(0, 0.1, vertices.shape)
        return vertices + noise
    
    
if __name__=='__main__':
    dataset = Cloth_in_Wind()
    dataloader = DataLoader(dataset, batch_size=7, shuffle=True)
    for i, data in enumerate(dataloader):
        vertices = data[0]
        sdist = data[1]
        print(f'Batch {i}: vertices: {vertices.shape}, sdist: {sdist.shape}')
