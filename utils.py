import torch
import trimesh 
import numpy as np

def mesh_normals(vertices, faces):
    # 获取顶点和面的数量
    num_vertices = vertices.shape[0]
    num_faces = faces.shape[0]

    # 提取每个面的三个顶点
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    # 计算每个面的两个边向量
    e1 = v1 - v0
    e2 = v2 - v0

    # 计算每个面的法线
    face_normals = torch.linalg.cross(e1, e2)
    face_normals = torch.nn.functional.normalize(face_normals, dim=1)

    # 初始化顶点法线
    vertex_normals = torch.zeros((num_vertices, 3), dtype=torch.float32)

    # 累加每个面的法线到对应的顶点
    for i in range(num_faces):
        vertex_normals[faces[i, 0]] += face_normals[i]
        vertex_normals[faces[i, 1]] += face_normals[i]
        vertex_normals[faces[i, 2]] += face_normals[i]

    # 归一化顶点法线
    vertex_normals = torch.nn.functional.normalize(vertex_normals, dim=1)

    return vertex_normals

# define normalized laplacian matrix
def laplacian_matrix(faces, normalize=True):
    """
    Compute the normalized Laplacian matrix of a mesh.
    Args:
        faces: np.ndarray of shape (N, 3) with dtype=np.int32
        normalize: bool, whether to normalize the Laplacian matrix
    Returns:
        torch.Tensor of shape (num_vertices, num_vertices) with dtype=torch.float32
    """

    num_faces = faces.shape[0]
    num_vertices = faces.max() + 1
    # adjacency matrix
    adj_matrix = torch.zeros((num_vertices, num_vertices), dtype=torch.float32)
    for i in range(num_faces):
        v0,v1,v2 = faces[i]
        adj_matrix[v0,v1] = 1.0
        adj_matrix[v1,v0] = 1.0
        adj_matrix[v0,v2] = 1.0
        adj_matrix[v2,v0] = 1.0
        adj_matrix[v1,v2] = 1.0
        adj_matrix[v2,v1] = 1.0
    # degree matrix
    degrees = adj_matrix.sum(dim=1)
    if normalize:
        degrees = torch.sqrt(1.0 / (degrees + 1e-5))
        degree_matrix = torch.diag(degrees)
        laplacian_mat = torch.eye(num_vertices) - degree_matrix @ adj_matrix @ degree_matrix
    else:
        laplacian_mat = torch.diag(degrees) - adj_matrix
    return laplacian_mat

def boundary_edges(mesh: trimesh.Trimesh):
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
    
def boundary_vertices(mesh: trimesh.Trimesh):
    """
    Calculate vertices not shared by faces in a mesh. If the mesh is watertight, the result should be empty.
    For non-watertight meshes, the result should be the boundary vertices.
    Args:
        mesh: trimesh.Trimesh object
    Returns:
        np.ndarray of shape (N, 3) with dtype=np.float32, representing the non-shared vertices
    """
    boundary_edges = boundary_edges(mesh)
    vertices = np.array(mesh.vertices,dtype=np.float32)
    boundary_vertices = vertices[np.unique(boundary_edges.flatten())]
    return boundary_vertices

def signed_distance(mesh:trimesh.Trimesh, points:np.ndarray,eps:float=1e-5):
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
    boundary_edges = boundary_edges(mesh)
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
            signed_distance[i] = -mesh_distance[i]
    return signed_distance
