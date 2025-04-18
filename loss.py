import torch
from torch import nn

# Signed distance loss
def loss_signed_distance(pred_distance, gt_distance, clamp=0.1):
    l1_loss = nn.L1Loss()
    #print(f' pred distance : {pred_distance}')
    #pred_distance = torch.clamp(pred_distance, -clamp, clamp)
    #gt_distance = torch.clamp(gt_distance, -clamp, clamp)
    loss = l1_loss(pred_distance, gt_distance)
    return loss

def loss_l2(pred, gt):
    l2_loss = nn.MSELoss()
    loss = l2_loss(pred, gt)
    return loss

# batched laplacian loss
def loss_laplacian(laplacian_matrix:torch.Tensor, pred_vertices:torch.Tensor, gt_vertices:torch.Tensor, ratio=0.1):
    laplacian_loss = nn.MSELoss()
    # assert pred_vertices shape == gt_vertices shape
    assert pred_vertices.shape == gt_vertices.shape # shape : batch_size, seq_length, num_vertices*3
    # reshape pred_vertices and gt_vertices to batch_size*seq_length, num_vertices*3
    pred_vertices = pred_vertices.view(-1, pred_vertices.shape[2])
    gt_vertices = gt_vertices.view(-1, gt_vertices.shape[2])
    #print(f' gt_vertices : {gt_vertices}')
    #print(f' pred_vertices : {pred_vertices}')
    #print(f' laplacian_matrix : {laplacian_matrix}')
    laplacian_pred = torch.matmul(pred_vertices, laplacian_matrix)
    laplacian_gt = torch.matmul(gt_vertices, laplacian_matrix)
    #print(f' laplacian pred : {laplacian_pred}')
    #print(f' laplacian gt : {laplacian_gt}')
    laplacian_loss = laplacian_loss(laplacian_pred,laplacian_gt)
    #print(f' laplsaian loss : {laplacian_loss}')
    vertex_loss = loss_l2(pred_vertices, gt_vertices)
    return ratio * laplacian_loss + (1-ratio) * vertex_loss
