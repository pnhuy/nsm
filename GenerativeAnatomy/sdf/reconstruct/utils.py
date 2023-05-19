import numpy as np
from scipy.spatial import cKDTree as KDTree


def compute_chamfer(
    pts1, 
    pts2, 
    num_samples=None,
    power=1,
):
    if num_samples is not None:
        sample1 = np.random.choice(
            pts1.shape[0],
            size=num_samples,
            replace=True if pts1.shape[0] < num_samples else False 
        )
        pts1 = pts1[sample1,:]
        
        sample2 = np.random.choice(
            pts2.shape[0],
            size=num_samples,
            replace=True if pts2.shape[0] < num_samples else False 
        )
        pts2 = pts2[sample2,:]
        
        
    kd1 = KDTree(pts1)
    kd2 = KDTree(pts2)
    
    d1, _ = kd1.query(pts2)
    d2, _ = kd2.query(pts1)
    
    return np.mean(d1**power) + np.mean(d2**power)

# Update LR
def adjust_learning_rate(
    initial_lr, optimizer, iteration, decreased_by, adjust_lr_every
):
    lr = initial_lr * ((1 / decreased_by) ** (iteration // adjust_lr_every))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr