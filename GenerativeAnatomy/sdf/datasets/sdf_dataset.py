import os
import pymskt as mskt
import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk
import torch
import hashlib
from datetime import datetime

today_date = datetime.now().strftime("%b_%d_%Y")

def get_pts_rel_surface(pts, mean=[0,0,0], sigma=1, n_pts=200000, function='normal'):
    """
    sigma is actually variance...
    """
    if function == 'normal':
        rand_gen = np.random.default_rng().multivariate_normal
    elif function =='laplacian':
        rand_gen = np.random.default_rng().laplace
    
    repeats = n_pts // pts.shape[0]
    n_extra_pts = n_pts % pts.shape[0]
    
    base_pts = np.tile(pts, [repeats, 1])
    if n_extra_pts > 0:
        base_pts = np.concatenate((base_pts, pts[:n_extra_pts, :]))
    
    if (function == 'normal') & (sigma is not None):
        cov = np.identity(len(mean)) * sigma
        rand_pts = rand_gen(mean, cov, n_pts)
    elif function == 'laplacian':
        rand_pts = np.zeros((n_pts, len(mean)))
        sigma = np.sqrt(sigma / 2)
        for axis in range(len(mean)):
            rand_pts[:, axis] = rand_gen(mean[axis], sigma, n_pts)
    
    return base_pts + rand_pts

def get_rand_uniform_pts(pts, n_pts, p_bigger_object=0.1):
    rand_gen = np.random.uniform

    pts = np.zeros((n_pts, pts.shape[1]))
    mins = np.min(pts, axis=0)
    maxs = np.max(pts, axis=0)    

    for axis in range(pts.shape[1]):
        axis_min = mins[axis]
        axis_max = maxs[axis]
        range_ = axis_max - axis_min
        axis_min -= range_ * p_bigger_object
        axis_max += range_ * p_bigger_object
        
        pts[:,axis]=rand_gen(axis_min, axis_max, n_pts)
    return pts

def norm(x, highdim=False, torch=False):
    """
    Computes norm of an array of vectors. Given (shape,d), returns (shape) after norm along last dimension
    """
    if torch is True:
        norm_ = torch.norm(x, dim=len(x.shape) - 1)
    else:
        norm_ = np.linalg.norm(x, axis=len(x.shape)-1)
    return norm_

def get_sdfs(pts, mesh):
    implicit_distance = vtk.vtkImplicitPolyDataDistance()
    implicit_distance.SetInput(mesh)
    sdfs = np.zeros(pts.shape[0])
    
    for pt_idx in range(pts.shape[0]):
        sdfs[pt_idx] = implicit_distance.EvaluateFunction(pts[pt_idx,:])
    return sdfs

def get_pts_center_and_scale(
    pts,
    center=True,
    scale=True,
    scale_method='max_rad',
    return_pts=False
):
    center = np.mean(pts, axis=0)
    pts -= center

    if scale_method == 'max_rad':
        scale = np.max(norm(pts), axis=-1)
        pts /= scale
    else:
        raise Exception(f'Scale Method ** {scale_method} ** Not Implemented')
    
    if return_pts is True:
        return center, scale, pts
    else:
        return center, scale


# def read_mesh_get_scaled_pts(
#     path,
#     center_pts=True,
#     axis_align=False,
#     norm_pts=False,
#     scale_method='max_rad',
#     return_orig_mesh=False
# ):
#     mesh = mskt.mesh.io.read_vtk(path)
#     pts = mskt.mesh.get_mesh_physical_point_coords(mesh)
    
#     center, scale = get_pts_center_and_scale(pts, scale_method=scale_method)

#     if center_pts is True:
#         pts -= center
#     if axis_align is True:
#         #TODO: If add axis align, must include in get_pts_center_and_scale
#         raise Exception('Axis Align Not implemented! Use PCA To Align pts to axes')
#         PCs, Vs = mskt.statistics.pca.pca_svd(pts.T)
#         pts = pts @ PCs
#         # This will turn the shape based on the PCA result
#         # If using a cropped femur or other bone, this might produce weird results.
#     if norm_pts is True:
#         pts /= scale
    
#     # mesh.GetPoints().SetData(numpy_to_vtk(pts))
#     if return_orig_mesh is True:
#         return pts, mesh
#     else:
#         return pts



def read_mesh_get_sampled_pts(
    path, 
    mean=[0,0,0], 
    sigma=1, 
    n_pts=200000, 
    rand_function='normal', 
    center_pts=True,
    #TODO: Add axis align back in - see code commented above for example
    # axis_align=False,
    norm_pts=False,
    scale_method='max_rad',
    get_random=True,
    return_orig_mesh=False,
    return_new_mesh=False,
    return_orig_pts=False,
    return_scale=False,
    return_center=False,
):
    orig_mesh = mskt.mesh.io.read_vtk(path)
    orig_pts = mskt.mesh.get_mesh_physical_point_coords(orig_mesh)
    
    if (center_pts is True) or (norm_pts is True):
        center, scale, new_pts = get_pts_center_and_scale(
            np.copy(orig_pts),
            center=center_pts,
            scale=norm_pts,
            scale_method=scale_method,
            return_pts=True)
    else:
        new_pts = np.copy(orig_pts)
        scale = 1
        center = np.zeros(3)
    
    new_mesh = mskt.mesh.vtk_deep_copy(orig_mesh)
    new_mesh.GetPoints().SetData(numpy_to_vtk(new_pts))

    results = {}

    if get_random is True:
        if sigma is not None:
            rand_pts = get_pts_rel_surface(new_pts, mean=mean, sigma=sigma, n_pts=n_pts, function=rand_function)
        else:
            rand_pts = get_rand_uniform_pts(new_pts, n_pts)

        rand_sdf = get_sdfs(rand_pts, new_mesh)

        results['pts'] = rand_pts
        results['sdf'] = rand_sdf
    else:
        results['pts'] = new_pts
        results['sdf'] = np.zeros(new_pts.shape[0])
    
    if return_orig_mesh is True:
        results['orig_mesh'] = orig_mesh
    if return_orig_pts is True:
        results['orig_pts'] = orig_pts
    if return_new_mesh is True:
        results['new_mesh'] = new_mesh
    if return_scale is True:
        results['scale'] = scale
    if return_center is True:
        results['center'] = center
    
    return results

class SDFSamples(torch.utils.data.Dataset):
    def __init__(
        self,
        list_mesh_paths,
        subsample=None,
        print_filename=False,
        n_pts=500000,
        p_near_surface=0.4,
        p_further_from_surface=0.4,
        sigma_near=0.01,
        sigma_far=0.1,
        rand_function='normal', 
        center_pts=True,
        axis_align=False,
        norm_pts=False,
        scale_method='max_rad',
        loc_save=os.environ['LOC_SDF_CACHE'],
        include_seed_in_hash=True,
        cache=True,
        random_seed=None,
    ):
        self.list_mesh_paths = list_mesh_paths
        self.subsample = subsample
        self.n_pts = n_pts
        self.p_near_surface = p_near_surface
        self.p_further_from_surface = p_further_from_surface
        self.sigma_near = sigma_near
        self.sigma_far = sigma_far
        self.rand_function = rand_function
        self.center_pts = center_pts
        self.axis_align = axis_align
        self.norm_pts = norm_pts
        self.scale_method = scale_method
        self.loc_save = loc_save
        self.include_seed_in_hash = include_seed_in_hash
        self.random_seed = random_seed

        if cache is True:
            cache_folder = os.path.join(self.loc_save, today_date)
            os.makedirs(cache_folder, exist_ok=True)

        n_p_near_surface = int(n_pts * p_near_surface)
        n_p_further_from_surface = int(n_pts * p_further_from_surface)
        n_p_random = n_pts - n_p_near_surface - n_p_further_from_surface

        pt_sample_combos = [
            [n_p_near_surface, sigma_near],
            [n_p_further_from_surface, sigma_far],
            [n_p_random, None]
        ]       

        self.data = []
        for loc_mesh in list_mesh_paths:
            if print_filename is True:
                print(loc_mesh)

            # Create hash and filename 
            file_hash = self.create_hash(loc_mesh)
            cached_file = self.find_hash(filename=f'{file_hash}.npz')
            #preallocate torch array
            pts_array = torch.zeros((n_pts, 4))

            if len(cached_file) > 0:
                # if hashed file exists, load it. 
                data = np.load(cached_file[0])
                pts_array[:, :3] = torch.from_numpy(data['pts']).float()
                pts_array[:, 3] = torch.from_numpy(data['sdfs']).float()
            else:
                # otherwise, load the mesh and create SDF samples. 
                print('Creating SDF Samples')
                pts_idx = 0
                for n_pts_, sigma_ in pt_sample_combos:
                    result_ = read_mesh_get_sampled_pts(
                        loc_mesh, 
                        mean=[0,0,0], 
                        sigma=sigma_, 
                        n_pts=n_pts_, 
                        rand_function=rand_function, 
                        center_pts=center_pts,
                        # axis_align=axis_align,
                        norm_pts=norm_pts,
                        scale_method=scale_method,
                        get_random=True,
                        return_orig_mesh=False,
                        return_new_mesh=False,
                        return_orig_pts=False
                    )
                    pts_ = result_['pts'] 
                    sdfs_ = result_['sdf']

                    pts_array[pts_idx:pts_idx + n_pts_, :3] = torch.from_numpy(pts_).float()
                    pts_array[pts_idx:pts_idx + n_pts_, 3] = torch.from_numpy(sdfs_).float()
                    pts_idx += n_pts_
                if cache is True:
                    # if want to cache, and new... then save. 
                    filepath = os.path.join(cache_folder, f'{file_hash}.npz')
                    np.savez(filepath, pts=pts_array[:, :3], sdfs=pts_array[:, 3])

            self.data.append(pts_array)
        
    def find_hash(self, filename='hashed_filename.npz'):
        files = []
        for p, d, f in os.walk(self.loc_save):
            for filename_ in f:
                if filename_ == filename:
                    files.append(os.path.join(p, filename_))
                    print('File found in cache:', files[-1])
                    return files
                    
        return files

    def create_hash(self, loc_mesh):
        list_params = [
            loc_mesh, 
            self.n_pts,
            self.p_near_surface, self.sigma_near,
            self.p_further_from_surface, self.sigma_far,
            self.center_pts,
            self.axis_align,
            self.norm_pts,
            self.scale_method,
            self.rand_function,
        ]
        if (self.include_seed_in_hash is True) & (self.random_seed is not None):
            list_params.append(self.random_seed) # random seed state
        print('List Params', list_params)
        list_params = [str(x) for x in list_params]
        file_params_string = '_'.join(list_params)
        hash_str = hashlib.md5(file_params_string.encode()).hexdigest()
        return hash_str
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_ = self.data[idx]
        if self.subsample is not None:
            perm = torch.randperm(data_.size(0))
            idx_ = perm[: self.subsample]
            data_ = data_[idx_]
        
        return data_, idx