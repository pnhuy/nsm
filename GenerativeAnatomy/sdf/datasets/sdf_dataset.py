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
    if issubclass(type(mesh), mskt.mesh.Mesh):
        mesh = mesh.mesh
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
    return_pts=False,
    pts_center=None
):

    # If pts_cener is not None, then use that to center the points
    # and use all of the points in pts to scale. This is used for
    # based on the bone only, and then scaling based on bone + cartilage
    if pts_center is None:
        center = np.mean(pts, axis=0)
    else:
        center = np.mean(pts_center, axis=0)
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
    register_to_mean_first=False,
    mean_mesh=None,
    **kwargs,
):
    results = {}

    orig_mesh = mskt.mesh.Mesh(path)
    orig_pts = orig_mesh.point_coords

    if (register_to_mean_first is True):
        # Rigid + scaling alginment of the original mesh to the mean mesh
        # of the model. This allows all downstream scaling to occur as expected
        # it also aligns the new bone with the mean/expected bone of the shape model
        # to maximize fidelity of the reconstruction.

        if mean_mesh is None:
            raise Exception('Must provide mean mesh to register to')
        icp_transform = orig_mesh.rigidly_register(
            other_mesh=mean_mesh,
            as_source=True,
            apply_transform_to_mesh=True,
            return_transformed_mesh=False,
            max_n_iter=100,
            n_landmarks=1000,
            reg_mode='similarity',
            return_transform=True,
        )
        results['icp_transform'] = icp_transform
    else:
        results['icp_transform'] = None
    
    if (center_pts is True) or (norm_pts is True):
        center, scale, new_pts = get_pts_center_and_scale(
            np.copy(orig_mesh.point_coords),
            center=center_pts,
            scale=norm_pts,
            scale_method=scale_method,
            return_pts=True)
    
    else:
        new_pts = np.copy(orig_mesh.point_coords)
        scale = 1
        center = np.zeros(3)
    
    new_mesh = orig_mesh.copy()
    new_mesh.point_coords = new_pts

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
    
    # Theres no reason we shouldnt include these... they are always there and take up effectively no space. 
    results['scale'] = scale
    results['center'] = center

    if return_orig_mesh is True:
        results['orig_mesh'] = orig_mesh
    if return_orig_pts is True:
        results['orig_pts'] = orig_pts
    if return_new_mesh is True:
        results['new_mesh'] = new_mesh
    
    if ('return_scale' in kwargs):
        print('return_scale is deprecated and not used in this function - always returns scale')
    if ('return_center' in kwargs):
        print('return_center is deprecated and not used in this function - always returns center')
    
    return results



def read_meshes_get_sampled_pts(
    paths, 
    mean=[0,0,0], 
    sigma=1, 
    n_pts=200000, 
    rand_function='normal', 
    center_pts=True,
    #TODO: Add axis align back in - see code commented above for example
    # axis_align=False,
    scale_all_meshes=False,
    center_all_meshes=False,
    mesh_to_scale=0,
    norm_pts=False,
    scale_method='max_rad',
    get_random=True,
    return_orig_mesh=False,
    return_new_mesh=False,
    return_orig_pts=False,
    return_scale=False,
    return_center=False,
    register_to_mean_first=False,
    mean_mesh=None,
):
    """
    Function to read in and sample points from multiple meshes.
    """

    results = {}

    orig_meshes = []
    orig_pts = []
    for path in paths:
        mesh = mskt.mesh.Mesh(path)
        orig_meshes.append(mesh)
        orig_pts.append(mesh.point_coords)

    if (register_to_mean_first is True):
        # Rigid + scaling alginment of the original mesh to the mean mesh
        # of the model. This allows all downstream scaling to occur as expected
        # it also aligns the new bone with the mean/expected bone of the shape model
        # to maximize fidelity of the reconstruction.

        if mean_mesh is None:
            raise Exception('Must provide mean mesh to register to')
        icp_transform = orig_meshes[mesh_to_scale].rigidly_register(
            other_mesh=mean_mesh,
            as_source=True,
            apply_transform_to_mesh=True,
            return_transformed_mesh=False,
            max_n_iter=100,
            n_landmarks=1000,
            reg_mode='similarity',
            return_transform=True,
        )
        results['icp_transform'] = icp_transform
        
        # apply transform to all other meshes
        mesh_indices = list(range(len(orig_meshes)))
        mesh_indices.remove(mesh_to_scale)
        for idx in mesh_indices:
            orig_meshes[idx].apply_transform(icp_transform)
            orig_pts[idx] = orig_meshes[idx].point_coords

    else:
        results['icp_transform'] = None

    if (center_pts is True) or (norm_pts is True):
        if scale_all_meshes is True:
            pts_ = np.concatenate(orig_pts, axis=0)
            if center_all_meshes is True:
                pts_center = None
            else:
                pts_center = orig_pts[mesh_to_scale]
        else:
            pts_ = orig_pts[mesh_to_scale]

        center, scale, _ = get_pts_center_and_scale(
            np.copy(pts_),
            center=center_pts,
            scale=norm_pts,
            scale_method=scale_method,
            return_pts=False,
            pts_center=pts_center,
        )

        new_pts = []
        for i in range(len(orig_pts)):
            new_pts.append((orig_pts[i] - center)/scale)
    else:
        new_pts = [np.copy(orig_pts_) for orig_pts_ in orig_pts]
        scale = 1
        center = np.zeros(3)
    
    new_meshes = []
    for mesh_idx, orig_mesh in enumerate(orig_meshes):
        new_mesh_ = orig_mesh.copy()
        new_mesh_.point_coords = new_pts[mesh_idx]
        new_meshes.append(new_mesh_)

    if get_random is True:
        rand_pts = []
        rand_sdf = []

        for new_pts_ in new_pts:
            if sigma is not None:
                rand_pts_ = get_pts_rel_surface(new_pts_, mean=mean, sigma=sigma, n_pts=n_pts, function=rand_function)
            else:
                rand_pts_ = get_rand_uniform_pts(new_pts_, n_pts)
            rand_pts.append(rand_pts_)
        
        rand_pts = np.concatenate(rand_pts, axis=0)

        for new_mesh in new_meshes:
            rand_sdf.append(get_sdfs(rand_pts, new_mesh))

        results['pts'] = rand_pts
        results['sdf'] = rand_sdf
    else:
        sdfs = []
        # Need to set SDFs for the same mesh to be 0
        # but need to actually calculate the SDFs for the other
        # meshes.
        for pts_idx, new_pts_ in enumerate(new_pts):
            sdfs_ = []
            for mesh_idx, new_mesh in enumerate(new_meshes):
                if pts_idx == mesh_idx:
                    # same mesh, set SDFs to 0
                    sdfs_.append(np.zeros(new_pts_.shape[0]))
                else:
                    # different mesh, calculate SDFs
                    sdfs_.append(get_sdfs(new_pts_, new_mesh))
            sdfs.append(np.concatenate(sdfs_, axis=0))       

        new_pts = np.concatenate(new_pts, axis=0)

        results['pts'] = new_pts
        results['sdf'] = sdfs
    
    results['scale'] = scale
    results['center'] = center

    if return_orig_mesh is True:
        results['orig_mesh'] = orig_meshes
    if return_orig_pts is True:
        results['orig_pts'] = orig_pts
    if return_new_mesh is True:
        results['new_mesh'] = new_meshes
    
    if ('return_scale' in kwargs):
        print('return_scale is deprecated and not used in this function - always returns scale')
    if ('return_center' in kwargs):
        print('return_center is deprecated and not used in this function - always returns center')
    
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
    
class MultiSurfaceSDFSamples(SDFSamples):
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
        scale_all_meshes=True,
        mesh_to_scale=0,
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
        self.scale_all_meshes = scale_all_meshes
        self.mesh_to_scale = mesh_to_scale
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
        for loc_meshes in list_mesh_paths:
            if type(loc_meshes) not in (tuple, list):
                loc_meshes = [loc_meshes]
            
            #preallocate torch array
            pts_array = torch.zeros((n_pts * len(loc_meshes), 3 + len(loc_mesh)))

            if print_filename is True:
                print(loc_meshes)
            
            # Create hash and filename 
            file_hash = self.create_hash('_'.join(loc_meshes))
            cached_file = self.find_hash(filename=f'{file_hash}.npz')
            
            if len(cached_file) > 0:
                # if hashed file exists, load it. 
                data = np.load(cached_file[0])
                pts_array[:, :3] = torch.from_numpy(data['pts']).float()
                pts_array[:, 3:] = torch.from_numpy(data['sdfs']).float()
            else:
                # otherwise, load the mesh and create SDF samples. 
                print('Creating SDF Samples')
                pts_idx = 0
                for n_pts_, sigma_ in pt_sample_combos:
                    result_ = read_meshes_get_sampled_pts(
                        loc_meshes, 
                        mean=[0,0,0], 
                        sigma=sigma_, 
                        n_pts=n_pts_, 
                        rand_function=rand_function, 
                        center_pts=center_pts,
                        # axis_align=axis_align,
                        scale_all_meshes=scale_all_meshes,
                        mesh_to_scale=mesh_to_scale,
                        norm_pts=norm_pts,
                        scale_method=scale_method,
                        get_random=True,
                        return_orig_mesh=False,
                        return_new_mesh=False,
                        return_orig_pts=False
                    )
                    pts_ = result_['pts'] 
                    sdfs_ = result_['sdf']
                    
                    for mesh_idx in range(len(loc_meshes)):
                        pts_array[pts_idx:pts_idx + (n_pts_ * 2), :3] = torch.from_numpy(pts_).float()
                        pts_array[pts_idx:pts_idx + (n_pts_ * 2), 3+mesh_idx] = torch.from_numpy(sdfs_[mesh_idx]).float()
                        pts_idx += (n_pts_ * 2)
                if cache is True:
                    # if want to cache, and new... then save. 
                    filepath = os.path.join(cache_folder, f'{file_hash}.npz')
                    np.savez(filepath, pts=pts_array[:, :3], sdfs=pts_array[:, 3:])

                self.data.append(pts_array)

    def __getitem__(self, idx):
        data_ = self.data[idx]
        if self.subsample is not None:
            perm = torch.randperm(data_.size(0))
            idx_ = perm[: self.subsample]
            data_ = data_[idx_]
        
        return data_, idx