import os
import pymskt as mskt
import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk
import torch
import hashlib
from datetime import datetime
import warnings

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
    
    if (function == 'normal') and (sigma is not None):
        cov = np.identity(len(mean)) * sigma
        rand_pts = rand_gen(mean, cov, n_pts)
    elif function == 'laplacian':
        rand_pts = np.zeros((n_pts, len(mean)))
        sigma = np.sqrt(sigma / 2)
        for axis in range(len(mean)):
            rand_pts[:, axis] = rand_gen(mean[axis], sigma, n_pts)
    
    return base_pts + rand_pts

def get_rand_uniform_pts(n_pts, mins=(-1, -1, -1), maxs=(1, 1, 1), p_bigger_object=0.):
    rand_gen = np.random.uniform

    pts = np.zeros((n_pts, len(mins)))
    # mins = np.min(pts, axis=0)
    # maxs = np.max(pts, axis=0)    

    # for axis in range(pts.shape[1]):
    #     axis_min = mins[axis]
    #     axis_max = maxs[axis]
    #     range_ = axis_max - axis_min
    #     axis_min -= range_ * p_bigger_object
    #     axis_max += range_ * p_bigger_object
        
    #     pts[:,axis]=rand_gen(axis_min, axis_max, n_pts)
    for axis in range(pts.shape[1]):
        axis_min = mins[axis]
        axis_max = maxs[axis]
        if p_bigger_object > 0:
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
    """
    Given a set of points, returns the center and scale of the points.
    """

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
        raise NotImplementedError(f'Scale Method ** {scale_method} ** Not Implemented')
    
    if return_pts is True:
        return center, scale, pts
    
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
    return_point_cloud=False,
    fix_mesh=True,
    include_surf_in_pts=False,
    **kwargs,
):
    results = {}

    orig_mesh = mskt.mesh.Mesh(path)
    if fix_mesh is True:
        n_pts_orig = orig_mesh.point_coords.shape[0]
        orig_mesh.fix_mesh()
        n_pts_fixed = orig_mesh.point_coords.shape[0]
        # Asserting that no more than 1% of the mesh points were removed
        print(f'Fixed mesh, {n_pts_orig} -> {n_pts_fixed} ({(n_pts_fixed - n_pts_orig) / n_pts_orig * 100:.2f}%)')
        # assert (n_pts_fixed - n_pts_orig) > -0.01 * n_pts_orig, f'Mesh was not fixed correctly, {n_pts_orig} -> {n_pts_fixed}'
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
            # rand_pts = get_rand_uniform_pts(new_pts, n_pts)
            rand_pts = get_rand_uniform_pts(n_pts)
        
        if include_surf_in_pts is True:
            rand_pts = np.concatenate([rand_pts, new_pts], axis=0)

        if norm_pts is True:
            rand_pts[rand_pts > 1] = 1
            rand_pts[rand_pts < -1] = -1

        rand_sdf = get_sdfs(rand_pts, new_mesh)

        results['xyz'] = rand_pts
        results['gt_sdf'] = rand_sdf
        results['pts_surface'] = [0] * rand_pts.shape[0]
    else:
        results['pts'] = new_pts
        results['sdf'] = np.zeros(new_pts.shape[0])
        results['pts_surface'] = [0] * new_pts.shape[0]
        
    
    if return_point_cloud is True:
        results['point_cloud'] = new_pts

    # Theres no reason we shouldnt include these... they are always there and take up effectively no space. 
    results['scale'] = scale
    results['center'] = center

    if return_orig_mesh is True:
        results['orig_mesh'] = [orig_mesh]
    if return_orig_pts is True:
        results['orig_pts'] = [orig_pts]
    if return_new_mesh is True:
        results['new_mesh'] = [new_mesh]
    
    if ('return_scale' in kwargs):
        print('return_scale is deprecated and not used in this function - always returns scale')
    if ('return_center' in kwargs):
        print('return_center is deprecated and not used in this function - always returns center')
    
    return results

def unpack_numpy_data(
    data_,
    point_cloud=False
):
    data = {}

    if 'pts' in data_:
        data['xyz'] = torch.from_numpy(data_['pts']).float()
    elif 'xyz' in data_:
        data['xyz'] = torch.from_numpy(data_['xyz']).float()
    else:
        raise ValueError('No pts or xyz in cached file')
    
    if 'sdfs' in data_:
        data['gt_sdf'] = torch.from_numpy(data_['sdfs']).float()
    elif 'gt_sdf' in data_:
        data['gt_sdf'] = torch.from_numpy(data_['gt_sdf']).float()
    elif 'sdf' in data_:
        data['gt_sdf'] = torch.from_numpy(data_['sdf']).float()
    else:
        raise ValueError('No sdfs or gt_sdf or sdf in cached file')
    
    if point_cloud is True:
        data['point_cloud'] = torch.from_numpy(data_['point_cloud']).float()

    return data

def read_meshes_get_sampled_pts(
    paths, 
    mean=[0,0,0], 
    sigma=[1, 1], 
    n_pts=[200000, 200000], 
    rand_function='normal', 
    center_pts=True,
    scale_all_meshes=True,
    center_all_meshes=False,
    mesh_to_scale=0,
    norm_pts=False,
    scale_method='max_rad',
    get_random=True,
    return_orig_mesh=False,
    return_new_mesh=False,
    return_orig_pts=False,
    register_to_mean_first=False,
    mean_mesh=None,
    verbose=False,
    fix_mesh=True,
    include_surf_in_pts=False,
    **kwargs,
):
    """
    Function to read in and sample points from multiple meshes.
    """
    list_deprecated = ['return_scale', 'return_center']
    for kwarg in kwargs:
        if kwarg in list_deprecated:
            print(f'{kwarg} is deprecated and not used in this function - always True')

    results = {}

    # Read all meshes and store in list
    orig_meshes = []
    orig_pts = []
    for path in paths:
        mesh = mskt.mesh.Mesh(path)
        # fixing meshes ensures they are not degenerate
        # degenerate meshes will cause issues fitting SDFs. 
        if fix_mesh is True:
            n_pts_orig = mesh.point_coords.shape[0]
            mesh.fix_mesh()
            n_pts_fixed = mesh.point_coords.shape[0]
            # Warning if more than 1% of the mesh points were removed
            warnings.warn(f'Fixed mesh, {n_pts_orig} -> {n_pts_fixed} ({(n_pts_fixed - n_pts_orig) / n_pts_orig * 100:.2f}%)', Warning)
            #TODO: Update to an assertion?
            # assert (n_pts_fixed - n_pts_orig) > -0.01 * n_pts_orig, f'Mesh was not fixed correctly, {n_pts_orig} -> {n_pts_fixed}'

        orig_meshes.append(mesh)
        orig_pts.append(mesh.point_coords)

    # Copy all meshes & points to new lists
    new_meshes = []
    new_pts = []
    for mesh_idx, orig_mesh in enumerate(orig_meshes):
        new_mesh_ = orig_mesh.copy()
        new_pts.append(new_mesh_.point_coords)
        new_meshes.append(new_mesh_)

    if (register_to_mean_first is True):
        # Rigid + scaling (similarity) alginment of the original mesh to the mean mesh
        # of the model. This allows all downstream scaling to occur as expected
        # it also aligns the new bone with the mean/expected bone of the shape model
        # to maximize fidelity of the reconstruction.

        if mean_mesh is None:
            raise ValueError('Must provide mean mesh to register to')
        icp_transform = orig_meshes[mesh_to_scale].rigidly_register(
            other_mesh=mean_mesh,
            as_source=True,
            apply_transform_to_mesh=False,
            return_transformed_mesh=False,
            max_n_iter=100,
            n_landmarks=1000,
            reg_mode='similarity',
            return_transform=True,
        )
        results['icp_transform'] = icp_transform

        # apply transform to all other meshes
        for idx, new_mesh in enumerate(new_meshes):
            new_mesh.apply_transform_to_mesh(icp_transform)
            new_pts[idx] = new_mesh.point_coords

    else:
        results['icp_transform'] = None

    if (center_pts is True) or (norm_pts is True):
        if scale_all_meshes is True:
            pts_ = np.concatenate(new_pts, axis=0)
            if center_all_meshes is True:
                # Set as None - becuasse scaling and centering on same data
                pts_center = None
            else:
                # set specific points to center becuase they are not the same
                # for centering as they are for caling (pts_)
                pts_center = new_pts[mesh_to_scale]
        else:
            pts_ = new_pts[mesh_to_scale]
            if center_all_meshes is True:
                # set specific points to center because scale/center are not on
                # the same data
                pts_center = np.concatenate(new_pts, axis=0)
            else:
                # Set as None - becuasse scaling and centering on same data
                pts_center = None

        center, scale = get_pts_center_and_scale(
            np.copy(pts_),
            center=center_pts,
            scale=norm_pts,
            scale_method=scale_method,
            return_pts=False,
            pts_center=pts_center,
        )

        for pts_idx, new_pts_ in enumerate(new_pts):
            new_pts[pts_idx] = (new_pts_ - center)/scale
    else:
        # Do nothing to the points because they are left the same.
        scale = 1
        center = np.zeros(3)
    
    # new_meshes = []
    for mesh_idx, new_mesh in enumerate(new_meshes):
        new_mesh.point_coords = new_pts[mesh_idx]

    if get_random is True:
        rand_pts = []
        rand_sdf = []
        pts_surface = []

        for new_pts_idx, new_pts_ in enumerate(new_pts):
            if n_pts[new_pts_idx]  > 0:
                if sigma[new_pts_idx] is not None:
                    rand_pts_ = get_pts_rel_surface(new_pts_, mean=mean, sigma=sigma[new_pts_idx], n_pts=n_pts[new_pts_idx], function=rand_function)
                else:
                    print('Getting random uniform points')
                    # rand_pts_ = get_rand_uniform_pts(new_pts_, n_pts[new_pts_idx])
                    rand_pts_ = get_rand_uniform_pts(n_pts[new_pts_idx])
                    print(rand_pts_)
                
                if include_surf_in_pts is True:
                    rand_pts_ = np.concatenate([rand_pts_, new_pts_], axis=0)

                rand_pts.append(rand_pts_)
                pts_surface.append([new_pts_idx] * rand_pts_.shape[0])
            else:
                rand_pts.append(np.zeros((0,3)))
        
        rand_pts = np.concatenate(rand_pts, axis=0)
        pts_surface = np.concatenate(pts_surface, axis=0)

        for new_mesh in new_meshes:
            rand_sdf.append(get_sdfs(rand_pts, new_mesh))

        results['pts'] = rand_pts
        results['sdf'] = rand_sdf
        results['pts_surface'] = pts_surface
    else:
        sdfs = []
        # Need to set SDFs for the same mesh to be 0
        # but need to actually calculate the SDFs for the other
        # meshes.
        for mesh_idx, new_mesh in enumerate(new_meshes):
            sdfs_ = []

            for pts_idx, new_pts_ in enumerate(new_pts):
                if verbose is True:
                    print('mesh_idx, new_mesh point_coords shape', mesh_idx, new_mesh.point_coords.shape)
                if pts_idx == mesh_idx:
                    if verbose is True:
                        print('adding zeros new_pts_ shape (zero)', new_pts_.shape)
                    # same mesh, set SDFs to 0
                    sdfs_.append(np.zeros(new_pts_.shape[0]))
                else:
                    # different mesh, calculate SDFs
                    _sdfs_ = get_sdfs(new_pts_, new_mesh)
                    if verbose is True:
                        print('caculating SDFs for new_pts_ ', _sdfs_.shape)
                    sdfs_.append(_sdfs_)

            sdfs.append(np.concatenate(sdfs_, axis=0))       

        pts_surface = []
        for pts_idx, new_pts_ in enumerate(new_pts):
            pts_surface.append([pts_idx] * new_pts_.shape[0])
        pts_surface = np.concatenate(pts_surface, axis=0)

        new_pts = np.concatenate(new_pts, axis=0)



        results['pts'] = new_pts
        results['sdf'] = sdfs
        results['pts_surface'] = pts_surface

    results['scale'] = scale
    results['center'] = center

    if return_orig_mesh is True:
        results['orig_mesh'] = orig_meshes
    if return_orig_pts is True:
        results['orig_pts'] = orig_pts
    if return_new_mesh is True:
        results['new_mesh'] = new_meshes

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
        save_cache=True,
        load_cache=True,
        random_seed=None,
        verbose=False,
        equal_pos_neg=True,
        fix_mesh=True,
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
        self.verbose = verbose
        self.equal_pos_neg = equal_pos_neg
        self.fix_mesh = fix_mesh
        self.load_cache = load_cache

        self.list_hash_params = [
            self.n_pts,
            self.p_near_surface, self.sigma_near,
            self.p_further_from_surface, self.sigma_far,
            self.center_pts,
            self.axis_align,
            self.norm_pts,
            self.scale_method,
            self.rand_function,
            self.fix_mesh,
        ]

        if save_cache is True:
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

            if (len(cached_file) > 0) and (load_cache is True):
                # if hashed file exists, load it. 
                data_ = np.load(cached_file[0])
                data = unpack_numpy_data(data_)
                
            else:
                # otherwise, load the mesh and create SDF samples. 
                print('Creating SDF Samples')
                data = {
                    'xyz': torch.zeros((n_pts, 3)),
                    'gt_sdf': torch.zeros((n_pts)),
                }
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
                        return_orig_pts=False,
                        fix_mesh=self.fix_mesh,
                    )
                
                    xyz_ = result_['pts'] if 'pts' in result_ else result_['xyz']
                    sdfs_ = result_['sdf'] if 'sdf' in result_ else result_['gt_sdf']

                    data['xyz'][pts_idx:pts_idx + n_pts_, :] = torch.from_numpy(xyz_).float()
                    data['gt_sdf'][pts_idx:pts_idx + n_pts_] = torch.from_numpy(sdfs_).float()
                    pts_idx += n_pts_
                if save_cache is True:
                    # if want to cache, and new... then save. 
                    filepath = os.path.join(cache_folder, f'{file_hash}.npz')
                    np.savez(filepath, pts=data['xyz'], sdfs=data['gt_sdf'])

            pos_idx, neg_idx, surf_idx = self.sdf_pos_neg_idx(data)
            data['pos_idx'] = pos_idx
            data['neg_idx'] = neg_idx
            data['surf_idx'] = surf_idx

            self.data.append(data)
    
    def sdf_pos_neg_idx(self, data):
        pos_idx = (data['gt_sdf'] > 0).nonzero(as_tuple=True)[0]
        neg_idx = (data['gt_sdf'] < 0).nonzero(as_tuple=True)[0]
        surf_idx = (data['gt_sdf'] == 0).nonzero(as_tuple=True)[0]

        return pos_idx, neg_idx, surf_idx
        
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
        list_hash_params = self.list_hash_params.copy()
        if isinstance(loc_mesh, str):
            list_hash_params.insert(0, loc_mesh)
        elif isinstance(loc_mesh, (list, tuple)):
            for path in loc_mesh:
                if self.verbose is True:
                    print(loc_mesh)
                list_hash_params.insert(0, path)
    
        if (self.include_seed_in_hash is True) and (self.random_seed is not None):
            list_hash_params.append(self.random_seed) # random seed state
        if self.verbose is True:
            print('List Params', list_hash_params)
        list_hash_params = [str(x) for x in list_hash_params]
        file_params_string = '_'.join(list_hash_params)
        hash_str = hashlib.md5(file_params_string.encode()).hexdigest()
        return hash_str
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_ = self.data[idx]
        if self.subsample is not None:
            if self.equal_pos_neg is True:
                samples_per_sign = int(self.subsample/2)
                
                idx_pos = data_['pos_idx'].repeat(data_['pos_idx'].size(0)//samples_per_sign + 1)
                perm_pos = torch.randperm(idx_pos.size(0))
                idx_pos = perm_pos[:samples_per_sign]

                idx_neg = data_['neg_idx'].repeat(data_['neg_idx'].size(0)//samples_per_sign + 1)
                perm_neg = torch.randperm(idx_neg.size(0))
                idx_neg = perm_neg[:samples_per_sign]

                idx_ = torch.cat((idx_pos, idx_neg), dim=0)

                if len(idx_) < self.subsample:
                    # if we don't have enough points, then just take random points
                    perm = torch.randperm(data_['xyz'].size(0))
                    _idx_ = perm[:self.subsample-len(idx_)]
                    idx_ = torch.cat([idx_, _idx_], dim=0)
            
            else:
                perm = torch.randperm(data_['xyz'].size(0))
                idx_ = perm[:self.subsample]

            data_ = {
                'xyz': data_['xyz'][idx_, :],
                'gt_sdf': data_['gt_sdf'][idx_],
            }
        
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
        center_all_meshes=False,
        mesh_to_scale=0,
        norm_pts=False,
        scale_method='max_rad',
        loc_save=os.environ['LOC_SDF_CACHE'],
        include_seed_in_hash=True,
        save_cache=True,
        load_cache=True,
        random_seed=None,
        reference_mesh=None,
        reference_object=0,
        verbose=False,
        equal_pos_neg=True,
        fix_mesh=True,
    ):
        
        self.list_mesh_paths = list_mesh_paths
        self.subsample = subsample
        self.n_pts = n_pts
        self.total_n_pts = sum(n_pts)
        self.p_near_surface = p_near_surface
        self.p_further_from_surface = p_further_from_surface
        self.sigma_near = sigma_near
        self.sigma_far = sigma_far
        self.rand_function = rand_function
        self.center_pts = center_pts
        self.axis_align = axis_align
        self.scale_all_meshes = scale_all_meshes
        self.center_all_meshes = center_all_meshes
        self.mesh_to_scale = mesh_to_scale
        self.norm_pts = norm_pts
        self.scale_method = scale_method
        self.loc_save = loc_save
        self.include_seed_in_hash = include_seed_in_hash
        self.random_seed = random_seed
        self.reference_mesh = reference_mesh
        self.reference_object = reference_object
        self.verbose = verbose
        self.equal_pos_neg = equal_pos_neg
        self.n_meshes = len(list_mesh_paths[0])
        self.fix_mesh = fix_mesh
        self.load_cache = load_cache

        if isinstance(self.list_mesh_paths[0], (list, tuple)):
            self.n_meshes = len(list_mesh_paths[0])
        elif isinstance(self.list_mesh_paths[0], (str, mskt.mesh.Mesh)):
            self.n_meshes = len(list_mesh_paths)
        

        if not isinstance(self.p_near_surface, (list, int)):
            self.p_near_surface = [self.p_near_surface] * self.n_meshes
        if not isinstance(self.p_further_from_surface, (list, int)):
            self.p_further_from_surface = [self.p_further_from_surface] * self.n_meshes
        if not isinstance(self.sigma_near, (list, int)):
            self.sigma_near = [self.sigma_near] * self.n_meshes
        if not isinstance(self.sigma_far, (list, int)):
            self.sigma_far = [self.sigma_far] * self.n_meshes
        if not isinstance(self.n_pts, (list, int)):
            self.n_pts = [self.n_pts] * self.n_meshes


        self.list_hash_params = [
            self.center_pts,
            self.axis_align,
            self.norm_pts,
            self.scale_method,
            self.rand_function,
            self.scale_all_meshes,
            self.center_all_meshes,
            self.reference_mesh,
            self.reference_object,
            False,
            self.fix_mesh,
        ]
        for n_pts_ in self.n_pts:
            self.list_hash_params.append(n_pts_)
        for p_near in self.p_near_surface:
            self.list_hash_params.append(p_near)
        for p_far in self.p_further_from_surface:
            self.list_hash_params.append(p_far)
        for sigma_near in self.sigma_near:
            self.list_hash_params.append(sigma_near)
        for sigma_far in self.sigma_far:
            self.list_hash_params.append(sigma_far)
        
        if save_cache is True:
            cache_folder = os.path.join(self.loc_save, today_date)
            os.makedirs(cache_folder, exist_ok=True)

        n_p_near_surface = [int(n_pts_ * p_near) for n_pts_, p_near in zip(n_pts, self.p_near_surface)]
        n_p_further_from_surface = [int(n_pts_ * p_far) for n_pts_, p_far in zip(n_pts, self.p_further_from_surface)]
        n_p_random = [n_pts_ - n_p_near - n_p_far for n_pts_, n_p_near, n_p_far in zip(n_pts, n_p_near_surface, n_p_further_from_surface)]

        self.pt_sample_combos = [
            [n_p_near_surface, self.sigma_near],
            [n_p_further_from_surface, self.sigma_far],
            [n_p_random, [None,] * self.n_meshes]
        ]

        if self.reference_mesh is not None:
            if issubclass(type(self.reference_mesh), mskt.mesh.Mesh):
                pass
            elif isinstance(self.reference_mesh, int):
                if isinstance(self.list_mesh_paths[0], (list, tuple)):
                    mesh = self.list_mesh_paths[self.reference_mesh][mesh_to_scale]
                elif isinstance(self.list_mesh_paths[0], (str, mskt.mesh.Mesh)):
                    mesh = self.list_mesh_paths[self.reference_mesh]
                else:
                    raise Exception('provided list_meshes wrong type')
                self.reference_mesh = mskt.mesh.Mesh(mesh)
            elif isinstance(self.reference_mesh, str):
                self.reference_mesh = mskt.mesh.Mesh(self.reference_mesh)
            elif isinstance(self.reference_mesh, list):
                self.reference_mesh = mskt.mesh.Mesh(self.reference_mesh[self.reference_object])
            else:
                raise ValueError('Reference mesh must be a string, list of strings, or mesh.Mesh object, not', type(self.reference_mesh))

        self.data = []
        for loc_meshes in list_mesh_paths:
            if self.verbose is True:
                print('Loading mesh:', loc_meshes)
            if type(loc_meshes) not in (tuple, list):
                loc_meshes = [loc_meshes]
            
            #preallocate torch array
            # pts_array = torch.zeros((sum(n_pts), 3 + len(loc_meshes)))

            if print_filename is True:
                print(loc_meshes)
            
            # Create hash and filename 
            file_hash = self.create_hash(loc_meshes)
            cached_file = self.find_hash(filename=f'{file_hash}.npz')
            
            if (len(cached_file) > 0) and (load_cache is True):
                # if hashed file exists, load it. 
                data_ = np.load(cached_file[0])
                data = {}
                data = unpack_numpy_data(data_)
                
            else:
                # otherwise, load the mesh and create SDF samples. 
                print('Creating SDF Samples')
                data = {
                    'xyz': torch.zeros((sum(n_pts), 3)),
                    'gt_sdf': torch.zeros((sum(n_pts), len(loc_meshes))),
                }
                pts_idx = 0
                for n_pts_, sigma_ in self.pt_sample_combos:
                    result_ = read_meshes_get_sampled_pts(
                        loc_meshes, 
                        mean=[0,0,0], 
                        sigma=sigma_, 
                        n_pts=n_pts_, 
                        rand_function=rand_function, 
                        center_pts=center_pts,
                        # axis_align=axis_align,
                        scale_all_meshes=scale_all_meshes,
                        center_all_meshes=center_all_meshes,
                        mesh_to_scale=mesh_to_scale,
                        norm_pts=norm_pts,
                        scale_method=scale_method,
                        get_random=True,
                        return_orig_mesh=False,
                        return_new_mesh=False,
                        return_orig_pts=False,
                        register_to_mean_first=False if self.reference_mesh is None else True,
                        mean_mesh=self.reference_mesh,
                        fix_mesh=self.fix_mesh,
                    )
                    xyz_ = result_['pts'] if 'pts' in result_ else result_['xyz']
                    sdfs_ = result_['sdf'] if 'sdf' in result_ else result_['gt_sdf']

                    data['xyz'][pts_idx:pts_idx + sum(n_pts_), :] = torch.from_numpy(xyz_).float()
                    # data['gt_sdf'][pts_idx:pts_idx + sum(n_pts_), :] = torch.from_numpy(sdfs_).float()

                    # pts_array[pts_idx:pts_idx + sum(n_pts_), :3] = torch.from_numpy(pts_).float()

                    for mesh_idx in range(len(sdfs_)):
                        data['gt_sdf'][pts_idx:pts_idx + sum(n_pts_), mesh_idx] = torch.from_numpy(sdfs_[mesh_idx]).float()
                        # pts_array[pts_idx:pts_idx + sum(n_pts_), 3+mesh_idx] = torch.from_numpy(sdfs_[mesh_idx]).float()
                    pts_idx += sum(n_pts_)
                if save_cache is True:
                    # if want to cache, and new... then save. 
                    filepath = os.path.join(cache_folder, f'{file_hash}.npz')
                    np.savez(filepath, pts=data['xyz'], sdfs=data['gt_sdf'])

            if data['gt_sdf'].shape[1] == 2:
                sdf_ = data['gt_sdf'].clone()
                sdf_[sdf_ < 0] = -1
                sdf_[sdf_ > 0] = 1
                sdf_[sdf_ == 0] = 0
                total = torch.sum(sdf_, axis=1)

                out_out = torch.sum(total == 2)
                out_in = torch.sum(total == 0)
                in_in = torch.sum(total == -2)

                # Drop overlapping points
                data['gt_sdf'] = data['gt_sdf'][total != -2, :]
                data['xyz'] = data['xyz'][total != -2, :]


                if self.verbose is True:
                    print('total', total.shape)
                    print('total', total)
                    print('out_out', out_out)
                    print('out_in', out_in)
                    print('in_in', in_in)

            
            

            # print(data)
            # print('sdf max', torch.max(data['gt_sdf'][:,0]), torch.max(data['gt_sdf'][:,1]))
            # print('sdf min', torch.min(data['gt_sdf'][:,0]), torch.min(data['gt_sdf'][:,1]))
            # print('xyz max', torch.max(data['xyz'][:,0]), torch.max(data['xyz'][:,1]), torch.max(data['xyz'][:,2]))
            # print('xyz min', torch.min(data['xyz'][:,0]), torch.min(data['xyz'][:,1]), torch.min(data['xyz'][:,2]))

            if self.equal_pos_neg is True:
                pos_idx, neg_idx, surf_idx = self.sdf_pos_neg_idx(data)
                data['pos_idx'] = pos_idx
                data['neg_idx'] = neg_idx
                data['surf_idx'] = surf_idx            

            self.data.append(data)


    def sdf_pos_neg_idx(self, data):
        '''
        - iterate over each mesh
        - get number of points for that mesh and get: 
            - points positive (outside mesh)
            - points negative (inside mesh)
        - return list of indices
        '''

        self.samples_per_mesh = [int((n_pts_/self.total_n_pts) * self.subsample) for n_pts_ in self.n_pts]

        pos_idx = []
        neg_idx = []
        surf_idx = []
        pts_idx_ = 0
        if self.verbose is True:
            print('data', data['xyz'].shape, data['gt_sdf'].shape)
        
        self.samples_per_sign = []
        for mesh_idx, subsample_ in enumerate(self.samples_per_mesh):
            
            samples_per_sign = int(subsample_/2)
            print(samples_per_sign)
            self.samples_per_sign.append(samples_per_sign)

            # BELOW NEEDS LOGIC TO UNPACK  1/2 pos/neg pts for each mesh
            # mesh_sdfs = data['gt_sdf'][pts_idx_:pts_idx_ + n_pts_, mesh_idx]
            mesh_sdfs = data['gt_sdf'][:, mesh_idx].clone()
            pos_idx_ = (mesh_sdfs > 0).nonzero(as_tuple=True)[0] #+ pts_idx_
            neg_idx_ = (mesh_sdfs < 0).nonzero(as_tuple=True)[0] #+ pts_idx_
            surf_idx_ = (mesh_sdfs == 0).nonzero(as_tuple=True)[0] #+ pts_idx_

            # print('pre repeat:')
            # print('pos_idx_', pos_idx_.shape)
            # print('neg_idx_', neg_idx_.shape)

            # Repeat +/- indices if either of them does not have enough for a full batch. 
            pos_idx_ = pos_idx_.repeat(samples_per_sign//pos_idx_.size(0) + 1)
            neg_idx_ = neg_idx_.repeat(samples_per_sign//neg_idx_.size(0) + 1)

            # print('pos_idx_', pos_idx_.shape)
            # print('neg_idx_', neg_idx_.shape)

            pos_idx.append(pos_idx_)
            neg_idx.append(neg_idx_)
            surf_idx.append(surf_idx_)
        
        return pos_idx, neg_idx, surf_idx

    def __getitem__(self, idx):
        #TODO: get rid of pts_array from above
        # replace with self.data[idx] = [pts, sdfs_]
        # this will simplify everything downstream because this is something that we are
        # constantly undoing/re-doing elsewhere in the code - and it even lookts like
        # the sdfs/pts are stroed separately in the npy files. 
        data_ = self.data[idx]
        if self.subsample is not None:
            if self.equal_pos_neg is True:
                # get number of points for each mesh
                # this is weighted by the number of points in the mesh 
                # relative to the total number of points in the dataset
                # samples_per_mesh = [int((n_pts_/self.total_n_pts) * self.subsample) for n_pts_ in self.n_pts]
                idx_ = []
                for mesh_idx, samples_per_sign in enumerate(self.samples_per_sign):
                    # get number of positive and negative points for this mesh
                    # samples_per_sign = int(subsample_/2)
                    if self.verbose is True:
                        print('samples_per_sign', samples_per_sign)
                    
                    if samples_per_sign == 0:
                        continue
                    # get random indices for positive and negative points
                    # idx_pos = data_['pos_idx'][mesh_idx].repeat(data_['pos_idx'][mesh_idx].size(0)//samples_per_sign + 1)
                    perm_pos = torch.randperm(data_['pos_idx'][mesh_idx].size(0))[:samples_per_sign]
                    idx_pos = data_['pos_idx'][mesh_idx][perm_pos]

                    # idx_neg = data_['neg_idx'][mesh_idx].repeat(data_['neg_idx'][mesh_idx].size(0)//samples_per_sign + 1)
                    perm_neg = torch.randperm(data_['neg_idx'][mesh_idx].size(0))[:samples_per_sign]
                    idx_neg = data_['neg_idx'][mesh_idx][perm_neg]

                    # combine positive and negative indices
                    idx_ += [idx_pos, idx_neg]
                    
                
                # combine indices for all meshes
                idx_ = torch.cat(idx_, dim=0)

                if len(idx_) < self.subsample:
                    # if we don't have enough points, then just take random points
                    perm = torch.randperm(data_['xyz'].size(0))
                    _idx_ = perm[:self.subsample-len(idx_)]
                    idx_ = torch.cat([idx_, _idx_], dim=0)
                
            else:
                perm = torch.randperm(data_['xyz'].size(0))
                idx_ = perm[: self.subsample]
            
            if self.verbose is True:
                print('idx_ size:', idx_.size(), 'idx_ min:', idx_.min(),'idx_ max:', idx_.max())
                print('equal neg pos', self.equal_pos_neg)

            data_ = {
                'xyz': data_['xyz'][idx_, :],
                'gt_sdf': data_['gt_sdf'][idx_, :],
            }
        
        return data_, idx


class DiffusionSDFSamples(SDFSamples):
    def __init__(
        self,
        list_mesh_paths,
        subsample=None,
        print_filename=False,
        n_pts=500000,
        p_near_surface=0.7,
        p_further_from_surface=0.2,
        sigma_near=0.01,
        sigma_far=0.1,
        rand_function='normal', 
        center_pts=True,
        axis_align=False,
        norm_pts=False,
        scale_method='max_rad',
        loc_save=os.environ['LOC_SDF_CACHE'],
        include_seed_in_hash=True,
        save_cache=True,
        random_seed=None,
        verbose=False,
        load_cache=True,
        point_cloud_size=1024,
        equal_pos_neg=True,
        fix_mesh=True,
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
        self.verbose = verbose
        self.point_cloud_size = point_cloud_size
        self.equal_pos_neg = equal_pos_neg
        self.fix_mesh = fix_mesh

        self.list_hash_params = [
            'diffusion_sdf', #just to make unique for this data class
            self.n_pts,
            self.p_near_surface, self.sigma_near,
            self.p_further_from_surface, self.sigma_far,
            self.center_pts,
            self.axis_align,
            self.norm_pts,
            self.scale_method,
            self.rand_function,
            self.fix_mesh
        ]

        if save_cache is True:
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

            if (len(cached_file) > 0) and (load_cache is True):
                # if hashed file exists, load it. 
                data_ = np.load(cached_file[0])
                data = unpack_numpy_data(data_, point_cloud=True)
                
            else:
                # otherwise, load the mesh and create SDF samples. 
                print('Creating SDF Samples')
                data = {
                    'xyz': torch.zeros((n_pts, 3)),
                    'gt_sdf': torch.zeros((n_pts)),
                }
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
                        return_orig_pts=False,
                        return_point_cloud=True,
                        fix_mesh=self.fix_mesh,
                    )
                    xyz_ = result_['xyz']
                    sdfs_ = result_['gt_sdf']

                    data['xyz'][pts_idx:pts_idx + n_pts_, :] = torch.from_numpy(xyz_).float()
                    data['gt_sdf'][pts_idx:pts_idx + n_pts_] = torch.from_numpy(sdfs_).float()
                    pts_idx += n_pts_
                data['point_cloud'] = torch.from_numpy(result_['point_cloud']).float()
                if save_cache is True:
                    # if want to cache, and new... then save. 
                    filepath = os.path.join(cache_folder, f'{file_hash}.npz')
                    np.savez(filepath, xyz=data['xyz'], gt_sdf=data['gt_sdf'], point_cloud=data['point_cloud'])

            pos_idx, neg_idx, surf_idx = self.sdf_pos_neg_idx(data)
            data['pos_idx'] = pos_idx
            data['neg_idx'] = neg_idx
            data['surf_idx'] = surf_idx

            self.data.append(data)
    
    def sdf_pos_neg_idx(self, data):
        pos_idx = (data['gt_sdf'] > 0).nonzero(as_tuple=True)[0]
        neg_idx = (data['gt_sdf'] < 0).nonzero(as_tuple=True)[0]
        surf_idx = (data['gt_sdf'] == 0).nonzero(as_tuple=True)[0]

        return pos_idx, neg_idx, surf_idx
    # def sdf_pos_neg(self, data):
    #     # print('data', data.shape)
    #     pos_idx = data['gt_sdf'] > 0
    #     neg_idx = data['gt_sdf'] < 0
    #     surf_idx = data['gt_sdf'] == 0

    #     # print('pos_idx', pos_idx.shape)
    #     # print('neg_idx', neg_idx.shape)

    #     pos_data = {
    #         'xyz': data['xyz'][pos_idx, :],
    #         'gt_sdf': data['gt_sdf'][pos_idx],
    #         'point_cloud': data['point_cloud']
    #     }

    #     neg_data = {
    #         'xyz': data['xyz'][neg_idx, :],
    #         'gt_sdf': data['gt_sdf'][neg_idx],
    #         'point_cloud': data['point_cloud']
    #     }

    #     return pos_data, neg_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_ = self.data[idx]
        if self.subsample is not None:
            # print('Subsampling', self.subsample)

            if self.equal_pos_neg is True:
                samples_per_sign = int(self.subsample/2)
                
                idx_pos = data_['pos_idx'].repeat(data_['pos_idx'].size(0)//samples_per_sign + 1)
                perm_pos = torch.randperm(idx_pos.size(0))
                idx_pos = perm_pos[:samples_per_sign]

                idx_neg = data_['neg_idx'].repeat(data_['neg_idx'].size(0)//samples_per_sign + 1)
                perm_neg = torch.randperm(idx_neg.size(0))
                idx_neg = perm_neg[:samples_per_sign]

                idx_ = torch.cat((idx_pos, idx_neg), dim=0)

                if len(idx_) < self.subsample:
                    # if we don't have enough points, then just take random points
                    perm = torch.randperm(data_['xyz'].size(0))
                    _idx_ = perm[:self.subsample-len(idx_)]
                    idx_ = torch.cat([idx_, _idx_], dim=0)
            
            else:
                perm = torch.randperm(data_['xyz'].size(0))
                idx_ = perm[:self.subsample]

            
            pt_cloud_perm = torch.randperm(data_['point_cloud'].size(0))
            idx_pc = pt_cloud_perm[:self.point_cloud_size]

            # print('idx_pc', idx_pc.shape)
            # print('data_ point cloud', data_['point_cloud'].shape)

            # print('idx_', idx_.shape)
            # print('data_ xyz', data_['xyz'].shape)

            data_ = {
                'xyz': data_['xyz'][idx_, :],
                'gt_sdf': data_['gt_sdf'][idx_],
                'point_cloud': data_['point_cloud'][idx_pc, :],
            }

            # print('min xyz', data_['xyz'].min(axis=0))
            # print('max xyz', data_['xyz'].max(axis=0))
            # print('min sdf', data_['gt_sdf'].min())
            # print('max sdf', data_['gt_sdf'].max())
            # print('min pc', data_['point_cloud'].min(axis=0))
            # print('max pc', data_['point_cloud'].max(axis=0))

            # print(data_)
        return data_, idx