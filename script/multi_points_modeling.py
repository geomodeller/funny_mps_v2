
from numba import jit
import numpy as np
import sys
sys.path.insert(0, './script')
from curate_training_image import curate_training_image

@jit(nopython=True)
def fast_bincount(arr, minlength):
    counts = np.zeros(minlength, dtype=np.int64)
    for val in arr:
        counts[val] += 1
    return counts

def predictive_model(data_x, data_y, 
                     input_x, 
                     facies_ratio, 
                     unique_facies) -> np.array:

    if np.all(input_x == -1):
        return facies_ratio

    # Identify valid columns (features that are not -1)
    valid_cols = np.where(input_x != -1)[1]
    input_x_filtered = input_x[:, valid_cols]
    data_x_filtered = data_x[:, valid_cols]

    # Find matching rows
    matching_rows = np.all(data_x_filtered == input_x_filtered, axis=1)
    matched_labels = data_y[matching_rows]

    # Count occurrences using np.bincount
    counts = fast_bincount(matched_labels.astype(np.int32).flatten(), minlength=len(unique_facies))

    # If counts sum to zero, fall back to facies_ratio
    if counts.sum() == 0:
        return facies_ratio

    return counts / counts.sum()

def multi_points_modeling(TI, template_size, random_seed, real_nx, real_ny, real_nz, hard_data = None, verbose = False):


    unique_facies = list(np.unique(TI).astype(np.int8))
    facies_ratio = [np.sum(TI==f)/np.prod(TI.shape) for f in unique_facies]
    padding_x, padding_y, padding_z = int((template_size[0]-1)/2), int((template_size[1]-1)/2), int((template_size[2]-1)/2)

    data_x, data_y, flag = curate_training_image(TI, template_size, 1.0)

    # TODO: generate model
    realization = np.ones((real_nx+2*padding_x, real_ny+2*padding_x, real_nz+2*padding_z))*-1
    if hard_data is not None:
        if padding_z != 0:
            realization[padding_x:-padding_x, padding_y:-padding_y, padding_z:-padding_z] = hard_data
        else:
            realization[padding_x:-padding_x, padding_y:-padding_y, :] = hard_data
        if verbose:
            print('hard data is conditioned')
    x_0, x_1 = int(0 +padding_x), int(realization.shape[0] - padding_x)
    y_0, y_1 = int(0 +padding_y), int(realization.shape[1] - padding_y)
    z_0, z_1 = int(0 +padding_z), int(realization.shape[2] - padding_z)
    xx, yy, zz = np.meshgrid(range(x_0, x_1), range(y_0, y_1), range(z_0, z_1))
    random_path = np.array([i.flatten() for i in [xx, yy, zz]])
    np.random.seed(random_seed)
    np.random.shuffle(random_path.T)

    
    for ii, jj, kk in zip(random_path[0].T, random_path[1].T, random_path[2].T):
        if realization[ii, jj, kk] != -1:
            continue
        template = realization[ii-padding_x:ii+(padding_x+1),
                            jj-padding_y:jj+(padding_y+1),
                            kk-padding_z:kk+(padding_z+1)].copy().flatten()
        input_x = template[flag].reshape(1,-1)  
        realization[ii, jj, kk] = np.random.choice(unique_facies, p=predictive_model(data_x, data_y, input_x, facies_ratio, unique_facies))
    if padding_z != 0:
        return realization[padding_x:-padding_x, padding_y:-padding_y, padding_z:-padding_z]
    else:
        return realization[padding_x:-padding_x, padding_y:-padding_y]
    
def multi_points_modeling_multi_scaled(TI, n_level, level_size,
                                      template_size, 
                                      random_seed, 
                                      real_nx, real_ny, real_nz, 
                                      hard_data = None, verbose = False):
    
    TI_s, grid_size_s = [], []
    nx, ny, nz = real_nx, real_ny, real_nz
    for level in range(n_level):
        TI_s.append(TI[::level_size**level, ::level_size**level, ::level_size**level])
        grid_size_s.append((nx, ny, nz))
        nx, ny, nz = round(nx/level_size), round(ny/level_size), 1


    real_s = []
    if hard_data is None:
        real = np.ones(grid_size_s[-1]) * -1
    else:
        real = hard_data
        
    for idx, (level, TI_at_level, grid_size_at_level) in enumerate(zip(range(n_level)[::-1],TI_s[::-1], grid_size_s[::-1])):
        real = multi_points_modeling(TI_at_level, 
                                    template_size, 
                                    random_seed, 
                                    grid_size_at_level[0], grid_size_at_level[1], grid_size_at_level[2], 
                                    real, 
                                    verbose)
        real_s.append(real)
        if level == 0:
            break
        real_next = np.ones(grid_size_s[level-1]) * -1
        real_next[1::level_size, 1::level_size, :] = real
        real = real_next.copy()
    return real