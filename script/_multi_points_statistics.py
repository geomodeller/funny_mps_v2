import numpy as np
import itertools
from numba import jit
from typing import List, Tuple, Optional, Union # Added for type hints

# Removed: sys.path.insert(0, './script')
# It's better practice to manage dependencies through installation (pip install .)
# or by ensuring the script is run from a location where Python can find 'curate_training_image'.
# If curate_training_image is in the same directory or a standard location,
# the direct import below should work. If it's truly in './script',
# consider restructuring your project or adding './script' to PYTHONPATH environment variable.
# Assuming curate_training_image functions are moved into this script or handled by Python path
# from curate_training_image import curate_training_image # If it was meant to be a separate file

# --- Improved curate_training_image ---
# Note: The original nested loop approach is very slow in Python.
# A vectorized approach using strides is much faster but requires care.
# If scikit-image is available, view_as_windows is often safer and easier.
# Below is a numpy-strides based implementation for performance.

def get_window_view(arr: np.ndarray, window_shape: Tuple[int, ...]) -> np.ndarray:
    """
    Creates a view of an array as (overlapping) windows.
    Uses stride tricks, be careful with the output array as it shares memory.
    Requires the input array to be C-contiguous.

    Args:
        arr (np.ndarray): The input array (must be C-contiguous).
        window_shape (Tuple[int, ...]): The shape of the desired window (template).

    Returns:
        np.ndarray: A view of the array with an added dimension for windows.
                    Shape: arr.shape[:-len(window_shape)] + tuple(arr.shape[i] - window_shape[i] + 1 for i in range(len(window_shape))) + window_shape
    """
    if not arr.flags['C_CONTIGUOUS']:
        arr = np.ascontiguousarray(arr) # Ensure C-contiguity

    arr_shape = np.array(arr.shape)
    window_shape = np.array(window_shape)

    if any(arr_shape < window_shape):
        raise ValueError("Window shape cannot be larger than array shape in any dimension.")

    new_shape = tuple(arr_shape - window_shape + 1) + tuple(window_shape)
    new_strides = arr.strides + arr.strides

    return np.lib.stride_tricks.as_strided(arr, shape=new_shape, strides=new_strides)

def curate_training_image_vectorized(
    TI: np.ndarray,
    template_shape: Union[List[int], Tuple[int, int, int]],
    fraction_to_use: float = 1.0,
    use_cross_stencil: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Curates a training image efficiently using vectorized operations (stride tricks).

    Extracts data patterns (context) and central target values from a 3D training image (TI)
    based on a specified template shape.

    Args:
        TI (np.ndarray): The 3D training image (e.g., facies). Should contain integer labels.
        template_shape (Union[List[int], Tuple[int, int, int]]):
            The dimensions (x, y, z) of the template/window. Must be odd numbers.
        fraction_to_use (float, optional): The fraction of extracted patterns to randomly sample.
                                         Defaults to 1.0 (use all patterns).
        use_cross_stencil (bool, optional): If True, uses only neighbors within a '+' shaped
                                         cross stencil (Manhattan distance <= radius) relative
                                         to the center within the template as features.
                                         If False, uses all neighbors except the center.
                                         Defaults to True.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - data_x (np.ndarray): The input features (context neighbours), shape (n_samples, n_features). dtype=int32.
            - data_y (np.ndarray): The target outputs (central pixel), shape (n_samples, 1). dtype=int32.
            - feature_indices (np.ndarray): The indices within the flattened template used as features (data_x).
    """
    template_shape = tuple(template_shape)
    if len(template_shape) != 3:
        raise ValueError("template_shape must have 3 dimensions (x, y, z)")
    if any(s % 2 == 0 for s in template_shape):
        raise ValueError(f"All dimensions in template_shape must be odd: {template_shape}")

    if not issubclass(TI.dtype.type, np.integer):
         print(f"Warning: Training image dtype is {TI.dtype}. Casting to int.")
         TI = TI.astype(np.int32) # Ensure integer type for facies

    # Create a view of the TI as overlapping windows (templates)
    # Shape: (n_win_x, n_win_y, n_win_z, win_x, win_y, win_z)
    all_windows = get_window_view(TI, template_shape)

    # Flatten the windows and the window grid dimensions
    # Shape: (n_total_windows, win_x * win_y * win_z)
    n_total_windows = np.prod(all_windows.shape[:3])
    template_size = np.prod(template_shape)
    data_flat = all_windows.reshape(n_total_windows, template_size)

    # Identify the center index of the flattened template
    center_index = template_size // 2 # Works for odd dimensions

    # Create template indices grid to calculate distances if needed
    indices = np.arange(template_size)
    coords = np.unravel_index(indices, template_shape) # Get (x,y,z) coords for each index
    center_coord = np.array(np.unravel_index(center_index, template_shape))

    if use_cross_stencil:
        radius_x = template_shape[0] // 2
        # Calculate Manhattan distance from center for each point in the template
        distances = np.sum(np.abs(np.array(coords).T - center_coord), axis=1)
        # Select indices that are not the center and within the cross stencil radius
        feature_indices = indices[(indices != center_index) & (distances <= radius_x)]
        # Adjust feature count if needed (original code seemed to hardcode half size?)
        # n_features = len(feature_indices) # This is the actual number of features
    else:
        # Select all indices except the center
        feature_indices = indices[indices != center_index]
        # n_features = template_size - 1

    # Extract features (data_x) and target (data_y)
    data_x = data_flat[:, feature_indices].astype(np.int32)
    data_y = data_flat[:, center_index].reshape(-1, 1).astype(np.int32)

    # Subsample if requested
    if fraction_to_use < 1.0:
        if fraction_to_use <= 0.0:
             raise ValueError("fraction_to_use must be greater than 0.")
        n_samples = data_x.shape[0]
        n_select = max(1, int(fraction_to_use * n_samples)) # Ensure at least 1 sample
        mask = np.random.choice(n_samples, n_select, replace=False)
        data_x = data_x[mask]
        data_y = data_y[mask]
    elif fraction_to_use > 1.0:
        print("Warning: fraction_to_use > 1.0. Using all data (fraction_to_use = 1.0).")


    return data_x, data_y, feature_indices # Return indices used

# --- fast_bincount ---
@jit(nopython=True)
def fast_bincount(arr: np.ndarray, minlength: int) -> np.ndarray:
    """
    Numba-accelerated bincount for non-negative integer arrays.

    Args:
        arr (np.ndarray): 1D array of non-negative integers.
        minlength (int): The minimum number of bins.

    Returns:
        np.ndarray: The count for each bin. dtype=int64.

    Note:
        Equivalent to np.bincount(arr, minlength=minlength) but potentially faster
        when called repeatedly within Numba-compiled functions. Assumes arr
        contains values >= 0 and < minlength.
    """
    # Consider input validation if necessary, e.g., checking for negatives
    # or values exceeding minlength, although Numba handles index errors.
    counts = np.zeros(minlength, dtype=np.int64)
    # Use .ravel() or .flatten() in case input is not 1D, ensures the loop works
    for val in arr.ravel():
        if val >= 0 and val < minlength: # Basic check
             counts[val] += 1
    return counts

# --- predictive_model ---
def predictive_model(
    training_patterns: np.ndarray,
    training_labels: np.ndarray,
    input_pattern: np.ndarray,
    global_facies_ratio: np.ndarray,
    unique_facies: np.ndarray
) -> np.ndarray:
    """
    Predicts facies probability based on matching patterns in training data.

    Finds rows in `training_patterns` that match the `input_pattern` (ignoring -1 values)
    and calculates the probability distribution of the corresponding `training_labels`.
    Falls back to `global_facies_ratio` if no matches are found or input is all -1.

    Args:
        training_patterns (np.ndarray): The input features from the training image (data_x).
                                        Shape (n_samples, n_features).
        training_labels (np.ndarray): The target outputs from the training image (data_y).
                                      Shape (n_samples, 1) or (n_samples,).
        input_pattern (np.ndarray): The pattern from the realization grid to predict.
                                    Shape (1, n_features). Should contain -1 for unknown values.
        global_facies_ratio (np.ndarray): The overall proportion of each facies in the TI.
                                          Used as a fallback. Shape (n_facies,).
        unique_facies (np.ndarray): Array of unique facies codes present in the TI.
                                    Used to determine the size of the output distribution.

    Returns:
        np.ndarray: A probability distribution over the `unique_facies`. Shape (n_facies,).
    """
    n_facies = len(unique_facies)
    # Ensure labels are flat for bincounting
    training_labels_flat = training_labels.ravel()

    # Handle case where input pattern is completely unknown
    if np.all(input_pattern == -1):
        return global_facies_ratio

    # Identify known features in the input pattern
    # input_pattern has shape (1, n_features)
    valid_cols_mask = (input_pattern != -1).flatten() # Boolean mask for columns

    # Handle case where there are no known features (should ideally not happen if not all -1)
    if not np.any(valid_cols_mask):
         return global_facies_ratio # Or raise an error?

    # Filter input pattern and training patterns based on known features
    input_pattern_filtered = input_pattern[:, valid_cols_mask]
    training_patterns_filtered = training_patterns[:, valid_cols_mask]

    # Find rows in training data matching the filtered input pattern
    # Use broadcasting for efficient comparison
    matching_rows_mask = np.all(training_patterns_filtered == input_pattern_filtered, axis=1)

    # Get labels corresponding to matching patterns
    matched_labels = training_labels_flat[matching_rows_mask]

    # If no matches found, return global ratio
    if matched_labels.size == 0:
        return global_facies_ratio

    # Count occurrences of each facies label among matches
    # Use standard np.bincount - it's highly optimized in C
    # Ensure matched_labels are integers and provide minlength
    counts = np.bincount(matched_labels.astype(np.int32), minlength=n_facies)

    # Alternative using the Numba version (if preferred/tested faster)
    # counts = fast_bincount(matched_labels.astype(np.int32), minlength=n_facies)

    # Normalize counts to get probabilities
    count_sum = counts.sum()
    if count_sum == 0: # Should ideally not happen if matched_labels.size > 0
        return global_facies_ratio
    else:
        return counts / count_sum

# --- multi_points_modeling ---
def multi_points_modeling(
    TI: np.ndarray,
    template_shape: Union[List[int], Tuple[int, int, int]],
    random_seed: int,
    real_nx: int,
    real_ny: int,
    real_nz: int,
    hard_data: Optional[np.ndarray] = None,
    verbose: bool = False,
    training_fraction: float = 1.0, # Added parameter
    use_cross_stencil: bool = True # Added parameter
) -> np.ndarray:
    """
    Performs single-grid Multi-Point Simulation (MPS).

    Generates a realization grid mimicking the patterns found in the Training Image (TI),
    optionally conditioned to hard data.

    Args:
        TI (np.ndarray): The 3D Training Image (facies).
        template_shape (Union[List[int], Tuple[int, int, int]]):
            Dimensions (x, y, z) of the template. Must be odd.
        random_seed (int): Seed for the random number generator for reproducibility.
        real_nx (int): X-dimension of the desired realization grid.
        real_ny (int): Y-dimension of the desired realization grid.
        real_nz (int): Z-dimension of the desired realization grid.
        hard_data (Optional[np.ndarray], optional): Grid of known values to condition the simulation.
                                                 Shape should match (real_nx, real_ny, real_nz).
                                                 Use a sentinel value (e.g., -1) for unknown nodes.
                                                 Defaults to None (unconditional simulation).
        verbose (bool, optional): If True, print progress messages. Defaults to False.
        training_fraction (float, optional): Fraction of TI patterns to use for training the model.
                                             Defaults to 1.0.
        use_cross_stencil (bool, optional): Passed to curate_training_image. Defaults to True.


    Returns:
        np.ndarray: The generated realization grid with shape (real_nx, real_ny, real_nz).
    """
    np.random.seed(random_seed)
    template_shape = tuple(template_shape)

    # --- 1. Pre-processing ---
    unique_facies = np.unique(TI[TI >= 0]) # Ignore potential negative sentinels in TI
    if len(unique_facies) == 0:
        raise ValueError("Training Image does not contain any valid facies (non-negative integers).")

    n_facies = len(unique_facies)
    facies_counts = np.bincount(TI[TI >= 0].astype(np.int32).ravel(), minlength=n_facies)
    global_facies_ratio = facies_counts / facies_counts.sum()

    # Calculate padding based on template shape
    padding_x, padding_y, padding_z = [(s - 1) // 2 for s in template_shape]
    padding = (padding_x, padding_y, padding_z) # Tuple for consistency

    if verbose: print("Curating training image...")
    # Use the improved, vectorized curator
    data_x, data_y, feature_indices = curate_training_image_vectorized(
        TI, template_shape, fraction_to_use=training_fraction, use_cross_stencil=use_cross_stencil
    )
    if verbose: print(f"Curated {data_x.shape[0]} patterns with {data_x.shape[1]} features.")
    if data_x.shape[0] == 0:
         raise ValueError("No training patterns could be extracted. Check TI and template_shape.")


    # --- 2. Initialization ---
    # Create realization grid with padding, initialized to -1 (unknown)
    real_shape_padded = (real_nx + 2 * padding_x, real_ny + 2 * padding_y, real_nz + 2 * padding_z)
    realization = np.full(real_shape_padded, -1, dtype=np.int32)

    # Define slices for the core (non-padded) area and padding
    core_slice = (slice(padding_x, padding_x + real_nx),
                  slice(padding_y, padding_y + real_ny),
                  slice(padding_z, padding_z + real_nz))

    # Condition to hard data if provided
    if hard_data is not None:
        if hard_data.shape != (real_nx, real_ny, real_nz):
            raise ValueError(f"Hard data shape {hard_data.shape} must match realization dimensions ({real_nx}, {real_ny}, {real_nz})")
        # Ensure hard data uses -1 for unknown values as well
        realization[core_slice] = np.where(hard_data != -1, hard_data.astype(np.int32), -1)
        if verbose: print('Hard data conditioned.')

    # --- 3. Simulation Path ---
    # Define coordinates for the core grid
    x_coords = np.arange(padding_x, padding_x + real_nx)
    y_coords = np.arange(padding_y, padding_y + real_ny)
    z_coords = np.arange(padding_z, padding_z + real_nz)

    # Create grid of indices and shuffle for random path
    zz, yy, xx = np.meshgrid(z_coords, y_coords, x_coords, indexing='ij')
    simulation_path_indices = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T
    np.random.shuffle(simulation_path_indices)

    # --- 4. Sequential Simulation Loop ---
    n_total = len(simulation_path_indices)
    if verbose: print(f"Starting simulation for {n_total} nodes...")
    for i, (ix, iy, iz) in enumerate(simulation_path_indices):
        # Skip if node is already informed (hard data or previously simulated)
        if realization[ix, iy, iz] != -1:
            continue

        # Extract the data event (template centered at the current node)
        template_slice = (slice(ix - padding_x, ix + padding_x + 1),
                          slice(iy - padding_y, iy + padding_y + 1),
                          slice(iz - padding_z, iz + padding_z + 1))
        current_template = realization[template_slice].flatten()

        # Select the neighbours defined by feature_indices for the input pattern
        input_pattern = current_template[feature_indices].reshape(1, -1) # Shape (1, n_features)

        # Get probability distribution from the predictive model
        probabilities = predictive_model(data_x, data_y, input_pattern, global_facies_ratio, unique_facies)

        # Draw a value based on the calculated probabilities
        # Ensure probabilities sum to 1 (can have minor floating point issues)
        probabilities = probabilities / probabilities.sum()
        simulated_value = np.random.choice(unique_facies, p=probabilities)

        # Assign simulated value to the realization grid
        realization[ix, iy, iz] = simulated_value

        if verbose and (i + 1) % (max(1, n_total // 10)) == 0:
             print(f"   ... simulated {i+1}/{n_total} nodes ({((i+1)/n_total)*100:.1f}%)")

    if verbose: print("Simulation complete.")

    # --- 5. Final Result ---
    # Return the core (unpadded) realization
    final_realization = realization[core_slice]

    # Adjust shape if Z dimension was originally 1
    if real_nz == 1:
        return final_realization.reshape((real_nx, real_ny))
    else:
        return final_realization


# --- multi_points_modeling_multi_scaled ---
def multi_points_modeling_multi_scaled(
    TI: np.ndarray,
    n_levels: int,
    level_scale_factor: int,
    template_shape: Union[List[int], Tuple[int, int, int]],
    random_seed: int,
    real_nx: int, real_ny: int, real_nz: int,
    hard_data: Optional[np.ndarray] = None,
    verbose: bool = False,
    training_fraction: float = 1.0, # Added parameter
    use_cross_stencil: bool = True # Added parameter
) -> np.ndarray:
    """
    Performs multi-scale Multi-Point Simulation (MPS).

    Generates a realization by simulating on progressively finer grids, using coarser
    simulations to condition finer ones.

    Args:
        TI (np.ndarray): The 3D Training Image (facies) at the finest scale.
        n_levels (int): The number of grid levels (including the finest). n_levels=1 is single grid.
        level_scale_factor (int): The scaling factor between grid levels (e.g., 2 or 3).
        template_shape (Union[List[int], Tuple[int, int, int]]):
            Dimensions (x, y, z) of the template. Must be odd. Used at all levels.
        random_seed (int): Seed for the random number generator.
        real_nx (int): X-dimension of the desired realization grid (finest level).
        real_ny (int): Y-dimension of the desired realization grid (finest level).
        real_nz (int): Z-dimension of the desired realization grid (finest level).
        hard_data (Optional[np.ndarray], optional): Grid of known values at the finest scale.
                                                 Shape should match (real_nx, real_ny, real_nz).
                                                 Defaults to None.
        verbose (bool, optional): If True, print progress messages. Defaults to False.
        training_fraction (float, optional): Passed to multi_points_modeling. Defaults to 1.0.
        use_cross_stencil (bool, optional): Passed to multi_points_modeling. Defaults to True.

    Returns:
        np.ndarray: The generated realization grid at the finest scale.
    """
    if n_levels < 1:
        raise ValueError("n_levels must be at least 1.")
    if level_scale_factor < 2:
         raise ValueError("level_scale_factor must be 2 or greater.")

    # --- 1. Prepare grids and TIs for each level ---
    TI_at_level = [TI] # Level 0 (finest)
    grid_dims_at_level = [(real_nx, real_ny, real_nz)] # Level 0

    current_nx, current_ny, current_nz = real_nx, real_ny, real_nz
    current_ti = TI

    for level in range(1, n_levels):
        # Calculate dimensions for the next coarser level
        # Use ceiling division equivalent to ensure coverage
        next_nx = (current_nx + level_scale_factor - 1) // level_scale_factor
        next_ny = (current_ny + level_scale_factor - 1) // level_scale_factor
        next_nz = (current_nz + level_scale_factor - 1) // level_scale_factor

        # Downsample TI (simple nearest neighbor by slicing)
        # Ensure dimensions don't go below 1
        stride = level_scale_factor**level
        next_ti = TI[::stride, ::stride, ::stride]
        # Handle potential empty slices if TI is small
        if next_ti.size == 0:
             raise ValueError(f"TI becomes empty at level {level}. Check TI size and n_levels.")

        TI_at_level.append(next_ti)
        grid_dims_at_level.append((next_nx, next_ny, next_nz))

        current_nx, current_ny, current_nz = next_nx, next_ny, next_nz

    # Reverse lists to go from coarsest (index -1) to finest (index 0)
    TI_at_level.reverse()
    grid_dims_at_level.reverse()

    # --- 2. Multi-Scale Simulation Loop ---
    realization_coarse = None # Start with no conditioning from coarser levels

    # Loop from coarsest level (level_idx = 0) up to finest (level_idx = n_levels - 1)
    for level_idx in range(n_levels):
        current_level = n_levels - 1 - level_idx # Actual scale level (0=finest)
        nx, ny, nz = grid_dims_at_level[level_idx]
        current_ti = TI_at_level[level_idx]

        if verbose:
            print(f"\n--- Simulating Level {current_level} (Grid: {nx}x{ny}x{nz}) ---")
            print(f"--- Using TI shape: {current_ti.shape} ---")


        # Prepare hard data for this level
        current_hard_data = np.full((nx, ny, nz), -1, dtype=np.int32)

        # Condition with simulation from the previous (coarser) level
        if realization_coarse is not None:
            # Upsample coarse realization to act as hard data points
            # Place coarse points at the start of each block in the finer grid
            rz_c, ry_c, rx_c = realization_coarse.shape
            temp_hd = np.full((nz, ny, nx), -1, dtype=np.int32) # Use ZYX order for easier slicing
            temp_hd[:rz_c, :ry_c, :rx_c] = realization_coarse # Assign coarse data

            # Create slices to map coarse grid onto current grid
            z_slice = slice(None, nz, level_scale_factor)
            y_slice = slice(None, ny, level_scale_factor)
            x_slice = slice(None, nx, level_scale_factor)

            # Use advanced indexing to get target locations - more robust
            zg, yg, xg = np.meshgrid(np.arange(nz)[z_slice],
                                     np.arange(ny)[y_slice],
                                     np.arange(nx)[x_slice], indexing='ij')

            # Ensure we don't try to write outside the bounds of temp_hd
            valid_mask = (zg < rz_c) & (yg < ry_c) & (xg < rx_c)
            current_hard_data[zg[valid_mask], yg[valid_mask], xg[valid_mask]] = temp_hd[zg[valid_mask], yg[valid_mask], xg[valid_mask]]


        # Condition with original hard data if at the finest level
        if level_idx == n_levels - 1 and hard_data is not None:
             if hard_data.shape != (nx, ny, nz): # Should match finest grid dims
                 raise ValueError("Original hard data shape mismatch at finest level.")
             # Combine: original hard data overrides coarser simulation where available
             current_hard_data = np.where(hard_data != -1, hard_data.astype(np.int32), current_hard_data)


        # Perform simulation at the current level
        realization_fine = multi_points_modeling(
            TI=current_ti,
            template_shape=template_shape,
            random_seed=random_seed + level_idx, # Vary seed per level slightly
            real_nx=nx, real_ny=ny, real_nz=nz,
            hard_data=current_hard_data, # Use the combined hard data
            verbose=verbose,
            training_fraction=training_fraction,
            use_cross_stencil=use_cross_stencil
        )

        # Prepare for the next level (finer grid)
        if len(realization_fine.shape) != 3:
            realization_coarse = realization_fine.reshape((realization_fine.shape[0], realization_fine.shape[1], 1))
        else:
            realization_coarse = realization_fine # This result conditions the next level

    # The final realization is the one from the last iteration (finest level)
    return realization_fine


# --- Example Usage (Illustrative) ---
if __name__ == "__main__":

    # Create a dummy 3D Training Image (e.g., 100x100x20 with 3 facies)
    print("Generating dummy Training Image...")
    ti_shape = (50, 50, 10)
    dummy_ti = np.random.randint(0, 3, size=ti_shape, dtype=np.int32)
    # Add some structure maybe
    dummy_ti[10:20, :, :] = 0
    dummy_ti[:, 20:30, :] = 1
    dummy_ti[:, :, 5:] = 2
    dummy_ti[25:35, 25:35, :5] = 0
    print(f"TI Shape: {dummy_ti.shape}, Facies: {np.unique(dummy_ti)}")

    # --- Test curate_training_image ---
    print("\nTesting curate_training_image_vectorized...")
    template = (5, 5, 3)
    dx, dy, feat_idx = curate_training_image_vectorized(dummy_ti, template, fraction_to_use=0.5, use_cross_stencil=True)
    print(f"data_x shape: {dx.shape}") # (n_samples, n_features)
    print(f"data_y shape: {dy.shape}") # (n_samples, 1)
    print(f"Feature indices count: {len(feat_idx)}")

    # --- Test single-grid MPS ---
    print("\nTesting multi_points_modeling (single grid)...")
    real_dims = (30, 30, 5)
    realization_single = multi_points_modeling(
        TI=dummy_ti,
        template_shape=template,
        random_seed=42,
        real_nx=real_dims[0], real_ny=real_dims[1], real_nz=real_dims[2],
        hard_data=None, # Example: could create dummy hard data here
        verbose=True,
        training_fraction=0.8 # Use 80% of patterns
    )
    print(f"Single-grid realization shape: {realization_single.shape}")
    print(f"Realization facies counts: {np.unique(realization_single, return_counts=True)}")


    # --- Test multi-scale MPS ---
    print("\nTesting multi_points_modeling_multi_scaled...")
    real_dims_ms = (40, 40, 8) # Potentially different dimensions for multi-scale test
    # Create some dummy hard data for the finest grid
    hard_data_ms = np.full(real_dims_ms, -1, dtype=np.int32)
    hard_data_ms[::5, ::5, :] = np.random.randint(0, 3, size=hard_data_ms[::5, ::5, :].shape) # Some points known

    realization_multi = multi_points_modeling_multi_scaled(
        TI=dummy_ti,
        n_levels=3, # Coarse, Medium, Fine
        level_scale_factor=2,
        template_shape=template,
        random_seed=123,
        real_nx=real_dims_ms[0], real_ny=real_dims_ms[1], real_nz=real_dims_ms[2],
        hard_data=hard_data_ms,
        verbose=True,
        training_fraction=0.8
    )
    print(f"Multi-scale realization shape: {realization_multi.shape}")
    print(f"Multi-scale realization facies counts: {np.unique(realization_multi, return_counts=True)}")

    # Verify hard data conditioning (optional check)
    if hard_data_ms is not None:
         mismatches = np.sum((realization_multi != hard_data_ms) & (hard_data_ms != -1))
         print(f"Hard data mismatches: {mismatches}") # Should be 0