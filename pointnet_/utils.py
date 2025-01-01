import torch

from einops import repeat, rearrange


def exists(x):
    return x is not None


def default(*vals):
    for val in vals:
        if exists(val):
            return val

<<<<<<< HEAD

def farthest_point_sampling(x: torch.Tensor, n_sample: int, start_idx: int = None):
    # x: (b, n, 3)
    print("FPS input shape:", x.shape)
    print("n_sample:", n_sample)
    b, n = x.shape[:2]
    assert n_sample <= n, "not enough points to sample"
    b, n = x.shape[:2]
    assert n_sample <= n, "not enough points to sample"

    if n_sample == n:
        return repeat(torch.arange(n_sample, dtype=torch.long, device=x.device), 'm -> b m', b=b)

    # start index
    if exists(start_idx):
        sel_idx = torch.full((b, n_sample), start_idx, dtype=torch.long, device=x.device)
    else:
        sel_idx = torch.randint(n, (b, n_sample), dtype=torch.long, device=x.device)

    cur_x = rearrange(x[torch.arange(b), sel_idx[:, 0]], 'b c -> b 1 c')
    min_dists = torch.full((b, n), dtype=x.dtype, device=x.device, fill_value=float('inf'))
    for i in range(1, n_sample):
        # update distance
        dists = torch.linalg.norm(x - cur_x, dim=-1)
        min_dists = torch.minimum(dists, min_dists)

        # take the farthest
        idx_farthest = torch.max(min_dists, dim=-1).indices
        sel_idx[:, i] = idx_farthest
        cur_x[:, 0, :] = x[torch.arange(b), idx_farthest]

    return sel_idx


def ball_query_pytorch(src, query, radius, k):
=======
# Original Points there 
# def farthest_point_sampling(x: torch.Tensor, n_sample: int, start_idx: int = None):
#     # x: (b, n, 3)
#     b, n = x.shape[:2]
#     assert n_sample <= n, "not enough points to sample"

#     if n_sample == n:
#         return repeat(torch.arange(n_sample, dtype=torch.long, device=x.device), 'm -> b m', b=b)

#     # start index
#     if exists(start_idx):
#         sel_idx = torch.full((b, n_sample), start_idx, dtype=torch.long, device=x.device)
#     else:
#         sel_idx = torch.randint(n, (b, n_sample), dtype=torch.long, device=x.device)

#     cur_x = rearrange(x[torch.arange(b), sel_idx[:, 0]], 'b c -> b 1 c')
#     min_dists = torch.full((b, n), dtype=x.dtype, device=x.device, fill_value=float('inf'))
#     for i in range(1, n_sample):
#         # update distance
#         dists = torch.linalg.norm(x - cur_x, dim=-1)
#         min_dists = torch.minimum(dists, min_dists)

#         # take the farthest
#         idx_farthest = torch.max(min_dists, dim=-1).indices
#         sel_idx[:, i] = idx_farthest
#         cur_x[:, 0, :] = x[torch.arange(b), idx_farthest]

#     return sel_idx

### new Farthest function
def farthest_point_sampling(x, n_sample, start_idx=0):
    """
    Perform farthest point sampling on a batch of point clouds.

    Parameters:
    - x (numpy.ndarray): Input points of shape (b, n, 3), where `b` is batch size, `n` is number of points.
    - n_sample (int): Number of points to sample.
    - start_idx (int): Starting index for sampling (default: 0).

    Returns:
    - numpy.ndarray: Indices of sampled points of shape (b, n_sample).
    """
    b, n = x.shape[:2]

    # Handle case where n_sample exceeds available points
    if n_sample > n:
        print(f"Warning: Requested {n_sample} points, but only {n} available. Using all points instead.")
        n_sample = n  # Use all points if not enough points are available

    # Initialize arrays
    idx = np.zeros((b, n_sample), dtype=np.int32)  # To store sampled indices
    distances = np.ones((b, n), dtype=np.float32) * 1e10  # Max distances
    farthest = np.random.randint(0, n, size=(b,))  # Randomly initialize farthest point

    # Perform sampling
    for i in range(n_sample):
        idx[:, i] = farthest
        centroid = x[np.arange(b), farthest, :].reshape(b, 1, -1)  # Selected farthest point
        dist = np.sum((x - centroid) ** 2, axis=-1)  # Compute squared distances
        distances = np.minimum(distances, dist)  # Update minimum distances
        farthest = np.argmax(distances, axis=-1)  # Select next farthest point

    return idx



# def ball_query_pytorch(src, query, radius, k):
#     # src: (b, n, 3)
#     # query: (b, m, 3)
#     b, n = src.shape[:2]
#     m = query.shape[1]
#     dists = torch.cdist(query, src)  # (b, m, n)
#     idx = repeat(torch.arange(n, device=src.device), 'n -> b m n', b=b, m=m)
#     idx = torch.where(dists > radius, n, idx)
#     idx = idx.sort(dim=-1).values[:, :, :k]  # (b, m, k)
#     idx = torch.where(idx == n, idx[:, :, [0]], idx)
#     _dists = dists.gather(-1, idx)  # (b, m, k)
#     return idx, _dists

#My Debug modification one 
def ball_query_pytorch(src, query, radius, k):
    """
    Ball query in PyTorch: Finds the indices of the nearest neighbors within a given radius.

    Args:
        src: Source points of shape (b, n, 3).
        query: Query points of shape (b, m, 3).
        radius: Radius within which neighbors are considered.
        k: Maximum number of neighbors to find.

    Returns:
        idx: Indices of the neighbors (b, m, k).
        _dists: Distances to the neighbors (b, m, k).
    """
>>>>>>> 922317e4f08b7622d352f933e83e9f2d64a3f308
    # src: (b, n, 3)
    # query: (b, m, 3)
    b, n = src.shape[:2]
    m = query.shape[1]
<<<<<<< HEAD
    dists = torch.cdist(query, src)  # (b, m, n)
    idx = repeat(torch.arange(n, device=src.device), 'n -> b m n', b=b, m=m)
    idx = torch.where(dists > radius, n, idx)
    idx = idx.sort(dim=-1).values[:, :, :k]  # (b, m, k)
    idx = torch.where(idx == n, idx[:, :, [0]], idx)
    _dists = dists.gather(-1, idx)  # (b, m, k)
    return idx, _dists
=======

    # Compute pairwise distances
    dists = torch.cdist(query, src)  # (b, m, n)

    # Create index array
    idx = repeat(torch.arange(n, device=src.device), 'n -> b m n', b=b, m=m)

    # Mask distances greater than radius
    idx = torch.where(dists > radius, n, idx)

    # Sort indices by distance and select the top-k neighbors
    idx = idx.sort(dim=-1).values[:, :, :k]  # (b, m, k)

    # Handle case where fewer than k neighbors are found
    idx = torch.where(idx == n, idx[:, :, [0]], idx)  # Fallback to the first index if no valid neighbors

    # Validate index bounds
    idx = torch.clamp(idx, min=0, max=n - 1)

    # Debugging: Ensure indices are within valid range
    assert (idx >= 0).all() and (idx < n).all(), "Index out of bounds in ball_query_pytorch"

    # Gather distances for the selected indices
    _dists = dists.gather(-1, idx)  # (b, m, k)

    return idx, _dists

>>>>>>> 922317e4f08b7622d352f933e83e9f2d64a3f308
