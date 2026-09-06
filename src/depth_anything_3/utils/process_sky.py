import numpy as np
import cv2

def set_sky_regions_to_max_depth_np(depth: np.ndarray, depth_conf: np.ndarray | None, non_sky_mask: np.ndarray, max_depth: float = 200.0,) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Set sky regions to maximum depth and high confidence using pure NumPy.

    Args:
        depth: Depth array (N, H, W) or (H, W)
        depth_conf: Optional depth confidence array (N, H, W) or (H, W)
        non_sky_mask: Boolean mask where True indicates non-sky regions
        max_depth: Maximum depth value for sky regions

    Returns:
        Tuple of (updated_depth, updated_depth_conf)
    """
    depth = depth.copy()
    sky_mask = ~non_sky_mask

    # Set sky regions to max_depth
    depth[sky_mask] = max_depth

    if depth_conf is not None:
        depth_conf = depth_conf.copy()
        depth_conf[sky_mask] = 1.0
        return depth, depth_conf
    else:
        return depth, None


def process_mono_sky_estimation_np(depth: np.ndarray, depth_conf: np.ndarray | None, sky: np.ndarray | None, sky_threshold: float = 0.3,) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Process mono sky estimation in NumPy.
        """
        if sky is None:
            print("No sky estimation found in outputs. Skipping sky processing.")
            return depth, depth_conf

        # non_sky_mask is True where sky prediction is below threshold
        non_sky_mask = sky < sky_threshold

        # Guard against dynamic shape / small pixel count edge cases
        if np.sum(non_sky_mask) <= 10 or np.sum(~non_sky_mask) <= 10:
            return depth, depth_conf

        non_sky_depth = depth[non_sky_mask]
        
        # Subsample if large (replicates torch.randint)
        if non_sky_depth.size > 100000:
            idx = np.random.choice(non_sky_depth.size, size=100000, replace=False)
            sampled_depth = non_sky_depth[idx]
        else:
            sampled_depth = non_sky_depth

        # Replicates torch.quantile(sampled_depth, 0.99)
        non_sky_max = float(np.percentile(sampled_depth, 99))

        # Apply sky depth and confidence updates
        updated_depth, updated_conf = set_sky_regions_to_max_depth_np(depth=depth, depth_conf=depth_conf, non_sky_mask=non_sky_mask, max_depth=non_sky_max,)

        return updated_depth, updated_conf
