"""Feature extraction module for converting images to feature vectors."""

import numpy as np
from PIL import Image
import config

def extract_lbp_features(image):
    """Extract Local Binary Pattern features for texture analysis."""
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = np.dot(image[...,:3], [0.299, 0.587, 0.114]).astype(np.uint8)
    else:
        gray = image.astype(np.uint8)
    
    rows, cols = gray.shape
    lbp_image = np.zeros((rows-2, cols-2), dtype=np.uint8)
    
    # LBP computation (simplified 8-neighbor version)
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            center = gray[i, j]
            code = 0
            # 8 neighbors clockwise
            code |= (gray[i-1, j-1] >= center) << 7
            code |= (gray[i-1, j] >= center) << 6
            code |= (gray[i-1, j+1] >= center) << 5
            code |= (gray[i, j+1] >= center) << 4
            code |= (gray[i+1, j+1] >= center) << 3
            code |= (gray[i+1, j] >= center) << 2
            code |= (gray[i+1, j-1] >= center) << 1
            code |= (gray[i, j-1] >= center) << 0
            lbp_image[i-1, j-1] = code
    
    # Create histogram of LBP values
    hist, _ = np.histogram(lbp_image.flatten(), bins=64, range=(0, 256))
    hist = hist.astype(np.float32)
    hist = hist / (hist.sum() + 1e-7)
    
    return hist

def extract_hog_features(image):
    """Extract HOG (Histogram of Oriented Gradients) features using manual implementation."""
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = np.dot(image[...,:3], [0.299, 0.587, 0.114]).astype(np.uint8)
    else:
        gray = image
    
    # Compute gradients
    gx = np.zeros_like(gray, dtype=np.float32)
    gy = np.zeros_like(gray, dtype=np.float32)
    gx[:, :-1] = np.diff(gray, axis=1).astype(np.float32)
    gy[:-1, :] = np.diff(gray, axis=0).astype(np.float32)
    
    # Compute gradient magnitude and orientation
    magnitude = np.sqrt(gx**2 + gy**2)
    orientation = np.arctan2(gy, gx) * (180 / np.pi) % 180
    
    # HOG parameters
    cell_size = config.HOG_PIXELS_PER_CELL[0]
    block_size = config.HOG_CELLS_PER_BLOCK[0]
    n_bins = config.HOG_ORIENTATIONS
    
    # Calculate cells
    n_cells_x = gray.shape[1] // cell_size
    n_cells_y = gray.shape[0] // cell_size
    
    # Compute histogram for each cell
    cell_histograms = np.zeros((n_cells_y, n_cells_x, n_bins))
    
    for i in range(n_cells_y):
        for j in range(n_cells_x):
            cell_mag = magnitude[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]
            cell_ori = orientation[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]
            
            # Create histogram
            hist, _ = np.histogram(cell_ori, bins=n_bins, range=(0, 180), weights=cell_mag)
            cell_histograms[i, j, :] = hist
    
    # Block normalization
    n_blocks_x = n_cells_x - block_size + 1
    n_blocks_y = n_cells_y - block_size + 1
    
    normalized_blocks = []
    for i in range(n_blocks_y):
        for j in range(n_blocks_x):
            block = cell_histograms[i:i+block_size, j:j+block_size, :].flatten()
            # L2 normalization
            norm = np.linalg.norm(block) + 1e-5
            normalized_blocks.append(block / norm)
    
    return np.concatenate(normalized_blocks) if normalized_blocks else np.zeros(100)

def extract_color_histogram(image):
    """Extract color histogram features from RGB and HSV color spaces."""
    hist_features = []
    
    # RGB histogram
    for i in range(3):  # RGB channels
        channel = image[:, :, i].flatten()
        hist, _ = np.histogram(channel, bins=config.COLOR_HIST_BINS, range=(0, 256))
        hist = hist.astype(np.float32)
        hist = hist / (hist.sum() + 1e-7)  # Normalize
        hist_features.extend(hist)
    
    # Convert to HSV-like representation (simplified)
    r, g, b = image[:,:,0].astype(np.float32), image[:,:,1].astype(np.float32), image[:,:,2].astype(np.float32)
    
    max_val = np.maximum(np.maximum(r, g), b)
    min_val = np.minimum(np.minimum(r, g), b)
    diff = max_val - min_val + 1e-7
    
    # Value (brightness)
    v = max_val
    
    # Saturation
    s = np.where(max_val > 0, diff / (max_val + 1e-7), 0)
    
    # Hue (simplified)
    h = np.zeros_like(max_val)
    mask_r = (max_val == r)
    mask_g = (max_val == g) & ~mask_r
    mask_b = ~mask_r & ~mask_g
    
    h[mask_r] = 60 * (((g[mask_r] - b[mask_r]) / diff[mask_r]) % 6)
    h[mask_g] = 60 * (((b[mask_g] - r[mask_g]) / diff[mask_g]) + 2)
    h[mask_b] = 60 * (((r[mask_b] - g[mask_b]) / diff[mask_b]) + 4)
    
    # HSV histograms
    h_hist, _ = np.histogram(h.flatten(), bins=32, range=(0, 360))
    s_hist, _ = np.histogram((s * 255).flatten(), bins=32, range=(0, 256))
    v_hist, _ = np.histogram(v.flatten(), bins=32, range=(0, 256))
    
    for hist in [h_hist, s_hist, v_hist]:
        hist = hist.astype(np.float32)
        hist = hist / (hist.sum() + 1e-7)
        hist_features.extend(hist)
    
    return np.array(hist_features)

def extract_texture_features(image):
    """Extract texture features using edge detection and gradient analysis."""
    gray = np.dot(image[...,:3], [0.299, 0.587, 0.114]).astype(np.float32)
    
    # Sobel edge detection
    sobelx = np.abs(np.diff(gray, axis=1, prepend=0))
    sobely = np.abs(np.diff(gray, axis=0, prepend=0))
    edge_magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    features = [
        np.mean(sobelx),
        np.std(sobelx),
        np.mean(sobely),
        np.std(sobely),
        np.mean(edge_magnitude),
        np.std(edge_magnitude),
        np.percentile(edge_magnitude, 25),
        np.percentile(edge_magnitude, 50),
        np.percentile(edge_magnitude, 75),
        np.percentile(edge_magnitude, 90),
    ]
    
    # Edge histogram (8 directions)
    edge_hist, _ = np.histogram(edge_magnitude.flatten(), bins=16, range=(0, 255))
    edge_hist = edge_hist.astype(np.float32)
    edge_hist = edge_hist / (edge_hist.sum() + 1e-7)
    features.extend(edge_hist)
    
    return np.array(features)

def extract_statistical_features(image):
    """Extract comprehensive statistical features from image."""
    features = []
    
    # Per-channel statistics
    for i in range(3):
        channel = image[:, :, i].astype(np.float32)
        features.extend([
            np.mean(channel),
            np.std(channel),
            np.median(channel),
            np.percentile(channel, 10),
            np.percentile(channel, 90),
            np.min(channel),
            np.max(channel),
        ])
    
    # Overall grayscale statistics
    gray = np.dot(image[...,:3], [0.299, 0.587, 0.114]).astype(np.float32)
    features.extend([
        np.mean(gray),
        np.std(gray),
        np.median(gray),
        np.percentile(gray, 10),
        np.percentile(gray, 90),
    ])
    
    # Color ratios (useful for material identification)
    eps = 1e-7
    r_mean = np.mean(image[:,:,0])
    g_mean = np.mean(image[:,:,1])
    b_mean = np.mean(image[:,:,2])
    total = r_mean + g_mean + b_mean + eps
    
    features.extend([
        r_mean / (g_mean + eps),  # R/G ratio
        r_mean / (b_mean + eps),  # R/B ratio
        g_mean / (b_mean + eps),  # G/B ratio
        r_mean / total,           # R proportion
        g_mean / total,           # G proportion
        b_mean / total,           # B proportion
    ])
    
    # Color variance features
    features.extend([
        np.var(image[:,:,0]),
        np.var(image[:,:,1]),
        np.var(image[:,:,2]),
    ])
    
    return np.array(features)

def extract_gabor_features(image):
    """Extract fast approximated Gabor-like texture features."""
    gray = np.dot(image[...,:3], [0.299, 0.587, 0.114]).astype(np.float32)
    
    # Fast approximation: directional edge responses at different scales
    features = []
    
    # Horizontal gradients (approximates 0° and 180°)
    grad_h = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    features.extend([np.mean(grad_h), np.std(grad_h)])
    
    # Vertical gradients (approximates 90° and 270°)
    grad_v = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
    features.extend([np.mean(grad_v), np.std(grad_v)])
    
    # Diagonal gradients (approximates 45° and 135°)
    h, w = gray.shape
    diag1 = np.abs(gray[1:, 1:] - gray[:-1, :-1])
    diag2 = np.abs(gray[1:, :-1] - gray[:-1, 1:])
    features.extend([np.mean(diag1), np.std(diag1), np.mean(diag2), np.std(diag2)])
    
    # Multi-scale responses (downsample and repeat)
    small = gray[::2, ::2]  # Half resolution
    grad_h_small = np.abs(np.diff(small, axis=1, prepend=small[:, :1]))
    grad_v_small = np.abs(np.diff(small, axis=0, prepend=small[:1, :]))
    features.extend([np.mean(grad_h_small), np.mean(grad_v_small)])
    
    return np.array(features)

def extract_spatial_features(image):
    """Extract spatial arrangement features with pyramid representation."""
    gray = np.dot(image[...,:3], [0.299, 0.587, 0.114]).astype(np.float32)
    
    h, w = gray.shape
    features = []
    
    # Spatial pyramid: 1x1, 2x2, 4x4 grids
    for grid_size in [1, 2, 4]:
        grid_h, grid_w = h // grid_size, w // grid_size
        for i in range(grid_size):
            for j in range(grid_size):
                region = gray[i*grid_h:(i+1)*grid_h, j*grid_w:(j+1)*grid_w]
                features.extend([
                    np.mean(region),
                    np.std(region),
                    np.percentile(region, 50)
                ])
    
    # Center vs edge brightness ratio
    center = gray[h//4:3*h//4, w//4:3*w//4]
    edge_top = gray[:h//4, :]
    edge_bottom = gray[3*h//4:, :]
    edge_left = gray[:, :w//4]
    edge_right = gray[:, 3*w//4:]
    
    center_mean = np.mean(center)
    edge_mean = (np.mean(edge_top) + np.mean(edge_bottom) + np.mean(edge_left) + np.mean(edge_right)) / 4
    
    features.append(center_mean / (edge_mean + 1e-7))
    features.append(np.std(center) / (np.std(gray) + 1e-7))
    
    # Diagonal features
    h_mid, w_mid = h // 2, w // 2
    top_left = gray[:h_mid, :w_mid]
    top_right = gray[:h_mid, w_mid:]
    bottom_left = gray[h_mid:, :w_mid]
    bottom_right = gray[h_mid:, w_mid:]
    
    features.extend([
        np.mean(top_left), np.mean(top_right),
        np.mean(bottom_left), np.mean(bottom_right)
    ])
    
    return np.array(features)

def extract_features(image):
    """
    Extract optimized feature vector for fast, accurate classification.
    
    Uses only the most important features:
    - HOG: Shape and edge information (~1500 features)
    - LBP: Texture patterns (64 features)
    - Color histograms: RGB and HSV distributions (288 features)
    Total: ~1850 features (optimized for speed and accuracy)
    """
    # Extract core feature types (most discriminative)
    hog_features = extract_hog_features(image)
    lbp_features = extract_lbp_features(image)
    color_features = extract_color_histogram(image)
    
    # Concatenate features into single vector
    feature_vector = np.concatenate([
        hog_features, 
        lbp_features,
        color_features
    ])
    
    return feature_vector

def preprocess_image(image):
    """Preprocess image before feature extraction."""
    # Resize to standard size using PIL
    if isinstance(image, np.ndarray):
        pil_img = Image.fromarray(image.astype('uint8'))
    else:
        pil_img = image
    
    pil_img = pil_img.resize(config.IMAGE_SIZE, Image.Resampling.LANCZOS)
    image = np.array(pil_img)
    
    # Simple blur using convolution
    kernel = np.ones((3, 3), np.float32) / 9
    if len(image.shape) == 3:
        for i in range(3):
            from scipy.ndimage import convolve
            try:
                image[:, :, i] = convolve(image[:, :, i].astype(np.float32), kernel, mode='reflect')
            except:
                pass  # Skip blur if scipy not available
    
    return image

if __name__ == "__main__":
    # Test feature extraction
    test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    features = extract_features(test_img)
    print(f"Feature vector shape: {features.shape}")
    print(f"Feature vector length: {len(features)}")
