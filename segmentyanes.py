import cv2 as cv
import imageio
from sklearn.cluster import KMeans
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import NMF
from skimage.feature import peak_local_max
import os
import time

# ============================================================================
# PREPROCESSING FUNCTIONS (dari Kode 2)
# ============================================================================

def convert_hsv_circular(image_rgb, v_thresh=20):
    """
    Convert RGB to HSV and return hsv image + mask for non-dark pixels.
    
    Args:
        image_rgb: RGB uint8 image
        v_thresh: threshold on V channel (0-255) to exclude dark background
    
    Returns:
        hsv_image: HSV image
        mask: boolean mask where V > v_thresh
    """
    hsv_image = cv.cvtColor(image_rgb, cv.COLOR_RGB2HSV)
    v = hsv_image[:, :, 2]
    mask = v > v_thresh
    return hsv_image, mask

def apply_median_filter(img_rgb, kernel_size=3):
    """Step 1: Median Filter untuk mengurangi noise"""
    img_denoised = cv.medianBlur(img_rgb, kernel_size)
    return img_denoised

def rgb2od(I):
    """Helper: convert RGB (0..255) to Optical Density (OD)"""
    I = I.astype(np.float32)
    I = np.clip(I, 1.0, 255.0)
    return -np.log(I / 255.0)

def od2rgb(OD):
    """Helper: convert OD back to RGB"""
    I = np.exp(-OD) * 255.0
    I = np.clip(I, 0, 255).astype(np.uint8)
    return I

def get_stain_matrix_macenko(I, beta=0.15, alpha=1.0):
    """Macenko stain matrix estimator (robustified)"""
    OD = rgb2od(I).reshape((-1, 3))
    mask = (OD.max(axis=1) > beta)
    ODhat = OD[mask]
    if ODhat.shape[0] < 10:
        raise ValueError("Not enough stained pixels found. Try lowering beta or use a different method.")
    _, _, Vt = np.linalg.svd(ODhat.T, full_matrices=False)
    V = Vt[:2, :].T
    projected = np.dot(ODhat, V)
    angles = np.arctan2(projected[:, 1], projected[:, 0])
    low, high = np.percentile(angles, [alpha, 100.0 - alpha])
    v1 = np.dot(V, [np.cos(low), np.sin(low)])
    v2 = np.dot(V, [np.cos(high), np.sin(high)])
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    W = np.vstack((v1, v2)).T
    return W

def get_concentrations_nnls(W, OD_flat):
    """Compute concentrations (enforce non-negativity)"""
    C, _, _, _ = np.linalg.lstsq(W, OD_flat, rcond=None)
    C = np.clip(C, 0, None)
    return C

def macenko_normalize(source_rgb, target_rgb, beta=0.15, alpha=1.0):
    """Macenko normalization with robustness"""
    Ws = get_stain_matrix_macenko(source_rgb, beta=beta, alpha=alpha)
    Wt = get_stain_matrix_macenko(target_rgb, beta=beta, alpha=alpha)
    h, w, _ = source_rgb.shape
    OD_s = rgb2od(source_rgb).reshape((-1, 3)).T
    Cs = get_concentrations_nnls(Ws, OD_s)
    OD_t = rgb2od(target_rgb).reshape((-1, 3)).T
    Ct = get_concentrations_nnls(Wt, OD_t)
    maxCt = np.percentile(Ct, 99, axis=1)
    maxCs = np.percentile(Cs, 99, axis=1)
    scaling = (maxCt / (maxCs + 1e-8))[:, np.newaxis]
    Cs_scaled = Cs * scaling
    OD_norm_flat = np.dot(Wt, Cs_scaled)
    OD_norm = OD_norm_flat.T.reshape((h, w, 3))
    img_norm_local = od2rgb(OD_norm)
    return img_norm_local

def nmf_normalize(source_rgb, target_rgb, n_components=2):
    """NMF-based fallback (Vahadane-like)"""
    h, w, _ = source_rgb.shape
    OD_s = rgb2od(source_rgb).reshape((-1, 3))
    OD_t = rgb2od(target_rgb).reshape((-1, 3))
    mask_s = OD_s.max(axis=1) > 0.15
    mask_t = OD_t.max(axis=1) > 0.15
    OD_s_hat = OD_s[mask_s]
    OD_t_hat = OD_t[mask_t]
    if OD_s_hat.shape[0] < 10 or OD_t_hat.shape[0] < 10:
        raise ValueError("Not enough stained pixels for NMF fallback.")
    nmf_t = NMF(n_components=n_components, init='nndsvda', random_state=0, max_iter=500)
    Ht = nmf_t.fit_transform(OD_t_hat)
    Wt = nmf_t.components_
    nmf_s = NMF(n_components=n_components, init='nndsvda', random_state=0, max_iter=500)
    Hs = nmf_s.fit_transform(OD_s_hat)
    Ws = nmf_s.components_
    maxHt = np.percentile(Ht, 99, axis=0)
    maxHs = np.percentile(Hs, 99, axis=0)
    scale = (maxHt / (maxHs + 1e-8))
    Hs_full = nmf_s.transform(OD_s)
    Hs_scaled = Hs_full * scale
    OD_norm = np.dot(Hs_scaled, Wt)
    OD_norm_img = OD_norm.reshape((h, w, 3))
    img_norm_local = od2rgb(OD_norm_img)
    return img_norm_local

def apply_macenko_normalization(img_denoised, ref_img_rgb):
    """Step 2: Macenko Stain Normalization with fallback to NMF"""
    img_macenko = None
    last_error = None
    
    for beta in (0.15, 0.10, 0.05):
        for alpha in (1.0, 5.0, 10.0):
            try:
                img_macenko = macenko_normalize(img_denoised, ref_img_rgb, beta=beta, alpha=alpha)
                last_error = None
                break
            except Exception as e:
                last_error = e
        if img_macenko is not None:
            break
    
    if img_macenko is None:
        try:
            img_macenko = nmf_normalize(img_denoised, ref_img_rgb, n_components=2)
            last_error = None
        except Exception as e:
            last_error = e
    
    if img_macenko is None:
        print(f"⚠️ Normalization failed: {last_error}. Using denoised image.")
        return img_denoised
    
    return img_macenko

def apply_clahe(img_macenko, clip_limit=2.0, tile_grid_size=(8, 8)):
    """Step 3: CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
    lab = cv.cvtColor(img_macenko, cv.COLOR_RGB2LAB)
    L, A, B = cv.split(lab)
    clahe = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    L_cl = clahe.apply(L)
    lab_cl = cv.merge((L_cl, A, B))
    img_clahe = cv.cvtColor(lab_cl, cv.COLOR_LAB2RGB)
    return img_clahe

def apply_log_enhancement(img_clahe, sigma=1.5, alpha=0.5):
    """Step 4: LoG (Laplacian of Gaussian) with Edge Enhancement"""
    img_clahe_float = img_clahe.astype(np.float32) / 255.0
    ksize = int(2 * np.ceil(3 * sigma) + 1)
    
    img_log_edges = np.zeros_like(img_clahe_float)
    for i in range(3):
        blurred = cv.GaussianBlur(img_clahe_float[:,:,i], (ksize, ksize), sigma)
        laplacian = cv.Laplacian(blurred, cv.CV_32F, ksize=3)
        img_log_edges[:,:,i] = laplacian
    
    img_sharpened = img_clahe_float - alpha * img_log_edges
    img_sharpened = np.clip(img_sharpened, 0, 1)
    img_log_result = (img_sharpened * 255).astype(np.uint8)
    
    return img_log_result

def preprocess_image(img_rgb, ref_img_rgb=None):
    """Complete preprocessing pipeline for blood smear images"""
    if ref_img_rgb is None:
        ref_img_rgb = img_rgb.copy()
    
    img_denoised = apply_median_filter(img_rgb, kernel_size=3)
    img_macenko = apply_macenko_normalization(img_denoised, ref_img_rgb)
    img_clahe = apply_clahe(img_macenko, clip_limit=2.0, tile_grid_size=(8, 8))
    img_preprocessed = apply_log_enhancement(img_clahe, sigma=1.5, alpha=0.5)
    
    return img_preprocessed

# ============================================================================
# K-MEANS CLUSTERING (simplified from Kode 1)
# ============================================================================

def kmeans_segmentation(image, k, use_preprocessing=True, v_thresh=20):
    """
    Perform K-means segmentation on the image with optional preprocessing.
    
    Args:
        image: Input image (HSV or RGB)
        k: Number of clusters
        use_preprocessing: Whether to apply preprocessing pipeline
        v_thresh: Threshold on V channel to exclude dark background
    
    Returns:
        segmented_images: List of segmented images for each cluster (HSV format)
        labels: Cluster labels for each pixel
    """
    if use_preprocessing:
        print("🔬 Applying preprocessing pipeline...")
        img_rgb = cv.cvtColor(image, cv.COLOR_HSV2RGB)
        img_preprocessed = preprocess_image(img_rgb, ref_img_rgb=None)
        print("✅ Preprocessing completed!")
    else:
        img_rgb = cv.cvtColor(image, cv.COLOR_HSV2RGB)
        img_preprocessed = img_rgb
    
    hsv_preprocessed = cv.cvtColor(img_preprocessed, cv.COLOR_RGB2HSV)
    pixels = hsv_preprocessed.reshape(-1, 3)
    
    print(f"🎯 Running K-Means clustering with k={k}...")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(pixels)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    
    segmented_images = []
    for i in range(k):
        cluster_image = np.zeros_like(pixels)
        cluster_image[labels == i] = centers[i]
        segmented_image = cluster_image.reshape(hsv_preprocessed.shape)
        segmented_image = np.clip(segmented_image, 0, 255).astype(np.uint8)
        segmented_images.append(segmented_image)
    
    print("✅ Clustering completed!")
    return segmented_images, labels

# ============================================================================
# REMOVE UNWANTED CELLS (improved version)
# ============================================================================

def bounded_opening(image, num_openings=3):
    """Apply bounded opening with increasing kernel size"""
    kernel_size = 5
    processed = image.copy()
    
    for _ in range(num_openings):
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        processed = cv.morphologyEx(processed, cv.MORPH_OPEN, kernel)
        kernel_size += 2
    
    return processed

def remove_unwanted_cells(clustered_image, selected_cluster, image):
    """
    Remove unwanted cells based on selected clusters (ORIGINAL FUNCTION - KEPT FOR COMPATIBILITY).
    
    Args:
        clustered_image: List of clustered images (HSV format)
        selected_cluster: List of selected cluster indices
        image: Original RGB image (preprocessed)
    
    Returns:
        rbc_only_image: Image with only RBC cells
    """
    if not selected_cluster:
        raise ValueError("No clusters selected. Please select at least one cluster.")
    
    # Combine selected clusters
    segmented_mask = clustered_image[selected_cluster[0]].copy()
    for index_cluster in selected_cluster[1:]:
        segmented_mask = cv.bitwise_or(segmented_mask, clustered_image[index_cluster])
    
    # Convert HSV mask to RGB then to grayscale
    segmented_mask = cv.cvtColor(segmented_mask, cv.COLOR_HSV2RGB)
    segmented_mask = cv.cvtColor(segmented_mask, cv.COLOR_RGB2GRAY)
    _, binary_mask = cv.threshold(segmented_mask, 1, 255, cv.THRESH_BINARY)

    # Apply mask to original image
    rbc_segment = cv.bitwise_and(image, image, mask=binary_mask)
    rbc_segment_gray = cv.cvtColor(rbc_segment, cv.COLOR_RGB2GRAY)

    # Morphological operations
    kernel_open = np.ones((5, 5), np.uint8)
    kernel_close = np.ones((5, 5), np.uint8)
    rbc_segment_gray = cv.morphologyEx(rbc_segment_gray, cv.MORPH_OPEN, kernel_open)
    rbc_segment_gray = cv.morphologyEx(rbc_segment_gray, cv.MORPH_CLOSE, kernel_close)
    
    # Find contours and filter by area
    contours, _ = cv.findContours(rbc_segment_gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    MIN_AREA = 120  # Smaller threshold for individual RBC detection

    filtered_mask = np.zeros_like(rbc_segment_gray)
    for contour in contours:
        area = cv.contourArea(contour)
        if area >= MIN_AREA:
            cv.drawContours(filtered_mask, [contour], -1, 255, thickness=cv.FILLED)

    # Apply filtered mask
    rbc_only_image = cv.bitwise_and(rbc_segment, rbc_segment, mask=filtered_mask)
    
    # Remove low-intensity pixels per channel
    for c in range(rbc_only_image.shape[2]):
        _, rbc_only_image[:, :, c] = cv.threshold(
            rbc_only_image[:, :, c], 15, 255, cv.THRESH_TOZERO
        )
    
    return rbc_only_image

def remove_unwanted_cells_extended(clustered_images, selected_cluster, original_image):
    """
    Remove unwanted cells by creating mask from selected clusters (EXTENDED VERSION).
    Returns additional outputs for BO-FRS pipeline.
    
    Args:
        clustered_images: list of segmented images (HSV) from kmeans_segmentation
        selected_cluster: list of cluster indices chosen by user
        original_image: RGB uint8 image to mask (preprocessed image)
    
    Returns:
        rbc_only_image: RGB image with only RBC cells
        filtered_mask: Binary mask after area filtering
        binary_mask: Initial binary mask from clusters
    """
    if not selected_cluster:
        raise ValueError("No clusters selected. Please select at least one cluster.")
    
    # Combine selected clusters
    segmented_mask = clustered_images[selected_cluster[0]].copy()
    for index_cluster in selected_cluster[1:]:
        segmented_mask = cv.bitwise_or(segmented_mask, clustered_images[index_cluster])
    
    # Convert HSV mask to RGB then to grayscale
    segmented_mask = cv.cvtColor(segmented_mask, cv.COLOR_HSV2RGB)
    segmented_mask = cv.cvtColor(segmented_mask, cv.COLOR_RGB2GRAY)
    _, binary_mask = cv.threshold(segmented_mask, 1, 255, cv.THRESH_BINARY)

    # Apply mask to original image
    rbc_segment = cv.bitwise_and(original_image, original_image, mask=binary_mask)
    rbc_segment_gray = cv.cvtColor(rbc_segment, cv.COLOR_RGB2GRAY)

    # Morphological operations
    kernel_open = np.ones((5, 5), np.uint8)
    kernel_close = np.ones((5, 5), np.uint8)
    rbc_segment_gray = cv.morphologyEx(rbc_segment_gray, cv.MORPH_OPEN, kernel_open)
    rbc_segment_gray = cv.morphologyEx(rbc_segment_gray, cv.MORPH_CLOSE, kernel_close)
    
    # Find contours and filter by area
    contours, _ = cv.findContours(rbc_segment_gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    MIN_AREA = 120  # Smaller threshold for individual RBC detection

    filtered_mask = np.zeros_like(rbc_segment_gray)
    for contour in contours:
        area = cv.contourArea(contour)
        if area >= MIN_AREA:
            cv.drawContours(filtered_mask, [contour], -1, 255, thickness=cv.FILLED)

    # Apply filtered mask
    rbc_only_image = cv.bitwise_and(rbc_segment, rbc_segment, mask=filtered_mask)
    
    # Remove low-intensity pixels per channel
    for c in range(rbc_only_image.shape[2]):
        _, rbc_only_image[:, :, c] = cv.threshold(
            rbc_only_image[:, :, c], 15, 255, cv.THRESH_TOZERO
        )
    
    return rbc_only_image, filtered_mask, binary_mask

# ============================================================================
# BO-FRS: BOUNDED OPENING + FAST RADIAL SYMMETRY TRANSFORM
# ============================================================================

def bounded_opening_frs(binary_mask, num_openings=3):
    """
    Bounded Opening + Fast Radial Symmetry Transform
    - Detects RBC centers
    - Refines RBC edges
    - Identifies overlapping RBCs
    - Estimates uniform radius
    """
    print("🔬 BO-FRS: Bounded Opening + Fast Radial Symmetry Transform")
    
    # STEP 1: Bounded Opening
    print("   1️⃣ Applying Bounded Opening...")
    kernel_size = 5
    processed_mask = binary_mask.copy()
    
    for iteration in range(num_openings):
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        processed_mask = cv.morphologyEx(processed_mask, cv.MORPH_OPEN, kernel)
        kernel_size += 2
        print(f"      - Iteration {iteration+1}: kernel size {kernel_size-2}x{kernel_size-2}")
    
    # STEP 2: Distance Transform
    print("   2️⃣ Computing Distance Transform...")
    start_time = time.time()
    dist_transform = cv.distanceTransform(processed_mask, cv.DIST_L2, 5)
    print(f"      - Completed in {time.time() - start_time:.2f}s")
    
    # STEP 3: Fast Radial Symmetry Transform
    print("   3️⃣ Fast Radial Symmetry (FRS) for center detection...")
    
    dist_norm = cv.normalize(dist_transform, None, 0, 1.0, cv.NORM_MINMAX)
    
    radii = [5, 7, 9, 11, 13]
    frs_maps = []
    
    for radius in radii:
        grad_x = cv.Sobel(dist_norm, cv.CV_64F, 1, 0, ksize=3)
        grad_y = cv.Sobel(dist_norm, cv.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        orientation = np.arctan2(grad_y, grad_x)
        
        symmetry_map = np.zeros_like(dist_norm)
        
        y_coords, x_coords = np.where(processed_mask > 0)
        if len(y_coords) > 0:
            angles = orientation[y_coords, x_coords]
            grads = grad_mag[y_coords, x_coords]
            
            px = (x_coords + radius * np.cos(angles)).astype(int)
            py = (y_coords + radius * np.sin(angles)).astype(int)
            nx = (x_coords - radius * np.cos(angles)).astype(int)
            ny = (y_coords - radius * np.sin(angles)).astype(int)
            
            valid_p = (px >= 0) & (px < symmetry_map.shape[1]) & (py >= 0) & (py < symmetry_map.shape[0])
            valid_n = (nx >= 0) & (nx < symmetry_map.shape[1]) & (ny >= 0) & (ny < symmetry_map.shape[0])
            
            np.add.at(symmetry_map, (py[valid_p], px[valid_p]), grads[valid_p])
            np.add.at(symmetry_map, (ny[valid_n], nx[valid_n]), grads[valid_n])
        
        frs_maps.append(symmetry_map)
    
    frs_combined = np.mean(frs_maps, axis=0)
    frs_combined = cv.normalize(frs_combined, None, 0, 1.0, cv.NORM_MINMAX)
    
    print(f"      - Processed {len(radii)} radii: {radii}")
    
    # STEP 4: Center Detection
    print("   4️⃣ Detecting RBC centers...")
    
    combined_map = 0.6 * dist_norm + 0.4 * frs_combined
    
    # Adaptive min_distance based on median radius estimate
    # First pass: rough estimate
    rough_coords = peak_local_max(
        combined_map,
        min_distance=6,
        threshold_abs=0.08,
        exclude_border=False
    )
    
    # Estimate median radius from distance transform
    if len(rough_coords) > 0:
        rough_radii = [dist_transform[y, x] for y, x in rough_coords]
        rough_median_radius = np.median(rough_radii)
        # Adaptive min_distance: 60-80% of expected diameter
        adaptive_min_dist = max(8, int(rough_median_radius * 1.4))
    else:
        adaptive_min_dist = 10
    
    print(f"      - Adaptive min_distance: {adaptive_min_dist} pixels")
    
    # Second pass: refined detection
    coordinates = peak_local_max(
        combined_map,
        min_distance=adaptive_min_dist,
        threshold_abs=0.1,  # Increased threshold to reduce false positives
        exclude_border=False
    )
    
    centers = [(int(x), int(y)) for y, x in coordinates]
    
    # STEP 5: Radius Estimation
    print("   5️⃣ Estimating candidate radius...")
    
    radii_list = []
    for (cx, cy) in centers:
        if 0 <= cx < dist_transform.shape[1] and 0 <= cy < dist_transform.shape[0]:
            radius = dist_transform[cy, cx]
            radii_list.append(radius)
    
    if len(radii_list) > 0:
        # Use median for robustness against outliers
        candidate_radius = int(np.median(radii_list))
        radius_std = np.std(radii_list)
        
        # Filter outlier centers (too close or far from median)
        min_acceptable_radius = candidate_radius * 0.5
        max_acceptable_radius = candidate_radius * 1.8
        
        filtered_centers = []
        for (cx, cy), r in zip(centers, radii_list):
            if min_acceptable_radius <= r <= max_acceptable_radius:
                filtered_centers.append((cx, cy))
        
        centers = filtered_centers
        print(f"      - Filtered {len(coordinates) - len(centers)} outlier centers")
    else:
        candidate_radius = 15
        radius_std = 0
    
    print(f"      - Candidate radius: {candidate_radius} ± {radius_std:.1f} pixels")
    print(f"      - Valid radius range: {candidate_radius*0.5:.1f} - {candidate_radius*1.8:.1f} pixels")
    
    center_map = np.zeros_like(processed_mask)
    for (cx, cy) in centers:
        cv.circle(center_map, (cx, cy), 3, 255, -1)
    
    print(f"\n✅ BO-FRS Completed:")
    print(f"   - RBC centers detected: {len(centers)}")
    print(f"   - Candidate radius: {candidate_radius} pixels")
    print(f"   - Refined boundary mask created")
    
    return {
        'refined_mask': processed_mask,
        'dist_transform': dist_transform,
        'frs_map': frs_combined,
        'combined_map': combined_map,
        'centers': centers,
        'center_map': center_map,
        'candidate_radius': candidate_radius,
        'radius_std': radius_std
    }

# ============================================================================
# PIXEL REPLICATION + GMM (NEW METHOD - Cell Separation)
# ============================================================================

def separate_overlapping_rbc_with_gmm(bofrs_results, cells_image):
    """
    Pixel Replication + GMM untuk pemisahan RBC yang overlap (NEW METHOD).
    This is an ALTERNATIVE to the original separate_cells function.
    
    INPUT dari BO-FRS:
    - centers: list of (x, y) RBC centers  
    - dist_transform: distance transform dari cluster
    - refined_mask: RBC boundary yang sudah refined
    
    OUTPUT:
    - Individual RBC cells (sudah dipisahkan)
    - Per-cell masks
    - Bounding boxes
    """
    print("🔬 Pixel Replication + GMM Separation (New Method)")
    print("=" * 80)
    
    centers_global = bofrs_results['centers']
    dist_transform = bofrs_results['dist_transform']
    refined_mask = bofrs_results['refined_mask']
    candidate_radius = bofrs_results['candidate_radius']
    radius_std = bofrs_results.get('radius_std', 0)
    
    all_cropped_cells = []
    all_bounding_boxes = []
    all_cell_masks = []
    
    contours, _ = cv.findContours(refined_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    total = len(contours)
    
    print(f"🔍 Processing {total} contours...")
    print(f"📏 Using candidate radius: {candidate_radius} ± {radius_std:.1f} pixels")
    
    single_cells = 0
    gmm_separated = 0
    skipped = 0
    
    # Adaptive thresholds based on candidate radius
    min_area = max(100, int(np.pi * (candidate_radius * 0.6) ** 2))
    expected_cell_area = int(np.pi * candidate_radius ** 2)
    max_area_for_single = expected_cell_area * 1.8  # Single cell shouldn't be much larger
    
    print(f"🎯 Adaptive thresholds:")
    print(f"   - Min area: {min_area} pixels²")
    print(f"   - Expected single cell area: {expected_cell_area} pixels²")
    print(f"   - Max area for single cell: {max_area_for_single:.0f} pixels²")
    
    for idx, contour in enumerate(contours):
        if (idx + 1) % 20 == 0:
            print(f"   Progress: {idx+1}/{total} ({(idx+1)*100//total}%)")
        
        x_offset, y_offset, w, h = cv.boundingRect(contour)
        area = cv.contourArea(contour)
        
        # Skip too small contours
        if w < 10 or h < 10 or area < min_area:
            skipped += 1
            continue
        
        single_mask = np.zeros_like(refined_mask)
        cv.drawContours(single_mask, [contour], -1, 255, thickness=cv.FILLED)
        
        cropped_mask = single_mask[y_offset:y_offset+h, x_offset:x_offset+w]
        cropped_image = cells_image[y_offset:y_offset+h, x_offset:x_offset+w]
        cropped_dist = dist_transform[y_offset:y_offset+h, x_offset:x_offset+w]
        
        # Find local centers
        local_centers = []
        for (cx, cy) in centers_global:
            if x_offset <= cx < x_offset+w and y_offset <= cy < y_offset+h:
                local_centers.append((cx - x_offset, cy - y_offset))
        
        # Decision logic: Single cell or overlapping?
        k = len(local_centers)
        
        # If no centers detected but small area → likely single cell
        if k == 0 and area <= max_area_for_single:
            all_bounding_boxes.append((x_offset, y_offset, w, h))
            all_cropped_cells.append(cropped_image)
            all_cell_masks.append(cropped_mask)
            single_cells += 1
            continue
        
        # If exactly 1 center and reasonable area → single cell
        if k == 1 and area <= max_area_for_single:
            all_bounding_boxes.append((x_offset, y_offset, w, h))
            all_cropped_cells.append(cropped_image)
            all_cell_masks.append(cropped_mask)
            single_cells += 1
            continue
        
        # If area suggests single cell despite multiple centers → validate
        if area <= max_area_for_single and k > 1:
            # Check if centers are too close (false positives)
            min_dist_between_centers = candidate_radius * 1.2
            valid_centers = [local_centers[0]]
            for center in local_centers[1:]:
                is_far_enough = all(
                    np.sqrt((center[0] - vc[0])**2 + (center[1] - vc[1])**2) >= min_dist_between_centers
                    for vc in valid_centers
                )
                if is_far_enough:
                    valid_centers.append(center)
            
            if len(valid_centers) == 1:
                all_bounding_boxes.append((x_offset, y_offset, w, h))
                all_cropped_cells.append(cropped_image)
                all_cell_masks.append(cropped_mask)
                single_cells += 1
                continue
            else:
                k = len(valid_centers)
                local_centers = valid_centers
        
        # OVERLAPPING CELLS: Use Pixel Replication + GMM
        X_replicated = []
        
        # Adaptive replication based on distance and radius consistency
        for (lx, ly) in local_centers:
            if 0 <= lx < cropped_dist.shape[1] and 0 <= ly < cropped_dist.shape[0]:
                local_radius = cropped_dist[ly, lx]
                # Weight based on how close the radius is to the expected value
                weight = int(local_radius)
                # Cap replication to prevent single center domination
                max_reps = max(20, int(candidate_radius * 2))
                weight = min(weight, max_reps)
                X_replicated.extend([(lx, ly)] * max(1, weight))
        
        # Need enough samples for GMM
        min_samples = k * 10
        if len(X_replicated) < min_samples:
            all_bounding_boxes.append((x_offset, y_offset, w, h))
            all_cropped_cells.append(cropped_image)
            all_cell_masks.append(cropped_mask)
            single_cells += 1
            continue
        
        # GMM: Model overlapping regions
        try:
            gmm = GaussianMixture(
                n_components=k,
                covariance_type="tied",
                max_iter=100,  # Increased iterations
                n_init=3,      # More initializations
                random_state=42,
                tol=1e-3
            )
            
            X_replicated_array = np.array(X_replicated)
            gmm.fit(X_replicated_array)
            
            ys, xs = np.where(cropped_mask == 255)
            
            if len(ys) == 0:
                continue
            
            foreground_pixels = np.column_stack((xs, ys))
            pixel_labels = gmm.predict(foreground_pixels)
            
            labeled_mask = np.zeros_like(cropped_mask, dtype=np.uint8)
            for (x, y), label in zip(foreground_pixels, pixel_labels):
                labeled_mask[y, x] = label + 1
            
            # Extract individual cells
            unique_labels = np.unique(labeled_mask)
            unique_labels = unique_labels[unique_labels != 0]
            
            for label in unique_labels:
                cell_mask = (labeled_mask == label).astype(np.uint8) * 255
                cell_image = cv.bitwise_and(cropped_image, cropped_image, mask=cell_mask)
                
                coords = cv.findNonZero(cell_mask)
                if coords is not None and len(coords) > 50:
                    x_cell, y_cell, w_cell, h_cell = cv.boundingRect(coords)
                    
                    global_x = x_cell + x_offset
                    global_y = y_cell + y_offset
                    
                    cell_img_cropped = cell_image[y_cell:y_cell+h_cell, x_cell:x_cell+w_cell]
                    cell_mask_cropped = cell_mask[y_cell:y_cell+h_cell, x_cell:x_cell+w_cell]
                    
                    all_bounding_boxes.append((global_x, global_y, w_cell, h_cell))
                    all_cropped_cells.append(cell_img_cropped)
                    all_cell_masks.append(cell_mask_cropped)
            
            gmm_separated += 1
            
        except Exception as e:
            print(f"⚠️ GMM failed for contour {idx+1}: {e}")
            all_bounding_boxes.append((x_offset, y_offset, w, h))
            all_cropped_cells.append(cropped_image)
            all_cell_masks.append(cropped_mask)
            single_cells += 1
    
    print(f"\n✅ Separation completed!")
    print(f"   📊 Summary:")
    print(f"      - Total contours processed: {total}")
    print(f"      - Single cells (no separation): {single_cells}")
    print(f"      - Contours separated by GMM: {gmm_separated}")
    print(f"      - Skipped (too small): {skipped}")
    print(f"      - Final individual cells: {len(all_bounding_boxes)}")
    print("=" * 80)
    
    return all_cropped_cells, all_bounding_boxes, all_cell_masks

# Pemisahan

def separate_cells(
    k, dist_transform, cell_contour_mask, cells_image, x_offset, y_offset, seed_map_cell
):
    """
    Separate overlapping cells using Gaussian Mixture Model (ORIGINAL METHOD).
    This function is kept for backward compatibility.
    
    Args:
        k: Number of cells to separate
        dist_transform: Distance transform of cell mask
        cell_contour_mask: Binary mask of cell contour
        cells_image: Original cell image
        x_offset: X offset for global coordinates
        y_offset: Y offset for global coordinates
        seed_map_cell: List of seed points
    
    Returns:
        cropped_cell: List of separated cell images
        bounding_boxes: List of bounding boxes [x, y, w, h]
    """
    X_mat = []
    cropped_cell = []
    bounding_boxes = []
    
    for yc, xc in seed_map_cell:
        X_mat.extend([(yc, xc)] * int(dist_transform[xc, yc]))

    gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=0)
    gmm.fit(X_mat)

    ys, xs = np.where(cell_contour_mask == 255)
    foreground_coords = np.column_stack((xs, ys))

    labels = gmm.predict(foreground_coords)
    labeled_cell = np.zeros_like(cell_contour_mask, dtype=np.uint8)
    for (x, y), label in zip(foreground_coords, labels):
        labeled_cell[y, x] = int((label + 1))

    unique_labels = np.unique(labeled_cell)
    unique_labels = unique_labels[unique_labels != 0]

    for label in unique_labels:
        mask = np.uint8(labeled_cell == label)
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            x, y, w, h = cv.boundingRect(cnt)
            global_x = x + x_offset
            global_y = y + y_offset
            bounding_boxes.append((global_x, global_y, w, h))

    for i in range(k):
        mask = (labeled_cell == (i + 1)).astype(np.uint8)
        indiv_cell = cv.bitwise_and(cells_image, cells_image, mask=mask)
        coords = cv.findNonZero(mask)
        x, y, w, h = cv.boundingRect(coords)
        cropped = indiv_cell[y : y + h, x : x + w]
        cropped_cell.append(cropped)

    return cropped_cell, bounding_boxes

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def sobel_edge_detect(image):
    """Detect edges using Sobel operator"""
    sobel_x = cv.Sobel(image, cv.CV_64F, 1, 0, ksize=5)
    sobel_y = cv.Sobel(image, cv.CV_64F, 0, 1, ksize=5)
    sobel_edges = cv.magnitude(sobel_x, sobel_y)
    sobel_edges = np.uint8(255 * (sobel_edges / np.max(sobel_edges)))
    _, sobel_binary = cv.threshold(sobel_edges, 50, 255, cv.THRESH_BINARY)
    contours_sobel, _ = cv.findContours(
        sobel_binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    return sobel_edges, contours_sobel

def draw_bounding_boxes(image, contours):
    """Draw bounding boxes around detected contours"""
    bbox_image = image.copy()
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        cv.rectangle(bbox_image, (x, y), (x + w, y + h), (0, 255, 0), 5)
    return bbox_image

def extract_contours(image, edge_map):
    """Extract contours from edge map"""
    contours, _ = cv.findContours(edge_map, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contour_mask = np.zeros_like(image)
    cv.drawContours(contour_mask, contours, -1, 255, thickness=cv.FILLED)
    return contours, contour_mask

def find_seed(contour_mask):
    """Find seed points for cell separation using distance transform"""
    dist_transform = cv.distanceTransform(contour_mask, cv.DIST_L2, 5)
    _, center_map = cv.threshold(
        dist_transform, 0.8 * dist_transform.max(), 255, cv.THRESH_BINARY
    )
    center_map = np.uint8(center_map)

    contours_centers, _ = cv.findContours(
        center_map, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )
    centers = [cv.moments(c) for c in contours_centers]
    seed_map = [
        (int(c["m10"] / c["m00"]), int(c["m01"] / c["m00"]))
        for c in centers
        if c["m00"] != 0
    ]
    return dist_transform, seed_map