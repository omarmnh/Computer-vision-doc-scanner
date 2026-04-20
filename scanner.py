import cv2
import numpy as np
import sys
import os


# ═══════════════════════════════════════════════════════
# STEP 1 — Load & Resize
# ═══════════════════════════════════════════════════════

def load_image(image_path: str, target_width: int = 800):
    if not os.path.exists(image_path):
        print(f"[ERROR] File not found: {image_path}")
        sys.exit(1)

    original = cv2.imread(image_path)
    if original is None:
        print(f"[ERROR] Could not read image: {image_path}")
        sys.exit(1)

    h, w    = original.shape[:2]
    scale   = target_width / w
    resized = cv2.resize(original,
                         (target_width, int(h * scale)),
                         interpolation=cv2.INTER_AREA)

    print(f"[1] Loaded: {w}×{h}  →  resized: {resized.shape[1]}×{resized.shape[0]}")
    return original, resized, scale


# ═══════════════════════════════════════════════════════
# STEP 2 — Preprocess
# ═══════════════════════════════════════════════════════

def preprocess(image, blur_kernel: int = 5):
    gray    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray,
                               (blur_kernel, blur_kernel),
                               sigmaX=0)
    print(f"[2] Grayscale + blur (kernel={blur_kernel})")
    return gray, blurred


# ═══════════════════════════════════════════════════════
# STEP 3 — Sobel Edge Detection
# ═══════════════════════════════════════════════════════

def detect_edges(blurred, threshold: int = 60):
    # Gradients in X and Y
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

    # Combine: magnitude ≈ |Gx| + |Gy|
    magnitude = np.abs(sobelx) + np.abs(sobely)

    # Normalize to 0–255
    magnitude_norm = np.uint8(255 * magnitude / magnitude.max())

    # Threshold → binary edge map
    _, edges = cv2.threshold(magnitude_norm, threshold,
                             255, cv2.THRESH_BINARY)

    pct = (np.sum(edges == 255) / edges.size) * 100
    print(f"[3] Sobel edges — threshold={threshold}, "
          f"edge density={pct:.1f}%")
    return edges


# ═══════════════════════════════════════════════════════
# STEP 4 — Contour Detection
# ═══════════════════════════════════════════════════════

def find_contours(edges, min_area_pct: float = 1.0):
    contours, _ = cv2.findContours(edges.copy(),
                                   cv2.RETR_LIST,
                                   cv2.CHAIN_APPROX_SIMPLE)

    image_area  = edges.shape[0] * edges.shape[1]
    min_area    = image_area * (min_area_pct / 100)
    filtered    = [c for c in contours
                   if cv2.contourArea(c) > min_area]
    sorted_cnts = sorted(filtered,
                         key=cv2.contourArea,
                         reverse=True)

    print(f"[4] Contours found: {len(contours)} total, "
          f"{len(sorted_cnts)} above {min_area_pct}% area threshold")
    return sorted_cnts


# ═══════════════════════════════════════════════════════
# STEP 5 — Document Detection
# ═══════════════════════════════════════════════════════

def find_document(contours, image_area: int,
                  min_area_pct: float = 10.0):
    """Try epsilon values until a valid 4-corner contour is found."""
    for epsilon_factor in [0.02, 0.03, 0.04, 0.05]:
        for contour in contours:
            perimeter    = cv2.arcLength(contour, True)
            approximated = cv2.approxPolyDP(
                contour, epsilon_factor * perimeter, True)
            area_pct     = (cv2.contourArea(contour) / image_area) * 100

            if (len(approximated) == 4
                    and area_pct >= min_area_pct
                    and cv2.isContourConvex(approximated)):
                print(f"[5] Document found — epsilon={epsilon_factor}, "
                      f"area={area_pct:.1f}%")
                return approximated

    print("[5] WARNING: No document contour found.")
    return None


# ═══════════════════════════════════════════════════════
# STEP 6 — Perspective Transform
# ═══════════════════════════════════════════════════════

def order_points(pts):
    """Sort 4 points into [TL, TR, BR, BL] order."""
    ordered = np.zeros((4, 2), dtype=np.float32)
    s               = pts.sum(axis=1)
    ordered[0]      = pts[np.argmin(s)]   # TL
    ordered[2]      = pts[np.argmax(s)]   # BR
    diff            = np.diff(pts, axis=1)
    ordered[1]      = pts[np.argmin(diff)]  # TR
    ordered[3]      = pts[np.argmax(diff)]  # BL
    return ordered


def perspective_transform(document_contour, original, scale):
    # Scale corners from resized → original coordinates
    pts      = document_contour.reshape(4, 2).astype(np.float32)
    pts_orig = pts / scale
    ordered  = order_points(pts_orig)

    TL, TR, BR, BL = ordered
    W = int(max(np.linalg.norm(TR - TL),
                np.linalg.norm(BR - BL)))
    H = int(max(np.linalg.norm(BL - TL),
                np.linalg.norm(BR - TR)))

    dst = np.array([[0,W,W,0],[0,0,H,H]],
                   dtype=np.float32).T   # shape (4,2)

    # dst rows: TL(0,0) TR(W,0) BR(W,H) BL(0,H)
    dst = np.float32([[0,0],[W,0],[W,H],[0,H]])

    M      = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(original, M, (W, H),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)

    print(f"[6] Perspective transform — output: {W}×{H}px")
    return warped


# ═══════════════════════════════════════════════════════
# STEP 7 — Enhancement
# ═══════════════════════════════════════════════════════

def enhance(warped, block_size: int = 21, C: int = 10):
    gray    = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    scan = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size, C
    )

    kernel = np.ones((2, 2), np.uint8)
    scan   = cv2.morphologyEx(scan, cv2.MORPH_OPEN,  kernel)
    scan   = cv2.morphologyEx(scan, cv2.MORPH_CLOSE, kernel)

    white_pct = (np.sum(scan == 255) / scan.size) * 100
    print(f"[7] Enhancement done — "
          f"white={white_pct:.1f}%, black={100-white_pct:.1f}%")
    return scan


# ═══════════════════════════════════════════════════════
# STEP 8 — Save & Display
# ═══════════════════════════════════════════════════════

def show_image(title, image, max_h=700):
    h, w = image.shape[:2]
    if h > max_h:
        image = cv2.resize(image,
                           (int(w * max_h / h), max_h),
                           interpolation=cv2.INTER_AREA)
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def save_results(original, warped, scan, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)

    cv2.imwrite(os.path.join(output_dir, "warped.jpg"),
                warped, [cv2.IMWRITE_JPEG_QUALITY, 95])

    cv2.imwrite(os.path.join(output_dir, "scan.png"), scan)

    # Build comparison panel
    TARGET_H = 900
    def fit(img):
        h, w = img.shape[:2]
        r = cv2.resize(img, (int(w*TARGET_H/h), TARGET_H),
                       interpolation=cv2.INTER_AREA)
        if len(r.shape) == 2:
            r = cv2.cvtColor(r, cv2.COLOR_GRAY2BGR)
        return r

    panels = [fit(original), fit(warped), fit(scan)]
    labels = ["1. Original", "2. Warped", "3. Scanned"]
    colors = [(0,200,200), (0,255,0), (200,200,200)]

    for img, label, color in zip(panels, labels, colors):
        cv2.putText(img, label, (12, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,0,0), 4)
        cv2.putText(img, label, (10, 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, color,  2)

    sep        = np.zeros((TARGET_H, 6, 3), dtype=np.uint8)
    sep[:]     = (0, 200, 0)
    comparison = np.hstack([panels[0], sep, panels[1], sep, panels[2]])

    cv2.imwrite(os.path.join(output_dir, "comparison.jpg"),
                comparison, [cv2.IMWRITE_JPEG_QUALITY, 90])

    scan_kb = os.path.getsize(
        os.path.join(output_dir, "scan.png")) // 1024

    print(f"\n{'═'*50}")
    print(f"  📄 DOCUMENT SCANNER — COMPLETE")
    print(f"{'═'*50}")
    print(f"  Input:          {original.shape[1]}×{original.shape[0]}")
    print(f"  Warped:         {warped.shape[1]}×{warped.shape[0]}")
    print(f"  Scan:           {scan.shape[1]}×{scan.shape[0]}")
    print(f"  Scan file size: {scan_kb} KB")
    print(f"  Output folder:  ./{output_dir}/")
    print(f"{'─'*50}")
    print(f"  ✅ warped.jpg     — color, perspective fixed")
    print(f"  ✅ scan.png       — final B&W scan (lossless)")
    print(f"  ✅ comparison.jpg — before/after overview")
    print(f"{'═'*50}\n")


# ═══════════════════════════════════════════════════════
# MAIN — Full Pipeline
# ═══════════════════════════════════════════════════════

def main():

    # ── Config — tweak these if results aren't great ──────────────────────
    IMAGE_PATH   = "test.jpeg"   # ← change to your image
    OUTPUT_DIR   = "output"
    RESIZE_WIDTH = 800
    BLUR_KERNEL  = 5
    SOBEL_THRESH = 60
    ENHANCE_BS   = 21               # adaptive threshold block size
    ENHANCE_C    = 10               # adaptive threshold C constant
    SHOW_STEPS   = True             # set False to skip intermediate windows

    print("\n" + "═"*50)
    print("  🔍 DOCUMENT SCANNER STARTING")
    print("═"*50)

    # Step 1
    original, resized, scale = load_image(IMAGE_PATH, RESIZE_WIDTH)

    # Step 2
    gray, blurred = preprocess(resized, BLUR_KERNEL)

    # Step 3
    edges = detect_edges(blurred, SOBEL_THRESH)

    # Step 4
    contours = find_contours(edges)

    # Step 5
    image_area       = resized.shape[0] * resized.shape[1]
    document_contour = find_document(contours, image_area)

    if document_contour is None:
        print("\n[FATAL] Document not detected.")
        print("  Tips:")
        print("  → Ensure document is on a contrasting background")
        print("  → Document should fill at least 40% of the frame")
        print("  → Try lowering SOBEL_THRESH (e.g. 40)")
        print("  → Try increasing BLUR_KERNEL (e.g. 7)")
        sys.exit(1)

    # Step 6
    warped = perspective_transform(document_contour, original, scale)

    # Step 7
    scan = enhance(warped, ENHANCE_BS, ENHANCE_C)

    # Step 8
    save_results(original, warped, scan, OUTPUT_DIR)

    # Display final results
    if SHOW_STEPS:
        show_image("Original",           original)
        show_image("Edges (Sobel)",       edges)
        show_image("Warped",              warped)
        show_image("Final Scan",          scan)

    print("  Done! Check the ./output/ folder for your results.\n")


if __name__ == "__main__":
    main()