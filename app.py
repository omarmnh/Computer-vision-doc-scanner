import io
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import streamlit as st

# Import your existing algorithm module (DO NOT modify it)
import scanner


@dataclass(frozen=True)
class ScanParams:
    resize_width: int = 900
    blur_kernel: int = 5
    sobel_threshold: int = 60
    enhance_block_size: int = 21
    enhance_c: int = 10


def _decode_uploaded_image(uploaded_file) -> np.ndarray:
    """Decode Streamlit uploaded file -> OpenCV BGR image (np.ndarray)."""
    data = uploaded_file.read()
    if not data:
        raise ValueError("Empty file.")

    file_bytes = np.frombuffer(data, dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Could not decode image. Please upload a valid JPG/PNG.")
    return img_bgr


def _resize_keep_aspect(image_bgr: np.ndarray, target_width: int) -> Tuple[np.ndarray, float]:
    """Resize to target width, return resized image and scale factor (target_width/original_width)."""
    h, w = image_bgr.shape[:2]
    if w == 0 or h == 0:
        raise ValueError("Invalid image dimensions.")

    scale = float(target_width) / float(w)
    resized = cv2.resize(
        image_bgr,
        (target_width, int(h * scale)),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def _bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _gray_to_rgb(image_gray: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB)


def scan_document(
    image_bgr: np.ndarray,
    params: ScanParams,
    *,
    return_intermediates: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Thin wrapper around the existing functions in `scanner.py`.

    Returns:
      warped_bgr: perspective-warped color image (BGR)
      enhanced_gray: final enhanced scan (grayscale)
      intermediates: optional debug images (edges, contour overlay, etc.)
    """
    resized_bgr, scale = _resize_keep_aspect(image_bgr, params.resize_width)

    # Step 2–5: detect document on resized image
    gray, blurred = scanner.preprocess(resized_bgr, params.blur_kernel)
    edges = scanner.detect_edges(blurred, params.sobel_threshold)
    contours = scanner.find_contours(edges)

    image_area = resized_bgr.shape[0] * resized_bgr.shape[1]
    doc_contour = scanner.find_document(contours, image_area)
    if doc_contour is None:
        raise RuntimeError(
            "No document detected. Try a clearer image, increase contrast, or adjust the threshold."
        )

    # Step 6–7: warp using original-resolution image
    warped_bgr = scanner.perspective_transform(doc_contour, image_bgr, scale)

    # Adaptive threshold needs odd block size >= 3
    bs = int(params.enhance_block_size)
    if bs < 3:
        bs = 3
    if bs % 2 == 0:
        bs += 1

    enhanced_gray = scanner.enhance(warped_bgr, bs, int(params.enhance_c))

    intermediates: Dict[str, Any] = {}
    if return_intermediates:
        overlay = resized_bgr.copy()
        cv2.drawContours(overlay, [doc_contour], -1, (0, 255, 0), 3)
        intermediates = {
            "resized_bgr": resized_bgr,
            "gray": gray,
            "blurred": blurred,
            "edges": edges,
            "document_overlay_bgr": overlay,
        }

    return warped_bgr, enhanced_gray, intermediates


def _encode_image_bytes(image: np.ndarray, ext: str = ".png") -> bytes:
    """Encode image to bytes for download (supports .png / .jpg)."""
    ext = ext.lower().strip()
    if ext not in {".png", ".jpg", ".jpeg"}:
        ext = ".png"

    if ext in {".jpg", ".jpeg"}:
        ok, buf = cv2.imencode(ext, image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    else:
        ok, buf = cv2.imencode(ext, image)

    if not ok:
        raise RuntimeError("Failed to encode image for download.")
    return buf.tobytes()


def _inject_css() -> None:
    st.markdown(
        """
        <style>
          .app-title { font-weight: 800; letter-spacing: -0.02em; margin-bottom: 0.25rem; }
          .app-subtitle { color: rgba(49,51,63,0.75); margin-top: 0; }
          [data-testid="stSidebar"] { border-right: 1px solid rgba(49,51,63,0.12); }
          .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
          .stButton>button { border-radius: 10px; padding: 0.6rem 1rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Document Scanner",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_css()

    st.markdown('<div class="app-title" style="font-size: 2rem;">📄 Document Scanner</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="app-subtitle">Upload a photo of a document, scan it, and download a clean enhanced result.</p>',
        unsafe_allow_html=True,
    )

    if "input_bgr" not in st.session_state:
        st.session_state.input_bgr = None
    if "warped_bgr" not in st.session_state:
        st.session_state.warped_bgr = None
    if "enhanced_gray" not in st.session_state:
        st.session_state.enhanced_gray = None
    if "intermediates" not in st.session_state:
        st.session_state.intermediates = None

    with st.sidebar:
        st.header("Controls")
        uploaded = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

        st.divider()
        st.subheader("Tuning (optional)")
        sobel_threshold = st.slider("Edge threshold", min_value=5, max_value=160, value=60, step=1)
        blur_kernel = st.select_slider("Blur kernel", options=[3, 5, 7, 9, 11], value=5)

        st.subheader("Enhancement")
        enhance_block_size = st.slider("Adaptive block size (odd)", min_value=3, max_value=51, value=21, step=2)
        enhance_c = st.slider("Adaptive C", min_value=0, max_value=30, value=10, step=1)

        show_steps = st.toggle("Show intermediate steps", value=False)

        st.divider()
        scan_clicked = st.button("Scan Document", type="primary", use_container_width=True)

        st.caption(
            "Tips: use a well-lit photo, avoid shadows, and keep the document filling most of the frame."
        )

    # Decode uploaded image (if any)
    if uploaded is not None:
        try:
            st.session_state.input_bgr = _decode_uploaded_image(uploaded)
            # Reset old results when a new file is uploaded
            st.session_state.warped_bgr = None
            st.session_state.enhanced_gray = None
            st.session_state.intermediates = None
        except Exception as e:
            st.session_state.input_bgr = None
            st.error(str(e))

    # Run scan on click
    if scan_clicked:
        if st.session_state.input_bgr is None:
            st.error("Please upload an image first.")
        else:
            params = ScanParams(
                resize_width=900,
                blur_kernel=int(blur_kernel),
                sobel_threshold=int(sobel_threshold),
                enhance_block_size=int(enhance_block_size),
                enhance_c=int(enhance_c),
            )

            with st.spinner("Scanning…"):
                try:
                    warped_bgr, enhanced_gray, intermediates = scan_document(
                        st.session_state.input_bgr,
                        params,
                        return_intermediates=bool(show_steps),
                    )
                    st.session_state.warped_bgr = warped_bgr
                    st.session_state.enhanced_gray = enhanced_gray
                    st.session_state.intermediates = intermediates
                    st.success("Scan complete.")
                except Exception as e:
                    st.session_state.warped_bgr = None
                    st.session_state.enhanced_gray = None
                    st.session_state.intermediates = None
                    st.error(f"Scan failed: {e}")

    # Main layout: Original / Warped / Enhanced
    col1, col2, col3 = st.columns([1, 1, 1], gap="large")

    with col1:
        st.subheader("Original")
        if st.session_state.input_bgr is not None:
            st.image(_bgr_to_rgb(st.session_state.input_bgr), use_container_width=True, caption="Uploaded image")
        else:
            st.info("Upload an image to preview it here.")

    with col2:
        st.subheader("Warped")
        if st.session_state.warped_bgr is not None:
            st.image(_bgr_to_rgb(st.session_state.warped_bgr), use_container_width=True, caption="Perspective corrected")
        else:
            st.info("Run a scan to generate the warped image.")

    with col3:
        st.subheader("Enhanced")
        if st.session_state.enhanced_gray is not None:
            st.image(_gray_to_rgb(st.session_state.enhanced_gray), use_container_width=True, caption="Final scanned result")

            # Download
            out_ext = st.radio("Download format", options=["PNG", "JPG"], horizontal=True, index=0)
            ext = ".png" if out_ext == "PNG" else ".jpg"
            download_bytes = _encode_image_bytes(st.session_state.enhanced_gray, ext=ext)
            st.download_button(
                label="Download Scanned Document",
                data=download_bytes,
                file_name=f"scanned{ext}",
                mime="image/png" if ext == ".png" else "image/jpeg",
                use_container_width=True,
            )
        else:
            st.info("Run a scan to generate the final enhanced result.")

    # Optional intermediate steps
    if st.session_state.intermediates:
        st.divider()
        st.subheader("Intermediate steps")
        a, b, c = st.columns([1, 1, 1], gap="large")
        with a:
            st.image(_gray_to_rgb(st.session_state.intermediates["edges"]), use_container_width=True, caption="Edges (Sobel)")
        with b:
            st.image(_bgr_to_rgb(st.session_state.intermediates["document_overlay_bgr"]), use_container_width=True, caption="Detected document contour")
        with c:
            st.image(_gray_to_rgb(st.session_state.intermediates["blurred"]), use_container_width=True, caption="Blurred grayscale")


if __name__ == "__main__":
    main()

