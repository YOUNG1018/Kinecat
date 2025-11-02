#!/usr/bin/env python3
# viewer.py — Kinect v2 RGB/Depth/IR inspector using pylibfreenect2 + OpenCV
# Run: python viewer.py
#
# Controls:
#   [q] or [ESC]  : quit
#   [space]       : pause/resume stream
#   [s]           : save RGB/depth/IR/registered frames to ./captures/
#   [g]           : toggle registration (color aligned to depth)
#   [1] [2] [3]   : layout (1=grid, 2=RGB+Depth, 3=IR+Depth)
#   [+]/[-]       : increase/decrease depth MAX (meters)
#   [=]/[r]       : increase depth MIN / reset both to defaults
#
# Notes:
#   - Depth range defaults to 0.5–4.5 m; adjust for better contrast.
#   - Registration outputs undistorted depth and color→depth registered frame.

import os
import time
import cv2
import numpy as np

WINDOW_NAME = "Kinect v2 Viewer (RGB/Depth/IR)"
DEPTH_MIN_M_DEFAULT = 0.1
DEPTH_MAX_M_DEFAULT = 4.5

def try_import_pipeline():
    """Pick the best available packet pipeline for pylibfreenect2."""
    from pylibfreenect2 import OpenGLPacketPipeline, OpenCLPacketPipeline, CpuPacketPipeline
    try:
        return OpenGLPacketPipeline()
    except Exception:
        try:
            return OpenCLPacketPipeline()
        except Exception:
            return CpuPacketPipeline()

def depth_to_colormap(depth_m: np.ndarray, dmin_m: float, dmax_m: float) -> np.ndarray:
    """Convert depth (meters, float32) to a BGR colormap image."""
    depth = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)
    depth = np.clip(depth, dmin_m, dmax_m)
    norm = (depth - dmin_m) / max(1e-6, (dmax_m - dmin_m))
    gray8 = (norm * 255.0).astype(np.uint8)
    return cv2.applyColorMap(gray8, cv2.COLORMAP_JET)

def ir_to_gray(ir_frame: np.ndarray) -> np.ndarray:
    """Normalize IR frame to uint8 0..255 using robust percentile scaling."""
    ir = ir_frame.astype(np.float32)
    valid = ir[np.isfinite(ir)]
    if valid.size > 0:
        p1, p99 = np.percentile(valid, [1, 99])
        if p99 <= p1:
            p1, p99 = float(valid.min()), float(valid.max())
    else:
        p1, p99 = 0.0, 1.0
    ir = np.clip((ir - p1) / max(1e-6, (p99 - p1)), 0.0, 1.0)
    return (ir * 255.0).astype(np.uint8)

def bgra_to_bgr(bgra: np.ndarray) -> np.ndarray:
    """Drop alpha channel if present."""
    return bgra[..., :3].copy() if bgra.shape[-1] == 4 else bgra

def _pad_to_same_height(imgs):
    H = max(i.shape[0] for i in imgs)
    return [cv2.copyMakeBorder(i, 0, H - i.shape[0], 0, 0, cv2.BORDER_CONSTANT) for i in imgs]

def _center_single(img, width):
    pad = max(0, (width - img.shape[1]) // 2)
    return cv2.copyMakeBorder(img, 0, 0, pad, width - img.shape[1] - pad, cv2.BORDER_CONSTANT)

def make_grid(imgs, titles, cell_w=640):
    """Tile up to 4 BGR images into a labeled grid canvas."""
    processed = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    for img, title in zip(imgs, titles):
        if img is None or title is None:
            continue
        h, w = img.shape[:2]
        scale = cell_w / float(w)
        resized = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        overlay = resized.copy()
        cv2.rectangle(overlay, (0, 0), (overlay.shape[1], 28), (0, 0, 0), thickness=-1)
        cv2.addWeighted(overlay, 0.35, resized, 0.65, 0, resized)
        cv2.putText(resized, title, (10, 20), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        processed.append(resized)

    if not processed:
        return None

    if len(processed) == 1:
        return processed[0]
    if len(processed) == 2:
        row = _pad_to_same_height(processed[:2])
        return np.hstack(row)
    if len(processed) == 3:
        row1 = np.hstack(_pad_to_same_height(processed[:2]))
        row2 = _center_single(processed[2], row1.shape[1])
        return np.vstack([row1, row2])
    row1 = np.hstack(_pad_to_same_height(processed[:2]))
    row2 = np.hstack(_pad_to_same_height(processed[2:4]))
    W = max(row1.shape[1], row2.shape[1])
    row1 = cv2.copyMakeBorder(row1, 0, 0, 0, W - row1.shape[1], cv2.BORDER_CONSTANT)
    row2 = cv2.copyMakeBorder(row2, 0, 0, 0, W - row2.shape[1], cv2.BORDER_CONSTANT)
    return np.vstack([row1, row2])

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_all(rgb_bgr, depth_m, ir_f32, reg_bgr):
    """Save current frames to ./captures/ as PNGs (depth/IR use 16-bit)."""
    ensure_dir("captures")
    ts = time.strftime("%Y%m%d_%H%M%S")
    if rgb_bgr is not None:
        cv2.imwrite(f"captures/{ts}_rgb.png", rgb_bgr)
    if depth_m is not None:
        depth_mm_u16 = np.clip(np.round(depth_m * 1000.0), 0, 65535).astype(np.uint16)
        cv2.imwrite(f"captures/{ts}_depth_mm.png", depth_mm_u16)
    if ir_f32 is not None:
        valid = ir_f32[np.isfinite(ir_f32)]
        if valid.size > 0:
            p1, p99 = np.percentile(valid, [1, 99])
            scaled = np.clip((ir_f32 - p1) / max(1e-6, (p99 - p1)), 0.0, 1.0)
            ir_u16 = (scaled * 65535.0).astype(np.uint16)
        else:
            ir_u16 = np.zeros_like(ir_f32, dtype=np.uint16)
        cv2.imwrite(f"captures/{ts}_ir.png", ir_u16)
    if reg_bgr is not None:
        cv2.imwrite(f"captures/{ts}_registered.png", reg_bgr)
    print(f"[INFO] Saved frames to captures/ with timestamp {ts}")

def handle_key(key: int, state: dict, on_save):
    """Update state based on keyboard input. Return True to request quit."""
    if key in (27, ord('q')):
        return True
    if key == ord(' '):
        state['pause'] = not state['pause']
    if key == ord('s'):
        on_save()
    if key == ord('g'):
        state['use_registration'] = not state['use_registration']
    if key == ord('1'):
        state['layout_mode'] = 1
    if key == ord('2'):
        state['layout_mode'] = 2
    if key == ord('3'):
        state['layout_mode'] = 3
    if key in (ord('+'), ord(']')):
        state['depth_max_m'] = min(state['depth_max_m'] + 0.1, 10.0)
    if key in (ord('-'), ord('['), ord('_')):
        state['depth_max_m'] = max(state['depth_max_m'] - 0.1, state['depth_min_m'] + 0.1)
    if key == ord('='):
        state['depth_min_m'] = min(state['depth_min_m'] + 0.1, state['depth_max_m'] - 0.1)
    if key == ord('r'):
        state['depth_min_m'] = DEPTH_MIN_M_DEFAULT
        state['depth_max_m'] = DEPTH_MAX_M_DEFAULT
    return False

def main():
    from pylibfreenect2 import (
        Freenect2, SyncMultiFrameListener, FrameType, Frame, FrameMap, Registration
    )

    # Choose packet pipeline and open device
    pipeline = try_import_pipeline()
    fn = Freenect2()
    if fn.enumerateDevices() == 0:
        print("[FATAL] No Kinect v2 devices found.")
        return
    serial = fn.getDeviceSerialNumber(0)
    dev = fn.openDevice(serial, pipeline=pipeline)

    # Subscribe to color, depth, IR
    listener = SyncMultiFrameListener(FrameType.Color | FrameType.Depth | FrameType.Ir)
    dev.setColorFrameListener(listener)
    dev.setIrAndDepthFrameListener(listener)
    dev.start()

    # Prepare registration — IMPORTANT: (ir_params, rgb_params) order
    ir_params = dev.getIrCameraParams()
    rgb_params = dev.getColorCameraParams()
    registration = Registration(ir_params, rgb_params)

    # Buffers for registration outputs (depth resolution)
    undistorted = Frame(512, 424, 4)  # float
    registered  = Frame(512, 424, 4)  # BGRA

    # Viewer state
    state = dict(
        pause=False,
        use_registration=True,
        layout_mode=1,
        depth_min_m=DEPTH_MIN_M_DEFAULT,
        depth_max_m=DEPTH_MAX_M_DEFAULT,
        fps=0.0,
        t_prev=time.time(),
        alpha=0.9,
    )

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        last_canvas = None
        while True:
            if state['pause']:
                # Still process keys while paused
                if last_canvas is not None:
                    cv2.imshow(WINDOW_NAME, last_canvas)
                key = cv2.waitKey(30) & 0xFF
                if handle_key(key, state, on_save=lambda: None):
                    break
                continue

            # Acquire a synchronized frame set
            fmap = FrameMap()
            ok = listener.waitForNewFrame(fmap, 2000)
            if not ok:
                continue

            # Frames
            color = fmap[FrameType.Color]   # BGRA uint8 (1080x1920)
            depth = fmap[FrameType.Depth]   # float32 mm (424x512)
            ir    = fmap[FrameType.Ir]      # float32 (424x512)

            # --- when reading frames from fmap ---
            color_bgra = color.asarray(np.uint8)          # was: color.asarray()
            depth_mm   = depth.asarray(np.float32)        # was: depth.asarray()
            ir_f32     = ir.asarray(np.float32)           # was: ir.asarray()

            # --- convert BGRA -> BGR (drop alpha) ---
            color_bgr = bgra_to_bgr(color_bgra)

            depth_m   = depth_mm * 0.001

            # Registration (color aligned to depth)
            if state['use_registration']:
                registration.apply(color, depth, undistorted, registered)
                reg_bgra  = registered.asarray(np.uint8)      # was: registered.asarray()
                reg_bgr   = bgra_to_bgr(reg_bgra)

                undist_m  = undistorted.asarray(np.float32) * 0.001  # was: undistorted.asarray() * 0.001
                depth_for_viz = undist_m  # nicer visualization when undistorted
            else:
                reg_bgr = None
                depth_for_viz = depth_m

            # Build visualization images
            depth_viz = depth_to_colormap(depth_for_viz, state['depth_min_m'], state['depth_max_m'])
            ir_viz    = cv2.cvtColor(ir_to_gray(ir_f32), cv2.COLOR_GRAY2BGR)
            rgb_viz   = color_bgr
            reg_viz   = reg_bgr

            # Compose layout
            if state['layout_mode'] == 1:
                imgs   = [rgb_viz, depth_viz, ir_viz, reg_viz]
                titles = [
                    "RGB 1080p",
                    f"Depth {state['depth_min_m']:.2f}-{state['depth_max_m']:.2f} m",
                    "IR",
                    "Registered (Color→Depth)" if reg_viz is not None else None
                ]
            elif state['layout_mode'] == 2:
                imgs   = [rgb_viz, depth_viz]
                titles = ["RGB 1080p", f"Depth {state['depth_min_m']:.2f}-{state['depth_max_m']:.2f} m"]
            else:
                imgs   = [ir_viz, depth_viz]
                titles = ["IR", f"Depth {state['depth_min_m']:.2f}-{state['depth_max_m']:.2f} m"]

            pair = [(i, t) for i, t in zip(imgs, titles) if i is not None and t is not None]
            if pair:
                imgs, titles = zip(*pair)
                canvas = make_grid(list(imgs), list(titles), cell_w=640)
            else:
                canvas = None

            if canvas is None:
                canvas = np.zeros((480, 640, 3), np.uint8)

            # FPS overlay
            t_now = time.time()
            inst_fps = 1.0 / max(1e-6, (t_now - state['t_prev']))
            state['fps'] = state['alpha'] * state['fps'] + (1 - state['alpha']) * inst_fps
            state['t_prev'] = t_now
            hud = (f"FPS: {state['fps']:.1f} | Reg: {'ON' if state['use_registration'] else 'OFF'} | "
                   f"Layout: {state['layout_mode']} | "
                   f"DepthRange: {state['depth_min_m']:.2f}-{state['depth_max_m']:.2f} m")
            cv2.putText(canvas, hud, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

            cv2.imshow(WINDOW_NAME, canvas)
            last_canvas = canvas

            key = cv2.waitKey(1) & 0xFF
            if handle_key(key, state, on_save=lambda: save_all(rgb_viz, depth_m, ir_f32, reg_viz)):
                listener.release(fmap)
                break

            listener.release(fmap)

    finally:
        cv2.destroyAllWindows()
        try:
            dev.stop()
        except Exception:
            pass
        try:
            dev.close()
        except Exception:
            pass

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] {e}")
        raise
