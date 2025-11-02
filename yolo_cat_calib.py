# yolo_cat_calibrated.py — Kinect v2 (pylibfreenect2) + YOLO tracking + 4-corner homography + UDP to Unity
# Run: python yolo_cat_calibrated.py
from ultralytics import YOLO
import numpy as np
import cv2
import time
import json
import sys
import socket
import os

# -------------------- Config (YOLO / Tracking) --------------------
MODEL_PATH = "yolo11n.pt"   # Ultralytics model (n = small, fast)
IMG_SIZE   = 960
CONF_THR   = 0.40
IOU_THR    = 0.50
TRACKER    = "bytetrack.yaml"
MAX_DET    = 1              # single cat
CAT_ID_FALLBACK = 15        # COCO default for "cat" if lookup fails
WINDOW_NAME = "YOLO Cat (Kinect/Webcam)"

# -------------------- Config (Calibration / Mapping / UDP) --------------------
PROJ_W, PROJ_H = 1920, 1080              # Your Unity projection canvas resolution
H_FILE         = "homography_cam_to_unity.npz"
EMA_ALPHA      = 0.6                      # Smoothing (higher = follow new value more)
DROPOUT_HOLD_S = 0.3                      # Hold last (U,V) if detection briefly drops
UNITY_IP       = "127.0.0.1"
UNITY_PORT     = 9000

# -------------------- UDP socket --------------------
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def send_to_unity(U, V, W, Ht, track_id, conf):
    """Send projection-space position and optional bbox size to Unity via UDP."""
    msg = {
        "t": time.time(),
        "id": int(track_id),
        "conf": float(conf),
        "Ux": float(U),
        "Uy": float(V),
        "Ux_n": float(U / PROJ_W),
        "Uy_n": float(V / PROJ_H),
        "Bw": float(W),
        "Bh": float(Ht),
    }
    sock.sendto(json.dumps(msg).encode("utf-8"), (UNITY_IP, UNITY_PORT))

# -------------------- Kinect v2 Grabber --------------------
def make_kinect_grabber():
    """
    Returns (dev, fn, grab_fn, close_fn) if Kinect v2 is available,
    else (None, None, None, None).
    grab_fn() -> np.ndarray BGR frame or None
    close_fn() -> safely stop/close device
    """
    try:
        from pylibfreenect2 import (
            Freenect2, SyncMultiFrameListener, FrameType,
            OpenGLPacketPipeline, OpenCLPacketPipeline, CpuPacketPipeline,
            FrameMap
        )
    except Exception:
        return None, None, None, None  # pylibfreenect2 not present

    # Choose best packet pipeline available
    try:
        pipeline = OpenGLPacketPipeline()
    except Exception:
        try:
            pipeline = OpenCLPacketPipeline()
        except Exception:
            pipeline = CpuPacketPipeline()

    fn = Freenect2()
    if fn.enumerateDevices() == 0:
        return None, None, None, None

    serial = fn.getDeviceSerialNumber(0)
    dev = fn.openDevice(serial, pipeline=pipeline)

    # Color only; add FrameType.Depth | FrameType.Ir if needed later
    listener = SyncMultiFrameListener(FrameType.Color)
    dev.setColorFrameListener(listener)
    dev.start()

    # Warm-up
    t0 = time.time()
    while time.time() - t0 < 0.3:
        fmap = FrameMap()
        ok = listener.waitForNewFrame(fmap, 1000)
        if ok:
            listener.release(fmap)

    def grab():
        fmap = FrameMap()
        ok = listener.waitForNewFrame(fmap, 2000)
        if not ok:
            return None
        color = fmap[FrameType.Color]     # pylibfreenect2.Frame
        arr = color.asarray()             # (1080, 1920, 4) BGRA uint8
        bgr = arr[..., :3].copy()         # drop alpha, keep BGR
        listener.release(fmap)
        return bgr

    def close_fn():
        try:
            dev.stop()
        except Exception:
            pass
        try:
            dev.close()
        except Exception:
            pass

    return dev, fn, grab, close_fn

# -------------------- Webcam Grabber --------------------
def make_webcam_grabber(index=0):
    """
    Returns (cap, grab_fn) if webcam opens, else (None, None).
    grab_fn() -> np.ndarray BGR frame or None
    """
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        return None, None

    def grab():
        ok, frame = cap.read()
        if not ok:
            return None
        return frame

    return cap, grab

# -------------------- Calibration / Homography helpers --------------------
def load_homography(path=H_FILE):
    if not os.path.exists(path):
        return None, None, None
    data = np.load(path, allow_pickle=True)
    return data["H"], data["cam_pts"], data["proj_pts"]

def save_homography(H, cam_pts, proj_pts, path=H_FILE):
    np.savez(path, H=H, cam_pts=np.array(cam_pts, dtype=np.float32), proj_pts=np.array(proj_pts, dtype=np.float32))
    print(f"[OK] Homography saved to {path}")

def cam_to_proj(u, v, H, clamp=True):
    """Map camera pixel (u,v) -> projection pixel (U,V) via homography H."""
    p = np.array([[u, v, 1.0]], dtype=np.float32).T
    q = H @ p
    U = float((q[0] / q[2])[0])
    V = float((q[1] / q[2])[0])
    if clamp:
        U = float(np.clip(U, 0, PROJ_W - 1))
        V = float(np.clip(V, 0, PROJ_H - 1))
    return U, V

def inside_quad(pt, quad):
    """Point-in-quad test (quad = TL,TR,BR,BL camera points)."""
    if quad is None:
        return True
    poly = np.array(quad, dtype=np.int32)
    return cv2.pointPolygonTest(poly, pt, False) >= 0

def calibrate_homography_with_grabber(grab_frame, save_path=H_FILE, proj_w=PROJ_W, proj_h=PROJ_H):
    """
    Calibrate using the current video source (Kinect or Webcam).
    Click TL -> TR -> BR -> BL in the camera view; Enter to save; 'r' to reset; 'q' to abort.
    Returns (H, cam_pts, proj_pts).
    """
    pts = []
    win = "Calibration - click TL, TR, BR, BL; Enter=save, r=reset, q=quit"
    cv2.namedWindow(win)

    def on_mouse(event, x, y, flags, param):
        nonlocal pts
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(pts) < 4:
                pts.append((x, y))

    cv2.setMouseCallback(win, on_mouse)

    while True:
        frame = grab_frame()
        if frame is None:
            continue
        vis = frame.copy()
        cv2.putText(vis, "Click 4 projection corners in camera view: TL -> TR -> BR -> BL",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 200, 50), 2)
        cv2.putText(vis, f"Points: {pts}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 200, 50), 2)
        for i, (x, y) in enumerate(pts):
            cv2.circle(vis, (x, y), 6, (0, 255, 255), -1)
            cv2.putText(vis, f"{i}", (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow(win, vis)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('r'):
            pts = []
        elif k == 13:  # Enter
            if len(pts) == 4:
                cam_pts = np.float32(pts)
                proj_pts = np.float32([[0, 0], [proj_w, 0], [proj_w, proj_h], [0, proj_h]])
                H, _ = cv2.findHomography(cam_pts, proj_pts, method=cv2.RANSAC)
                save_homography(H, cam_pts, proj_pts, save_path)
                cv2.destroyWindow(win)
                return H, cam_pts, proj_pts
            else:
                print("[!] Need 4 points before Enter.")
        elif k == ord('q') or k == 27:
            cv2.destroyWindow(win)
            raise SystemExit("Calibration aborted")

# -------------------- Main --------------------
def main():
    model = YOLO(MODEL_PATH)

    # Resolve cat class id robustly (prefer model.names lookup)
    try:
        inv_names = {v: k for k, v in model.names.items()}
        CAT_ID = int(inv_names.get("cat", CAT_ID_FALLBACK))
    except Exception:
        CAT_ID = CAT_ID_FALLBACK

    # Prefer Kinect; fallback to webcam
    dev = fn = None
    cap = None
    grabber = None
    kinect_close = None
    source_name = ""

    dev, fn, kinect_grab, kinect_close = make_kinect_grabber()
    if kinect_grab is not None:
        print("[INFO] Using Kinect v2 as source.")
        grabber = kinect_grab
        source_name = "kinect"
    else:
        print("[WARN] No Kinect v2 found; falling back to webcam 0.")
        cap, cam_grab = make_webcam_grabber(0)
        if cam_grab is None:
            raise RuntimeError("No valid video source (Kinect or webcam).")
        grabber = cam_grab
        source_name = "webcam0"

    # Load or create homography
    H, cam_quad, proj_quad = load_homography(H_FILE)
    if H is None:
        print("[*] No homography file found; starting calibration on current source...")
        H, cam_quad, proj_quad = calibrate_homography_with_grabber(grabber, save_path=H_FILE)

    # FPS HUD + EMA smoothing
    fps = 0.0
    alpha_fps = 0.9
    t_prev = time.time()
    ema_U = None
    ema_V = None
    last_seen_ts = 0.0

    try:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        while True:
            frame = grabber()
            if frame is None:
                continue

            # Run tracking on the current frame (single-call mode to keep your structure)
            results_list = model.track(
                source=frame,
                imgsz=IMG_SIZE,
                conf=CONF_THR,
                iou=IOU_THR,
                classes=[CAT_ID],
                tracker=TRACKER,
                persist=True,
                verbose=False,
                stream=False,
                max_det=MAX_DET
            )
            res = results_list[0]

            # Default overlay text
            hud_lines = []

            # Draw detection and compute mapped coordinates
            sent_this_frame = False
            if hasattr(res, "boxes") and res.boxes is not None and len(res.boxes) > 0:
                # Choose the largest bbox (single cat use-case)
                xyxy = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                ids = res.boxes.id.cpu().numpy() if res.boxes.id is not None else np.full((len(xyxy),), -1)
                areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
                i = int(np.argmax(areas))

                x1, y1, x2, y2 = xyxy[i].astype(int).tolist()
                conf = float(confs[i])
                tid = int(ids[i])

                # Camera center
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0

                # Optional: ignore points outside calibrated quad
                if cam_quad is None or inside_quad((cx, cy), cam_quad):
                    # Camera -> Projection
                    U, V = cam_to_proj(cx, cy, H, clamp=True)

                    # EMA smoothing
                    if ema_U is None:
                        ema_U, ema_V = U, V
                    else:
                        ema_U = EMA_ALPHA * U + (1 - EMA_ALPHA) * ema_U
                        ema_V = EMA_ALPHA * V + (1 - EMA_ALPHA) * ema_V

                    # Send to Unity
                    send_to_unity(ema_U, ema_V, (x2 - x1), (y2 - y1), tid, conf)
                    last_seen_ts = time.time()
                    sent_this_frame = True

                    # Draw bbox + labels
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"cat {tid if tid>=0 else '-'} {conf:.2f}"
                    cv2.putText(frame, label, (x1, max(0, y1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    # Draw camera center
                    cv2.circle(frame, (int(cx), int(cy)), 5, (0, 255, 0), -1)

                    # Show projected (U,V) on HUD
                    hud_lines.append(f"Proj(U,V)=({int(ema_U)},{int(ema_V)})")
                else:
                    hud_lines.append("Center outside calibrated quad — skipped send.")
            else:
                # Brief dropout: keep sending the last smoothed point to avoid flicker
                if (time.time() - last_seen_ts) < DROPOUT_HOLD_S and ema_U is not None and ema_V is not None:
                    send_to_unity(ema_U, ema_V, 0, 0, -1, 0.0)
                    sent_this_frame = True
                    hud_lines.append("Dropout hold → sent last (U,V).")

            # FPS HUD
            t_now = time.time()
            inst_fps = 1.0 / max(1e-6, (t_now - t_prev))
            fps = alpha_fps * fps + (1 - alpha_fps) * inst_fps
            t_prev = t_now

            # Compose HUD
            hud_head = f"{source_name} | {fps:.1f} FPS"
            if sent_this_frame:
                hud_head += " | UDP✓"
            else:
                hud_head += " | UDP-"
            cv2.putText(frame, hud_head, (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_base = 50
            for j, line in enumerate(hud_lines[:3]):
                cv2.putText(frame, line, (10, y_base + 22 * j),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Optional: visualize calibrated quad in camera space
            if cam_quad is not None:
                quad = np.array(cam_quad, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(frame, [quad], isClosed=True, color=(255, 200, 0), thickness=2)

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):  # ESC or q
                break
            elif key in (ord('c'), ord('C')):  # Re-calibrate on demand
                print("[*] Starting re-calibration...")
                H, cam_quad, _ = calibrate_homography_with_grabber(grabber, save_path=H_FILE)

    finally:
        cv2.destroyAllWindows()
        if cap is not None:
            cap.release()
        if kinect_close is not None:
            kinect_close()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] {e}", file=sys.stderr)
        sys.exit(1)
