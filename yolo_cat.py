# yolo_cat.py — Kinect v2 (pylibfreenect2) + YOLO tracking with webcam fallback
# Run: python yolo_cat.py
from ultralytics import YOLO
import numpy as np
import cv2
import time
import json
import sys

# -------------------- Config --------------------
MODEL_PATH = "yolo11n.pt"  # Ultralytics model
CAT_ID = 15                # COCO: 15 = cat
IMG_SIZE = 960
CONF_THR = 0.40
IOU_THR = 0.50
TRACKER = "bytetrack.yaml"
WINDOW_NAME = "YOLO Cat (Kinect/Webcam)"

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

    # Choose best packet pipeline
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

    # Color only; add FrameType.Depth | FrameType.Ir if you need them
    listener = SyncMultiFrameListener(FrameType.Color)
    dev.setColorFrameListener(listener)
    dev.start()

    # Warm-up using FrameMap API
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
        bgr = arr[..., :3].copy()         # drop alpha
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

# -------------------- Main --------------------
def main():
    model = YOLO(MODEL_PATH)

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

    # FPS HUD
    fps = 0.0
    alpha = 0.9  # smoothing
    t_prev = time.time()

    try:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        while True:
            frame = grabber()
            if frame is None:
                continue

            # Single-frame tracking (no generator)
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
            )
            res = results_list[0]

            # Draw detection if any
            if hasattr(res, "boxes") and res.boxes is not None and len(res.boxes) > 0:
                i = 0
                xyxy = res.boxes.xyxy[i].cpu().numpy().astype(int)
                conf = float(res.boxes.conf[i])
                tid = int(res.boxes.id[i]) if res.boxes.id is not None else -1

                x1, y1, x2, y2 = xyxy.tolist()
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"cat {tid if tid>=0 else '-'} {conf:.2f}"
                cv2.putText(frame, label, (x1, max(0, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # (optional) normalized payload
                # H, W = frame.shape[:2]
                # w = (x2 - x1) / W; h = (y2 - y1) / H
                # cx = (x1 + x2) / (2 * W); cy = (y1 + y2) / (2 * H)
                # print(json.dumps(dict(cx_n=cx, cy_n=cy, w_n=w, h_n=h, conf=conf, track_id=tid)))

            # FPS HUD
            t_now = time.time()
            inst_fps = 1.0 / max(1e-6, (t_now - t_prev))
            fps = alpha * fps + (1 - alpha) * inst_fps
            t_prev = t_now
            hud = f"{source_name} | {fps:.1f} FPS"
            cv2.putText(frame, hud, (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):  # ESC or q
                break

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
