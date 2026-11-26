#!/usr/bin/env python3
"""Live (frame-by-frame) YOLOv11 prediction on a webcam, video file or RTSP/HTTP stream.

Examples
--------
    python live_predict.py --weights yolo11s_wildlife_best.pt --source 0                 # webcam, live window
    python live_predict.py --weights yolo11s_wildlife_best.pt --source clip.mp4 --save out.mp4
    python live_predict.py --weights yolo11s_wildlife_best.pt --source rtsp://host/stream --track

Press `q` in the preview window to quit. Use --no-show on headless machines (e.g. Kaggle/servers).
"""
import argparse
import time

import cv2
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default="yolo11s_wildlife_best.pt")
    ap.add_argument("--source", default="0", help="camera index, video path or stream URL")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None, help="0 for first GPU, 'cpu', ... (default: auto)")
    ap.add_argument("--save", default=None, help="write annotated video to this path (.mp4)")
    ap.add_argument("--track", action="store_true", help="enable ByteTrack multi-object tracking (persistent IDs)")
    ap.add_argument("--no-show", action="store_true", help="do not open a preview window")
    a = ap.parse_args()

    model = YOLO(a.weights)
    src = int(a.source) if a.source.isdigit() else a.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"cannot open source: {a.source}")
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
    writer = cv2.VideoWriter(a.save, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (w, h)) if a.save else None

    t_start, n, lat = time.perf_counter(), 0, []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = time.perf_counter()
        kw = dict(conf=a.conf, imgsz=a.imgsz, verbose=False, device=a.device)
        r = (model.track(frame, persist=True, tracker="bytetrack.yaml", **kw) if a.track else model.predict(frame, **kw))[0]
        dt = (time.perf_counter() - t) * 1000
        lat.append(dt)
        n += 1
        vis = r.plot(line_width=3)
        fps = n / (time.perf_counter() - t_start)
        cv2.rectangle(vis, (0, 0), (470, 40), (0, 0, 0), -1)
        cv2.putText(vis, f"{dt:5.1f} ms | {fps:5.1f} FPS | {len(r.boxes)} obj", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        if writer:
            writer.write(vis)
        if not a.no_show:
            cv2.imshow("YOLO11 live prediction (q = quit)", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    if n:
        print(f"{n} frames | mean latency {sum(lat)/n:.1f} ms | end-to-end {n/(time.perf_counter()-t_start):.1f} FPS")


if __name__ == "__main__":
    main()
