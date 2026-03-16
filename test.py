#!/usr/bin/env python3
"""
A* Pathfinder with OpenCV Camera Input + Flask Web Interface
------------------------------------------------------------
Supports: PC webcam (default) and Raspberry Pi Camera (--picamera flag)

Usage:
  python astar_pathfinder.py                  # PC webcam
  python astar_pathfinder.py --picamera       # Raspberry Pi Camera
  python astar_pathfinder.py --port 5000      # Custom port (default 5000)
  python astar_pathfinder.py --width 640 --height 480

Install deps:
  pip install flask opencv-python numpy
  # On Raspberry Pi also: pip install picamera2
"""

import cv2
import numpy as np
import heapq
import threading
import time
import base64
import json
import argparse
from flask import Flask, Response, jsonify, request

# ══════════════════════════════════════════════════════════════════════════════
#  CAMERA ABSTRACTION  –  PC webcam  OR  Raspberry Pi Camera
# ══════════════════════════════════════════════════════════════════════════════

def find_available_cameras(max_index=10):
    """Scan camera indices and return a list of working ones."""
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(i)
        if cap and cap.isOpened():
            ok, _ = cap.read()
            if ok:
                found.append(i)
            cap.release()
    return found


def make_test_pattern(width, height, t):
    """Animated test pattern used when no real camera is available."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    # Scrolling gradient background
    for y in range(height):
        v = int(((y + t * 2) % height) / height * 80)
        frame[y, :] = (v, 0, v // 2)
    # Fake obstacles: static rectangles + a moving circle
    cv2.rectangle(frame, (80, 80),   (200, 180), (0, 80, 160), -1)
    cv2.rectangle(frame, (350, 200), (500, 320), (0, 60, 120), -1)
    cv2.rectangle(frame, (150, 300), (280, 420), (0, 80, 160), -1)
    cx = int(width * 0.5 + np.cos(t * 0.05) * 120)
    cy = int(height * 0.55 + np.sin(t * 0.07) * 80)
    cv2.circle(frame, (cx, cy), 45, (0, 50, 110), -1)
    # Label
    cv2.putText(frame, "NO CAMERA — TEST PATTERN", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 200, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Run with --camera <index> to use a real camera", (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (120, 120, 180), 1, cv2.LINE_AA)
    return frame


class Camera:
    def __init__(self, use_picamera=False, width=640, height=480, cam_index=None):
        self.use_picamera = use_picamera
        self.width     = width
        self.height    = height
        self.cam_index = cam_index   # None = auto-scan
        self._picam    = None
        self._cap      = None
        self._test_mode = False
        self._tick      = 0
        self._init()

    def _init(self):
        # ── Raspberry Pi Camera ──────────────────────────────────────────────
        if self.use_picamera:
            try:
                from picamera2 import Picamera2
                self._picam = Picamera2()
                cfg = self._picam.create_preview_configuration(
                    main={"size": (self.width, self.height), "format": "RGB888"}
                )
                self._picam.configure(cfg)
                self._picam.start()
                time.sleep(0.5)
                print("[Camera] ✓ Raspberry Pi Camera (picamera2)")
                return
            except Exception as e:
                print(f"[Camera] picamera2 failed ({e}), falling back to webcam.")
                self.use_picamera = False

        # ── PC Webcam ────────────────────────────────────────────────────────
        # Determine index to try
        if self.cam_index is not None:
            indices = [self.cam_index]
        else:
            print("[Camera] Scanning for available cameras…")
            indices = find_available_cameras()
            if indices:
                print(f"[Camera] Found cameras at indices: {indices}")
            else:
                indices = []

        for idx in indices:
            cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            if not cap.isOpened():
                cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                ok, _ = cap.read()
                if ok:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                    self._cap = cap
                    print(f"[Camera] ✓ Webcam opened at index {idx}")
                    return
                cap.release()

        # ── Fallback: animated test pattern ─────────────────────────────────
        print("\n[Camera] ⚠  No camera found – running in TEST PATTERN mode.")
        print("[Camera]    Pass --camera <index> or --picamera to use a real camera.\n")
        self._test_mode = True

    def read(self):
        if self._picam:
            frame = self._picam.capture_array()
            return True, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if self._cap and self._cap.isOpened():
            return self._cap.read()
        # Test pattern
        self._tick += 1
        return True, make_test_pattern(self.width, self.height, self._tick)

    def release(self):
        if self._picam:  self._picam.stop()
        if self._cap:    self._cap.release()


# ══════════════════════════════════════════════════════════════════════════════
#  OBSTACLE DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def build_obstacle_map(frame, dilate=18):
    """
    Returns a binary uint8 array the same size as frame (H×W).
    255 = blocked, 0 = free.
    Uses Canny edge detection + dilation so the robot keeps clearance.
    """
    gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur   = cv2.GaussianBlur(gray, (9, 9), 0)
    edges  = cv2.Canny(blur, 40, 120)
    kernel = np.ones((dilate, dilate), np.uint8)
    return cv2.dilate(edges, kernel, iterations=1)


# ══════════════════════════════════════════════════════════════════════════════
#  A* ALGORITHM
# ══════════════════════════════════════════════════════════════════════════════

def heuristic(a, b):
    """Octile distance – optimal heuristic for 8-directional grid."""
    dx, dy = abs(a[0]-b[0]), abs(a[1]-b[1])
    return max(dx, dy) + (1.4142 - 1) * min(dx, dy)

DIRS  = [(-1,-1),(0,-1),(1,-1),(-1,0),(1,0),(-1,1),(0,1),(1,1)]
COSTS = [ 1.4142,  1.0,  1.4142,  1.0,  1.0, 1.4142,  1.0, 1.4142]

def astar(obs_map, start, goal):
    """
    obs_map : H×W uint8, non-zero = blocked
    start, goal : (x, y) pixel tuples
    Returns list of (x,y) or None if no path.
    """
    h, w = obs_map.shape
    sx, sy = int(np.clip(start[0], 0, w-1)), int(np.clip(start[1], 0, h-1))
    gx, gy = int(np.clip(goal[0],  0, w-1)), int(np.clip(goal[1],  0, h-1))

    if obs_map[gy, gx] != 0:
        return None   # goal inside obstacle

    open_q   = [(0.0, (sx, sy))]
    came_from = {}
    g_cost    = {(sx, sy): 0.0}

    while open_q:
        _, cur = heapq.heappop(open_q)
        if cur == (gx, gy):
            path = []
            node = cur
            while node in came_from:
                path.append(node)
                node = came_from[node]
            path.append((sx, sy))
            path.reverse()
            return path

        cx, cy = cur
        for (dx, dy), cost in zip(DIRS, COSTS):
            nx, ny = cx+dx, cy+dy
            if 0 <= nx < w and 0 <= ny < h and obs_map[ny, nx] == 0:
                ng = g_cost[cur] + cost
                nb = (nx, ny)
                if ng < g_cost.get(nb, float('inf')):
                    g_cost[nb]    = ng
                    came_from[nb] = cur
                    heapq.heappush(open_q, (ng + heuristic(nb, (gx, gy)), nb))
    return None

def smooth_path(path, window=8):
    """Moving-average smoothing to make the path less jagged."""
    if not path or len(path) < window:
        return path
    out = []
    for i in range(len(path)):
        lo, hi = max(0, i-window//2), min(len(path), i+window//2+1)
        seg = path[lo:hi]
        out.append((int(sum(p[0] for p in seg)/len(seg)),
                    int(sum(p[1] for p in seg)/len(seg))))
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  SHARED APPLICATION STATE
# ══════════════════════════════════════════════════════════════════════════════

class State:
    def __init__(self):
        self.lock             = threading.Lock()
        self.frame            = None    # latest raw BGR frame
        self.obs_map          = None    # latest obstacle map (H×W uint8)
        self.prev_obs         = None    # previous map for change detection
        self.start            = None    # (x, y) or None
        self.goal             = None    # (x, y) or None
        self.path             = None    # list[(x,y)] or None
        self.path_error       = None    # error message string
        self.show_obs         = True    # overlay obstacle viz
        self.show_change      = True    # highlight changed regions
        self.dilate           = 18      # obstacle dilation radius
        self.computing        = False
        self.replanning       = False   # auto-replan in progress
        self.replan_requested = threading.Event()
        self.frame_w          = 640
        self.frame_h          = 480
        # Fraction of path waypoints that must become blocked to trigger replan
        self.replan_threshold = 0.04    # 4 % of waypoints
        self.replan_cooldown  = 0.5     # seconds between auto replans
        self._last_replan_t   = 0.0
        # Change heatmap for visualisation (H×W float32, decays each frame)
        self.change_heat      = None

S = State()


# ══════════════════════════════════════════════════════════════════════════════
#  CHANGE DETECTION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def path_obstruction_fraction(path, obs_map):
    """Return fraction of path waypoints that are now inside obstacles."""
    if not path or obs_map is None:
        return 0.0
    h, w = obs_map.shape
    blocked = sum(
        1 for (x, y) in path
        if 0 <= x < w and 0 <= y < h and obs_map[y, x] != 0
    )
    return blocked / len(path)


def significant_change_near_path(old_obs, new_obs, path, margin=30):
    """
    True when obstacle pixels changed substantially within `margin` px of path.
    Fast: only samples every 4th waypoint, dilates a mask, then XORs.
    """
    if old_obs is None or new_obs is None or not path:
        return False
    h, w = new_obs.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    for (x, y) in path[::4]:   # sample every 4th point for speed
        x1, y1 = max(0, x-margin), max(0, y-margin)
        x2, y2 = min(w, x+margin), min(h, y+margin)
        mask[y1:y2, x1:x2] = 255
    diff = cv2.bitwise_and(cv2.absdiff(old_obs, new_obs), mask)
    changed_px = cv2.countNonZero(diff)
    total_px   = max(1, cv2.countNonZero(mask))
    return (changed_px / total_px) > 0.02   # >2 % of near-path area changed


# ══════════════════════════════════════════════════════════════════════════════
#  CAMERA CAPTURE THREAD  – also drives change detection
# ══════════════════════════════════════════════════════════════════════════════

def camera_thread(cam: Camera):
    print("[Thread] Camera thread started.")
    _dilate_cache = None    # cache so we only rebuild map when dilate changes
    while True:
        ok, frame = cam.read()
        if not ok or frame is None:
            time.sleep(0.05)
            continue

        with S.lock:
            dilate = S.dilate
        obs = build_obstacle_map(frame, dilate=dilate)

        with S.lock:
            prev_obs = S.obs_map          # snapshot before overwrite
            S.frame   = frame.copy()
            S.obs_map = obs.copy()
            S.frame_h, S.frame_w = frame.shape[:2]

            # ── Update change heatmap ──────────────────────────────────────
            if prev_obs is not None and prev_obs.shape == obs.shape:
                diff = cv2.absdiff(prev_obs, obs).astype(np.float32)
                if S.change_heat is None or S.change_heat.shape != obs.shape:
                    S.change_heat = np.zeros(obs.shape, dtype=np.float32)
                S.change_heat = S.change_heat * 0.80 + diff   # exponential decay
            S.prev_obs = prev_obs

            # ── Check if the current path is still valid ───────────────────
            now = time.time()
            should_replan = False
            if (S.path is not None
                    and not S.computing
                    and not S.replanning
                    and (now - S._last_replan_t) > S.replan_cooldown):

                frac = path_obstruction_fraction(S.path, obs)
                if frac >= S.replan_threshold:
                    should_replan = True
                elif (prev_obs is not None
                        and significant_change_near_path(prev_obs, obs, S.path)):
                    should_replan = True

            if should_replan:
                S.replanning = True
                S._last_replan_t = now
                S.replan_requested.set()


# ══════════════════════════════════════════════════════════════════════════════
#  PATH COMPUTATION  – manual trigger + automatic replanner
# ══════════════════════════════════════════════════════════════════════════════

def _run_astar():
    """Core: grab state, run A*, write result back. Called by both triggers."""
    with S.lock:
        if S.start is None or S.goal is None or S.obs_map is None:
            S.computing  = False
            S.replanning = False
            return
        obs = S.obs_map.copy()
        st  = S.start
        go  = S.goal

    path = astar(obs, st, go)

    with S.lock:
        S.computing  = False
        S.replanning = False
        if path is None:
            S.path_error = "No path found – goal may be inside an obstacle."
            S.path       = None
        else:
            S.path_error = None
            S.path       = smooth_path(path)


def compute_path():
    """Called when user sets new points."""
    with S.lock:
        S.computing  = True
        S.path       = None
        S.path_error = None
    _run_astar()


def auto_replan_thread():
    """
    Dedicated thread that wakes up whenever the camera thread sets the
    replan_requested event, then reruns A* in the background.
    """
    print("[Thread] Auto-replanner thread started.")
    while True:
        S.replan_requested.wait()       # sleep until triggered
        S.replan_requested.clear()
        time.sleep(0.05)                # tiny debounce
        _run_astar()


# ══════════════════════════════════════════════════════════════════════════════
#  FRAME ANNOTATION
# ══════════════════════════════════════════════════════════════════════════════

def annotate(frame, obs_map, start, goal, path, show_obs,
             change_heat=None, replanning=False):
    out = frame.copy()

    # Change heatmap overlay (cyan glow where scene changed)
    if change_heat is not None and change_heat.shape[:2] == out.shape[:2]:
        heat_norm = np.clip(change_heat / 80.0, 0, 1)
        cyan_layer = np.zeros_like(out)
        cyan_layer[:, :, 0] = (heat_norm * 200).astype(np.uint8)  # B
        cyan_layer[:, :, 1] = (heat_norm * 220).astype(np.uint8)  # G
        out = cv2.addWeighted(out, 1.0, cyan_layer, 0.55, 0)

    # Obstacle overlay (semi-transparent red)
    if show_obs and obs_map is not None:
        red_layer = np.zeros_like(out)
        red_layer[:, :, 2] = obs_map
        out = cv2.addWeighted(out, 1.0, red_layer, 0.30, 0)

    # Path – colour shifts to orange while replanning
    if path and len(path) > 1:
        pts = np.array(path, dtype=np.int32).reshape(-1, 1, 2)
        if replanning:
            cv2.polylines(out, [pts], False, (0, 140, 255), 6, cv2.LINE_AA)  # orange glow
            cv2.polylines(out, [pts], False, (40, 200, 255), 2, cv2.LINE_AA)
        else:
            cv2.polylines(out, [pts], False, (40, 200, 40), 6, cv2.LINE_AA)  # green glow
            cv2.polylines(out, [pts], False, (180, 255, 180), 2, cv2.LINE_AA)

    # Start marker
    if start:
        cv2.circle(out, start, 12, (0, 230, 80),   -1, cv2.LINE_AA)
        cv2.circle(out, start, 14, (255, 255, 255),  2, cv2.LINE_AA)
        cv2.putText(out, "S", (start[0]-6, start[1]+5),
                    cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)

    # Goal marker
    if goal:
        cv2.circle(out, goal, 12, (0, 80, 255),    -1, cv2.LINE_AA)
        cv2.circle(out, goal, 14, (255, 255, 255),   2, cv2.LINE_AA)
        cv2.putText(out, "G", (goal[0]-6, goal[1]+5),
                    cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    # Replanning banner
    if replanning:
        h, w = out.shape[:2]
        cv2.rectangle(out, (0, 0), (w, 34), (0, 0, 0), -1)
        cv2.putText(out, "⟳  Replanning path…", (10, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 200, 255), 2, cv2.LINE_AA)

    return out


# ══════════════════════════════════════════════════════════════════════════════
#  FLASK APP
# ══════════════════════════════════════════════════════════════════════════════

flask_app = Flask(__name__)

# ── MJPEG stream ──────────────────────────────────────────────────────────────
def gen_frames():
    while True:
        with S.lock:
            frame      = S.frame
            obs        = S.obs_map
            start      = S.start
            goal       = S.goal
            path       = S.path
            show       = S.show_obs
            heat       = S.change_heat.copy() if S.change_heat is not None else None
            replanning = S.replanning

        if frame is None:
            time.sleep(0.05)
            continue

        annotated = annotate(frame, obs, start, goal, path, show,
                              change_heat=heat, replanning=replanning)
        _, buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
               buf.tobytes() + b'\r\n')
        time.sleep(0.033)   # ~30 fps cap

@flask_app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ── REST endpoints ─────────────────────────────────────────────────────────────
@flask_app.route('/set_points', methods=['POST'])
def set_points():
    """
    JSON body: { "start": [x, y], "goal": [x, y] }
    Coordinates are in VIDEO pixels (not canvas pixels).
    The web UI does the scaling before sending.
    """
    data = request.get_json(force=True)
    with S.lock:
        if 'start' in data and data['start']:
            S.start = tuple(int(v) for v in data['start'])
        if 'goal' in data and data['goal']:
            S.goal  = tuple(int(v) for v in data['goal'])
        S.path = None
        S.path_error = None

    threading.Thread(target=compute_path, daemon=True).start()
    return jsonify(ok=True)

@flask_app.route('/clear', methods=['POST'])
def clear():
    with S.lock:
        S.start = S.goal = S.path = S.path_error = None
    return jsonify(ok=True)

@flask_app.route('/toggle_obstacles', methods=['POST'])
def toggle_obstacles():
    with S.lock:
        S.show_obs = not S.show_obs
    return jsonify(show=S.show_obs)

@flask_app.route('/set_dilate', methods=['POST'])
def set_dilate():
    d = int(request.get_json(force=True).get('dilate', 18))
    with S.lock:
        S.dilate = max(1, min(d, 60))
    return jsonify(dilate=S.dilate)

@flask_app.route('/set_replan', methods=['POST'])
def set_replan():
    d = request.get_json(force=True)
    with S.lock:
        if 'threshold' in d:
            S.replan_threshold = max(0.01, min(float(d['threshold']), 1.0))
        if 'cooldown' in d:
            S.replan_cooldown  = max(0.2,  min(float(d['cooldown']),  10.0))
    return jsonify(ok=True)

@flask_app.route('/status')
def status():
    with S.lock:
        return jsonify(
            start      = list(S.start) if S.start else None,
            goal       = list(S.goal)  if S.goal  else None,
            has_path   = S.path is not None,
            path_len   = len(S.path) if S.path else 0,
            computing  = S.computing,
            replanning = S.replanning,
            error      = S.path_error,
            show_obs   = S.show_obs,
            dilate     = S.dilate,
            threshold  = S.replan_threshold,
            cooldown   = S.replan_cooldown,
            frame_w    = S.frame_w,
            frame_h    = S.frame_h,
        )

# ── Main HTML page ─────────────────────────────────────────────────────────────
@flask_app.route('/')
def index():
    return Response(HTML, mimetype='text/html')

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>A* Pathfinder</title>
<style>
:root {
  --bg:     #0a0a0a;
  --surface:#111111;
  --border: #222222;
  --accent: #ffffff;
  --dim:    #444444;
  --muted:  #666666;
  --text:   #cccccc;
  --green:  #4ade80;
  --red:    #f87171;
}
* { box-sizing:border-box; margin:0; padding:0; }
body { background:var(--bg); color:var(--text); font-family:'Segoe UI',system-ui,sans-serif;
       display:flex; flex-direction:column; height:100vh; overflow:hidden;
       font-size:13px; }

/* Header */
header { padding:10px 16px; display:flex; align-items:center; gap:10px;
         border-bottom:1px solid var(--border); flex-shrink:0; }
header h1 { font-size:.85rem; font-weight:500; color:var(--text); letter-spacing:.02em; }
.live-dot { width:6px; height:6px; border-radius:50%; background:var(--green);
            box-shadow:0 0 5px var(--green); flex-shrink:0; }

/* Layout */
.main { display:flex; flex:1; gap:0; min-height:0; }

/* Video */
#canvas-wrap { position:relative; flex:1; background:#000; cursor:crosshair; }
#videoEl { width:100%; height:100%; object-fit:contain; display:block; pointer-events:none; }
#clickCanvas { position:absolute; top:0; left:0; width:100%; height:100%; cursor:crosshair; }

/* Side panel */
.panel { width:220px; display:flex; flex-direction:column; border-left:1px solid var(--border);
         overflow-y:auto; background:var(--surface); }
.section { padding:14px 16px; border-bottom:1px solid var(--border); }
.section-title { font-size:.65rem; text-transform:uppercase; letter-spacing:.1em;
                 color:var(--muted); margin-bottom:10px; }

/* Buttons */
.btn { display:block; width:100%; padding:7px 10px; border:1px solid var(--border);
       border-radius:4px; font-size:.8rem; font-weight:500; cursor:pointer;
       transition:background .1s, color .1s; background:transparent; color:var(--muted); }
.btn + .btn { margin-top:5px; }
.btn:hover { background:var(--border); color:var(--text); }
.btn.active { background:var(--accent); color:#000; border-color:var(--accent); }
.btn-ghost { color:var(--muted); }
.btn-ghost:hover { color:var(--red); border-color:var(--red); background:transparent; }

/* Point rows */
.pt-row { display:flex; align-items:center; gap:6px; padding:5px 0;
          border-bottom:1px solid var(--border); font-size:.78rem; }
.pt-row:last-of-type { border:none; }
.pt-label { color:var(--muted); width:36px; flex-shrink:0; }
.pt-coord { flex:1; color:var(--text); font-variant-numeric:tabular-nums; }
.dot { width:7px; height:7px; border-radius:50%; flex-shrink:0; background:var(--dim); }
.dot-green { background:var(--green); }
.dot-red   { background:var(--red); }

/* Sliders */
.slider-row { margin-bottom:10px; }
.slider-row:last-of-type { margin-bottom:0; }
.slider-header { display:flex; justify-content:space-between; margin-bottom:4px;
                 font-size:.75rem; color:var(--muted); }
.slider-header span:last-child { color:var(--text); }
input[type=range] { width:100%; accent-color:var(--accent); height:2px; }

/* Status bar */
#statusBar { padding:6px 16px; font-size:.72rem; color:var(--muted);
             border-top:1px solid var(--border); flex-shrink:0; white-space:nowrap;
             overflow:hidden; text-overflow:ellipsis; }
#statusText { color:var(--text); }
.computing-anim { color:#f59e0b; animation:pulse .9s infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.3} }
</style>
</head>
<body>

<header>
  <span class="live-dot"></span>
  <h1>A* Pathfinder</h1>
</header>

<div class="main">

  <div id="canvas-wrap">
    <img id="videoEl" src="/video_feed" alt="camera stream">
    <canvas id="clickCanvas"></canvas>
  </div>

  <div class="panel">

    <div class="section">
      <div class="section-title">Place</div>
      <button class="btn active" id="modeStart" onclick="setMode('start')">Start</button>
      <button class="btn" id="modeGoal" onclick="setMode('goal')">Goal</button>
    </div>

    <div class="section">
      <div class="section-title">Points</div>
      <div class="pt-row">
        <span class="pt-label">Start</span>
        <span class="pt-coord" id="startCoord">—</span>
        <span class="dot" id="startDot"></span>
      </div>
      <div class="pt-row">
        <span class="pt-label">Goal</span>
        <span class="pt-coord" id="goalCoord">—</span>
        <span class="dot" id="goalDot"></span>
      </div>
      <div class="pt-row">
        <span class="pt-label">Path</span>
        <span class="pt-coord" id="pathInfo">—</span>
        <span class="dot" id="pathDot"></span>
      </div>
      <button class="btn btn-ghost" style="margin-top:10px;" onclick="clearAll()">Clear</button>
    </div>

    <div class="section">
      <div class="section-title">Obstacles</div>
      <div class="slider-row">
        <div class="slider-header"><span>Clearance</span><span id="dilateVal">18</span></div>
        <input type="range" id="dilateSlider" min="2" max="50" value="18"
               oninput="document.getElementById('dilateVal').textContent=this.value"
               onchange="setDilate(this.value)">
      </div>
      <button class="btn" onclick="toggleObs()" id="obsBtn" style="margin-top:4px;">Hide overlay</button>
    </div>

    <div class="section">
      <div class="section-title">Replanning</div>
      <div class="pt-row" style="margin-bottom:8px;">
        <span class="pt-label">Status</span>
        <span class="pt-coord" id="replanStatus">Idle</span>
        <span class="dot" id="replanDot"></span>
      </div>
      <div class="slider-row">
        <div class="slider-header"><span>Sensitivity</span><span id="threshVal">4%</span></div>
        <input type="range" id="threshSlider" min="1" max="30" value="4"
               oninput="document.getElementById('threshVal').textContent=this.value+'%'"
               onchange="setReplan()">
      </div>
      <div class="slider-row">
        <div class="slider-header"><span>Min interval</span><span id="cooldownVal">0.5s</span></div>
        <input type="range" id="cooldownSlider" min="2" max="30" value="5"
               oninput="document.getElementById('cooldownVal').textContent=(this.value/10).toFixed(1)+'s'"
               onchange="setReplan()">
      </div>
    </div>

  </div>
</div>

<div id="statusBar">Status: <span id="statusText">Click the video to set Start and Goal</span></div>

<script>
let mode = 'start';
let showObs = true;
let frameW = 640, frameH = 480;

function setMode(m) {
  mode = m;
  document.getElementById('modeStart').classList.toggle('active', m === 'start');
  document.getElementById('modeGoal').classList.toggle('active', m === 'goal');
}

// Map canvas click → video pixel coordinates
const clickCanvas = document.getElementById('clickCanvas');
const videoEl     = document.getElementById('videoEl');

function resizeCanvas() {
  clickCanvas.width  = clickCanvas.offsetWidth;
  clickCanvas.height = clickCanvas.offsetHeight;
}
window.addEventListener('resize', resizeCanvas);
resizeCanvas();

clickCanvas.addEventListener('click', async (e) => {
  const rect = clickCanvas.getBoundingClientRect();
  // clickCanvas fills the wrap div; video is letterboxed inside
  const cw = rect.width, ch = rect.height;
  const videoAspect = frameW / frameH;
  const canvasAspect = cw / ch;

  let vidPxW, vidPxH, offX, offY;
  if (canvasAspect > videoAspect) {
    vidPxH = ch;
    vidPxW = ch * videoAspect;
    offX = (cw - vidPxW) / 2;
    offY = 0;
  } else {
    vidPxW = cw;
    vidPxH = cw / videoAspect;
    offX = 0;
    offY = (ch - vidPxH) / 2;
  }

  const cx = e.clientX - rect.left;
  const cy = e.clientY - rect.top;
  const vx = Math.round((cx - offX) / vidPxW * frameW);
  const vy = Math.round((cy - offY) / vidPxH * frameH);

  if (vx < 0 || vy < 0 || vx >= frameW || vy >= frameH) return;

  const body = mode === 'start'
    ? { start: [vx, vy] }
    : { goal:  [vx, vy] };

  await fetch('/set_points', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify(body)
  });
});

async function clearAll() {
  await fetch('/clear', { method:'POST' });
}

async function toggleObs() {
  const r = await fetch('/toggle_obstacles', { method:'POST' });
  const d = await r.json();
  showObs = d.show;
  document.getElementById('obsBtn').textContent = showObs ? 'Hide overlay' : 'Show overlay';
}

async function setDilate(v) {
  await fetch('/set_dilate', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ dilate: parseInt(v) })
  });
}

async function setReplan() {
  const threshold = parseInt(document.getElementById('threshSlider').value) / 100;
  const cooldown  = parseInt(document.getElementById('cooldownSlider').value) / 10;
  await fetch('/set_replan', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ threshold, cooldown })
  });
}

// Poll status
async function pollStatus() {
  try {
    const r = await fetch('/status');
    const d = await r.json();
    frameW = d.frame_w || 640;
    frameH = d.frame_h || 480;

    // Start dot
    if (d.start) {
      document.getElementById('startCoord').textContent = `(${d.start[0]}, ${d.start[1]})`;
      document.getElementById('startDot').className = 'dot dot-green';
    } else {
      document.getElementById('startCoord').textContent = '—';
      document.getElementById('startDot').className = 'dot';
    }

    // Goal dot
    if (d.goal) {
      document.getElementById('goalCoord').textContent = `(${d.goal[0]}, ${d.goal[1]})`;
      document.getElementById('goalDot').className = 'dot dot-green';
    } else {
      document.getElementById('goalCoord').textContent = '—';
      document.getElementById('goalDot').className = 'dot';
    }

    // Path & replanning
    if (d.computing || d.replanning) {
      const label = d.replanning ? 'Replanning…' : 'Computing…';
      document.getElementById('pathInfo').textContent = label;
      document.getElementById('pathInfo').className = 'computing-anim';
      document.getElementById('pathDot').className = 'dot';
      document.getElementById('pathDot').style.background = d.replanning ? '#f97316' : '#f59e0b';
      document.getElementById('statusText').textContent = label;
      document.getElementById('statusText').className = 'computing-anim';
      document.getElementById('replanStatus').textContent = d.replanning ? 'Replanning…' : 'Computing…';
      document.getElementById('replanDot').style.background = '#f97316';
      document.getElementById('replanDot').className = 'dot';
    } else if (d.error) {
      document.getElementById('pathInfo').textContent = 'No path';
      document.getElementById('pathInfo').className = '';
      document.getElementById('pathDot').className = 'dot dot-red';
      document.getElementById('statusText').textContent = d.error;
      document.getElementById('statusText').className = '';
      document.getElementById('replanStatus').textContent = 'No path';
      document.getElementById('replanDot').className = 'dot dot-red';
    } else if (d.has_path) {
      document.getElementById('pathInfo').textContent = `${d.path_len} pts`;
      document.getElementById('pathInfo').className = '';
      document.getElementById('pathDot').className = 'dot dot-green';
      document.getElementById('statusText').textContent = `Path found — ${d.path_len} waypoints`;
      document.getElementById('statusText').className = '';
      document.getElementById('replanStatus').textContent = 'Watching';
      document.getElementById('replanDot').className = 'dot dot-green';
    } else {
      document.getElementById('pathInfo').textContent = '—';
      document.getElementById('pathInfo').className = '';
      document.getElementById('pathDot').className = 'dot';
      if (!d.start || !d.goal) {
        document.getElementById('statusText').textContent = 'Click the video to set Start and Goal';
      }
      document.getElementById('statusText').className = '';
      document.getElementById('replanStatus').textContent = 'Idle';
      document.getElementById('replanDot').className = 'dot';
    }
  } catch(e) { /* ignore fetch errors */ }
}
setInterval(pollStatus, 400);
pollStatus();
</script>
</body>
</html>
"""

# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='A* Pathfinder – OpenCV + Flask',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python astar_pathfinder.py                    # auto-detect webcam
  python astar_pathfinder.py --camera 2         # force camera index 2
  python astar_pathfinder.py --list-cameras     # show available indices
  python astar_pathfinder.py --picamera         # Raspberry Pi Camera
  python astar_pathfinder.py --port 8080        # custom port
        """
    )
    parser.add_argument('--picamera', action='store_true',
                        help='Use Raspberry Pi Camera (picamera2)')
    parser.add_argument('--camera', type=int, default=None, metavar='INDEX',
                        help='Camera device index (e.g. 0, 1, 2). Auto-scans if omitted.')
    parser.add_argument('--list-cameras', action='store_true',
                        help='Scan and list available camera indices, then exit.')
    parser.add_argument('--width',  type=int, default=640, help='Frame width  (default 640)')
    parser.add_argument('--height', type=int, default=480, help='Frame height (default 480)')
    parser.add_argument('--port',   type=int, default=5000, help='Flask port (default 5000)')
    parser.add_argument('--host',   type=str, default='0.0.0.0',
                        help='Bind host (default 0.0.0.0 = all interfaces)')
    args = parser.parse_args()

    print("═" * 55)
    print("  A* Pathfinder  |  OpenCV + Flask")
    print("═" * 55)

    # ── List cameras and exit ────────────────────────────────────────────────
    if args.list_cameras:
        print("Scanning for cameras (this may take a few seconds)…")
        found = find_available_cameras()
        if found:
            print(f"\nAvailable camera indices: {found}")
            print(f"\nRun with:  python {__file__} --camera {found[0]}")
        else:
            print("\nNo cameras found. Check connections and drivers.")
            print("The program will still run in TEST PATTERN mode without --camera.")
        return

    # ── Open camera ──────────────────────────────────────────────────────────
    cam = Camera(
        use_picamera=args.picamera,
        width=args.width,
        height=args.height,
        cam_index=args.camera,
    )

    # Start camera grab thread
    t = threading.Thread(target=camera_thread, args=(cam,), daemon=True)
    t.start()

    # Start auto-replanner thread
    r = threading.Thread(target=auto_replan_thread, daemon=True)
    r.start()

    # Wait for first frame
    print("[Init] Waiting for first frame…")
    for _ in range(60):
        with S.lock:
            if S.frame is not None:
                break
        time.sleep(0.1)
    else:
        print("[Init] WARNING: No frame received yet – starting server anyway.")

    print(f"\n[Server] Open your browser at  http://localhost:{args.port}")
    print(f"[Server] On the same network:   http://<this-device-ip>:{args.port}")
    print("[Server] Press Ctrl+C to quit.\n")

    try:
        flask_app.run(host=args.host, port=args.port, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n[Server] Shutting down.")
    finally:
        cam.release()


if __name__ == '__main__':
    main()
