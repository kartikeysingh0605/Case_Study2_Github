"""Review build (v2) for Vinfotech traffic-intersection analytics.

NOT validated ground-truth. Explicit stop-lines and queue ROIs must be visually calibrated.
Retains original frame rate. No model training required.
"""
from __future__ import annotations
import argparse
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
import csv
import json
import math

import cv2
import numpy as np

ARMS = ('North', 'South', 'East', 'West')
COLORS = {'North': (255, 130, 50), 'South': (48, 170, 255),
          'East': (90, 220, 90), 'West': (220, 70, 210), 'junction': (40, 220, 230)}
CLASSES = [2, 3, 5, 7]  # COCO car, motorcycle, bus, truck


def scale_polygon(points, w, h, ref):
    sx, sy = w / ref[0], h / ref[1]
    return np.array([(round(x * sx), round(y * sy)) for x, y in points], np.int32)


def load_geometry(cfg, w, h):
    ref = cfg['reference_size']
    arms = {a: scale_polygon(cfg['arms'][a], w, h, ref) for a in ARMS}
    junction = scale_polygon(cfg['junction'], w, h, ref)
    queue_rois = {a: scale_polygon(cfg.get('queue_regions', cfg['arms'])[a], w, h, ref) for a in ARMS}
    lines = {}
    for a, data in cfg.get('stop_lines', {}).items():
        if a not in ARMS:
            continue
        pts = scale_polygon(data['points'], w, h, ref)
        before = scale_polygon([data['approach_point']], w, h, ref)[0]
        lines[a] = (pts[0], pts[1], before)
    return arms, junction, queue_rois, lines


def inside(poly, p):
    return cv2.pointPolygonTest(poly, (float(p[0]), float(p[1])), False) >= 0


def get_arm(p, arms):
    for a in ARMS:
        if inside(arms[a], p):
            return a
    return None


def cross_line_forward(prev_p, p, line, projection_margin=0.15):
    """True only for approach-side -> junction-side crossing of a FINITE line.

    The config's approach_point determines the direction. Uses vehicle ground point.
    """
    A, B, before = line
    ax, ay = map(float, A)
    bx, by = map(float, B)
    dx, dy = bx-ax, by-ay
    denominator = dx*dx + dy*dy
    if denominator < 1:
        return False
    def side(q):
        return dx*(float(q[1])-ay) - dy*(float(q[0])-ax)
    direction = side(before)
    if abs(direction) < 1:
        return False
    s0, s1 = side(prev_p)*direction, side(p)*direction
    if not (s0 > 0 and s1 <= 0):
        return False
    alpha = s0 / (s0 - s1) if (s0 - s1) else 0
    x = prev_p[0] + alpha*(p[0]-prev_p[0]); y = prev_p[1] + alpha*(p[1]-prev_p[1])
    projection = ((x-ax)*dx+(y-ay)*dy)/denominator
    return -projection_margin <= projection <= 1+projection_margin


@dataclass
class Vehicle:
    track_id: int
    vehicle_class: str
    first_seen: float
    last_seen: float
    entry: str | None = None
    exit: str | None = None
    junction_seen: bool = False
    counted_entry: bool = False
    counted_turn: bool = False
    junction_frames: int = 0
    candidate_exit: str | None = None
    candidate_frames: int = 0
    last_junction_point: tuple[int, int] | None = None
    positions: deque = field(default_factory=lambda: deque(maxlen=100))
    pending_stop: float | None = None
    wait_start: float | None = None
    waited_s: float | None = None
    stopped_now: bool = False
    line_crossed: bool = False
    line_cross_time: float | None = None
    frames: int = 0


def update_track(v, t, p, box_h, arm, in_junction, queue_rois, lines,
                 stopped_threshold_s=2.0, exit_frames=4, min_exit_distance_px=24):
    """Update vehicle FSM. Returns (new entry, completed movement, completed wait)."""
    prior = v.positions[-1][1] if v.positions else None
    v.positions.append((t, p))
    v.last_seen = t
    v.frames += 1

    # Set the origin BEFORE entering the junction; do not assign ambiguous born-in-junction tracks.
    if not v.junction_seen and not in_junction and v.entry is None and arm is not None:
        v.entry = arm

    # Stationary decision using a measured ~0.5s displacement (ground point).
    old = None
    for ts, pt in reversed(v.positions):
        if t-ts >= .5:
            old = (ts, pt)
            break
    speed = None
    if old is not None and t > old[0]:
        speed = math.dist(p, old[1]) / (t-old[0])
    v.stopped_now = (speed is not None and speed < max(2.5, 0.09*box_h))

    # Queue evidence is independent of future success at reaching the intersection.
    queue_valid = (v.entry is not None and not in_junction and not v.line_crossed
                   and inside(queue_rois[v.entry], p))
    if queue_valid and v.stopped_now:
        if v.pending_stop is None:
            v.pending_stop = t
        if t-v.pending_stop >= stopped_threshold_s and v.wait_start is None:
            v.wait_start = v.pending_stop
    else:
        v.pending_stop = None

    completed_wait = None
    if v.entry is not None and not v.line_crossed and prior is not None and v.entry in lines:
        if cross_line_forward(prior, p, lines[v.entry]):
            v.line_crossed = True
            v.line_cross_time = t
            if v.wait_start is not None:
                v.waited_s = max(0.0, t-v.wait_start)
                completed_wait = (v.entry, v.waited_s)

    entered, movement = None, None
    if in_junction:
        v.junction_frames += 1
        v.junction_seen = True
        v.last_junction_point = p
        v.candidate_exit = None
        v.candidate_frames = 0
        if v.entry is not None and not v.counted_entry and v.frames >= 3:
            v.counted_entry = True
            entered = v.entry
    elif v.junction_seen and v.counted_entry and not v.counted_turn:
        # An exit must be consistently observed away from the junction boundary.
        if arm is not None and v.junction_frames >= 3:
            dist = math.dist(p, v.last_junction_point) if v.last_junction_point else 0
            if dist >= min_exit_distance_px:
                if v.candidate_exit == arm:
                    v.candidate_frames += 1
                else:
                    v.candidate_exit = arm
                    v.candidate_frames = 1
                if v.candidate_frames >= exit_frames:
                    v.exit = arm
                    v.counted_turn = True
                    movement = (v.entry, v.exit)  # same arm = candidate U-turn; audit manually
            else:
                v.candidate_exit = None
                v.candidate_frames = 0
        else:
            v.candidate_exit = None
            v.candidate_frames = 0
    return entered, movement, completed_wait


def draw_geometry(frame, arms, junction, queue_rois, lines):
    lay = frame.copy()
    for a in ARMS:
        cv2.fillPoly(lay, [arms[a]], COLORS[a])
    cv2.fillPoly(lay, [junction], COLORS['junction'])
    cv2.addWeighted(lay, .13, frame, .87, 0, frame)
    for a in ARMS:
        cv2.polylines(frame, [arms[a]], True, COLORS[a], 2)
        mid = tuple(int(x) for x in np.mean(arms[a], axis=0))
        cv2.putText(frame, a.upper(), mid, cv2.FONT_HERSHEY_SIMPLEX, .7, COLORS[a], 2, cv2.LINE_AA)
        cv2.polylines(frame, [queue_rois[a]], True, (255, 255, 255), 1, cv2.LINE_AA)
        if a in lines:
            start, end, _ = lines[a]
            cv2.line(frame, tuple(start), tuple(end), (0, 0, 255), 3)
            center = (int((start[0]+end[0])/2), int((start[1]+end[1])/2))
            cv2.putText(frame, f'{a[0]} STOP', center, cv2.FONT_HERSHEY_SIMPLEX, .53, (0,0,255), 2, cv2.LINE_AA)
    cv2.polylines(frame, [junction], True, COLORS['junction'], 2)
    return frame


def write_csv(path, rows, columns):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description='Case Study 2 review build: explicit stop lines and queue metrics')
    parser.add_argument('--video', required=True)
    parser.add_argument('--zones', default=str(Path(__file__).parent/'config'/'zones.json'))
    parser.add_argument('--out', default='output_v2')
    parser.add_argument('--mode', choices=('preview', 'analyze'), default='preview')
    parser.add_argument('--model', default='yolo11n.pt')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--imgsz', type=int, default=960)
    parser.add_argument('--conf', type=float, default=.25)
    parser.add_argument('--preview-time', type=float, default=300)
    parser.add_argument('--min-box-height', type=int, default=0)
    args = parser.parse_args()
    if args.stride < 1:
        parser.error('--stride must be >= 1')
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(Path(args.zones).read_text(encoding='utf-8'))
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f'Cannot read video: {args.video}')
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0:
        raise SystemExit('Input FPS invalid')
    arms, junction, queue_rois, lines = load_geometry(cfg, width, height)
    print(f'Input: {width}x{height}; {fps:.2f} FPS; {frames} frames; {frames/fps:.2f}s')
    print('Geometry status:', cfg.get('status', 'Not documented'))
    if args.mode == 'preview':
        cap.set(cv2.CAP_PROP_POS_MSEC, args.preview_time*1000)
        ok, frame = cap.read()
        if not ok:
            raise SystemExit('Cannot read requested preview timestamp')
        cv2.imwrite(str(out/'zone_preview_v2.jpg'), draw_geometry(frame, arms, junction, queue_rois, lines))
        cap.release()
        print('Preview:', (out/'zone_preview_v2.jpg').resolve())
        return
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise SystemExit('Install dependencies from requirements.txt') from e
    model = YOLO(args.model)
    writer = cv2.VideoWriter(str(out/'annotated.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps/args.stride, (width,height))
    if not writer.isOpened():
        raise SystemExit('Could not create annotated video')
    states = {}
    entries = {a:0 for a in ARMS}
    matrix = {(a,b):0 for a in ARMS for b in ARMS}
    waits = defaultdict(list)
    maxima = {a:0 for a in ARMS}
    frame_idx, processed = -1, 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            frame_idx += 1
            if frame_idx % args.stride: continue
            t = frame_idx/fps
            result = model.track(frame, persist=True, tracker='bytetrack.yaml', classes=CLASSES,
                                 conf=args.conf, imgsz=args.imgsz, device=args.device, verbose=False)[0]
            active = {}
            if result.boxes is not None and result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                ids = result.boxes.id.int().cpu().numpy().tolist()
                classes = result.boxes.cls.int().cpu().numpy().tolist()
                for coords, tid, cid in zip(boxes, ids, classes):
                    x1,y1,x2,y2 = map(int, coords)
                    if y2-y1 < args.min_box_height: continue
                    p = (int((x1+x2)/2), int(y2))
                    arm = get_arm(p, arms)
                    in_junction = inside(junction, p)
                    if tid not in states:
                        states[tid] = Vehicle(int(tid), str(model.names[cid]), t, t)
                    v = states[tid]
                    entered, movement, wait = update_track(v, t, p, y2-y1, arm, in_junction, queue_rois, lines)
                    if entered is not None: entries[entered] += 1
                    if movement is not None: matrix[movement] += 1
                    if wait is not None: waits[wait[0]].append(wait[1])
                    active[tid] = (v, p, (x1,y1,x2,y2))
            q = {a:set() for a in ARMS}
            for tid, (v,p,box) in active.items():
                if (v.entry and not v.line_crossed and not inside(junction, p)
                    and inside(queue_rois[v.entry], p) and v.stopped_now
                    and v.pending_stop is not None and t-v.pending_stop >= 2.0):
                    q[v.entry].add(tid)
            for a in ARMS:
                maxima[a] = max(maxima[a], len(q[a]))
            draw_geometry(frame, arms, junction, queue_rois, lines)
            for tid, (v,p,box) in active.items():
                x1,y1,x2,y2 = box
                color = (0,255,0) if v.counted_turn else ((0,210,255) if v.junction_seen else (255,255,255))
                cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                label = f'#{tid} {v.vehicle_class}'
                if v.entry: label += f' {v.entry[0]}'
                if v.exit: label += f'>{v.exit[0]}'
                cv2.putText(frame, label, (x1,max(16,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, .46, color, 1, cv2.LINE_AA)
            overlay = '  '.join(f'{a[0]}:{entries[a]} Q:{len(q[a])}' for a in ARMS)
            cv2.rectangle(frame, (0,0), (min(width,1000),38), (0,0,0), -1)
            cv2.putText(frame, f'{t:.1f}s {overlay}', (7,26), cv2.FONT_HERSHEY_SIMPLEX, .62, (255,255,255), 2, cv2.LINE_AA)
            writer.write(frame)
            processed += 1
            if processed % 450 == 0:
                print(f'Processed {processed} frames; video timestamp {t:.0f}s', flush=True)
    finally:
        cap.release(); writer.release()

    rows = []
    for v in sorted(states.values(), key=lambda x:x.track_id):
        rows.append({'track_id':v.track_id,'vehicle_class':v.vehicle_class,
          'first_seen_s':round(v.first_seen,2),'last_seen_s':round(v.last_seen,2),
          'entry_arm':v.entry or '', 'exit_arm':v.exit or '',
          'entered_junction':v.counted_entry,'completed_turn':v.counted_turn,
          'stop_line_crossed':v.line_crossed,
          'wait_s':round(v.waited_s,2) if v.waited_s is not None else ''})
    write_csv(out/'vehicles.csv', rows, ['track_id','vehicle_class','first_seen_s','last_seen_s',
        'entry_arm','exit_arm','entered_junction','completed_turn','stop_line_crossed','wait_s'])
    write_csv(out/'turning_counts.csv',
        [{'entry_arm':a, **{b:matrix[(a,b)] for b in ARMS}} for a in ARMS], ['entry_arm',*ARMS])
    write_csv(out/'arm_summary.csv',
        [{'arm':a,'entered_vehicles':entries[a],'max_observed_queue':maxima[a],
          'avg_wait_s':round(float(np.mean(waits[a])),2) if waits[a] else '',
          'max_wait_s':round(max(waits[a]),2) if waits[a] else '',
          'observed_waits':len(waits[a])} for a in ARMS],
          ['arm','entered_vehicles','max_observed_queue','avg_wait_s','max_wait_s','observed_waits'])
    summary = {'warning':'REVIEW BUILD: counts unvalidated; stop lines/queue regions must be calibrated against the original footage.',
        'video':str(args.video), 'source_fps':fps, 'source_frames':frames,
        'processed_frames':processed, 'analyzed_fps':fps/args.stride,
        'entry_counts':entries,
        'turning_counts':{a:{b:matrix[(a,b)] for b in ARMS} for a in ARMS},
        'max_observed_queues':maxima,
        'wait_stats':{a:{'observed_waits':len(waits[a]),
               'mean_wait_s':round(float(np.mean(waits[a])),2) if waits[a] else None,
               'max_wait_s':round(max(waits[a]),2) if waits[a] else None} for a in ARMS},
        'unresolved_entered_tracks':sum(v.counted_entry and not v.counted_turn for v in states.values()),
        'completed_movement_tracks':sum(v.counted_turn for v in states.values()),
        'all_seen_track_ids':len(states),
        'tracks_under_half_second':sum(v.last_seen-v.first_seen<.5 for v in states.values()),
        'same_arm_movements_candidate':sum(matrix[(a,a)] for a in ARMS),
        'geometry_status':cfg.get('status','unknown')}
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2)); print('Saved:',out.resolve())

if __name__ == '__main__':
    main()
