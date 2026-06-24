"""
IoU-based multi-person tracker with runner/fielder role classifier.

Tracks person bboxes across frames and labels each track as either
'runner' (high velocity, approaching first base) or 'fielder' (stationary).
"""
import numpy as np
from typing import List, Dict, Optional, Tuple


def iou(a: List[float], b: List[float]) -> float:
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter + 1e-6)


class Track:
    """Single player track across frames."""
    def __init__(self, track_id: int, bbox: List[float], frame_idx: int):
        self.id = track_id
        self.bboxes: Dict[int, List[float]] = {frame_idx: bbox}
        self.role: Optional[str] = None  # 'runner' or 'fielder'

    def center(self, frame_idx: int) -> np.ndarray:
        b = self.bboxes[frame_idx]
        return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])

    def last_bbox(self) -> List[float]:
        return self.bboxes[max(self.bboxes)]

    def last_frame(self) -> int:
        return max(self.bboxes)

    def velocity(self, frame_window: int = 10) -> np.ndarray:
        """Mean velocity (pixels/frame) over recent frames."""
        frames = sorted(self.bboxes.keys())
        if len(frames) < 2:
            return np.zeros(2)
        recent = frames[-min(len(frames), frame_window):]
        vecs = []
        for i in range(1, len(recent)):
            dt = recent[i] - recent[i-1]
            if dt == 0:
                continue
            dc = self.center(recent[i]) - self.center(recent[i-1])
            vecs.append(dc / dt)
        return np.mean(vecs, axis=0) if vecs else np.zeros(2)

    def speed(self) -> float:
        return float(np.linalg.norm(self.velocity()))

    def displacement(self) -> np.ndarray:
        """Total displacement from first to last seen frame."""
        frames = sorted(self.bboxes.keys())
        if len(frames) < 2:
            return np.zeros(2)
        return self.center(frames[-1]) - self.center(frames[0])


class PlayerTracker:
    """
    Greedy IoU tracker. Associates detections across frames, then
    classifies each track as runner or fielder.
    """

    IOU_THRESHOLD = 0.25
    MAX_MISS_FRAMES = 5

    def __init__(self):
        self.tracks: List[Track] = []
        self._next_id = 0

    def update(self, frame_idx: int, bboxes: List[List[float]]):
        """Associate bboxes at frame_idx to existing tracks."""
        active = [t for t in self.tracks if frame_idx - t.last_frame() <= self.MAX_MISS_FRAMES]
        matched_track_ids = set()
        matched_det_ids = set()

        # Greedy matching: highest IoU first
        pairs = []
        for ti, track in enumerate(active):
            for di, bbox in enumerate(bboxes):
                score = iou(track.last_bbox(), bbox)
                if score >= self.IOU_THRESHOLD:
                    pairs.append((score, ti, di))
        pairs.sort(reverse=True)

        for score, ti, di in pairs:
            t = active[ti]
            if t.id in matched_track_ids or di in matched_det_ids:
                continue
            t.bboxes[frame_idx] = bboxes[di]
            matched_track_ids.add(t.id)
            matched_det_ids.add(di)

        # New tracks for unmatched detections
        for di, bbox in enumerate(bboxes):
            if di not in matched_det_ids:
                track = Track(self._next_id, bbox, frame_idx)
                self._next_id += 1
                self.tracks.append(track)

    def classify_roles(self, frame_width: int) -> Dict[int, str]:
        """
        Label each track as 'runner' or 'fielder'.

        Runner: high speed AND x-displacement toward first base (positive x in
        broadcast view where home is left, first base is right).
        Fielder: low speed OR moving in the opposite direction.

        Returns {track_id: role}.
        """
        # Only consider tracks seen in at least 10% of frames
        min_frames = 5
        long_tracks = [t for t in self.tracks if len(t.bboxes) >= min_frames]

        if not long_tracks:
            return {}

        speeds = np.array([t.speed() for t in long_tracks])
        displacements = np.array([t.displacement()[0] for t in long_tracks])  # x-component

        roles = {}
        if len(long_tracks) == 1:
            # Single person — assume runner
            roles[long_tracks[0].id] = "runner"
        elif len(long_tracks) == 2:
            # Two people: faster + moving right → runner; slower/stationary → fielder
            t0, t1 = long_tracks[0], long_tracks[1]
            # Runner has higher speed overall
            if speeds[0] > speeds[1]:
                roles[t0.id] = "runner"
                roles[t1.id] = "fielder"
            else:
                roles[t0.id] = "fielder"
                roles[t1.id] = "runner"

            # If neither moves much, pick the one closer to the right edge as fielder
            if max(speeds) < 1.0:
                centers = [t.center(max(t.bboxes)) for t in long_tracks]
                right_idx = int(np.argmax([c[0] for c in centers]))
                roles[long_tracks[right_idx].id] = "fielder"
                roles[long_tracks[1 - right_idx].id] = "runner"
        else:
            # 3+ tracks: fastest is runner, next most-stationary is fielder, rest unknown
            speed_order = np.argsort(speeds)[::-1]
            roles[long_tracks[speed_order[0]].id] = "runner"
            # Fielder: second slowest (not fastest, not unknown)
            speed_order_asc = np.argsort(speeds)
            roles[long_tracks[speed_order_asc[0]].id] = "fielder"
            for t in long_tracks:
                if t.id not in roles:
                    roles[t.id] = "other"

        # Store roles on tracks
        for t in self.tracks:
            t.role = roles.get(t.id, "unknown")

        return roles

    def get_role_at_frame(self, frame_idx: int) -> Dict[str, Optional[List[float]]]:
        """Return {role: bbox} for each classified track visible at frame_idx."""
        result: Dict[str, Optional[List[float]]] = {"runner": None, "fielder": None}
        for t in self.tracks:
            if t.role in result and frame_idx in t.bboxes:
                result[t.role] = t.bboxes[frame_idx]
        return result
