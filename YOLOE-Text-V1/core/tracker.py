"""
Unified Object Tracking Module for YOLOE Project.
Contains both a simple, permissive tracker for robot vision and an
advanced Kalman filter-based tracker for counting applications.
"""
import collections
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter

# Get a logger for this module
logger = logging.getLogger(__name__)


class SimpleTrack:
    """
    Represents a single tracked object for the SimpleObjectTracker.
    Tracks are immediately active and use simple velocity prediction.
    """
    def __init__(self, track_id: int, class_id: int, bbox: Tuple[int, int, int, int], confidence: float):
        self.id = track_id
        self.class_id = class_id
        self.bbox = bbox
        self.confidence = confidence
        
        self.age = 0
        self.misses = 0
        self.last_seen = datetime.now()
        
        self.center_history = collections.deque(maxlen=5)
        self.center_history.append(self.center)
        self.velocity = (0, 0)

    @property
    def center(self) -> Tuple[int, int]:
        """Get center point of bounding box."""
        return (self.bbox[0] + self.bbox[2]) // 2, (self.bbox[1] + self.bbox[3]) // 2

    def update(self, bbox: Tuple[int, int, int, int], confidence: float):
        """Update track with a new detection."""
        self.bbox = bbox
        self.confidence = confidence
        self.misses = 0
        self.last_seen = datetime.now()
        
        new_center = self.center
        if len(self.center_history) > 0:
            prev_center = self.center_history[-1]
            self.velocity = (new_center[0] - prev_center[0], new_center[1] - prev_center[1])
        self.center_history.append(new_center)

    def predict(self):
        """Predict the next position of the track."""
        self.age += 1
        self.misses += 1
        
        if self.misses <= 5: # Only predict for a few frames
            damping = 0.8
            self.velocity = (int(self.velocity[0] * damping), int(self.velocity[1] * damping))
            
            if abs(self.velocity[0]) > 1 or abs(self.velocity[1]) > 1:
                x1, y1, x2, y2 = self.bbox
                vx, vy = self.velocity
                self.bbox = (x1 + vx, y1 + vy, x2 + vx, y2 + vy)
                self.center_history.append(self.center)

class SimpleObjectTracker:
    """
    A simple, permissive object tracker ideal for robot vision.
    Uses a standard IoU-based matching algorithm for robustness.
    Tracks are persistent and not easily lost.
    """
    
    def __init__(self, max_misses: int = 30, iou_threshold: float = 0.2):
        """
        Initialize the simple tracker.
        
        Args:
            max_misses: Very high persistence. Track will survive for 30 frames without a detection.
            iou_threshold: Permissive matching. A 20% overlap is enough to match.
        """
        self.max_misses = max_misses
        self.iou_threshold = iou_threshold
        self.tracks: Dict[int, SimpleTrack] = {}
        self.next_id: int = 0
        logger.info(f"SimpleObjectTracker initialized: max_misses={max_misses}, iou_threshold={iou_threshold}")

    def _calculate_iou(self, bbox1, bbox2):
        x1_inter = max(bbox1[0], bbox2[0])
        y1_inter = max(bbox1[1], bbox2[1])
        x2_inter = min(bbox1[2], bbox2[2])
        y2_inter = min(bbox1[3], bbox2[3])
        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0
        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union_area = bbox1_area + bbox2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0.0

    def update(self, detections: List[Tuple[Tuple, int, float]]) -> List[SimpleTrack]:
        for track in self.tracks.values():
            track.predict()

        if not detections:
            active_tracks = []
            tracks_to_remove = []
            for track_id, track in self.tracks.items():
                if track.misses <= self.max_misses:
                    active_tracks.append(track)
                else:
                    tracks_to_remove.append(track_id)
            for track_id in tracks_to_remove:
                del self.tracks[track_id]
            return active_tracks

        track_ids = list(self.tracks.keys())
        det_indices = list(range(len(detections)))
        
        matched_track_ids = []
        matched_det_indices = []
        unmatched_det_indices = list(range(len(detections)))

        if self.tracks:
            track_bboxes = [self.tracks[tid].bbox for tid in track_ids]
            det_bboxes = [d[0] for d in detections]
            
            iou_matrix = np.zeros((len(track_bboxes), len(det_bboxes)))
            for i, trk_bb in enumerate(track_bboxes):
                for j, det_bb in enumerate(det_bboxes):
                    iou_matrix[i, j] = self._calculate_iou(trk_bb, det_bb)
            
            track_indices_matched, det_indices_matched = linear_sum_assignment(-iou_matrix) # maximize IoU

            for track_idx, det_idx in zip(track_indices_matched, det_indices_matched):
                if iou_matrix[track_idx, det_idx] >= self.iou_threshold:
                    track_id = track_ids[track_idx]
                    self.tracks[track_id].update(detections[det_idx][0], detections[det_idx][2])
                    matched_track_ids.append(track_id)
                    matched_det_indices.append(det_idx)
            
            unmatched_det_indices = list(set(det_indices) - set(matched_det_indices))

        for det_idx in unmatched_det_indices:
            bbox, class_id, conf = detections[det_idx]
            new_track = SimpleTrack(self.next_id, class_id, bbox, conf)
            self.tracks[self.next_id] = new_track
            self.next_id += 1

        active_tracks = []
        tracks_to_remove = []
        for track_id, track in self.tracks.items():
            if track.misses <= self.max_misses:
                 active_tracks.append(track)
            else:
                 tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
            
        return active_tracks

    def get_tracks(self) -> List[SimpleTrack]:
        return list(self.tracks.values())

    def reset(self):
        self.tracks.clear()
        self.next_id = 0
        logger.info("Simple tracker reset")


class KalmanTrack:
    """
    Represents a single tracked object with state, Kalman Filter, and history.
    """
    def __init__(self, track_id: int, class_id: int, initial_bbox: Tuple[int, int, int, int], initial_confidence: float):
        self.id = track_id
        self.class_id = class_id
        self.state = 'TENTATIVE'  # TENTATIVE, CONFIRMED, COASTING, LOST
        
        self.kf = self.init_kalman_filter(initial_bbox)
        self.bbox = initial_bbox
        self.confidence = initial_confidence
        
        self.age = 0
        self.misses = 0
        self.hits = 0
        self.history = collections.deque(maxlen=30)  
        
        self.last_seen_timestamp = datetime.now()

    @staticmethod
    def init_kalman_filter(bbox: Tuple[int, int, int, int]) -> KalmanFilter:
        """Initializes a Kalman Filter for a new track."""
        kf = KalmanFilter(dim_x=7, dim_z=4)
        kf.F = np.array([[1,0,0,0,1,0,0], [0,1,0,0,0,1,0], [0,0,1,0,0,0,1], [0,0,0,1,0,0,0],
                        [0,0,0,0,1,0,0], [0,0,0,0,0,1,0], [0,0,0,0,0,0,1]], dtype=float)
        kf.H = np.array([[1,0,0,0,0,0,0], [0,1,0,0,0,0,0], [0,0,1,0,0,0,0], [0,0,0,1,0,0,0]], dtype=float)
        

        kf.R[2:,2:] *= 10.
        
        kf.P[4:,4:] *= 1000. 
        kf.P *= 10.

        kf.Q[-1,-1] *= 0.1  
        kf.Q[4:6,4:6] *= 0.1 

        z = KalmanTrack.bbox_to_z(bbox)
        kf.x[:4] = z
        return kf

    def predict(self):
        """Advances the state vector and returns predicted bounding box."""
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] = 0.
        self.kf.predict()
        self.age += 1
        self.misses += 1
        self.bbox = self.to_bbox()
        self.history.append(self.center)
        return self.bbox

    def update(self, bbox: Tuple[int, int, int, int], confidence: float):
        """Updates the Kalman Filter with a new detection."""
        self.bbox = bbox
        self.confidence = confidence
        self.misses = 0
        self.hits += 1
        self.last_seen_timestamp = datetime.now()
        self.history.append(self.center)
        self.kf.update(self.bbox_to_z(bbox))

    @property
    def center(self) -> Tuple[int, int]:
        """Calculates the center of the current bounding box."""
        return (self.bbox[0] + self.bbox[2]) // 2, (self.bbox[1] + self.bbox[3]) // 2

    def to_bbox(self) -> Tuple[int, int, int, int]:
        """Converts the Kalman Filter's state to a bounding box."""
        x, y, s, r = self.kf.x[:4, 0]

        s = max(0, s)
        r = max(0, r)

        w = np.sqrt(s * r)
        h = np.sqrt(s / r) if r > 1e-6 else 0

        if np.isnan(w): w = 0
        if np.isnan(h): h = 0

        return int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)

    @staticmethod
    def bbox_to_z(bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Converts a bounding box to the measurement vector [x, y, s, r]."""
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        x, y = x1 + w / 2, y1 + h / 2
        s = w * h
        r = w / h if h > 0 else 0
        return np.array([x, y, s, r]).reshape((4, 1))

class KalmanObjectTracker:
    """
    Multi-object tracker using Kalman Filter and Hungarian Algorithm.
    Maintains object IDs across frames for consistent tracking.
    """
    
    def __init__(self, max_misses: int = 10, min_hits: int = 3, iou_threshold: float = 0.3):
        """
        Initialize tracker with configuration parameters.
        
        Args:
            max_misses: Maximum number of consecutive misses before deleting track
            min_hits: Minimum hits to confirm a track
            iou_threshold: IoU threshold for matching detections to tracks
        """
        self.max_misses = max_misses
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        
        self.tracks: Dict[int, KalmanTrack] = {}
        self.next_id: int = 0
        
        logger.info(f"KalmanObjectTracker initialized with max_misses={max_misses}, min_hits={min_hits}, iou_threshold={iou_threshold}")

    def update(self, detections: List[Tuple[Tuple[int, int, int, int], int, float]]) -> List[KalmanTrack]:
        """
        Update tracking state with new detections.
        
        Args:
            detections: List of (bbox, class_id, confidence) tuples
            
        Returns:
            List of active tracks
        """
        for track_id in list(self.tracks.keys()):
            track = self.tracks[track_id]
            track.predict()
            
            if track.misses > self.max_misses:
                logger.debug(f"Track {track_id} removed due to high misses ({track.misses})")
                del self.tracks[track_id]

        if detections and len(self.tracks) > 0:
            det_bboxes = [d[0] for d in detections]
            track_bboxes = [t.to_bbox() for t in self.tracks.values()]
            
            iou_matrix = self._calculate_iou_matrix(track_bboxes, det_bboxes)
            
            if iou_matrix.size > 0:
                track_indices, det_indices = linear_sum_assignment(-iou_matrix)  # Maximize IoU
                
                matched_det_indices = set()
                for track_idx, det_idx in zip(track_indices, det_indices):
                    if iou_matrix[track_idx, det_idx] >= self.iou_threshold:
                        track_id = list(self.tracks.keys())[track_idx]
                        bbox, class_id, conf = detections[det_idx]
                        self.tracks[track_id].update(bbox, conf)
                        self.tracks[track_id].class_id = class_id
                        matched_det_indices.add(det_idx)
                
                unmatched_det_indices = set(range(len(detections))) - matched_det_indices
                for det_idx in unmatched_det_indices:
                    bbox, class_id, conf = detections[det_idx]
                    self._create_new_track(class_id, bbox, conf)
        else:
            for bbox, class_id, conf in detections:
                self._create_new_track(class_id, bbox, conf)
        
        # 3. Update track states
        for track in self.tracks.values():
            self._update_track_state(track)
        
        # Return only confirmed tracks
        return [track for track in self.tracks.values() if track.state in ['CONFIRMED', 'COASTING']]

    def _create_new_track(self, class_id: int, bbox: Tuple[int, int, int, int], confidence: float):
        """Create a new track for an unmatched detection."""
        new_track = KalmanTrack(self.next_id, class_id, bbox, confidence)
        self.tracks[self.next_id] = new_track
        logger.debug(f"Created new TENTATIVE track {self.next_id}")
        self.next_id += 1

    def _update_track_state(self, track: KalmanTrack):
        """Update track state based on hits and misses."""
        if track.state == 'TENTATIVE' and track.hits >= self.min_hits:
            track.state = 'CONFIRMED'
            logger.debug(f"Track {track.id} promoted to CONFIRMED")
        elif track.state == 'CONFIRMED' and track.misses > 0:
            track.state = 'COASTING'
            logger.debug(f"Track {track.id} moved to COASTING")
        elif track.state == 'COASTING' and track.misses == 0:
            track.state = 'CONFIRMED'
            logger.debug(f"Track {track.id} returned to CONFIRMED")

    def _calculate_iou_matrix(self, track_bboxes: List, det_bboxes: List) -> np.ndarray:
        """Calculate IoU matrix between tracks and detections."""
        if not track_bboxes or not det_bboxes:
            return np.empty((0, 0))
        
        iou_matrix = np.zeros((len(track_bboxes), len(det_bboxes)), dtype=np.float32)
        for i, track_bbox in enumerate(track_bboxes):
            for j, det_bbox in enumerate(det_bboxes):
                iou_matrix[i, j] = self._calculate_iou(track_bbox, det_bbox)
        return iou_matrix

    def _calculate_iou(self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate IoU between two bounding boxes."""
        x1_inter = max(bbox1[0], bbox2[0])
        y1_inter = max(bbox1[1], bbox2[1])
        x2_inter = min(bbox1[2], bbox2[2])
        y2_inter = min(bbox1[3], bbox2[3])
        
        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0
        
        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union_area = bbox1_area + bbox2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0

    def get_tracks(self) -> List[KalmanTrack]:
        """Get all active tracks."""
        return list(self.tracks.values())

    def reset(self):
        """Reset tracker state."""
        self.tracks.clear()
        self.next_id = 0
        logger.info("KalmanObjectTracker reset")
