"""
Simple Object Tracker for Robot Vision
Designed for immediate tracking without strict state machines.
Perfect for robot navigation and object following applications.
"""
import collections
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

# Get a logger for this module
logger = logging.getLogger(__name__)

class SimpleTrack:
    """
    Simple track object without complex state management.
    Tracks are immediately active and easy to maintain.
    """
    def __init__(self, track_id: int, class_id: int, bbox: Tuple[int, int, int, int], confidence: float):
        self.id = track_id
        self.class_id = class_id
        self.bbox = bbox
        self.confidence = confidence

        self.age = 0  
        self.misses = 0 
        self.last_seen = datetime.now()
        
        # Position tracking for prediction
        self.center_history = collections.deque(maxlen=5) 
        self.center_history.append(self.center)
        
        self.velocity = (0, 0) 
        
        logger.debug(f"Created simple track {track_id} for class {class_id}")

    @property
    def center(self) -> Tuple[int, int]:
        """Get center point of bounding box."""
        return (self.bbox[0] + self.bbox[2]) // 2, (self.bbox[1] + self.bbox[3]) // 2

    def predict_next_position(self) -> Tuple[int, int]:
        """Simple position prediction based on velocity."""
        cx, cy = self.center
        vx, vy = self.velocity
        return int(cx + vx), int(cy + vy)

    def update(self, bbox: Tuple[int, int, int, int], confidence: float):
        """Update track with new detection."""
        old_center = self.center
        self.bbox = bbox
        self.confidence = confidence
        self.misses = 0
        self.last_seen = datetime.now()

        new_center = self.center
        self.center_history.append(new_center)
        
        # Simple velocity calculation
        if len(self.center_history) >= 2:
            prev_center = self.center_history[-2]
            self.velocity = (new_center[0] - prev_center[0], new_center[1] - prev_center[1])
        
        logger.debug(f"Updated track {self.id} at {new_center}")

    def predict(self):
        """Predict next position (called when no detection matched)."""
        self.age += 1
        self.misses += 1

        if self.velocity != (0, 0) and self.misses <= 3:  
            x1, y1, x2, y2 = self.bbox
            vx, vy = self.velocity
            damping = 0.7  # More aggressive damping
            self.velocity = (int(vx * damping), int(vy * damping))
            
            if abs(vx) > 1 or abs(vy) > 1:
                self.bbox = (x1 + vx, y1 + vy, x2 + vx, y2 + vy)
                self.center_history.append(self.center)


class SimpleObjectTracker:
    """
    Simple multi-object tracker for robot vision.
    - Immediate tracking (no confirmation needed)
    - Permissive matching
    - Easy to tune parameters
    """
    
    def __init__(self, max_misses: int = 15, iou_threshold: float = 0.1, distance_threshold: int = 150):
        """
        Initialize simple tracker with more persistent settings.
        
        Args:
            max_misses: Maximum frames without detection before deleting track (default: 15 - more persistent)
            iou_threshold: IoU threshold for matching (default: 0.1 - very permissive)
            distance_threshold: Maximum pixel distance for matching (default: 150 - larger area)
        """
        self.max_misses = max_misses
        self.iou_threshold = iou_threshold
        self.distance_threshold = distance_threshold
        
        self.tracks: Dict[int, SimpleTrack] = {}
        self.next_id: int = 0
        
        logger.info(f"SimpleObjectTracker initialized: max_misses={max_misses}, "
                   f"iou_threshold={iou_threshold}, distance_threshold={distance_threshold}")

    def update(self, detections: List[Tuple[Tuple[int, int, int, int], int, float]]) -> List[SimpleTrack]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of (bbox, class_id, confidence) tuples
            
        Returns:
            List of all active tracks
        """
        for track in self.tracks.values():
            track.predict()

        tracks_to_remove = []
        for track_id, track in self.tracks.items():
            if track.misses > self.max_misses:
                tracks_to_remove.append(track_id)
                logger.debug(f"Removing track {track_id} after {track.misses} misses")
        
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
        
        if detections and self.tracks:
            matched_tracks, matched_detections = self._match_detections_to_tracks(detections)
            
            for track_id, det_idx in zip(matched_tracks, matched_detections):
                bbox, class_id, conf = detections[det_idx]
                self.tracks[track_id].update(bbox, conf)
                self.tracks[track_id].class_id = class_id  # Allow class changes

            unmatched_detections = set(range(len(detections))) - set(matched_detections)
            for det_idx in unmatched_detections:
                bbox, class_id, conf = detections[det_idx]
                self._create_new_track(bbox, class_id, conf)
        
        elif detections:
            for bbox, class_id, conf in detections:
                self._create_new_track(bbox, class_id, conf)
        
        return list(self.tracks.values())

    def _match_detections_to_tracks(self, detections):
        """Match detections to tracks using both IoU and distance."""
        track_ids = list(self.tracks.keys())
        track_bboxes = [self.tracks[tid].bbox for tid in track_ids]
        det_bboxes = [det[0] for det in detections]
        
        cost_matrix = np.zeros((len(track_bboxes), len(det_bboxes)))
        
        for i, track_bbox in enumerate(track_bboxes):
            track_center = ((track_bbox[0] + track_bbox[2]) // 2, (track_bbox[1] + track_bbox[3]) // 2)
            
            for j, det_bbox in enumerate(det_bboxes):
                det_center = ((det_bbox[0] + det_bbox[2]) // 2, (det_bbox[1] + det_bbox[3]) // 2)
                
                # Calculate IoU
                iou = self._calculate_iou(track_bbox, det_bbox)
                
                # Calculate distance
                distance = np.sqrt((track_center[0] - det_center[0])**2 + 
                                 (track_center[1] - det_center[1])**2)

                if distance < self.distance_threshold:

                    cost_matrix[i, j] = distance / 100.0  
                elif iou > self.iou_threshold:
  
                    cost_matrix[i, j] = (1.0 - iou) * 2.0
                else:
                    cost_matrix[i, j] = 999.0
        
        track_indices, det_indices = linear_sum_assignment(cost_matrix)
        
        matched_tracks = []
        matched_detections = []
        
        for track_idx, det_idx in zip(track_indices, det_indices):
            if cost_matrix[track_idx, det_idx] < 50.0:  
                matched_tracks.append(track_ids[track_idx])
                matched_detections.append(det_idx)
        
        return matched_tracks, matched_detections

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

    def _create_new_track(self, bbox: Tuple[int, int, int, int], class_id: int, confidence: float):
        """Create a new track immediately (no confirmation needed)."""
        new_track = SimpleTrack(self.next_id, class_id, bbox, confidence)
        self.tracks[self.next_id] = new_track
        logger.debug(f"Created new track {self.next_id} immediately")
        self.next_id += 1

    def get_tracks(self) -> List[SimpleTrack]:
        """Get all active tracks."""
        return list(self.tracks.values())

    def reset(self):
        """Reset tracker state."""
        self.tracks.clear()
        self.next_id = 0
        logger.info("Simple tracker reset")

    def get_track_by_id(self, track_id: int) -> Optional[SimpleTrack]:
        """Get specific track by ID."""
        return self.tracks.get(track_id)

    def get_closest_track_to_point(self, point: Tuple[int, int]) -> Optional[SimpleTrack]:
        """Get the closest track to a given point (useful for robot targeting)."""
        if not self.tracks:
            return None
        
        closest_track = None
        min_distance = float('inf')
        
        for track in self.tracks.values():
            track_center = track.center
            distance = np.sqrt((track_center[0] - point[0])**2 + (track_center[1] - point[1])**2)
            
            if distance < min_distance:
                min_distance = distance
                closest_track = track
        
        return closest_track
