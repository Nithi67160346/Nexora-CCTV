"""One shared YOLO+tracker invocation per frame, with optional pose output."""
import numpy as np
from .person_filter import PersonObservationFilter


class SharedYoloCore:
    def __init__(self, model, device="cpu", confidence=0.25):
        self.model = model
        self.device = device
        self.confidence = confidence
        self.person_filter = PersonObservationFilter()

    def reset(self):
        """Discard tracker history after a seek, loop, or source change."""
        for tracker in getattr(getattr(self.model, 'predictor', None), 'trackers', ()):
            tracker.reset()
        self.person_filter.reset()

    def process(self, frame, *, source_id, frame_id, timestamp_ms, fps):
        results = self.model.track(frame, persist=True, classes=[0], conf=self.confidence,
                                   tracker="bytetrack.yaml", device=self.device, verbose=False)
        persons = []
        require_pose = getattr(self.model, 'task', None) == 'pose'
        if results:
            result = results[0]
            boxes = result.boxes
            if boxes is not None and boxes.id is not None:
                poses = None
                if getattr(result, "keypoints", None) is not None:
                    require_pose = True
                    xy = result.keypoints.xy.cpu().numpy()
                    conf = result.keypoints.conf
                    if conf is not None:
                        poses = np.concatenate([xy, conf.cpu().numpy()[..., None]], axis=-1).tolist()
                for i, (bbox, confidence, tid) in enumerate(zip(boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist(), boxes.id.cpu().tolist())):
                    x1, y1, x2, y2 = bbox
                    persons.append(dict(track_id=int(tid), confidence=float(confidence), bbox_xyxy=bbox,
                        center_xy=[(x1 + x2) / 2, (y1 + y2) / 2],
                        pose=poses[i] if poses is not None else None))
        context = dict(schema_version="1.0", source_id=source_id, frame_id=frame_id,
                       timestamp_ms=timestamp_ms, fps=fps, persons=persons)
        filtered = self.person_filter.process(context, frame.shape, require_pose=require_pose)
        boxes = results[0].boxes if results else None
        scores = getattr(boxes, 'conf', None)
        raw_count = len(scores) if scores is not None else len(persons)
        filtered['person_filter'].update(raw_detections=raw_count,
            tracked_detections=len(persons), untracked_detections=max(0, raw_count-len(persons)))
        return filtered
