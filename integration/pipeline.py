"""Combine existing feature modules; capture and tracking belong to the host."""
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from uuid import uuid4
import math

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


class EventAggregator:
    """Upsert lifecycle updates, preserving feature events without cross-type merging.

    Bounded current view; the host persists the update stream separately.
    """
    def __init__(self, capacity=10000):
        if capacity < 1:
            raise ValueError("Event capacity must be positive")
        self.capacity = capacity
        self.events = OrderedDict()

    def upsert(self, event):
        key = (event["source_id"], event["event_id"])
        self.events[key] = deepcopy(event)
        self.events.move_to_end(key)
        while len(self.events) > self.capacity:
            self.events.popitem(last=False)

    def snapshot(self):
        return list(deepcopy(self.events).values())


class WanderingAdapter:
    """Convert SharedContext to the existing Wandering FrameResult contract.

    Separate BoundaryFeature per source avoids collisions between camera track IDs.
    Coordinates here are relative [0,1]; they are scaled to each frame.
    """
    def setup(self, config):
        self.config = config
        self.sources = {}
        for polygon in config.get("zones_relative", {}).values():
            if len(polygon) < 3 or any(len(p) != 2 or not all(0 <= v <= 1 for v in p) for p in polygon):
                raise ValueError("Wandering polygons require at least three relative [0,1] points")

    def reset(self, source_id):
        self.sources.pop(source_id, None)

    def process(self, frame, context):
        from Wandering.ai_camera_system.core.schema import BoundingBox, FrameResult, TrackedObject
        from Wandering.ai_camera_system.features.boundary import BoundaryFeature
        from Wandering.ai_camera_system.utils.geometry import polygon_edges

        height, width = frame.shape[:2]
        source = context["source_id"]
        saved = self.sources.get(source)
        if saved is None or saved[0] != (width, height):
            zones = {name: [(x * width, y * height) for x, y in poly]
                     for name, poly in self.config.get("zones_relative", {}).items()}
            lines = {f"{name}_edge_{i}": edge for name, poly in zones.items()
                     for i, edge in enumerate(polygon_edges(poly))}
            saved = ((width, height), BoundaryFeature(zones=zones, lines=lines), {})
            self.sources[source] = saved
        _, module, last_seen = saved
        ts = context["timestamp_ms"]
        timeout = float(self.config.get("track_timeout_sec", 2)) * 1000
        for tid, last in list(last_seen.items()):
            if ts - last >= timeout:
                module._prev_position.pop(tid, None)
                module._inside_zones.pop(tid, None)
                del last_seen[tid]
        objects = []
        for person in context["persons"]:
            tid = person["track_id"]
            if tid is None:
                continue
            last_seen[tid] = ts
            objects.append(TrackedObject(tid, BoundingBox(*person["bbox_xyxy"]), person["confidence"], "person"))
        result = FrameResult(context["frame_id"], ts / 1000, width, height, objects)
        module.process(frame, result)
        events = []
        for event in result.events:
            events.append(dict(schema_version="1.0", event_id=str(uuid4()),
                event_type="wandering_" + event.event_type, source_id=source,
                track_ids=[event.track_id], start_timestamp_ms=ts, end_timestamp_ms=ts,
                confidence=1.0, severity={"warning": "medium", "critical": "high"}.get(event.severity, "info"),
                message=event.message, status="needs_review",
                metadata=dict(event.extra, feature="wandering", confidence_kind="geometric_rule",
                              original_event_type=event.event_type)))
        return events

    def flush(self):
        self.sources.clear()
        return []


class ProgressPipeline:
    def __init__(self, modules, aggregator=None):
        self.modules = dict(modules)
        self.aggregator = aggregator if aggregator is not None else EventAggregator()
        self._clocks = {}
        self.health = {name: dict(state='ready', error=None, failures=0, last_success_ms=None)
                       for name in self.modules}

    def _module_failed(self, name, error, source_id=None):
        health = self.health[name]
        health.update(state='error', error=str(error))
        health['failures'] += 1
        # Discard interrupted evidence; it cannot bridge an unavailable frame.
        if source_id is not None:
            try:
                self.modules[name].reset(source_id)
            except Exception:
                pass

    @classmethod
    def from_config(cls, config):
        modules = {}
        features = config.get("features", {})
        if features.get("violence", {}).get("enabled", True):
            if features.get('violence', {}).get('backend') == 'resnet18_lstm':
                from integration.violence_lstm import ViolenceLSTM
                module = ViolenceLSTM().setup(features['violence'])
            else:
                from app.features.violence import HighRiskInteractionModule
                module = HighRiskInteractionModule()
                module.setup(load_config(ROOT / "app/features/violence/config.yaml"))
            modules["violence"] = module
        if features.get("location", {}).get("enabled", True):
            from Location.location import LocationModule
            # Explicit progress config: no sample residents/rooms or face models.
            cfg = deepcopy(features.get("location", {}).get("config", {}))
            cfg.setdefault("identity", {"enabled": False})
            modules["location"] = LocationModule().setup(cfg)
        if features.get("wandering", {}).get("enabled", True):
            module = WanderingAdapter()
            module.setup(features.get("wandering", {}))
            modules["wandering"] = module
        if features.get("fall", {}).get("enabled", False):
            if features['fall'].get('backend') == 'yolo_pose_rf_v2':
                from integration.fall_ml import FallMLAdapter
                module = FallMLAdapter()
            else:
                from integration.fall_adapter import FallAdapter
                module = FallAdapter()
            fall_config = deepcopy(features.get('fall', {}))
            fall_config['posture_review_exclusion_zones'] = [deepcopy(zone)
                for zone in features.get('location', {}).get('config', {}).get('zones', {}).values()
                if isinstance(zone, dict) and zone.get('type') == 'bed']
            module.setup(fall_config)
            modules["fall"] = module
        return cls(modules, EventAggregator(config.get("event_capacity", 10000)))

    def _collect(self, name, updates):
        result = []
        for original in updates:
            event = deepcopy(original)
            event.setdefault("metadata", {})["feature"] = name
            self.aggregator.upsert(event)
            result.append(event)
        return result

    def process(self, frame, context):
        source, ts = context["source_id"], float(context["timestamp_ms"])
        if not math.isfinite(ts) or ts < 0:
            raise ValueError("timestamp_ms must be finite and nonnegative")
        previous = self._clocks.get(source)
        if previous is not None and ts < previous:
            raise ValueError("Timestamp moved backwards; reset this source after seeking")
        if previous == ts:
            return []
        self._clocks[source] = ts
        updates = []
        for name, module in self.modules.items():
            try:
                module_context = context
                if name in ('fall', 'violence') and not getattr(module, 'uses_scene_pixels', False):
                    # Keep box visibility/location without treating failed pose
                    # validation as usable action evidence. Still send an empty
                    # current frame so temporal modules observe the dropout.
                    module_context = dict(context, persons=[p for p in context.get('persons', [])
                        if p.get('observation_kind') != 'bbox_only'])
                events = self._collect(name, module.process(frame, module_context))
                updates.extend(events)
                self.health[name].update(state='ready', error=None, last_success_ms=ts)
            except Exception as error:
                self._module_failed(name, error, source)
        return updates

    def reset(self, source_id):
        # Reset discards buffers; flush at pipeline shutdown for final updates.
        for name, module in self.modules.items():
            try:
                module.reset(source_id)
                self.health[name].update(state='ready', error=None, last_success_ms=None)
            except Exception as error:
                self._module_failed(name, error)
        self._clocks.pop(source_id, None)

    def flush(self):
        updates = []
        for name, module in self.modules.items():
            try:
                updates.extend(self._collect(name, module.flush()))
            except Exception as error:
                self._module_failed(name, error)
        self._clocks.clear()
        return updates
