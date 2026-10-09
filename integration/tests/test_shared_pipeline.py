import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from integration.core import SharedYoloCore
from integration.pipeline import EventAggregator, ProgressPipeline, ROOT, WanderingAdapter, load_config


def context(source="A", ts=0, bbox=(8, 8, 96, 88)):
    return dict(schema_version="1.0", source_id=source, frame_id=int(ts * 6 / 1000),
        timestamp_ms=ts, fps=6, persons=[dict(track_id=7, confidence=.95,
        bbox_xyxy=list(bbox), center_xy=[(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2],
        pose=[[40+(j%2)*12,16+j*3,.9] for j in range(17)])])


class SharedIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((96, 128, 3), np.uint8)

    def test_product_defaults_do_not_load_held_checkpoint(self):
        pipeline = ProgressPipeline.from_config(load_config(ROOT / "integration/progress.yaml"))
        self.assertEqual(set(pipeline.modules), {"violence", "location", "wandering"})
        for i in range(31):
            updates = pipeline.process(self.frame, context(ts=round(i*1000/6)))
            self.assertFalse(any(e["event_type"].startswith("wandering_") for e in updates))
        self.assertTrue(any(e["event_type"] == "location_update" for e in pipeline.aggregator.snapshot()))
        pipeline.flush()

    def test_old_config_does_not_reintroduce_archived_model(self):
        cfg = load_config(ROOT / 'integration/progress.yaml')
        cfg['features']['seizure'] = dict(enabled=True)
        with patch.dict('sys.modules', {'app.features.seizure': None}):
            pipeline = ProgressPipeline.from_config(cfg)
        self.assertNotIn('seizure', pipeline.modules)

    def test_wandering_sources_and_missing_track_do_not_share_history(self):
        module = WanderingAdapter()
        module.setup(dict(zones_relative={"zone":[[0,0],[1,0],[1,1],[0,1]]}))
        self.assertEqual(len(module.process(self.frame, context("A"))), 1)
        self.assertEqual(len(module.process(self.frame, context("B"))), 1)
        self.assertEqual(module.process(self.frame, context("A", 500)), [])
        absent = context("A", 3000);absent["persons"] = []
        module.process(self.frame, absent)
        self.assertEqual(len(module.process(self.frame, context("A", 3100))), 1)
        self.assertEqual(module.process(self.frame, context("B", 1000)), [])

    def test_shared_yolo_runs_once_and_preserves_pixel_pose_and_identity(self):
        boxes = SimpleNamespace(xyxy=torch.tensor([[8.,8.,96.,88.]]),
            conf=torch.tensor([.95]), id=torch.tensor([7.]))
        pose = SimpleNamespace(xy=torch.tensor([[[20.+j%2*40, 16.+j*3] for j in range(17)]]), conf=torch.ones((1,17))*.9)
        model = SimpleNamespace(track=lambda *args, **kwargs: [SimpleNamespace(boxes=boxes,keypoints=pose)])
        with patch.object(model, "track", wraps=model.track) as call:
            result = SharedYoloCore(model).process(self.frame, source_id="A", frame_id=0, timestamp_ms=0, fps=30)
            self.assertEqual(call.call_count, 1)
            self.assertTrue(call.call_args.kwargs["persist"])
        self.assertEqual(result["persons"][0]["track_id"], 7)
        self.assertEqual(result["persons"][0]["bbox_xyxy"], [8,8,96,88])
        self.assertEqual(len(result["persons"][0]["pose"]), 17)

    def test_untracked_detection_is_not_given_fake_id(self):
        model = SimpleNamespace(track=lambda *args, **kwargs: [SimpleNamespace(boxes=SimpleNamespace(id=None))])
        result = SharedYoloCore(model).process(self.frame, source_id="A", frame_id=0, timestamp_ms=0, fps=30)
        self.assertEqual(result["persons"], [])

    def test_aggregator_keeps_sources_separate_and_bounds_memory(self):
        store = EventAggregator(2)
        for source in ("A", "B"):
            store.upsert(dict(source_id=source, event_id="same", metadata={"lifecycle":"started"}))
        store.upsert(dict(source_id="A", event_id="same", metadata={"lifecycle":"ended"}))
        self.assertEqual(len(store.snapshot()), 2)
        store.upsert(dict(source_id="C", event_id="new", metadata={}))
        self.assertEqual([e["source_id"] for e in store.snapshot()], ["A", "C"])

    def test_reset_and_timestamp_guards_apply_to_all_features(self):
        module = SimpleNamespace(process=lambda *args: [], reset=lambda source: None, flush=lambda: [])
        pipeline = ProgressPipeline({"spy":module})
        pipeline.process(self.frame, context(ts=100))
        with patch.object(module, "process", wraps=module.process) as call:
            self.assertEqual(pipeline.process(self.frame, context(ts=100)), [])
            self.assertEqual(call.call_count, 0)
        with self.assertRaises(ValueError):pipeline.process(self.frame, context(ts=50))
        invalid=context();invalid['timestamp_ms']=float('inf')
        with self.assertRaises(ValueError):pipeline.process(self.frame, invalid)
        pipeline.reset("A")
        self.assertEqual(pipeline.process(self.frame, context(ts=0)), [])


if __name__ == "__main__":
    unittest.main()
