import threading
import unittest
from unittest.mock import Mock, patch

from fastapi import HTTPException
from web import server


class StreamLifecycleTests(unittest.TestCase):
    def test_stop_waits_for_inference_without_holding_processing_lock(self):
        worker = server.StreamWorker()
        entered = threading.Event()
        release = threading.Event()
        acquired = threading.Event()

        def inference_finishes():
            entered.set()
            release.wait(3)
            with worker.lock:
                acquired.set()

        worker.thread = threading.Thread(target=inference_finishes)
        worker.thread.start()
        self.assertTrue(entered.wait(1))
        original_join = worker.thread.join

        def join(timeout):
            self.assertGreater(timeout, 1.5)
            release.set()
            original_join(3)

        try:
            with patch.object(worker.thread, 'join', side_effect=join):
                worker.stop_stream()
            self.assertTrue(acquired.is_set())
            self.assertIsNone(worker.thread)
            self.assertEqual(worker.stage, 'idle')
        finally:
            release.set()
            original_join(3)

    def test_timeout_retains_old_thread_and_models_for_safe_retry(self):
        worker = server.StreamWorker()
        old = Mock()
        old.is_alive.side_effect = [True, True, False]
        worker.thread = old
        worker.pipeline = Mock()
        worker.pipeline.flush.return_value = []
        with self.assertRaises(RuntimeError):
            worker.stop_stream()
        self.assertIs(worker.thread, old)
        worker.pipeline.flush.assert_not_called()
        self.assertFalse(worker.is_running)
        self.assertEqual(worker.stage, 'stopping')
        worker.stop_stream()
        self.assertIsNone(worker.thread)
        worker.pipeline.flush.assert_called_once()

    def test_concurrent_start_cannot_reuse_models_before_stop_finishes(self):
        worker = server.StreamWorker()
        stopping = threading.Event()
        finish = threading.Event()
        started = threading.Event()

        def slow_stop():
            stopping.set()
            finish.wait(3)

        with patch.object(worker, '_stop_stream', side_effect=slow_stop), \
             patch.object(worker, '_start_stream', side_effect=lambda *args: started.set()):
            stop = threading.Thread(target=worker.stop_stream)
            start = threading.Thread(target=lambda: worker.start_stream('0'))
            stop.start()
            try:
                self.assertTrue(stopping.wait(1))
                start.start()
                self.assertFalse(started.wait(.05))
            finally:
                finish.set()
                stop.join(3)
                start.join(3)
            self.assertTrue(started.is_set())

    def test_start_timeout_is_recoverable_conflict_without_sticky_error(self):
        worker = server.StreamWorker()
        worker.stage = 'stopping'
        with patch.object(server, 'worker', worker), \
             patch.object(server.streams, 'active', return_value=False), \
             patch.object(worker, 'start_stream', side_effect=RuntimeError('waiting')):
            with self.assertRaises(HTTPException) as raised:
                server.start_stream(server.StartStreamRequest(source='browser://test'))
        self.assertEqual(raised.exception.status_code, 409)
        self.assertIsNone(worker.last_error)
