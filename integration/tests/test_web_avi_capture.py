"""AVI decoder crashes must stay outside the web process."""
import io
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from unittest.mock import Mock

import cv2
import numpy as np
from web import capture
from web.violence_clip import sample_frames


class AviCaptureTests(unittest.TestCase):
    def test_main_player_avi_never_opens_native_opencv_decoder(self):
        with patch.object(capture, 'AviCapture', return_value='isolated') as decoder, \
             patch.object(capture.cv2, 'VideoCapture', side_effect=AssertionError('unsafe decoder')):
            self.assertEqual(capture.open_capture('uploaded/My Video.AVI'), 'isolated')
        decoder.assert_called_once_with('uploaded/My Video.AVI')

    def test_gpu_ai_can_use_cpu_avi_but_explicit_nvdec_request_is_rejected(self):
        with patch.object(capture, 'AviCapture', return_value='isolated'):
            self.assertEqual(capture.open_selected_capture('clip.avi', 'cpu'), 'isolated')
        with self.assertRaisesRegex(ValueError, 'AVI'):
            capture.open_selected_capture('clip.avi', 'cuda')

    def test_ffprobe_metadata_reads_exact_frame_count_and_fractional_fps(self):
        info = {'streams':[dict(width=320,height=240,avg_frame_rate='25/1',nb_frames='181')],
                'format':{'duration':'7.24'}}
        result = SimpleNamespace(returncode=0,stdout=json.dumps(info).encode(),stderr=b'')
        with patch.object(capture.Path, 'is_file', return_value=True), patch.object(capture.subprocess, 'run', return_value=result):
            values = capture._avi_metadata('clip.avi', '/usr/bin/ffmpeg')
        self.assertEqual(values[cv2.CAP_PROP_FRAME_COUNT],181)
        self.assertEqual(values[cv2.CAP_PROP_FPS],25)

    def test_ffmpeg_only_installation_probes_without_decoding_frames(self):
        text = b'Duration: 00:00:07.24, start: 0\nStream #0:0: Video: rawvideo, bgr24, 320x240, 25 fps\n'
        result = SimpleNamespace(returncode=0,stdout=b'',stderr=text)
        with patch.object(capture.Path, 'is_file', return_value=False), patch.object(capture.shutil, 'which', return_value=None), \
             patch.object(capture.subprocess, 'run', return_value=result) as run:
            values = capture._avi_metadata('clip.avi', '/bundle/ffmpeg')
        self.assertEqual(values[cv2.CAP_PROP_FRAME_COUNT],181)
        args = run.call_args.args[0]
        self.assertEqual(args[args.index('-frames:v')+1],'0')

    def test_decoder_abort_raises_python_error_and_does_not_become_normal_eof(self):
        decoder = capture.AviCapture.__new__(capture.AviCapture)
        decoder.position = 0
        decoder.values = {cv2.CAP_PROP_FRAME_WIDTH:2,cv2.CAP_PROP_FRAME_HEIGHT:2}
        decoder.process = SimpleNamespace(stdout=io.BytesIO(b'\x00'*12),poll=lambda:134)
        decoder.error_file = io.BytesIO(b'corrupted double-linked list')
        ok, frame = decoder.read()
        self.assertTrue(ok); self.assertEqual(frame.shape,(2,2,3))
        with self.assertRaisesRegex(RuntimeError,'AVI.*corrupted'):
            decoder.read()

    def test_manual_lstm_sampling_uses_same_safe_entry_point(self):
        isolated = Mock()
        isolated.get.side_effect = lambda key: 181 if key==cv2.CAP_PROP_FRAME_COUNT else 25
        isolated.read.return_value = (True,np.ones((8,8,3),dtype=np.uint8))
        with patch.object(capture, 'AviCapture', return_value=isolated), \
             patch.object(capture.cv2, 'VideoCapture', side_effect=AssertionError('unsafe decoder')):
            frames, metadata = sample_frames('unsafe.avi')
        self.assertEqual(len(frames),16)
        self.assertEqual(metadata['total_frames'],181)
        isolated.release.assert_called_once()

    def test_incomplete_output_frame_cannot_be_reported_as_completed_clip(self):
        decoder = capture.AviCapture.__new__(capture.AviCapture)
        decoder.values = {cv2.CAP_PROP_FRAME_WIDTH:2,cv2.CAP_PROP_FRAME_HEIGHT:2}
        decoder.process = SimpleNamespace(stdout=io.BytesIO(b'\x00'*6),poll=lambda:0)
        with self.assertRaisesRegex(RuntimeError,'เฟรมไม่ครบ'):
            decoder.read()


if __name__ == '__main__':
    unittest.main()
