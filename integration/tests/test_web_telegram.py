import io
import json
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError

from web.telegram_service import TelegramDrafts, send_message
from web import server


def event(eid, source='cam_01', kind='fall_detected'):
    return dict(event_id=eid, source_id=source, event_type=kind,
                metadata=dict(people=[dict(identity_name='private name')]))


class TelegramTests(unittest.TestCase):
    def test_drafts_are_opt_in_and_deduplicate_lifecycle_with_source_cooldown(self):
        clock = Mock(return_value=0)
        drafts = TelegramDrafts(clock)
        drafts.record(event('a'), 'ห้อง A', 1)
        self.assertEqual(len(drafts.drafts), 0)
        drafts.settings(True, ['fall_detected'], 30)
        drafts.record(event('a'), 'ห้อง A', 1)
        drafts.record(event('a'), 'ห้อง A', 2)
        drafts.record(event('b'), 'ห้อง A', 2)
        drafts.record(event('a', 'cam_02'), 'ห้อง B', 2)
        clock.return_value = 31
        drafts.record(event('c'), 'ห้อง A', 3)
        self.assertEqual(len(drafts.drafts), 3)
        self.assertEqual(drafts.suppressed, 1)
        self.assertFalse(drafts.snapshot()['live_sending_available'])
        self.assertNotIn('private name', json.dumps(drafts.snapshot()))

    def test_invalid_settings_are_atomic_and_location_updates_do_not_alert(self):
        drafts = TelegramDrafts()
        before = drafts.snapshot()
        for types, delay in [(['location_update'], 30), ([], float('nan')), ([], -1)]:
            with self.assertRaises(ValueError): drafts.settings(True, types, delay)
            self.assertEqual(drafts.snapshot(), before)

    def test_web_preview_and_settings_never_call_transport(self):
        worker = server.StreamWorker()
        with patch.object(server, 'worker', worker), patch('web.telegram_service.request.build_opener') as network:
            server.configure_telegram(server.TelegramSettings(enabled=True, event_types=['fall_detected']))
            self.assertFalse(server.preview_telegram()['sent'])
            worker._record_updates([event('a')])
            self.assertEqual(len(server.get_telegram()['drafts']), 1)
            network.assert_not_called()

    def test_transport_requires_explicit_authorization_before_touching_network(self):
        opener = Mock()
        with self.assertRaises(PermissionError): send_message('token', 'chat', 'test', opener=opener)
        opener.open.assert_not_called()

    def test_transport_wire_shape_and_success_use_mock_only(self):
        opener = Mock(); response = io.StringIO('{"ok":true,"result":{"message_id":9}}')
        opener.open.return_value = response
        result = send_message('123:fake', '-1001', 'ข้อความ', live_authorized=True, opener=opener)
        req = opener.open.call_args.args[0]
        self.assertEqual(req.full_url, 'https://api.telegram.org/bot123:fake/sendMessage')
        self.assertEqual(json.loads(req.data)['text'], 'ข้อความ')
        self.assertNotIn('parse_mode', json.loads(req.data))
        self.assertEqual(result, dict(state='sent', message_id=9))

    def test_transport_errors_do_not_expose_token_or_retry_ambiguous_delivery(self):
        opener = Mock()
        opener.open.side_effect = HTTPError('secret token URL', 429, 'rate limit', {}, io.BytesIO(b'{"parameters":{"retry_after":7}}'))
        result = send_message('123:fake', '1', 'test', live_authorized=True, opener=opener)
        self.assertEqual(result, dict(state='failed', error_code=429, retry_after=7))
        opener.open.side_effect = TimeoutError('secret token URL')
        result = send_message('123:fake', '1', 'test', live_authorized=True, opener=opener)
        self.assertEqual(result['state'], 'delivery_unknown')
        self.assertNotIn('secret', str(result))
        self.assertEqual(opener.open.call_count, 2)
