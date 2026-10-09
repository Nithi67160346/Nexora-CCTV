"""Session-only notification drafts. The web application has no live send route.

Future transport follows https://core.telegram.org/bots/api#sendmessage.
It is deliberately separate from the video worker and requires explicit authorization.
"""
from collections import OrderedDict, deque
from copy import deepcopy
import json
import math
import time
from urllib import request, error


ALERT_NAMES = {
    'fall_detected': 'พบเหตุสงสัยการล้ม',
    'fall_posture_review': 'ท่าทางเสี่ยง — ยังยืนยันจังหวะล้มไม่ได้',
    'high_risk_interaction': 'พบการปะทะที่ควรตรวจสอบ',
    'wandering_entered_zone': 'เข้าพื้นที่ที่กำหนด',
    'wandering_exited_zone': 'ออกจากพื้นที่ที่กำหนด',
}


def message_text(event, camera_name, position_s):
    """Plain text only; no identity registry, clip path, image, or medical verdict."""
    lines = ['NEXORA • กรุณาตรวจสอบเหตุการณ์', f'ห้อง: {str(camera_name)[:100]}',
             f'เหตุ: {ALERT_NAMES[event["event_type"]]}', f'ตำแหน่งในแหล่งภาพ: {position_s:.1f} วินาที']
    zone = event.get('metadata', {}).get('zone')
    if zone:
        lines.append(f'โซน: {str(zone)[:100]}')
    lines.append('ผลวิเคราะห์อัตโนมัติ ต้องตรวจภาพยืนยัน')
    return '\n'.join(lines)


class TelegramDrafts:
    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.enabled = False
        self.event_types = [kind for kind in ALERT_NAMES if kind != 'fall_posture_review']
        self.cooldown_sec = 30.0
        self.drafts = deque(maxlen=100)
        self.seen = OrderedDict()
        self.last_draft = {}
        self.suppressed = 0

    def settings(self, enabled, event_types, cooldown_sec):
        if not isinstance(enabled, bool) or not isinstance(event_types, list) or any(t not in ALERT_NAMES for t in event_types):
            raise ValueError('ชนิดเหตุหรือสวิตช์ Telegram ไม่ถูกต้อง')
        value = float(cooldown_sec)
        if not math.isfinite(value) or not 0 <= value <= 3600:
            raise ValueError('ช่วงเว้นข้อความต้องอยู่ระหว่าง 0–3600 วินาที')
        self.enabled, self.event_types, self.cooldown_sec = enabled, list(dict.fromkeys(event_types)), value
        return self.snapshot()

    def record(self, event, camera_name, position_s):
        kind = event.get('event_type')
        if not self.enabled or kind not in self.event_types or not event.get('event_id'):
            return
        key = (event.get('source_id'), event['event_id'])
        if key in self.seen:
            return
        self.seen[key] = True
        if len(self.seen) > 2000:
            self.seen.popitem(last=False)
        scope = (event.get('source_id'), kind, str(event.get('metadata', {}).get('zone', '')))
        now = self.clock()
        if now - self.last_draft.get(scope, -math.inf) < self.cooldown_sec:
            self.suppressed += 1
            return
        self.last_draft[scope] = now
        if len(self.last_draft) > 2000:
            self.last_draft.pop(next(iter(self.last_draft)))
        self.drafts.appendleft(dict(event_id=event['event_id'], source_id=event.get('source_id'),
                                    event_type=kind, state='draft_only', text=message_text(event, camera_name, position_s)))

    def preview(self, camera_name):
        return message_text(dict(event_type='fall_detected'), camera_name, 12.5)

    def snapshot(self):
        return dict(mode='draft_only', live_sending_available=False, enabled=self.enabled,
                    event_types=self.event_types[:], cooldown_sec=self.cooldown_sec,
                    drafts=deepcopy(list(self.drafts)), suppressed=self.suppressed, storage='session')


def send_message(token, chat_id, text, *, live_authorized=False, opener=None):
    """Future integration seam, unused by web routes; never retries an ambiguous send.

    Callers must supply locally loaded credentials after live delivery is authorized.
    Errors never include token-bearing URLs or remote error descriptions.
    """
    if not live_authorized:
        raise PermissionError('Live Telegram delivery is not authorized')
    if not isinstance(token, str) or not token.strip() or any(c.isspace() for c in token):
        raise ValueError('Telegram credentials are missing or invalid')
    if not chat_id or not isinstance(text, str) or not 1 <= len(text) <= 4096:
        raise ValueError('Telegram destination or message is invalid')
    # Restrict token characters so credentials cannot change the host/path structure.
    if any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_:-' for c in token):
        raise ValueError('Telegram credentials are invalid')
    payload = dict(chat_id=chat_id, text=text, link_preview_options=dict(is_disabled=True))
    req = request.Request(f'https://api.telegram.org/bot{token}/sendMessage',
                          data=json.dumps(payload, ensure_ascii=False).encode('utf-8'),
                          headers={'Content-Type':'application/json'}, method='POST')
    transport = opener or request.build_opener()
    try:
        with transport.open(req, timeout=10) as response:
            body = json.load(response)
    except error.HTTPError as failure:
        try:
            body = json.loads(failure.read())
        except (ValueError, OSError):
            body = {}
        return dict(state='failed', error_code=failure.code, retry_after=body.get('parameters', {}).get('retry_after'))
    except (OSError, ValueError):
        # A timeout does not establish whether Telegram accepted the message.
        return dict(state='delivery_unknown', error_code=None, retry_after=None)
    if body.get('ok') is not True:
        return dict(state='failed', error_code=body.get('error_code'), retry_after=body.get('parameters', {}).get('retry_after'))
    return dict(state='sent', message_id=body.get('result', {}).get('message_id'))
