"""Report ready only when the web API and detector have initialized."""
import json
import sys
from urllib.request import ProxyHandler, build_opener


def ready(status):
    if not isinstance(status, dict) or not isinstance(status.get('health'), dict):
        return False
    detector = status['health'].get('detector')
    return isinstance(detector, dict) and detector.get('state') == 'ready'


def main():
    try:
        with build_opener(ProxyHandler({})).open('http://127.0.0.1:8080/api/status', timeout=3) as response:
            if ready(json.load(response)):
                return 0
    except (OSError, ValueError):
        pass
    print('NEXORA web API / detector is not ready yet.', file=sys.stderr)
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
