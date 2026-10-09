"""Product feature decisions; archived research modules remain available."""
from copy import deepcopy


ARCHIVED_FEATURES = {"seizure"}


def apply_web_feature_policy(config):
    """Old configs and recovery snapshots must not reactivate held features."""
    candidate = deepcopy(config)
    features = candidate.setdefault("features", {})
    for name in ARCHIVED_FEATURES:
        features.pop(name, None)
    return candidate


def is_archived_event(event):
    return (event.get("metadata", {}).get("feature") in ARCHIVED_FEATURES
            or str(event.get("event_type", "")).startswith("seizure"))
