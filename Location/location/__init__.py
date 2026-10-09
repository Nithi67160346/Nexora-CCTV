"""
Location / Bed / Zone / Last-Seen Feature (+ Optional Face Recognition)

    from location import LocationModule, load_config
    location = LocationModule().setup(load_config())
    events += location.process(frame, context)
"""

from .models import Identity, Location
from .module import DEFAULT_CONFIG_PATH, LocationModule, load_config

__all__ = ["LocationModule", "load_config", "DEFAULT_CONFIG_PATH", "Identity", "Location"]
