from hallu.calibration.base import Calibrator, Identity
from hallu.calibration.conformal import risk_control_threshold
from hallu.calibration.isotonic import Isotonic
from hallu.calibration.platt import Platt
from hallu.calibration.temperature import Temperature

_REGISTRY = {"identity": Identity, "none": Identity, "platt": Platt, "temperature": Temperature,
             "isotonic": Isotonic}


def get_calibrator(name: str) -> Calibrator:
    try:
        return _REGISTRY[name.lower()]()
    except KeyError:
        raise ValueError(f"unknown calibrator {name!r}; choose from {sorted(_REGISTRY)}") from None


__all__ = ["Calibrator", "Identity", "Platt", "Temperature", "Isotonic", "get_calibrator", "risk_control_threshold"]
