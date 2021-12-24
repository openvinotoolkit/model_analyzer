from pathlib import Path

from openvino.pyopenvino import Model
from openvino.runtime import Core


class SingletonType(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonType, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class OpenVINOCoreService(metaclass=SingletonType):
    def __init__(self):
        self._core = Core()

    @property
    def core(self) -> Core:
        return self._core

    def read_model(self, model_path: Path, weights_path: Path) -> Model:
        return self.core.read_model(str(model_path), str(weights_path))

    def compile_model(self, model: Model, device: str):
        return self.core.compile_model(model, device)


OPENVINO_CORE_SERVICE = OpenVINOCoreService()