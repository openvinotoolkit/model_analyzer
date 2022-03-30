# Copyright (C) 2019-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

# pylint: disable=import-error
from typing import Tuple

from openvino.runtime import CompiledModel, Core, Model


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

    @property
    def available_devices(self) -> Tuple[str]:
        return self._core.available_devices

    def read_model(self, model_path: Path, weights_path: Path) -> Model:
        return self.core.read_model(str(model_path), str(weights_path))

    def compile_model(self, model: Model, device: str) -> CompiledModel:
        return self.core.compile_model(model, device)


OPENVINO_CORE_SERVICE = OpenVINOCoreService()
