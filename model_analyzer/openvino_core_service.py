"""
 Model Analyzer

 Copyright (c) 2021 Intel Corporation

 LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”) is subject to
 the terms and conditions of the software license agreements for Software Package, which may also include
 notices, disclaimers, or license terms for third party or open source software
 included in or with the Software Package, and your use indicates your acceptance of all such terms.
 Please refer to the “third-party-programs.txt” or other similarly-named text file included with the Software Package
 for additional details.
 You may obtain a copy of the License at
      https://software.intel.com/content/dam/develop/external/us/en/documents/intel-openvino-license-agreements.pdf
"""
from pathlib import Path

# pylint: disable=import-error
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
