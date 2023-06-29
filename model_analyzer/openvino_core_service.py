# Copyright (C) 2019-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import logging
from pathlib import Path

# pylint: disable=import-error
from openvino.runtime import CompiledModel, Core, Model
from openvino.runtime.passes import ConstantFolding
from openvino.runtime.passes import Manager


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

    def compile_model(self, model: Model, device: str) -> CompiledModel:
        return self.core.compile_model(model, device)

    @staticmethod
    def pass_constant_folding(model: Model):
        try:
            pass_manager = Manager()
            pass_manager.register_pass(ConstantFolding())
            pass_manager.set_per_pass_validation(False)
            pass_manager.run_passes(model)
        # pylint: disable=broad-except
        except Exception as exception:
            logging.error(exception, exc_info=True)


OPENVINO_CORE_SERVICE = OpenVINOCoreService()
