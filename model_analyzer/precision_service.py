# Copyright (C) 2019-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from typing import Dict

from model_analyzer.singleton_metaclass import SingletonType


class Precision(Enum):
    fp32 = 'FP32'
    fp16 = 'FP16'
    int8 = 'INT8'
    bool = 'BOOL'
    unknown = 'UNKNOWN'


class PrecisionService(metaclass=SingletonType):
    _precisions_to_aliases = {
        Precision.fp32: ['FP32', 'float', 'F32', 'I32', 'INT32', 'int32', 'U32', 'UI32'],
        Precision.fp16: ['FP16', 'half', 'F16', 'I16', 'INT16', 'int16', 'U16', 'UI16'],
        Precision.int8: ['I8', 'INT8', 'int8', 'U8', 'UI8'],
        Precision.bool: ['BIN', 'bin', 'bool', 'INT1', 'int1', 'I1'],
    }

    def __init__(self):
        self._aliases_to_precisions = self._get_aliases_to_precisions()

    @staticmethod
    def _get_aliases_to_precisions() -> Dict[str, Precision]:
        return {
            precision: client_representation
            for client_representation, precisions in PrecisionService._precisions_to_aliases.items()
            for precision in precisions
        }

    def get_precision(self, value: str) -> Precision:
        return self._aliases_to_precisions.get(value, Precision.unknown)

    @staticmethod
    def is_int(precision: Precision) -> bool:
        return precision in (Precision.int8, Precision.bool)

    @staticmethod
    def is_fp(precision: Precision) -> bool:
        return precision in (Precision.fp32, Precision.fp16)


PRECISION_SERVICE = PrecisionService()
