# Copyright (C) 2019-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

class ValueConverter:

    @staticmethod
    def to_giga(value: float) -> float:
        return value / 1000000.0

    @staticmethod
    def to_percentage(value: float) -> float:
        return value / 100
