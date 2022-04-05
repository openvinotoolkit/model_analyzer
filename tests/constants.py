# Copyright (C) 2019-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

MODELS_PATH = Path(os.environ['MODELS_PATH']) / 'v11'
CONFIGS_FOLDER = Path(__file__).parent.resolve() / 'data'
