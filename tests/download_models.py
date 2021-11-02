"""
 Model Analyzer

 Copyright (c) 2019 Intel Corporation

 LEGAL NOTICE: Your use of this software and any required dependent software (the “Software Package”) is subject to
 the terms and conditions of the software license agreements for Software Package, which may also include
 notices, disclaimers, or license terms for third party or open source software
 included in or with the Software Package, and your use indicates your acceptance of all such terms.
 Please refer to the “third-party-programs.txt” or other similarly-named text file included with the Software Package
 for additional details.
 You may obtain a copy of the License at
      https://software.intel.com/content/dam/develop/external/us/en/documents/intel-openvino-license-agreements.pdf
"""
import json
import os
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path

from tests.utils import load_test_config

CURRENT_SCRIPT_DIRECTORY = Path.cwd()
openvino_dir = Path(os.environ['INTEL_OPENVINO_DIR'])

model_downloader_script = (
        openvino_dir / 'extras' / 'open_model_zoo' / 'tools' / 'model_tools' / 'downloader.py'
)


def parse_arguments():
    parser = ArgumentParser()

    parser.add_argument('-c', '--config',
                        help='Path to tests configuration file',
                        required=True,
                        type=Path)

    parser.add_argument('-o', '--output-dir',
                        help='Path to the directory where downloaded models will be stored',
                        required=False,
                        type=Path,
                        default=Path(CURRENT_SCRIPT_DIRECTORY) / 'data' / 'models')

    return parser.parse_args()


def load_config(config_file_path: Path):
    _, config = load_test_config(config_file_path)
    return config


def download_model(model_name: str, output_path: Path):
    subprocess.run(
        [sys.executable, model_downloader_script, '--name', model_name, '--output_dir', str(output_path), '--precision',
         'FP32'])


def download_models(model_name: str, output_directory: Path):
    print(model_name)
    print(output_directory)


if __name__ == '__main__':
    ARGUMENTS = parse_arguments()
    config = load_config(ARGUMENTS.config)
    for model_info in config:
        download_model(model_info['name'], ARGUMENTS.output_dir)
