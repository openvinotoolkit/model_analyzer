# Copyright (C) 2019-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path

from utils import load_test_config

CURRENT_SCRIPT_DIRECTORY = Path.cwd()


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


def load_config(config_file_path: Path) -> dict:
    _, config = load_test_config(config_file_path)
    return config


def download_model(model_name: str, output_path: Path):
    subprocess.run(
        ['omz_downloader', '--name', model_name, '--output_dir', str(output_path), '--precision',
         'FP32'])


def download_models(model_name: str, output_directory: Path):
    print(model_name)
    print(output_directory)


if __name__ == '__main__':
    ARGUMENTS = parse_arguments()
    config = load_config(ARGUMENTS.config)
    for model_info in config:
        download_model(model_info['name'], ARGUMENTS.output_dir)
