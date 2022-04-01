# Copyright (C) 2019-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Tuple

from model_analyzer.model_complexity import ModelComputationalComplexity
from model_analyzer.model_metadata import ModelMetaData


def parse_arguments():
    log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)
    parser = ArgumentParser()

    parser.add_argument('-m', '--model',
                        help='Path to an .xml file of the Intermediate Representation (IR) model'
                             'Or path to .onnx or .prototxt file of ONNX model',
                        required=True,
                        type=Path)

    parser.add_argument('-w', '--weights',
                        help='Path to the .bin file of the Intermediate Representation (IR) model. If not specified,'
                             'the weights file name expected to be the same as the .xml file passed'
                             'with --model option',
                        type=Path)

    parser.add_argument('-o', '--report_dir', '--report-dir',
                        help='Output directory',
                        type=Path,
                        default=Path.cwd())

    parser.add_argument('--model_report', '--model-report',
                        help='Name for the file where theoretical analysis results are stored',
                        type=str,
                        default='model_report.csv')

    parser.add_argument('--per_layer_mode', '--per-layer-mode',
                        help='Enables collecting per-layer complexity metrics',
                        action='store_true',
                        default=False)

    parser.add_argument('--per_layer_report', '--per-layer-report',
                        help='File name for the per-layer complexity metrics. '
                             'Should be specified only when --per-layer-mode option' +
                             ' is used',
                        default='per_layer_report.csv')

    parser.add_argument('--sparsity_ignored_layers', '--sparsity-ignored-layers',
                        help='Specifies ignored layers names separated by comma',
                        default='')

    parser.add_argument('--sparsity_ignore_first_conv', '--sparsity-ignore-first-conv',
                        help='Ignores first Convolution layer for sparsity computation',
                        action='store_true',
                        default=False)

    parser.add_argument('--sparsity_ignore_fc', '--sparsity-ignore-fc',
                        help='Ignores FullyConnected layers for sparsity computation',
                        action='store_true',
                        default=False)

    parser.add_argument('--ignore_unknown_layers', '--ignore-unknown-layers',
                        help='Ignores unknown types of layers when counting GFLOPs',
                        action='store_true',
                        default=False)

    arguments = parser.parse_args()
    arguments.model, arguments.weights = process_model_files(arguments)
    return arguments


def process_model_files(cli_args) -> Tuple[Path, Path]:
    if cli_args.model.suffix == '.xml' and not cli_args.weights:
        cli_args.weights = cli_args.model.with_suffix('.bin')
    return cli_args.model, cli_args.weights


def main(cli_args):
    log.info('Loading network files:\n\t%s\n\t%s', cli_args.model, cli_args.weights)

    model_metadata = ModelMetaData(cli_args.model, cli_args.weights)

    model_computational_complexity = ModelComputationalComplexity(model_metadata)
    model_computational_complexity.set_ignore_unknown_layers(cli_args.ignore_unknown_layers)

    sparsity_ignored_layers = cli_args.sparsity_ignored_layers.split(',')
    model_computational_complexity.set_ignored_layers(sparsity_ignored_layers,
                                                      cli_args.sparsity_ignore_first_conv,
                                                      cli_args.sparsity_ignore_fc)

    model_computational_complexity.print_network_info(cli_args.report_dir,
                                                      cli_args.model_report,
                                                      cli_args.per_layer_mode,
                                                      cli_args.per_layer_report)


if __name__ == '__main__':
    ARGUMENTS = parse_arguments()
    main(ARGUMENTS)
