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

import csv
import logging as log
import os
from typing import List

from model_analyzer.layer_provider import LayerTypesManager, LayerType, Constant, Result, Parameter
from model_analyzer.network_metadata import NetworkMetaData


# pylint: disable=too-many-instance-attributes
class NetworkComputationalComplexity:
    def __init__(self, metadata: NetworkMetaData):
        self.net_metadata = metadata
        self.layer_providers = [LayerTypesManager.provider(layer) for layer in self.net_metadata.ops]
        self.input_names = [layer_provider.name for layer_provider in self.layer_providers if
                            isinstance(layer_provider, Parameter)]
        self.output_names = [layer_provider.name for layer_provider in self.layer_providers if
                             isinstance(layer_provider, Result)]
        self.ignored_layers = []
        self.ignore_unknown_layers = True
        self.computational_complexity = {}
        net_precisions = set()
        for layer_provider in self.layer_providers:
            for i in range(layer_provider.get_outputs_number()):
                net_precisions.add(layer_provider.get_output_precision(i))
            self.computational_complexity[layer_provider.name] = {'layer_type': layer_provider.type,
                                                                  'layer_name': layer_provider.name}

            input_blob, output_blob = self.get_blob_sizes_and_precisions(layer_provider)
            self.computational_complexity[layer_provider.name]['input_blob'] = input_blob
            self.computational_complexity[layer_provider.name]['output_blob'] = output_blob

        self.net_precisions = list(net_precisions.union(self.net_metadata.int8precisions))
        self.params_const_layers = set()

    @property
    def network(self):
        return self.net_metadata.network

    def get_total_params(self) -> dict:
        parameters = {
            'total_params': 0,
            'zero_params': 0
        }
        for layer_provider in self.layer_providers:
            params = 0
            layer_params_dict = layer_provider.get_params()
            if layer_params_dict:
                for const_name in layer_params_dict.keys():
                    params, zeros = layer_params_dict[const_name]
                    # Avoid double counting
                    if const_name not in self.params_const_layers:
                        parameters['total_params'] += params
                        self.params_const_layers.add(const_name)
                        if layer_provider.name in self.ignored_layers:
                            continue
                        parameters['zero_params'] += zeros
            self.computational_complexity[layer_provider.name]['m_params'] = params / 1000000.0
        return parameters

    @staticmethod
    def get_blob_sizes_and_precisions(layer_provider) -> tuple:
        inputs = []
        for i in range(layer_provider.get_inputs_number()):
            input_precision = layer_provider.get_input_precision(i)
            input_shape_as_string = 'x'.join(map(str, layer_provider.get_input_shape(i)))
            input_str = f'{input_precision}({input_shape_as_string})'
            inputs.append(input_str)
        in_blob = ' '.join(inputs)

        outputs = []
        for i in range(layer_provider.get_outputs_number()):
            output_precision = layer_provider.get_output_precision(i)
            output_shape_as_string = 'x'.join(map(str, layer_provider.get_output_shape(i)))
            output_str = f'{output_precision}({output_shape_as_string})'
            outputs.append(output_str)
        out_blob = ' '.join(outputs)

        return in_blob, out_blob

    def get_maximum_memory_consumption(self) -> int:
        total_memory_size = 0
        for layer_provider in self.layer_providers:
            if not isinstance(layer_provider, Constant):
                total_memory_size += layer_provider.get_output_blobs_total_size()
        return total_memory_size

    def get_minimum_memory_consumption(self) -> int:
        input_layer_providers = list(filter(lambda x: x.name in self.input_names, self.layer_providers))
        all_layer_providers = list(
            filter(lambda x: not isinstance(x, Constant), self.layer_providers))
        is_computed = {layer_provider.name: False for layer_provider in all_layer_providers}

        direct_input_children_names = []
        for layer_provider in input_layer_providers:
            direct_input_children_names.extend(layer_provider.get_child_names())

        max_memory_size = 0
        for layer_provider in all_layer_providers:
            current_memory_size = layer_provider.get_output_blobs_total_size()

            for prev_layer_provider in all_layer_providers:
                if prev_layer_provider.name == layer_provider.name:
                    break
                memory_not_needed = True

                for child_name in prev_layer_provider.get_child_names():
                    memory_not_needed = memory_not_needed and is_computed.get(child_name)

                if not memory_not_needed:
                    current_memory_size += prev_layer_provider.get_output_blobs_total_size()

            max_memory_size = max(max_memory_size, current_memory_size)

            is_computed[layer_provider.name] = True
        return max_memory_size

    def print_network_info(self, output, file_name, complexity, complexity_filename):
        g_flops, g_iops = self.get_total_ops()

        parameters = self.get_total_params()
        total_parameters = parameters['total_params']
        zero_params = parameters['zero_params']

        total_params = total_parameters / 1000000.0
        sparsity = zero_params / total_parameters * 100
        min_mem_consumption = self.get_minimum_memory_consumption() / 1000000.0
        max_mem_consumption = self.get_maximum_memory_consumption() / 1000000.0
        net_precisions = self.net_precisions.pop() if len(self.net_precisions) == 1 else 'MIXED (' + '-'.join(
            sorted(self.net_precisions)) + ')'
        guessed_type = self.net_metadata.guess_topology_type()
        if guessed_type:
            guessed_type = guessed_type.value
        log.info('GFLOPs: %.4f', g_flops)
        log.info('GIOPs: %.4f', g_iops)
        log.info('MParams: %.4f', total_params)
        log.info('Sparsity: %.4f%%', sparsity)
        log.info('Minimum memory consumption: %.4f', min_mem_consumption)
        log.info('Maximum memory consumption: %.4f', max_mem_consumption)
        log.info('Guessed type: %s', guessed_type)
        export_network_into_csv(g_flops, g_iops, total_params, sparsity, min_mem_consumption, max_mem_consumption,
                                net_precisions, output, file_name, guessed_type)
        if complexity:
            self.export_layers_into_csv(output, complexity_filename)

    def export_layers_into_csv(self, output_dir, file_name):
        if output_dir:
            file_name = os.path.join(output_dir, file_name)
        with open(file_name, mode='w') as info_file:
            info_writer = csv.writer(info_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            info_writer.writerow(
                ['LayerType', 'LayerName', 'GFLOPs', 'GIOPs', 'MParams', 'LayerParams', 'InputBlobs', 'OutputBlobs'])
            layers_ids = self.net_metadata.get_layers_ids()
            try:
                sorted_layers = sorted(self.computational_complexity.keys(), key=lambda x: layers_ids[x])
            except (KeyError, TypeError):
                sorted_layers = sorted(self.computational_complexity.keys())
            core_layers = filter(lambda x: x not in self.input_names, sorted_layers)
            for layer_name in core_layers:
                cur_layer = self.computational_complexity[layer_name]
                info_writer.writerow([
                    cur_layer['layer_type'],
                    cur_layer['layer_name'],
                    '{:.4f}'.format(float(cur_layer['g_flops'])),
                    '{:.4f}'.format(float(cur_layer['g_iops'])),
                    '{:.4f}'.format(float(cur_layer['m_params'])),
                    cur_layer['layer_params'] if 'layer_params' in cur_layer.keys() else None,
                    cur_layer['input_blob'],
                    cur_layer['output_blob'],
                ])
        log.info('Complexity file name: %s', file_name)

    def get_ops(self, layer_provider) -> int:
        try:
            total_flops = layer_provider.get_ops() * pow(10, -9)
        except NotImplementedError as error:
            self.computational_complexity[layer_provider.name]['g_iops'] = -1
            self.computational_complexity[layer_provider.name]['g_flops'] = -1
            raise error
        self.computational_complexity[layer_provider.name]['layer_params'] = get_layer_params(layer_provider)
        self.computational_complexity[layer_provider.name]['g_iops'] = (
            total_flops if layer_provider.name in self.net_metadata.int8layers else 0
        )
        self.computational_complexity[layer_provider.name]['g_flops'] = (
            0 if layer_provider.name in self.net_metadata.int8layers else total_flops
        )
        return total_flops

    def get_total_ops(self) -> tuple:
        uncounted_layers = set()
        unknown_layers = set()
        total_flops = 0
        total_iops = 0
        for layer_provider in self.layer_providers:
            if layer_provider.__class__ == LayerType:
                unknown_layers.add(layer_provider.type)
            try:
                layer_flops = self.get_ops(layer_provider)
            except NotImplementedError:
                uncounted_layers.add(layer_provider.type)
                continue
            if layer_provider.name in self.net_metadata.int8layers:
                total_iops += layer_flops
                continue
            total_flops += layer_flops
        if not self.ignore_unknown_layers and unknown_layers:
            print(f'Unknown types: {", ".join(unknown_layers)}')
            raise Exception('Model contains unknown layers!')
        if uncounted_layers:
            print(f'Warning, GOPS for layer(s) was not counted - {", ".join(uncounted_layers)}')
        return total_flops, total_iops

    def set_ignored_layers(self, ignored_layers: List[str], ignore_first_conv: bool, ignore_fc: bool):
        self.ignored_layers.extend(ignored_layers)
        all_convs = []
        all_fcs = []
        for layer_provider in self.layer_providers:
            if layer_provider.type.lower() == 'convolution':
                all_convs.append(layer_provider.name)
            elif layer_provider.type.lower() == 'fullyconnected':
                all_fcs.append(layer_provider.name)
            elif layer_provider.type.lower() == 'scaleshift':
                self.ignored_layers.extend(layer_provider.name)
        if ignore_first_conv:
            self.ignored_layers.append(all_convs[0])
        if ignore_fc:
            self.ignored_layers.extend(all_fcs)

    def set_ignore_unknown_layers(self, ignore_unknown_layers: bool):
        self.ignore_unknown_layers = ignore_unknown_layers


# pylint: disable=too-many-arguments
def export_network_into_csv(g_flops, g_iops, total_params, sparsity, min_mem_consumption, max_mem_consumption,
                            net_precisions, output_dir, file_name, guessed_type):
    if output_dir:
        file_name = os.path.join(output_dir, file_name)
    with open(file_name, mode='w') as info_file:
        info_writer = csv.writer(info_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        info_writer.writerow(['GFLOPs', 'GIOPs', 'MParams', 'MinMem', 'MaxMem', 'Sparsity', 'Precision', 'GuessedType'])
        info_writer.writerow(
            [g_flops, g_iops, total_params, min_mem_consumption, max_mem_consumption, sparsity, net_precisions,
             guessed_type])
    log.info('Network status information file name: %s', file_name)


def get_layer_params(layer_provider):
    params = []

    layer_params = layer_provider.params
    layer_params.pop('element_type', None)
    layer_params.pop('shape', None)

    if not layer_params:
        return ''

    for param in sorted(layer_params):
        value = layer_params[param]
        if isinstance(value, list):
            value_string = f'({"xs".join(str(x) for x in value)})'
        elif isinstance(value, str) and ',' in value:
            value_string = f'({"x".join(value.split(","))})'
        else:
            value_string = value

        params.append(f'{param}: {value_string}')
    return f'[{"; ".join(params)}]'
