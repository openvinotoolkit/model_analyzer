# Copyright (C) 2019-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from model_analyzer import main, parse_arguments


if __name__ == '__main__':
    ARGUMENTS = parse_arguments()
    main(ARGUMENTS)
