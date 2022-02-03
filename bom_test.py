# Copyright (C) 2019-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import platform
import re
from itertools import islice

DIR_PATTERNS_TO_SKIP = [
    '.*__pycache__.*',
    '.*\\.git.*',
    '.*automation.*',
    'tests',
    '.*venv',
    '.*.pytest_cache',
    '.*.idea'
]
FILE_PATTERNS_TO_SKIP = [
    '.*_test\\.py$',
    '.*\\.pyc$',
    'server\\.log$'
]
FILES_TO_IGNORE = [
    '.*DEVELOPER.md*.',
    '.*requirements_dev.txt.*',
    '.*.gitignore.*',
    '.*.git.*',
    '.*.pylintrc.*',
    '.*Dockerfile.*',
    '.*generic_e2e_test_case.py*.'
]

FULL_NAME_PATTERNS_TO_SKIP = ['.*tests/.*']
if platform.system() == 'Windows':
    FULL_NAME_PATTERNS_TO_SKIP = [i.replace('/', '\\\\') for i in FULL_NAME_PATTERNS_TO_SKIP]
DIRS_TO_SEARCH = ['']


def is_match(name: str, patterns: ()):
    return any((re.match(pattern, name) for pattern in patterns))


class TestBOMFile:
    @classmethod
    def setup(cls):
        cls.existing_files = []
        cur_path = os.path.dirname(os.path.realpath(__file__))
        cls.output_dir = cur_path
        with open(os.path.join(cur_path, 'automation', 'package_BOM.txt'), 'r') as bom_file:
            if platform.system() == 'Windows':
                cls.existing_files = [name.rstrip().replace('/', '\\') for name in bom_file.readlines()]
            else:
                cls.existing_files = [name.rstrip() for name in bom_file.readlines()]

        # pylint:disable=W1401
        cls.expected_header = [re.compile(pattern) for pattern in [
            '^# Copyright (C) 2019-2022 Intel Corporation$',
            '^# SPDX-License-Identifier: Apache-2.0$'
        ]]

    def test_bom_file(self):
        missing_files = list()
        for src_dir in DIRS_TO_SEARCH:
            src_dir = os.path.join(self.output_dir, src_dir)
            if not os.path.isdir(src_dir):
                continue
            for root, _, files in os.walk(src_dir):
                if is_match(root, DIR_PATTERNS_TO_SKIP):
                    continue
                for file in files:
                    if is_match(file, FILES_TO_IGNORE):
                        continue
                    full_name = os.path.join(root, file)
                    full_name = full_name[len(self.output_dir) + 1:]
                    if is_match(file, FILE_PATTERNS_TO_SKIP) or is_match(full_name, FULL_NAME_PATTERNS_TO_SKIP):
                        continue
                    if full_name not in self.existing_files:
                        missing_files.append(full_name)

        if missing_files:
            print('{} files missed in BOM:'.format(len(missing_files)))
            for file in missing_files:
                print(file.replace('\\', '/'))
        assert not missing_files

    def test_deleted_files_still_stored_in_bom(self):
        deleted = list()
        ignores = [
            '^static'
        ]
        for file in self.existing_files:
            if is_match(file, ignores):
                continue
            if not os.path.isfile(os.path.join(self.output_dir, file)):
                deleted.append(file)
        if deleted:
            print('{} files deleted but still stored in BOM:'.format(len(deleted)))
            for file in deleted:
                print(file)
        assert not deleted

    def test_alphabetical_order_and_duplicates(self):
        sorted_bom = sorted([x for x in self.existing_files if self.existing_files.count(x) == 1], key=str.lower)
        if self.existing_files != sorted_bom:
            print("Wrong order. Alphabetical order of BOM is:")
            print(*sorted_bom, sep='\n')
            assert self.existing_files == sorted_bom

    def test_missed_intel_header(self):
        missing_files = list()
        for src_dir in DIRS_TO_SEARCH:
            src_dir = os.path.join(self.output_dir, src_dir)
            if not os.path.isdir(src_dir):
                continue
            for root, _, files in os.walk(src_dir):
                if is_match(root, DIR_PATTERNS_TO_SKIP):
                    continue
                for file in files:
                    ignores = [
                        '^__init__.py$',
                        '^.*.pyc$',
                        '^.*test.py$'
                    ]
                    if not is_match(file, ['.*.py$']) or is_match(file, ignores):
                        continue
                    full_name = os.path.join(root, file)
                    with open(full_name, 'r', encoding='utf-8') as source_f:
                        # read two more lines from the file because it can contain shebang and empty lines
                        source = [x.strip() for x in islice(source_f, len(self.expected_header) + 2)]
                        # skip shebang and empty lines in the beginning of the file
                        while source[0] in ('', '#!/usr/bin/env python3'):
                            source = source[1:]
                        for str_ind in range(0, len(self.expected_header)):
                            if not re.match(self.expected_header[str_ind], source[str_ind]):
                                missing_files.append(full_name)
                                break
        if missing_files:
            print('{} files with missed header: \n{}'.format(len(missing_files), '\n'.join(missing_files)))
        assert not missing_files
