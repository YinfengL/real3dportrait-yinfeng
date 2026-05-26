# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import glob
import hashlib
import importlib
import os
import re
import shutil
import uuid

import torch
import torch.utils.cpp_extension
from torch.utils.file_baton import FileBaton

#----------------------------------------------------------------------------
# Global options.

verbosity = 'brief'  # Verbosity level: 'none', 'brief', 'full'

#----------------------------------------------------------------------------
# Internal helper funcs.

def _find_compiler_bindir():
    patterns = [
        'C:/Program Files (x86)/Microsoft Visual Studio/*/Professional/VC/Tools/MSVC/*/bin/Hostx64/x64',
        'C:/Program Files (x86)/Microsoft Visual Studio/*/BuildTools/VC/Tools/MSVC/*/bin/Hostx64/x64',
        'C:/Program Files (x86)/Microsoft Visual Studio/*/Community/VC/Tools/MSVC/*/bin/Hostx64/x64',
        'C:/Program Files (x86)/Microsoft Visual Studio */vc/bin',
    ]
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if len(matches):
            return matches[-1]
    return None

def _get_mangled_gpu_name():
    name = torch.cuda.get_device_name().lower()
    out = []
    for c in name:
        if re.match('[a-z0-9_-]+', c):
            out.append(c)
        else:
            out.append('-')
    return ''.join(out)

#----------------------------------------------------------------------------
# Main entry point for compiling and loading C++/CUDA plugins.

_cached_plugins = dict()

def get_plugin(module_name, sources, headers=None, source_dir=None, **build_kwargs):
    assert verbosity in ['none', 'brief', 'full']
    if headers is None:
        headers = []
    if source_dir is not None:
        sources = [os.path.join(source_dir, fname) for fname in sources]
        headers = [os.path.join(source_dir, fname) for fname in headers]

    if module_name in _cached_plugins:
        return _cached_plugins[module_name]

    if verbosity == 'full':
        print(f'Setting up PyTorch plugin "{module_name}"...')
    elif verbosity == 'brief':
        print(f'Setting up PyTorch plugin "{module_name}"... ', end='', flush=True)
    verbose_build = (verbosity == 'full')

    module = None
    try:
        if os.name == 'nt' and os.system("where cl.exe >nul 2>nul") != 0:
            compiler_bindir = _find_compiler_bindir()
            if compiler_bindir is None:
                raise RuntimeError(f'Could not find MSVC/GCC/CLANG installation on this computer. Check _find_compiler_bindir() in "{__file__}".')
            os.environ['PATH'] += ';' + compiler_bindir

        os.environ['TORCH_CUDA_ARCH_LIST'] = ''

        all_source_files = sorted(sources + headers)
        all_source_dirs = set(os.path.dirname(fname) for fname in all_source_files)
        if len(all_source_dirs) == 1:
            hash_md5 = hashlib.md5()
            for src in all_source_files:
                with open(src, 'rb') as f:
                    hash_md5.update(f.read())

            source_digest = hash_md5.hexdigest()
            build_top_dir = torch.utils.cpp_extension._get_build_directory(module_name, verbose=verbose_build)  # pylint: disable=protected-access
            cached_build_dir = os.path.join(build_top_dir, f'{source_digest}-{_get_mangled_gpu_name()}')

            if not os.path.isdir(cached_build_dir):
                tmpdir = f'{build_top_dir}/srctmp-{uuid.uuid4().hex}'
                os.makedirs(tmpdir)
                for src in all_source_files:
                    shutil.copyfile(src, os.path.join(tmpdir, os.path.basename(src)))
                try:
                    os.replace(tmpdir, cached_build_dir)
                except OSError:
                    shutil.rmtree(tmpdir)
                    if not os.path.isdir(cached_build_dir):
                        raise

            cached_sources = [os.path.join(cached_build_dir, os.path.basename(fname)) for fname in sources]
            build_kwargs = dict(build_kwargs)
            build_kwargs.setdefault('extra_include_paths', [])
            build_kwargs['extra_include_paths'] = list(build_kwargs['extra_include_paths']) + [
                '/usr/local/cuda-12.1/include',
                '/usr/local/cuda-12.1/targets/x86_64-linux/include',
            ]
            build_kwargs.setdefault('extra_ldflags', [])
            build_kwargs['extra_ldflags'] = list(build_kwargs['extra_ldflags']) + [
                '-L/usr/local/cuda-12.1/lib64',
            ]
            
            torch.utils.cpp_extension.load(
                name=module_name,
                build_directory=cached_build_dir,
                verbose=verbose_build,
                sources=cached_sources,
                **build_kwargs
            )


            # 直接从生成的 .so 加载，而不是 importlib.import_module
            so_path = os.path.join(cached_build_dir, f'{module_name}.so')
            if not os.path.isfile(so_path):
                so_candidates = glob.glob(os.path.join(cached_build_dir, '*.so'))
                if len(so_candidates) == 0:
                    raise FileNotFoundError(f'Could not find compiled shared object for {module_name} under {cached_build_dir}')
                so_path = so_candidates[0]

            torch.ops.load_library(so_path)
            module = torch.ops
        else:
            # fallback: torch extension standard path
            build_dir = torch.utils.cpp_extension._get_build_directory(module_name, verbose=verbose_build)  # pylint: disable=protected-access
            torch.utils.cpp_extension.load(
                name=module_name,
                verbose=verbose_build,
                sources=sources,
                build_directory=build_dir,
                **build_kwargs
            )
            so_path = os.path.join(build_dir, f'{module_name}.so')
            if not os.path.isfile(so_path):
                so_candidates = glob.glob(os.path.join(build_dir, '*.so'))
                if len(so_candidates) == 0:
                    raise FileNotFoundError(f'Could not find compiled shared object for {module_name} under {build_dir}')
                so_path = so_candidates[0]
            torch.ops.load_library(so_path)
            module = torch.ops

    except:
        if verbosity == 'brief':
            print('Failed!')
        raise

    if verbosity == 'full':
        print(f'Done setting up PyTorch plugin "{module_name}".')
    elif verbosity == 'brief':
        print('Done.')
    _cached_plugins[module_name] = module
    return module
