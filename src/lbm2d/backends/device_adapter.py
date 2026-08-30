"""Centralized explicit device-memory compatibility API for CUDA-MLIR."""
from __future__ import annotations

from numba_cuda_mlir import cuda


def to_device(array):
    return cuda.to_device(array)


def device_array_like(array):
    return cuda.device_array_like(array)


def copy_to_host(device_array, host_array=None):
    return device_array.copy_to_host(host_array)


def copy_to_device(device_array, host_array):
    device_array.copy_to_device(host_array)


def synchronize():
    cuda.synchronize()

