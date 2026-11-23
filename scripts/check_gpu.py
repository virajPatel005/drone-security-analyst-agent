"""Confirm PyTorch can initialise and use the local NVIDIA GPU."""

from __future__ import annotations

import torch


def main() -> None:
    print(f"torch_version: {torch.__version__}")
    print(f"torch_cuda_build: {torch.version.cuda}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu_name: {torch.cuda.get_device_name(0)}")
        print(f"compute_capability: {torch.cuda.get_device_capability(0)}")
        probe = torch.ones(4, 4, device="cuda")
        print(f"cuda_matmul_ok: {bool((probe @ probe).sum().item() > 0)}")
    else:
        print("gpu_name: n/a")
        print("compute_capability: n/a")
        print("cuda_matmul_ok: False")


if __name__ == "__main__":
    main()
