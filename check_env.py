"""Sanity-check the ECR environment before training: library versions, the GPU, a tiny R-GCN run on CUDA,
and the data/backbone folders. Exits with status 1 if anything required is missing.

    python check_env.py
"""

import os
import platform
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
failures = []


def check(name, ok, detail="", required=True):
    mark = "ok  " if ok else ("FAIL" if required else "warn")
    print(f"[{mark}] {name}{': ' + detail if detail else ''}")
    if not ok and required:
        failures.append(name)


check("python 3.8", sys.version_info[:2] == (3, 8), platform.python_version())

try:
    import accelerate
    import torch
    import torch_geometric
    import transformers
except ImportError as e:
    check("imports", False, str(e))
    sys.exit(1)

for mod, want in [(torch, "1.8.1"), (transformers, "4.15.0"), (accelerate, "0.8.0"), (torch_geometric, "2.0.1")]:
    check(f"{mod.__name__} {want}", mod.__version__.startswith(want), mod.__version__)

cuda = torch.cuda.is_available()
check("CUDA available", cuda, f"torch built for CUDA {torch.version.cuda}")
if cuda:
    cap = torch.cuda.get_device_capability(0)
    mem = torch.cuda.get_device_properties(0).total_memory / 2**30
    check("GPU", True, f"{torch.cuda.get_device_name(0)}, compute capability {cap[0]}.{cap[1]}, {mem:.0f} GB")
    check("GPU memory >= 24 GB (authors' setup)", mem >= 23, f"{mem:.0f} GB; lower batch sizes if smaller",
          required=False)
    if cap > (8, 6):
        print("       this GPU is newer than CUDA 11.1 targets (max 8.6); the kernel test below shows if it works")
    try:
        from torch_geometric.nn import RGCNConv

        conv = RGCNConv(8, 8, num_relations=3, num_bases=2).cuda()
        x = torch.randn(5, 8, device="cuda", requires_grad=True)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], device="cuda")
        conv(x, edge_index, torch.tensor([0, 1, 2, 0], device="cuda")).sum().backward()
        check("R-GCN forward/backward on GPU", True)
    except Exception as e:  # incompatible GPU or broken torch-scatter/torch-sparse install
        check("R-GCN forward/backward on GPU", False, f"{type(e).__name__}: {e}")


def has_files(rel):
    path = os.path.join(ROOT, rel)
    return os.path.isdir(path) and any(not f.endswith(".md") for f in os.listdir(path))


for folder, hint, required in [
    ("src_emo/data/emo_data", "run: bash download_data.sh data", True),
    ("src_emo/data/saved", "authors' checkpoints, run: bash download_data.sh ckpt", False),
    ("src_emo/save/dialogpt", "created by setup_env.sh", True),
    ("src_emo/save/roberta", "created by setup_env.sh", True),
]:
    ok = has_files(folder)
    check(folder, ok, "" if ok else hint, required)

print("\nall required checks passed" if not failures else f"\n{len(failures)} required check(s) failed")
sys.exit(1 if failures else 0)
