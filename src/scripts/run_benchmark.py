import os
import time
import glob
import trimesh
import sys
import torch

# Add parent directory and src to path to handle package imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Monkey patch tqdm to be completely silent
import tqdm
import builtins
class DummyTqdm:
    def __init__(self, iterable=None, *args, **kwargs):
        self.iterable = iterable
    def __iter__(self):
        return iter(self.iterable) if self.iterable else iter([])
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def set_description(self, *args): pass
    def update(self, *args): pass
tqdm.tqdm = DummyTqdm
sys.modules['tqdm'].tqdm = DummyTqdm

from src.core.sdf_calculator import compute_sdf_cone
from src.core.custom_ortho_sdf import compute_custom_ortho_sdf
from src.core.custom_ortho_sdf_gpu import compute_custom_ortho_sdf_gpu

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w', encoding='utf-8')
    def __exit__(self, exc_type, exc_val, exc_tb):
        if sys.stdout != self._original_stdout:
            sys.stdout.close()
        sys.stdout = self._original_stdout

def main():
    # Only test on a few specific files to save time
    test_files = [
        'data/radio_0026.off',
        'data/wardrobe_0032.off',
        'data/bunny1.obj'
    ]
    
    results = []
    header = ["Model Name", "Vertex Count", "Standard (Meshlab)", "Vectorized (CPU)", "Vectorized GPU", "GPU vs CPU", "GPU vs Meshlab"]
    results.append(header)
    
    print("Starting Performance Benchmarking...\n")
    if torch.cuda.is_available():
        with HiddenPrints():
            dummy = trimesh.creation.box()
            compute_custom_ortho_sdf_gpu(dummy)
            torch.cuda.synchronize()
            
    for f in test_files:
        if not os.path.exists(f):
             continue
             
        filename = os.path.basename(f)
        print(f"[*] Benchmarking model: {filename} ... ", end="")
        sys.stdout.flush()
        
        try:
            with HiddenPrints():
                mesh = trimesh.load(f, force='mesh')
            num_vertices = len(mesh.vertices)
        except Exception:
            print(f"Mesh loading error!")
            continue
            
        # 1. Meshlab Style (Serial)
        try:
            with HiddenPrints():
                t0 = time.perf_counter()
                compute_sdf_cone(mesh, num_rays=30, cone_angle=120)
                t1 = time.perf_counter()
            time_meshlab = t1 - t0
        except Exception:
            time_meshlab = float('nan')
            
        # 2. CPU Vectorized
        try:
            with HiddenPrints():
                t0 = time.perf_counter()
                compute_custom_ortho_sdf(mesh, num_rays=30, batch_size=200)
                t1 = time.perf_counter()
            time_cpu = t1 - t0
        except Exception:
            time_cpu = float('nan')
            
        # 3. GPU Vectorized
        try:
            with HiddenPrints():
                t0 = time.perf_counter()
                compute_custom_ortho_sdf_gpu(mesh, num_rays=30, batch_size=2048)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
            time_gpu = t1 - t0
        except Exception:
            time_gpu = float('nan')
            
        speedup_cpu = time_cpu / time_gpu if time_gpu > 0 else 0
        speedup_mesh = time_meshlab / time_gpu if time_gpu > 0 else 0
        
        results.append([
            filename,
            f"{num_vertices:,}",
            f"{time_meshlab:.3f} s",
            f"{time_cpu:.3f} s",
            f"{time_gpu:.3f} s",
            f"{speedup_cpu:.1f}x faster",
            f"{speedup_mesh:.1f}x faster"
        ])
        print("Done!")
        
    print("\n" + "="*110)
    print(f"{'SDF PERFORMANCE BENCHMARK (30 RAYS)'.center(110)}")
    print("="*110)
    
    col_widths = [max(len(str(item)) for item in col) for col in zip(*results)]
    for i, row in enumerate(results):
        formatted_row = " | ".join(str(item).ljust(width) for item, width in zip(row, col_widths))
        print(f"| {formatted_row} |")
        if i == 0:
            separator = "-|-".join('-' * width for width in col_widths)
            print(f"| {separator} |")
            
    print("="*110)

if __name__ == "__main__":
    main()
