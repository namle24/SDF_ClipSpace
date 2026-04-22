import argparse
import pyvista as pv
import trimesh
import sys
import os

# Add parent directory and src to path to handle package imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import optimized GPU calculation function
from src.core.fast_sdf_gpu import compute_fast_sdf_gpu

def main():
    parser = argparse.ArgumentParser(description="3D Mesh Comparison Tool: Raw vs SDF-Colored")
    parser.add_argument("--input_file", type=str, default="data/bunny1.obj", help="Path to input model file (.obj, .off)")
    parser.add_argument("--fov", type=int, default=90, help="Cone FOV angle for SDF calculation")
    parser.add_argument("--batch_size", type=int, default=32, help="Number of triangles processed per GPU batch")
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"ERROR: File not found: {args.input_file}")
        sys.exit(1)

    print(f"[*] Loading model: {args.input_file} ...")
    try:
        mesh = trimesh.load(args.input_file, force='mesh')
        mesh.fix_normals()
        mesh.process()
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    print("[*] Computing SDF using Vectorized GPU Rasterization...")
    # Calculate SDF using our core library
    sdf_values = compute_fast_sdf_gpu(mesh, fov_deg=args.fov, batch_size=args.batch_size)

    # -------------------------------------------------------------
    # VISUALIZATION PHASE: SPLIT-SCREEN INTERFACE
    # -------------------------------------------------------------
    
    # Mesh 1: Raw gray mesh
    pv_mesh_raw = pv.wrap(mesh)
    
    # Mesh 2: SDF-mapped mesh
    pv_mesh_sdf = pv.wrap(mesh)
    pv_mesh_sdf.cell_data['SDF_Values'] = sdf_values

    # Initialize Plotter with 1x2 grid
    plotter = pv.Plotter(shape=(1, 2), title="SDF Visualization Comparison: Raw vs Heatmap")

    # LEFT VIEWPORT: Raw Mesh
    plotter.subplot(0, 0)
    plotter.add_text("Before: Raw Mesh", font_size=14, color='black')
    plotter.add_mesh(pv_mesh_raw, color='#d3d3d3', show_edges=True, edge_color='#555555', specular=0.5)
    
    # RIGHT VIEWPORT: SDF Heatmap
    plotter.subplot(0, 1)
    plotter.add_text("After: SDF (Vectorized Rasterization)", font_size=14, color='black')
    # Heatmap 'jet_r' (Red = Thick, Blue = Thin)
    plotter.add_mesh(pv_mesh_sdf, scalars='SDF_Values', cmap='jet_r', show_edges=True, smooth_shading=True)

    # Synchronize camera views for both viewports
    plotter.link_views()

    # Initial camera orientation
    plotter.view_isometric()

    # Professional white background
    plotter.set_background('white')
    
    print("\n[+] Visualization started. Use mouse to rotate/zoom.")
    plotter.show()

if __name__ == "__main__":
    main()
