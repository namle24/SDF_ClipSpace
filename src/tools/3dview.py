import pyvista as pv

mesh = pv.read(r"D:\CMSVTransformer\bunny1.obj")

plotter = pv.Plotter()
plotter.add_mesh(mesh)
plotter.show()