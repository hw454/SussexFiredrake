from firedrake import *
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
nx=40
ny=40
mesh = UnitSquareMesh(nx,ny,quadrilateral=True)
V = FunctionSpace(mesh, "CG", 1)
x, y = SpatialCoordinate(mesh)
cos
f=Function(V)
u=Function(V)
StSt=np.array([2,1])
perm=0.1
hx=1/nx
hy=1/ny
f.interpolate(StSt[1]+perm*(sin(pi*x/(2*hx))+sin(pi*y/(2*hy))))
print(f.dat.data[:])
#lot(f)
ns=10
fn_plotter=FunctionPlotter(mesh,num_sample_points=ns)
fig, axes = plt.subplots()
levels = np.linspace(0, 1, 51)
contours=tripcolor(f, axes=axes, cmap="inferno")
#contours = tricontourf(f,levels=levels, axes=axes, cmap="inferno")
axes.set_aspect("equal")
fig.colorbar(contours)
fig.show()
fig.savefig('TestPlot.jpg')
