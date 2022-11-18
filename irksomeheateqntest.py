from firedrake import *
from irksome import GaussLegendre, Dt, TimeStepper
from ufl.algorithms.ad import expand_derivatives
import FileFuncs as ff
import matplotlib.pyplot as plt
import os
def simulations():
    butcher_tableau = GaussLegendre(1)
    ns = butcher_tableau.num_stages
    N = 100
    x0 = 0.0
    x1 = 10.0
    y0 = 0.0
    y1 = 10.0

    msh = RectangleMesh(N, N, x1, y1)
    V = FunctionSpace(msh, "CG", 1)
    dt = Constant(10.0 / N)
    t = Constant(0.0)

    x, y = SpatialCoordinate(msh)
    u = interpolate(0.5*0.05*(cos(x)+cos(y)), V)
    v = TestFunction(V)
    F = inner(Dt(u), v)*dx + inner(grad(u), grad(v))*dx 
    luparams = {"mat_type": "aij",
            "ksp_type": "preonly",
            "pc_type": "lu"}
    stepper = TimeStepper(F, butcher_tableau, t, dt, u,
                      solver_parameters=luparams)
    var_hist=[]
    nt=0
    tmax=50.0
    vmin=min(u.vector())
    vmax=max(u.vector())
    while (float(t) < tmax):
        if (float(t) + float(dt) > tmax):
            dt.assign(tmax - float(t))
        stepper.advance()
        umin=min(u.vector())
        umax=max(u.vector())
        if umin<vmin: vmin=umin
        if umax>vmax: vmax=umax
        print(float(t))
        t.assign(float(t) + float(dt))
        nt+=1
        var_hist.append(u.copy(deepcopy=True))
    path=os.path.abspath(os.getcwd())
    plotdir='/TestPlots/'
    ff.check_folder(path,plotdir)
    Plotting(var_hist,nt,path,plotdir,vmin,vmax)
    return
def Plotting(var_hist,nt,path,PlotFoldername,vmin,vmax):
    ''' Plot an animation of a tricontour plot for the variable `var_cls`.
    Save plot at path+PlotFoldername+var_cls.name+'.mp4'
    :param var_cls: `SpatialVariable` object with results in.
    :param path: Location for the output
    :param PlotFoldername: directory for the results.
    :param consts_dict: dictionary of the parameters.'''
    fig, axes = plt.subplots()
    axes.set_aspect('equal')
    colors = tripcolor(var_hist[0], num_sample_points=1, axes=axes,cmap='inferno',vmin=vmin,vmax=vmax)
    fig.colorbar(colors)
    fig.savefig(path+PlotFoldername+'/'+'HeatEqnTest'+ff.numfix(0)+'.jpg')
    fig.clear()
    plt.close()
    print('length of t',nt)
    for tstep in range(nt-1):
        fig, axes = plt.subplots()
        axes.set_aspect('equal')
        colors = tripcolor(var_hist[tstep], num_sample_points=1, 
        axes=axes,cmap='inferno')#,vmin=0,vmax=0.04)
        fig.colorbar(colors)
        fig.savefig(path+PlotFoldername+'/'+'HeatEqnTest'+ff.numfix(tstep)+'.jpg',vmin=vmin,vmax=vmax)
        fig.clear()
        plt.close()
    return
if __name__=='__main__':
    simulations()
    exit()