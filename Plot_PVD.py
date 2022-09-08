from paraview.simple import *
import FileFuncs as ff

def plot_from_pvd(fname):
    pvd=PVDReader(Filename=fname)
    return
def main(filename,test=0):
    ResultsFolder='Plots'
    MeshFolder  ='Mesh'
    if test==1:
        ResultsFolder='./DiffusionOnly/'+ResultsFolder
        MeshFolder   ='./DiffusionOnly/'+MeshFolder
    dirlist=(ResultsFolder,MeshFolder)
    etaname     ='eta'
    thetaname   ='theta'
    tname       ='tvec'
    pars,pdes_pars=ff.read_inputs(filename)
    # nx,ny are number of elements on x and y axis
    # xm, ym are maximum x and y values
    # x0,y0 are minimum x and y values
    # hx, hy are meshwidth on the x and y axis.
    nx,ny,xm,ym,x0,y0=pars
    hx=(xm-x0)/nx
    hy=(ym-y0)/ny # mesh size
    # Parameters for the PDEs
    _,_,_,_,_,_,_,_,_,dt,T,perm=pdes_pars
    parstr=ff.par_string(dt,T,perm,hx,hy)
    ff.check_folder(ResultsFolder)
    ff.check_folder(MeshFolder)
    MeshFolder=MeshFolder+'/'+parstr
    ResultsFolder=ResultsFolder+'/'+parstr
    ff.check_folder(ResultsFolder)
    ff.check_folder(MeshFolder)
    plot_from_pvd(MeshFolder+etaname+'.pvd')
    plot_from_pvd(MeshFolder+thetaname+'.pvd')
    return

if __name__=='__main__':
    filename='Inputs.csv'
    heateqn=0
    main(filename,heateqn)
    exit()


