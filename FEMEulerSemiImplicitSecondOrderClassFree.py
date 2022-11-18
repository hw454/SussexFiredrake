#!/usr/bin/env python
# coding: utf-8
# Packages for runningfiredrake
import os
import sys
from itertools import product

#Plotting
import matplotlib.pyplot as plt
# Packages for managing data
import numpy as np
# Packages for loading data
import pandas as pd
import ufl
from matplotlib.animation import FuncAnimation

import FileFuncs as ff
import firedrake
import PlottingFunctions as pf
from firedrake import *


def check_steps(meshpars,dt):
    '''Check that :math:'dt<h**2/2'
    :param meshpars: Array foimport mpi4py as MPIr the axis limits and number of steps on each axis [nx,ny,xm,ym,x0,y0]
    :param dt: time step, float
    return True if valid False if not'''
    nx,ny,xm,ym,x0,y0=meshpars
    h=min((xm-x0)/nx,(ym-y0)/ny)
    print(meshpars)
    #assert h<1
    check=bool(dt<h**2/2)
    return check


def f_func(eta,theta,consts_dict,test=0):
    eps=consts_dict['eps']
    #Eps=fem.Constant(r_mesh, PETSc.ScalarType(eps))
    #return - eta*theta+eps*eta**2/(1+eta) 
    # Pure Diffusion (Heat Eqn)
    if test==1:
        print('Heat eqn test')
        return 0 #fem.Constant(r_mesh, PETSc.ScalarType(0))
    else:
        #ec=conditional(eta>0.0,eta,0.0)
        #tc=conditional(theta>0.0,theta,0.0)
        return - eta*theta+(eps*eta**2/(1+eta))
def g_func(eta,theta,consts_dict,test=0):
    eps=consts_dict['eps']
    a0=consts_dict['a0']
    A0=consts_dict['A0']
    b0=consts_dict['b0']
    #Eps  =fem.Constant(r_mesh, PETSc.ScalarType(eps))
    #a0cst=fem.Constant(r_mesh, PETSc.ScalarType(a0))
    #b0cst=fem.Constant(r_mesh, PETSc.ScalarType(b0))
    #A0cst=fem.Constant(r_mesh, PETSc.ScalarType(A0))
    if test==1:
        print('Heateqn test')
        return 0 #fem.Constant(r_mesh, PETSc.ScalarType(0))
    else:
        A1=A0*2/eps
        a1=a0-eps*np.log(2)
        #ec=conditional(eta>0.0,eta,0.0)
        #tc=conditional(theta>0.0,theta,0.0)
        return A0*exp(a0*eta+b0*theta)*(1-theta)-A1*exp(a1*eta+b0*theta)*theta


def check_output(varlist):
    nanlist=list()
    for v in varlist:
        arr=np.array(v)
        arrsum=np.sum(arr)
        nanlist+=arrsum
    check=any([np.isnan(n) for n in nanlist])
    return check

def check_zero(v):
    return np.all((v==0))


def norm_res(v,w):
    diff=np.linalg.norm(v.vector()-w.vector())
    bot=np.linalg.norm(v.vector())+np.linalg.norm(w.vector())
    return diff/bot

def time_div(v1,v0,dt):
    diff=np.linalg.norm(v1.vector()-v0.vector())
    return diff/dt

def Simulations(filename,path,MeshFolder,ResultsFolder,varnames,test=0,ind=0):
    ''' This function runs everything required to compute the simulations results.
    Note that this does not plot the results. The results are saved to numpy files which may be plotted later.
    :param filename: string containing the filename for the csv of input values. Includes path.
    :param MeshFolder: where to save the output files
    :param varnames: array of strings indicating variable names
    *Calls :py:func:'read_inputs(filename)' to load the inputs.
    *Computes mesh
    *Initialises the variables
    *Estimates initial conditions and time step 1.
    *Time stepping is a forwards Euler method.
    *Uses FEM implementation to estimate the variables for each time stepping.
    *Saves numpy arrays for the variables over discrete space and time.
    :returns: nothing computations are saved
    '''
    mesh_par_dict,pdes_par_dict=ff.read_inputs(path,filename,ind)
    etaname,thetaname,tname,domname,etaresname,theresname,etadivname,thedivname,vargsname=varnames
    # nx,ny are number of elements on x and y axis
    # xm, ym are maximum x and y values
    # x0,y0 are minimum x and y values
    # hx, hy are meshwidth on the x and y axis.
    nx=int(mesh_par_dict['nx'])
    ny=int(mesh_par_dict['ny'])
    xm=mesh_par_dict['xm']
    ym=mesh_par_dict['ym']
    x0=mesh_par_dict['x0']
    y0=mesh_par_dict['y0']
    hx=(xm-x0)/nx
    hy=(ym-y0)/ny # mesh size
    boundaries=np.array([x0,y0,xm,ym])
    # Parameters for the PDEs
    global StSt, perm
    StSt=pdes_par_dict['StSt']
    dt=pdes_par_dict['dt']
    T=pdes_par_dict['T']
    perm=pdes_par_dict['perm']
    parstr=ff.par_string(ind,pdes_par_dict)
    MeshFolder=MeshFolder+parstr
    ResultsFolder=ResultsFolder+parstr
    ff.check_folder(path,ResultsFolder)
    ff.check_folder(path,MeshFolder)    
    # Simulation parameters--------------------------------------------
    tvec = np.arange(0,T,dt)

    # Firedrake variables--------------------------------------------------
    # Initialise the space ---------------------------------------------
    r_mesh=RectangleMesh(nx,ny,xm-x0,ym-y0)
    V = FunctionSpace(r_mesh, 'CG',1)
    x, y = SpatialCoordinate(r_mesh)
    # sv.spatialvariable is a class which has a function on V and other information about the variable and the domain.
    eta0 = interpolate(StSt[0]+0.5*perm[0]*(cos(x)+cos(y)), V)
    theta0=interpolate(StSt[1]+0.5*perm[1]*(cos(x)+cos(y)), V)
    v1  =   TestFunction(V)
    v2  =   TestFunction(V)
    # Create outfiles
    etahist=[]
    etahist.append(eta0.copy(deepcopy=True))
    thehist=[]
    thehist.append(theta0.copy(deepcopy=True))
    
    # Define two-step variational problem ---------------------------------------
    eta1=Function(V)
    theta1=Function(V)
    eta2=Function(V)
    theta2=Function(V)
    tmpet  = TrialFunction(V)
    tmpth  = TrialFunction(V)
    det=Constant(pdes_par_dict['d_e'])
    dth=Constant(pdes_par_dict['d_t'])
    dtau=Constant(dt)
    rho=Constant(pdes_par_dict['rho'])
    sig=Constant(pdes_par_dict['sig'])
    ae0=inner(tmpet,v1)*dx+dtau*det*inner(grad(tmpet),grad(v1))*dx
    at0=inner(tmpth,v2)*dx+dtau*dth*inner(grad(tmpth),grad(v2))*dx
    Le0=inner(eta0,v1)*dx+rho*dtau*f_func(eta0,theta0,pdes_par_dict)*v1*dx
    Lt0=inner(theta0,v2)*dx+sig*dtau*g_func(eta0,theta0,pdes_par_dict)*v2*dx
    ae1=3*inner(tmpet,v1)*dx+2*dtau*det*inner(grad(tmpet),grad(v1))*dx
    at1=3*inner(tmpth,v2)*dx+2*dtau*dth*inner(grad(tmpth),grad(v2))*dx
    Le1=4*inner(eta1,v1)*dx-inner(eta0,v1)*dx+4*rho*dtau*f_func(eta1,theta1,pdes_par_dict)*v1*dx-2*rho*dtau*f_func(eta0,theta0,pdes_par_dict)*v1*dx
    Lt1=4*inner(theta1,v2)*dx-inner(theta0,v2)*dx+4*sig*dtau*g_func(eta1,theta1,pdes_par_dict)*v2*dx-2*sig*dtau*g_func(eta0,theta0,pdes_par_dict)*v2*dx
    spars={'snes_monitor': None, 
                'snes_max_it': 100,
                'snes_type': 'newtonls',
                'ksp_type': 'gmres',
                'pc_type': 'lu', 
                'mat_type': 'aij',
                'pc_factor_mat_solver_type': 'mumps'}
    pdes_par_dict['spars']=spars
    vmine=min(eta0.vector())
    vmaxe=max(eta0.vector())
    vmint=min(theta0.vector())
    vmaxt=max(theta0.vector())
    vargs={'eta':{'min':vmine,'max':vmaxe},'theta':{'min':vmint,'max':vmaxt}}
    try: 
        solve(ae0==Le0,eta1,solver_parameters=spars)
        solve(at0==Lt0,theta1,solver_parameters=spars)
        etahist.append(eta1.copy(deepcopy=True))
        thehist.append(theta1.copy(deepcopy=True))
        if DEBUG:
            print('solved for tstep',0)
    except: 
        print('Solving at step 1 failed')
        zeroarr=np.array([0])
        outvars={tname:tvec[0],domname:boundaries,
            etaresname:zeroarr,theresname:zeroarr,
            etadivname:zeroarr,thedivname:zeroarr,
            vargsname:vargs}
        save_outs(outvars,path,MeshFolder)
        return etahist,thehist,parstr,pdes_par_dict,vargs

    Norm_res    =np.zeros((len(tvec),2))
    tDiv        =np.zeros((len(tvec),2))
    Norm_res[0,0]   =norm_res(eta0,eta1) # Sqrt of the sum of the sqs
    Norm_res[0,1]   =norm_res(theta0,theta1) # Sqrt of the sum of the sqs
    tDiv[0,0]   =time_div(eta0,eta1,dt) # Sqrt of the sum of the sqs
    tDiv[0,1]   =time_div(theta0,theta1,dt) # Sqrt of the sum of the sqs
    outvars={tname:tvec,domname:boundaries,
            etaresname:Norm_res[:,0],theresname:Norm_res[:,1],
            etadivname:tDiv[:,0],thedivname:tDiv[:,1],
            vargsname:vargs}
    # TIME STEPPING COMPUTATIONS------------------------------------------------------

    # Begin Iterative Simulations --------------------------------------
    check_out=False
    # Create empty lists to store results in
    rescount=0
    res_fac=max(Norm_res[0])
    if min(eta1.vector())<vargs['eta']['min']:
        vargs['eta']['min']=min(eta1.vector())
    if max(eta1.vector())>vargs['eta']['max']:
        vargs['eta']['max']=max(eta1.vector())
    if min(theta1.vector())<vargs['theta']['min']:
        vargs['theta']['min']=min(theta1.vector())
    if max(theta1.vector())>vargs['theta']['max']:
        vargs['theta']['max']=max(theta1.vector())
    for tstep,t in enumerate(tvec):
        if check_out or np.isnan(res_fac):
             # Nans in output solution blown up
            print('Nans in output ')
            print('Final time %.3f on step %d'%((t-dt),tstep))
            break
        elif tstep+2>=len(tvec): 
            print('Time steps exceeded maximum')
            print('Final time %.3f on step %d'%((t-dt),tstep))
            break
        elif res_fac>1E+200 :
            print('Residual blown up')
            print('Final time %.3f on step %d'%((t-dt),tstep))
            break
        elif abs(res_fac)<1e+8*macheps:
            rescount+=1
            if rescount>10:
                print('Residual has set at 0')
                print('Final time %.3f on step %d'%((t-dt),tstep))
                break
        #print('Solving for time step',t)
        # Update the right hand side reusing the initial vector
        try: 
            solve(ae1==Le1,eta2,solver_parameters=spars)
            solve(at1==Lt1,theta2,solver_parameters=spars)
        except:
            print('Solve failed at time step %.3f'%t)
            break
        varlist             =(eta2.vector(),theta2.vector())
        check_out           =check_output(varlist)
        
        # Calculate the residual and time derivative
        Norm_res[tstep+1,0] =norm_res(eta2.vector(),eta1.vector()) # Sqrt of the sum of the sqs
        Norm_res[tstep+1,1] =norm_res(theta2.vector(),theta1.vector()) # Sqrt of the sum of the sqs
        tDiv[tstep+1,0]     =time_div(eta2.vector(),eta1.vector(),dt) # Sqrt of the sum of the sqs
        tDiv[tstep+1,1]     =time_div(theta2.vector(),theta1.vector(),dt) # Sqrt of the sum of the sqs
        res_fac=max(Norm_res[tstep+1])
        #DEBUGSTARTtime_
        if DEBUG:
            print('solved for tstep',tstep+1)
        #DEBUGEND
        # Reassign the variables before the next step
        etahist.append(eta2.copy(deepcopy=True))
        thehist.append(theta2.copy(deepcopy=True))
        eta0.assign(eta1.copy(deepcopy=True))
        eta1.assign(eta2.copy(deepcopy=True))
        theta0.assign(theta1.copy(deepcopy=True))
        theta1.assign(theta2.copy(deepcopy=True))
        if min(eta1.vector())<vargs['eta']['min']:
            vargs['eta']['min']=min(eta1.vector())
        if max(eta1.vector())>vargs['eta']['max']:
            vargs['eta']['max']=max(eta1.vector())
        if min(theta1.vector())<vargs['theta']['min']:
            vargs['theta']['min']=min(theta1.vector())
        if max(theta1.vector())>vargs['theta']['max']:
            vargs['theta']['max']=max(theta1.vector())
        # RESET the solution variables
        eta2=Function(V)
        theta2=Function(V)
        #Plot_tstep(var00_cls,tstep,path,PlotFolder,consts_dict)
    Norm_res=np.resize(Norm_res,(tstep-1,2))
    tDiv=np.resize(tDiv,(tstep-1,2))  
    tvec=np.resize(tvec,(tstep-2,1))
    #plotting.plot(eta1)
    # Save the outputs from the simulation
    outvars[etaresname]=Norm_res[:,0]
    outvars[theresname]=Norm_res[:,1]
    outvars[tname]=tvec
    outvars[etadivname]=tDiv[:,0]
    outvars[thedivname]=tDiv[:,1],
    outvars[vargsname]=vargs
    save_outs(outvars,path,MeshFolder)
    return etahist,thehist,parstr,pdes_par_dict,vargs
def save_outs(vardict,path,savedir):
    for k in vardict.keys():
        ff.save_var(vardict[k],k,path,savedir)
    print('Results saved in',path+savedir)
    return 0
def Plot_tstep(var,tstep,dt,varname,savedir,vargs):
    v0=vargs['min']
    v1=vargs['max']
    fig, axes = plt.subplots()
    axes.set_aspect('equal')
    colors = tripcolor(var, num_sample_points=1, axes=axes,cmap='inferno',vmin=v0,vmax=v1)
    plt.title('Time = %.3f'%(tstep*dt))
    fig.colorbar(colors)
    fig.savefig(savedir+'/'+varname+ff.numfix(tstep)+'.jpg')
    fig.clear()
    plt.close()
    return
def Plotting(var_list,varname,path,PlotFoldername,parstr,consts_dict,vargs):
    ''' Plot an animation of a tricontour plot for the variable `var_cls`.
    Save plot at path+PlotFoldername+var_cls.name+'.mp4'
    :param var_cls: `SpatialVariable` object with results in.
    :param path: Location for the output
    :param PlotFoldername: directory for the results.
    :param consts_dict: dictionary of the parameters.'''
    dt=consts_dict['dt']
    tstep=0.0
    savedir=path+PlotFoldername+parstr
    for var in var_list:
        Plot_tstep(var,tstep,dt,varname,savedir,vargs)
        tstep+=1.0
    print('final t',tstep*dt)
    return

def main(path,filename,simon=0,test=0,ind=0,simstep=2):
    '''Runs the simulations and plotting functions.
    Ouputs from the simulation are saved as numpy variables. These are then loaded before plotting.
    :param filename: The name of the file containing the input parameters
    :param simon: 0 or 1, if 1 then run simulations, if 0 then only plotting required. default 0.
    :param test: 0 or 1 if 1 then the simulation will run as heat equation
    :param simstep: 1 or 2 number of steps in spatial solver
    Uses functions :py:func:`Simulations(filename,MeshFolder,ResultsFolder,varnames,test)' and :py:func:`plotting(var1,var2,time,foldername)'
    :returns: nothing
    '''
    ResultsFolder='Plots'+ff.simstr(simstep)+'/'
    MeshFolder  ='Mesh'+ff.simstr(simstep)+'/'
    dirlist=(ResultsFolder,MeshFolder)
    dlistout=list()
    for d in dirlist:
        dlistout.append(ff.teststr(test)+d)
    ResultsFolder,MeshFolder=dlistout
    etaname     ='eta'
    thetaname   ='theta'
    tname       ='tvec'
    domname     ='boundaries'
    etaresname  ='eta_res'
    theresname  ='the_res'
    etadivname  ='eta_div'
    thedivname  ='the_div'
    vargsname   ='vargs'
    varnames=[etaname,thetaname,tname,domname,etaresname,theresname,etadivname,thedivname,vargsname]
    if simon: etahist,thehist,parstr,consts_dict,vargs=Simulations(filename,path,MeshFolder,ResultsFolder,varnames,test,ind)
    print('Simulations completed')
    tvec=ff.load_file(path,MeshFolder+parstr,tname)
    nt=len(tvec)
    varlist=(etaresname,theresname,etadivname,thedivname)
    for vname in varlist:
        var=ff.load_file(path,MeshFolder+parstr,vname)
        pf.line_plot(var,np.log(tvec+1),vname,'log'+tname+'Plus1',path,ResultsFolder+'/'+parstr)
        del var
    etadiv=ff.load_file(path,MeshFolder+parstr,etadivname)
    thediv=ff.load_file(path,MeshFolder+parstr,thedivname)
    pf.line_plot(etadiv,thediv,etadivname,thedivname,path,ResultsFolder+'/'+parstr)
    Plotting(etahist,etaname,path,ResultsFolder,parstr,consts_dict,vargs['eta'])
    Plotting(thehist,thetaname,path,ResultsFolder,parstr,consts_dict,vargs['theta'])
    #MeshFolder=MeshFolder+parstring
    #ResultsFolder=ResultsFolder+parstring
    return
def get_ind(argv):
    job=0 # default jobHeateqn t
    if len(argv)>1:
        job=int(argv[1])
    return job

if __name__=='__main__':
    ff.change_owner()
    global macheps
    DEBUG=True
    macheps=sys.float_info.epsilon
    path=os.path.abspath(os.getcwd())
    filename='/ParameterSets.csv'
    simon=1
    test=(0,)#,1,2)
    ind=get_ind(sys.argv)
    for te in test:
        main(path,filename,simon,te,ind)
    exit()