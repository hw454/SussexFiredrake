#!/usr/bin/env python
# coding: utf-8
# Packages for runningfiredrake
import firedrake
from firedrake import *
import ufl
# Packages for loading data
import pandas as pd
import FileFuncs as ff
import os
import sys

# Packages for managing data
import numpy as np
from itertools import product
import SpatialVariable as sv

#Plotting
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import PlottingFunctions as pf


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


def f_func(r_mesh,eta,theta,consts_dict,test=0):
    eps=consts_dict['eps']
    #Eps=fem.Constant(r_mesh, PETSc.ScalarType(eps))
    #return - eta*theta+eps*eta**2/(1+eta) 
    # Pure Diffusion (Heat Eqn)
    if test:
        return 0 #fem.Constant(r_mesh, PETSc.ScalarType(0))
    else:
        return - eta*theta+(eps*eta**2/(1+eta))
def g_func(r_mesh,eta,theta,consts_dict,test=0):
    eps=consts_dict['eps']
    a0=consts_dict['a0']
    A0=consts_dict['A0']
    b0=consts_dict['b0']
    #Eps  =fem.Constant(r_mesh, PETSc.ScalarType(eps))
    #a0cst=fem.Constant(r_mesh, PETSc.ScalarType(a0))
    #b0cst=fem.Constant(r_mesh, PETSc.ScalarType(b0))
    #A0cst=fem.Constant(r_mesh, PETSc.ScalarType(A0))
    if test==1:
        return 0 #fem.Constant(r_mesh, PETSc.ScalarType(0))
    else:
        return A0*ufl.exp(a0*eta+b0*theta)*(1-theta*(1+((2**(1-eps*eta))/eps)))
def linear_form_func(varkey,var_cls,othervar_cls,testvar,consts_dict,test=0):
    '''The function `linear_form_func` defines the linear form for the variational problem. 
    By defining this outside the main simulation the same main program can be used for 
    different variational problems.
    :param varkey: either 'e' or 't' this gives which variable to take the form form.
    :param nsteps: is an integer giving the number of timesteps featuring in the linear form.
    :param var: `if nsteps=1: Function. 
                `if nsteps>1:
                    var=(var0,var1,...,var_{nsteps-1})
                    type(var_i)= Function`
    :param testvar:  TestFunction
    :param consts: The constants required for the function declaration.
                   consts= [eps,rho,sig,a0,b0,A0,d,dt]
    :rtype: `ufl` expression
    :return: L=linear form
    '''
    rho=consts_dict['rho']
    sig=consts_dict['sig']
    d=consts_dict['d']
    dt=consts_dict['dt']
    othervar=othervar_cls.var
    initvar=var_cls.var
    r_mesh=var_cls.mesh
    print('Heateqn test',test)
    divterm=initvar*testvar
    if varkey=='e':
        L=(dt*rho*f_func(r_mesh,initvar,othervar,consts_dict,test)*testvar
           +divterm)*ufl.dx
    if varkey=='t':
        L=(dt*sig*g_func(r_mesh,initvar,othervar,consts_dict,test)*testvar
           +divterm)*ufl.dx
    return L

def bilinear_form_func(varkey,tmpvar,testvar,consts_dict,r_mesh,test=0):
    ''' bilinear form for variational problem.
    :param varkey: either 'e' or 't' this gives which variable to take the form form.
    :param tmpvar: is a `dolfinx.ufl` TrialFunction :math:`v`
    :param testvar: is a `dolfinx.ufl` Test Function. :math:`\nu`
    :param consts: The constants required for the function declaration.
                   consts= [tol,eps,rho,sig,a0,b0,A0,d,dt]
    :param nsteps: is an integer giving the number of timesteps featuring in the linear form.
    :param mesh:   `dolfinx.fem` mesh
    
    ..math::
        if nstep==1
            `a(v,\nu)=\int d*dt*\nabla u \cdot \nabla \nu \cdot dx+\int v\nu\cdot d x`
        elif nstep==2
            `a(v,\nu)=\int 2*d*dt*\nabla u \cdot \nabla \nu \cdot dx+\int v\nu\cdot d x`
    
    :rtype: `ufl` expression
    :returns: a = bilinear form
    '''
    # bilinear form -trial function must go before test function
    d=consts_dict['d']
    dt=consts_dict['dt']
    #Dt =fem.Constant(r_mesh, PETSc.ScalarType(dt))
    if test==2:
        al=0
    else:
        al=1
    if varkey=='e':
        D=1.0*al#fem.Constant(r_mesh, PETSc.ScalarType(1.0))
    elif varkey=='t':
        D=d*al#D  =fem.Constant(r_mesh, PETSc.ScalarType(d))
    a=D*dt*ufl.dot(ufl.grad(tmpvar),ufl.grad(testvar))*ufl.dx+tmpvar*testvar*ufl.dx
    return a

def convert_vec_to_array(vec,nx,ny):
    arr=np.zeros((nx+1,ny+1))
    for i,j in product(range(nx+1),range(ny+1)):
        k=nx*j+i
        arr[i,j]=abs(vec[k])
    return arr


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
    diff=np.linalg.norm(v-w)
    bot=np.linalg.norm(v)
    return diff/bot

def time_div(v1,v0,dt):
    diff=np.linalg.norm(v1-v0)
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
    etaname,thetaname,tname,domname,etaresname,theresname,etadivname,thedivname=varnames
    # nx,ny are number of elements on x and y axis
    # xm, ym are maximum x and y values
    # x0,y0 are minimum x and y values
    # hx, hy are meshwidth on the x and y axis.
    nx=mesh_par_dict['nx']
    ny=mesh_par_dict['ny']
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
    eps=pdes_par_dict['eps']
    A0=pdes_par_dict['A0']
    d=pdes_par_dict['d']
    dt=pdes_par_dict['dt']
    T=pdes_par_dict['T']
    perm=pdes_par_dict['perm']
    parstr=ff.par_string(ind,dt,T,perm,hx,hy,A0,d,eps)
    MeshFolder=MeshFolder+parstr
    ResultsFolder=ResultsFolder+parstr
    ff.check_folder(path,ResultsFolder)
    ff.check_folder(path,MeshFolder)    
    # Simulation parameters--------------------------------------------
    h = max(hx,hy)      # mesh size
    nt=int(T/dt)
    tvec = np.arange(0,T,dt)

    # Convert back to integer for array initialisation later.
    nx=int(nx)
    ny=int(ny)
    nt=int(nt)

    # Initialisation---------------------------------------------------
    # Regular python variables for outputs------------------------------
    #eta=np.zeros((nx+1,ny+1,nt))
    #the=np.zeros((nx+1,ny+1,nt))

    # FeNiCs variables--------------------------------------------------
    # Initialise the space ---------------------------------------------
    Ns=(nx,ny)
    r_mesh=RectangleMesh(nx,ny,xm-x0,ym-y0)
    V = FunctionSpace(r_mesh, 'CG',1)
    # Function(V) indicates a scalar variable over the domain.
    # TestFunction(V) is a scalar test function over the domain for the weak problem.
    # TrialFunction(V) is the scalar function to be found in the solver.
    eta0_cls=sv.spatial_variable(V,r_mesh,etaname)
    eta0_cls.set_char_key('e')
    the0_cls=sv.spatial_variable(V,r_mesh,thetaname)
    the0_cls.set_char_key('t')

    #Initialise time step 0
    eta0_cls.initialise(StSt,perm,hx,hy,'e')
    the0_cls.initialise(StSt,perm,hx,hy,'t')
    #pf.plot_function(0,eta0_cls,0,ResultsFolder)
    #pf.plot_function(0,the0_cls,0,ResultsFolder)
    
    # Create outfiles
    etaoutname=path+MeshFolder+'/'+etaname
    theoutname=path+MeshFolder+'/'+thetaname
    etaout=File(etaoutname+'.pvd')
    theout=File(theoutname+'.pvd')
    etaout.write(eta0_cls.var,time=0)
    theout.write(the0_cls.var,time=0)
    
    
    # Define one-step variational problem ---------------------------------------
    vars_cls=(eta0_cls,the0_cls)
    Lvec,avec=first_step(vars_cls,pdes_par_dict,test)
    eta1_cls=sv.spatial_variable(V,r_mesh,etaname+'1')
    eta1_cls.set_char_key('e')
    the1_cls=sv.spatial_variable(V,r_mesh,thetaname+'1')
    the1_cls.set_char_key('t')

    Outfiles=(etaout,theout)
    vars_cls, eNorm, tNorm,eDiv,tDiv,tvec=one_step_loop(vars_cls,Lvec,avec,pdes_par_dict,tvec,path,ResultsFolder,Outfiles)
    eta2_cls,the2_cls=vars_cls
    #plotting.plot(eta1)
    # Save the outputs from the simulation
    #eta_xdmf.close()
    #the_xdmf.close()
    ff.save_var(tvec,tname,path,MeshFolder)
    ff.save_var(boundaries,domname,path,MeshFolder)
    ff.save_var(eNorm,etaresname,path,MeshFolder)
    ff.save_var(tNorm,theresname,path,MeshFolder)
    ff.save_var(eDiv,etadivname,path,MeshFolder)
    ff.save_var(tDiv,thedivname,path,MeshFolder)
    print('Results saved in',MeshFolder)
    return eta2_cls,the2_cls,parstr,pdes_par_dict
    
def first_step(vars_cls,consts_dict,test=1):
    '''Define variational problem ---------------------------------------  
    linear form -function must go before test function.
    :param vars_cls: list of all the variables as thier `spatial_variable` class object. :math:`var0_cls,var1_cls=vars_cls`
    :param testvars: tuple for the two test functions. :math:`v0,v1=testvars`
    :param trialvars: tuple of the trial functions. :math:`tmpv0,tmpv1=trialvars`
    :param consts: The constants required for the functions
    :param test: Indictator of test case. 
    - Heat Eqn test test=1
    - Full Model test=0 

    
    Creates linear form using `linear_form_func` and bilinear form using `bilinear_form_func`. 
    Creates the one step form for each variable for these.
    
    Uses these forms to create the solvers which are returned.
    
    - Solvers=(Solver_0,Solver_1)
    
    :returns: Solvers,bvecs
    '''
    var0_cls,var1_cls=vars_cls
    V                =var0_cls.function_space
    r_mesh           =var0_cls.mesh
    tmpv0   = TrialFunction(V)
    tmpv1   = TrialFunction(V)
    v0      = TestFunction(V)
    v1      = TestFunction(V)
    L0=linear_form_func(var0_cls.char_key,var0_cls,var1_cls,v0,consts_dict,test) # Linear form for eta with time step 1
    L1=linear_form_func(var1_cls.char_key,var1_cls,var0_cls,v1,consts_dict,test)
    a0=bilinear_form_func(var0_cls.char_key,tmpv0,v0,consts_dict,r_mesh,test=0)
    a1=bilinear_form_func(var1_cls.char_key,tmpv1,v0,consts_dict,r_mesh,test=0)
    
    Lvec=(L0,L1)
    avec=(a0,a1)
    return Lvec,avec


def one_step_loop(vars_cls,Lvec,avec,consts_dict,tvec,path,MeshFolder="",Outfiles=()):
    ''' Solve the spatial problem for each time step using the one step solver. 
    Solutions are plotted at each time step and the output is the variable with the values for the final time step along with the resiudal vector.
    '''
    var00_cls,var10_cls=vars_cls
    V                  =var00_cls.function_space
    r_mesh             =var00_cls.mesh
    var01_cls=sv.spatial_variable(V,r_mesh,var00_cls.name+'1')
    var01_cls.set_char_key(var00_cls.char_key)
    var11_cls=sv.spatial_variable(V,r_mesh,var10_cls.name+'1')
    var11_cls.set_char_key(var10_cls.char_key)
    #Solver_0,Solver_1=Solvers
    L0,L1            =Lvec
    a0,a1            =avec
    Norm0=np.zeros(len(tvec))
    Norm1=np.zeros(len(tvec))
    Div0=np.zeros(len(tvec))
    Div1=np.zeros(len(tvec))
    Norm_t=1
    tol=consts_dict['tol']
    dt=consts_dict['dt']
    if not len(Outfiles)==2:
        out0name=path+MeshFolder+'/'+var00_cls.name
        Out0=File(out0name+'.pvd')
        out1name=path+MeshFolder+'/'+var10_cls.name
        Out1=File(out1name+'.pvd')
    else:
        Out0,Out1=Outfiles

    # TIME STEPPING COMPUTATIONS------------------------------------------------------

    # Begin Iterative Simulations --------------------------------------
    check_out=False
    # Create empty lists to store results in
    rescount=0
    res_fac=20
    tstep=0
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
        elif abs(res_fac)<10*macheps:
            rescount+=1
            if rescount>10:
                print('Residual has set at 0')
                print('Final time %.3f on step %d'%((t-dt),tstep))
                break
        #print('Solving for time step',t)
        # Update the right hand side reusing the initial vector
        solve(a0==L0,var01_cls.var)
        solve(a1==L1,var11_cls.var)
        
        varlist=(var01_cls.var_vals(),var11_cls.var_vals())
        check_out=check_output(varlist)
        
        # Calculate the residual and time derivative
        Norm0[tstep]=norm_res(var01_cls.var_vals(),var00_cls.var_vals()) # Sqrt of the sum of the sqs
        Norm1[tstep]=norm_res(var11_cls.var_vals(),var10_cls.var_vals()) # Sqrt of the sum of the sqs
        Div0[tstep]=time_div(var01_cls.var_vals(),var00_cls.var_vals(),dt) # Sqrt of the sum of the sqs
        Div1[tstep]=time_div(var11_cls.var_vals(),var10_cls.var_vals(),dt) # Sqrt of the sum of the sqs
        res_fac=max(Norm0[tstep],Norm1[tstep])
        #DEBUGSTARTtime_
        if DEBUG:
            print('solved for tstep',tstep)
        #DEBUGEND
        # Reassign the variables before the next step
        var00_cls.update_var(var01_cls)
        var10_cls.update_var(var11_cls)
        # RESET the solution variables
        var01_cls.reset_var()
        var11_cls.reset_var()
        # Write the estimation, this is done on the 0th variable instead of the first since the varname is then missing the index. 
        Out0.write(var00_cls.var,time=t+dt)
        Out1.write(var10_cls.var,time=t+dt)
    Divresarrs=[Norm0,Norm1,Div0,Div1]
    #for arr in Divresarrs:
    Norm0=np.resize(Norm0,tstep-1)
    Norm1=np.resize(Norm1,tstep-1)
    Div0=np.resize(Div0,tstep-1)
    Div1=np.resize(Div1,tstep-1)    
    tvec=np.arange(0,t,dt)
    vars_cls=(var01_cls,var11_cls)
    return vars_cls, Norm0, Norm1,Div0,Div1,tvec

def Plotting(var_cls,path,PlotFoldername,parstr,consts_dict):
    ''' Plot an animation of a tricontour plot for the variable `var_cls`.
    Save plot at path+PlotFoldername+var_cls.name+'.mp4'
    :param var_cls: `SpatialVariable` object with results in.
    :param path: Location for the output
    :param PlotFoldername: directory for the results.
    :param consts_dict: dictionary of the parameters.'''
    dt=consts_dict['dt']
    T=consts_dict['T']
    nt=int(T/dt)
    r_mesh=var_cls.mesh
    var_list=var_cls.var_hist
    var_init=var_cls.init_var
    ns=nt
    fn_plotter=FunctionPlotter(r_mesh,num_sample_points=ns)
    fig, axes = plt.subplots()
    axes.set_aspect('equal')
    colors = tripcolor(var_init, num_sample_points=ns, axes=axes,cmap='inferno')
    fig.colorbar(colors)
    fig.savefig(path+PlotFoldername+var_cls.name+'00.jpg')
    #interval = 1e3 * dt
    #def animate(q):
    #    colors.set_array(fn_plotter(q))
    #animation = FuncAnimation(fig, animate, frames=var_list, interval=interval)
    #savename=path+PlotFoldername+var_cls.name+'.mp4'
    #try:
    #    animation.save(savename, writer="ffmpeg")
    #except:
    #    print("Failed to write movie! Try installing `ffmpeg`.")
    print(var_cls.var_hist)
    for tstep in range(nt):
        fig, axes = plt.subplots()
        axes.set_aspect('equal')
        colors = tripcolor(var_cls.var_hist[tstep], num_sample_points=ns, axes=axes,cmap='inferno')
        fig.colorbar(colors)
        fig.savefig(path+PlotFoldername+var_cls.name+'%.3d.jpg'%tstep)
    return

def main(path,filename,simon=0,test=0,ind=0):
    '''Runs the simulations and plotting functions.
    Ouputs from the simulation are saved as numpy variables. These are then loaded before plotting.
    :param filename: The name of the file containing the input parameters
    :param simon: 0 or 1, if 1 then run simulations, if 0 then only plotting required. default 0.
    :param test: 0 or 1 if 1 then the simulation will run as heat equation
    :param simstep: 1 or 2 number of steps in spatial solver
    Uses functions :py:func:`Simulations(filename,MeshFolder,ResultsFolder,varnames,test)' and :py:func:`plotting(var1,var2,time,foldername)'
    :returns: nothing
    '''
    ResultsFolder='PlotsFirstOrder/'
    MeshFolder  ='MeshFirstOrder/'
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
    varnames=[etaname,thetaname,tname,domname,etaresname,theresname,etadivname,thedivname]
    if simon: eta_cls,the_cls,parstr,consts_dict=Simulations(filename,path,MeshFolder,ResultsFolder,varnames,test,ind)
    print('Simulations completed')
    varlist=(eta_cls,the_cls)
    #for var in varlist:
    #    Plotting(var,path,ResultsFolder,parstr,consts_dict)
    #MeshFolder=MeshFolder+parstring
    #ResultsFolder=ResultsFolder+parstring
    tvec=ff.load_file(path,MeshFolder+parstr,tname)
    varlist=(etaresname,theresname,etadivname,thedivname)
    for vname in varlist:
        var=ff.load_file(path,MeshFolder+parstr,vname)
        pf.line_plot(var,np.log(tvec+1),vname,'log'+tname+'Plus1',path,ResultsFolder+'/'+parstr)
        del var
    etadiv=ff.load_file(path,MeshFolder+parstr,etadivname)
    thediv=ff.load_file(path,MeshFolder+parstr,thedivname)
    pf.line_plot(etadiv,thediv,etadivname,thedivname,path,ResultsFolder+'/'+parstr)
    return
def get_ind(argv):
    job=0 # default jobHeateqn t
    if len(argv)>1:
        job=int(argv[1])
    return job

if __name__=='__main__':
    ff.change_owner()
    global DEBUG,macheps
    DEBUG=True
    macheps=sys.float_info.epsilon
    path=os.path.abspath(os.getcwd())
    filename='/ParameterSets.csv'
    simon=1
    test=(0,1,2)
    ind=get_ind(sys.argv)
    for te in test:
        main(path,filename,simon,te,ind)
    exit()