#!/usr/bin/env python
# coding: utf-8
# Packages for runningfiredrake
import firedrake
from firedrake import *
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


def f_func(eta,theta,consts_dict,test=0):
    eps=Constant(consts_dict['eps'])
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
    eps=Constant(consts_dict['eps'])
    a0=Constant(consts_dict['a0'])
    A0=Constant(consts_dict['A0'])
    b0=Constant(consts_dict['b0'])
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
def linear_form_func_init(varkey,var_cls,othervar_cls,testvar,consts_dict,test=0):
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
    rho=Constant(consts_dict['rho'])
    sig=Constant(consts_dict['sig'])
    dt=Constant(consts_dict['dt'])
    othervar0=othervar_cls.var
    initvar0=var_cls.var
    #c=conditional(initvar0>0,initvar0,0)
    divterm=initvar0*testvar
    if varkey=='e':
        L=(dt*rho*f_func(initvar0,othervar0,consts_dict,test)*testvar
           +divterm)*dx
    if varkey=='t':
        L=(dt*sig*g_func(initvar0,othervar0,consts_dict,test)*testvar
           +divterm)*dx
    return L

def bilinear_form_func_init(varkey,tmpvar,testvar,consts_dict,test=0):
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
    dt=Constant(consts_dict['dt'])
    #Dt =fem.Constant(r_mesh, PETSc.ScalarType(dt))
    if test==2:
        print('Source only test')
        al=0
    else:
        al=1
    if varkey=='e':
        d=consts_dict['d_e']
    elif varkey=='t':
        d=consts_dict['d_t']
    D=Constant(d*al)
    a=D*dt*inner(grad(tmpvar),grad(testvar))*dx+tmpvar*testvar*dx
    return a
def linear_form_func(varkey,var_clslist,othervar_clslist,testvar,consts_dict,test=0):
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
    rho=Constant(consts_dict['rho'])
    sig=Constant(consts_dict['sig'])
    dt=Constant(consts_dict['dt'])
    othervar0=othervar_clslist[0].var
    othervar1=othervar_clslist[1].var
    initvar0=var_clslist[0].var
    initvar1=var_clslist[1].var
    #c1=conditional(initvar1>0.0,initvar1,0.0)
    #c0=conditional(initvar0>0,initvar0,0.0)
    divterm=4*initvar1*testvar -initvar0*testvar
    if varkey=='e':
        L=(dt*rho*2*(2*f_func(initvar1,othervar1,consts_dict,test)
        -f_func(initvar0,othervar0,consts_dict,test))*testvar
           +divterm)*dx
    if varkey=='t':
        L=(dt*sig*2*(2*g_func(initvar1,othervar1,consts_dict,test)
        -g_func(initvar0,othervar0,consts_dict,test))*testvar
           +divterm)*dx
    return L

def bilinear_form_func(varkey,tmpvar,testvar,consts_dict,test=0):
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
    dt=Constant(consts_dict['dt'])
    if test==2:
        print('Source only test')
        al=0
    else:
        al=1
    if varkey=='e':
        d=consts_dict['d_e']
    elif varkey=='t':
        d=consts_dict['d_t']
    D=Constant(d*al)
    a=2*D*dt*inner(grad(tmpvar),grad(testvar))*dx+3*tmpvar*testvar*dx
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
    # sv.spatialvariable is a class which has a function on V and other information about the variable and the domain.
    eta0_cls=sv.spatial_variable(V,r_mesh,etaname)
    eta0_cls.set_char_key('e')
    the0_cls=sv.spatial_variable(V,r_mesh,thetaname)
    the0_cls.set_char_key('t')

    #Initialise time step 0
    eta0_cls.initialise(StSt,perm,hx,hy,'e')
    the0_cls.initialise(StSt,perm,hx,hy,'t')
    # Create outfiles
    etaoutname=path+MeshFolder+'/'+etaname
    theoutname=path+MeshFolder+'/'+thetaname
    etaout=File(etaoutname+'.pvd')
    theout=File(theoutname+'.pvd')
    etaout.write(eta0_cls.var,time=0)
    theout.write(the0_cls.var,time=0)
    
    
    # Define two-step variational problem ---------------------------------------
    vars_cls=(eta0_cls,the0_cls)
    Lvec,avec=first_step(vars_cls,pdes_par_dict,test)
    eta1_cls=sv.spatial_variable(V,r_mesh,etaname+'1')
    eta1_cls.set_char_key('e')
    the1_cls=sv.spatial_variable(V,r_mesh,thetaname+'1')
    the1_cls.set_char_key('t')
    vars_cls=(eta0_cls,the0_cls,eta1_cls,the1_cls)
    a0,a1=avec
    L0,L1=Lvec
    spars={'snes_monitor': None, 
                'snes_max_it': 100,
                'snes_type': 'newtonls',
                'ksp_type': 'gmres',
                'pc_type': 'lu', 
                'mat_type': 'aij',
                'pc_factor_mat_solver_type': 'mumps'}
    pdes_par_dict['spars']=spars
    try: solve(a0==L0,eta1_cls.var,solver_parameters=spars)
    except: 
        print('Solving at step 1 failed for eta')
        return eta0_cls,the0_cls,parstr,pdes_par_dict,vargs
    try: solve(a1==L1,the1_cls.var,solver_parameters=spars)
    except: 
        print('Solving at step 1 failed for theta')
        return eta0_cls,the0_cls,parstr,pdes_par_dict,vargs

    Outfiles=(etaout,theout)
    vars_cls, Norm_res,tDiv,tvec,vargs=two_step_loop(vars_cls,pdes_par_dict,tvec,path,MeshFolder,Outfiles,test)
    eta2_cls,the2_cls=vars_cls
    #plotting.plot(eta1)
    # Save the outputs from the simulation
    ff.save_var(tvec,tname,path,MeshFolder)
    ff.save_var(boundaries,domname,path,MeshFolder)
    ff.save_var(Norm_res[:,0],etaresname,path,MeshFolder)
    ff.save_var(Norm_res[:,1],theresname,path,MeshFolder)
    ff.save_var(tDiv[:,0],etadivname,path,MeshFolder)
    ff.save_var(tDiv[:,1],thedivname,path,MeshFolder)
    ff.save_var(vargs,vargsname,path,MeshFolder)
    print('Results saved in',MeshFolder)
    return eta2_cls,the2_cls,parstr,pdes_par_dict,vargs
    
def first_step(vars_cls,consts_dict,test=0):
    '''Define variational problem ---------------------------------------  
    linear form -function must go before test function.
    :param vars_cls: list of all the variables as thier `spatial_variable` class object. :math:`var0_cls,var1_cls=vars_cls`
    :param consts_dict: The constants required for the functions
    :param test: Indictator of test case. Default is 0
    - Heat Eqn test test=1
    - Full Model test=0 

    
    Creates linear form using `linear_form_func` and bilinear form using `bilinear_form_func`. 
    Creates the one step form for each variable for these.

    - Linear forms for variables var0 and var1 are L0 and L1 respectively.
    - bilinear forms for variables var0 and var1 are a0 and a1 respectively.
    
    :returns: (L0,L1),(a0,a1)
    '''
    var0_cls,var1_cls=vars_cls
    V                =var0_cls.function_space
    tmpv0   = TrialFunction(V)
    tmpv1   = TrialFunction(V)
    v0      = TestFunction(V)
    v1      = TestFunction(V)
    L0=linear_form_func_init(var0_cls.char_key,var0_cls,var1_cls,v0,consts_dict,test) # Linear form for eta with time step 1
    L1=linear_form_func_init(var1_cls.char_key,var1_cls,var0_cls,v1,consts_dict,test)
    a0=bilinear_form_func_init(var0_cls.char_key,tmpv0,v0,consts_dict,test)
    a1=bilinear_form_func_init(var1_cls.char_key,tmpv1,v1,consts_dict,test)
    
    Lvec=(L0,L1)
    avec=(a0,a1)
    return Lvec,avec
def update_step(vars_cls,consts_dict,test=0):
    '''Define variational problem for second order semi-implict scheme given results for time step m and m-1---------------------------------------  
    linear form -function must go before test function.
    :param vars_cls: list of all the variables as their `spatial_variable` class object. :math:`var00_cls,var10_cls,var01_cls,var11_cls=vars_cls`
    :param consts_dict: The constants required for the functions
    :param test: Indictator of test case. Default=0
    - Heat Eqn test test=1
    - Full Model test=0 

    
    Creates linear form using `linear_form_func` and bilinear form using `bilinear_form_func`. 
    Creates the one step form for each variable for these.
    
    Uses these forms to create the solvers which are returned.
    
    - Linear forms for variables var0 and var1 are L0 and L1 respectively.
    - bilinear forms for variables var0 and var1 are a0 and a1 respectively.
    
    :returns: (L0,L1),(a0,a1)
    '''
    var00_cls,var10_cls,var01_cls,var11_cls=vars_cls
    V                =var00_cls.function_space
    tmpv0   = TrialFunction(V)
    tmpv1   = TrialFunction(V)
    v0      = TestFunction(V)
    v1      = TestFunction(V)
    L0=linear_form_func(var00_cls.char_key,(var00_cls,var01_cls),
    (var10_cls,var11_cls),v0,consts_dict,test) # Linear form for eta with time step 1
    L1=linear_form_func(var10_cls.char_key,(var10_cls,var11_cls),
    (var00_cls,var01_cls),v1,consts_dict,test)
    a0=bilinear_form_func(var00_cls.char_key,tmpv0,v0,consts_dict,test)
    a1=bilinear_form_func(var10_cls.char_key,tmpv1,v1,consts_dict,test)
    Lvec=(L0,L1)
    avec=(a0,a1)
    return Lvec,avec


def two_step_loop(vars_cls,consts_dict,tvec,path,MeshFolder="",Outfiles=(),test=0):
    ''' Solve the spatial problem for each time step using the one step solver. 
    Solutions are plotted at each time step and the output is the variable with the values for the final time step along with the resiudal vector.
    '''
    var00_cls,var10_cls,var01_cls,var11_cls=vars_cls
    V                  =var00_cls.function_space
    r_mesh             =var00_cls.mesh
    tol         =consts_dict['tol']
    dtau          =consts_dict['dt']
    spars       =consts_dict['spars']
    var02_cls   =sv.spatial_variable(V,r_mesh,var00_cls.name+'1')
    var02_cls.set_char_key(var00_cls.char_key)
    var12_cls   =sv.spatial_variable(V,r_mesh,var10_cls.name+'1')
    var12_cls.set_char_key(var10_cls.char_key)
    Norm_res    =np.zeros((len(tvec),2))
    tDiv        =np.zeros((len(tvec),2))
    Norm_res[0,0]   =norm_res(var00_cls.var_vals(),var01_cls.var_vals()) # Sqrt of the sum of the sqs
    Norm_res[0,1]   =norm_res(var10_cls.var_vals(),var11_cls.var_vals()) # Sqrt of the sum of the sqs
    tDiv[0,0]   =time_div(var00_cls.var_vals(),var01_cls.var_vals(),dtau) # Sqrt of the sum of the sqs
    tDiv[0,1]   =time_div(var10_cls.var_vals(),var11_cls.var_vals(),dtau) # Sqrt of the sum of the sqs
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
    res_fac=max(Norm_res[0])
    vmin0=min(min(var00_cls.var_vals()),min(var01_cls.var_vals()))
    vmax0=max(max(var00_cls.var_vals()),max(var01_cls.var_vals()))
    vmin1=min(min(var10_cls.var_vals()),min(var11_cls.var_vals()))
    vmax1=max(max(var10_cls.var_vals()),max(var10_cls.var_vals()))
    vargs=(vmin0,vmax0,vmin1,vmax1)
    for tstep,t in enumerate(tvec):
        if check_out or np.isnan(res_fac):
             # Nans in output solution blown up
            print('Nans in output ')
            print('Final time %.3f on step %d'%((t-dtau),tstep))
            break
        elif tstep+2>=len(tvec): 
            print('Time steps exceeded maximum')
            print('Final time %.3f on step %d'%((t-dtau),tstep))
            break
        elif res_fac>1E+200 :
            print('Residual blown up')
            print('Final time %.3f on step %d'%((t-dtau),tstep))
            break
        elif abs(res_fac)<10*macheps:
            rescount+=1
            if rescount>10:
                print('Residual has set at 0')
                print('Final time %.3f on step %d'%((t-dtau),tstep))
                break
        #print('Solving for time step',t)
        # Update the right hand side reusing the initial vector
        Lvec,avec=update_step((var00_cls,var10_cls,var01_cls,var11_cls),
        consts_dict,test)
        a0,a1=avec
        L0,L1=Lvec
        try: solve(a0==L0,var02_cls.var,solver_parameters=spars)
        except:
            print('Solve failed for ',var02_cls.name,' at time step %.3f'%t)
            return (var01_cls,var11_cls), Norm_res[0:tstep,:], tDiv[0:tstep,:],tvec[0:tstep],vargs
        try: solve(a1==L1,var12_cls.var,solver_parameters=spars)
        except:
            print('Solve failed for ',var12_cls.name,' at time step %.3f'%t)
            return (var01_cls,var11_cls), Norm_res[0:tstep,:], tDiv[0:tstep,:],tvec[0:tstep],vargs
        
        varlist             =(var02_cls.var_vals(),var12_cls.var_vals())
        check_out           =check_output(varlist)
        
        # Calculate the residual and time derivative
        Norm_res[tstep+1,0] =norm_res(var02_cls.var_vals(),var01_cls.var_vals()) # Sqrt of the sum of the sqs
        Norm_res[tstep+1,1] =norm_res(var12_cls.var_vals(),var11_cls.var_vals()) # Sqrt of the sum of the sqs
        tDiv[tstep+1,0]     =time_div(var02_cls.var_vals(),var01_cls.var_vals(),dtau) # Sqrt of the sum of the sqs
        tDiv[tstep+1,1]     =time_div(var12_cls.var_vals(),var11_cls.var_vals(),dtau) # Sqrt of the sum of the sqs
        res_fac=max(Norm_res[tstep+1])
        #DEBUGSTARTtime_
        if DEBUG:
            print('solved for tstep',tstep)
        #DEBUGEND
        # Reassign the variables before the next step
        var00_cls.update_var(var01_cls)
        var10_cls.update_var(var11_cls)
        var01_cls.update_var(var02_cls)
        var11_cls.update_var(var12_cls)
        if min(var01_cls.var_vals())<vmin0:
            vmin0=min(var01_cls.var_vals())
        if max(var01_cls.var_vals())>vmax0:
            vmax0=max(var01_cls.var_vals())
        if min(var11_cls.var_vals())<vmin1:
            vmin1=min(var11_cls.var_vals())
        if max(var11_cls.var_vals())>vmax1:
            vmax1=max(var11_cls.var_vals())
        # RESET the solution variables
        var02_cls.reset_var()
        var12_cls.reset_var()
        # Write the estimation, this is done on the 0th variable instead of the first since the varname is then missing the index. 
        Out0.write(var00_cls.var,time=t+dtau)
        Out1.write(var10_cls.var,time=t+dtau)
        #Plot_tstep(var00_cls,tstep,path,PlotFolder,consts_dict)
    Norm_res=np.resize(Norm_res,(tstep-1,2))
    tDiv=np.resize(tDiv,(tstep-1,2))  
    tvec=np.arange(0,t,dtau)
    vars_cls=(var00_cls,var10_cls)
    vargs=(vmin0,vmax0,vmin1,vmax1)
    return vars_cls, Norm_res,tDiv,tvec,vargs

def Plot_tstep(var_cls,tstep,path,PlotFoldername,consts_dict):
    ''' Plot an animation of a tricontour plot for the variable `var_cls`.
    Save plot at path+PlotFoldername+var_cls.name+'.mp4'
    :param var_cls: `SpatialVariable` object with results in.
    :param path: Location for the output
    :param PlotFoldername: directory for the results.
    :param consts_dict: dictionary of the parameters.'''
    r_mesh=var_cls.mesh
    var_list=var_cls.var_hist
    var_init=var_cls.init_var
    fn_plotter=FunctionPlotter(r_mesh,num_sample_points=1)
    fig, axes = plt.subplots()
    axes.set_aspect('equal')
    colors = tripcolor(var_cls.var, axes=axes,cmap='inferno')
    fig.colorbar(colors)
    fig.savefig(path+PlotFoldername+var_cls.name+'%3d.jpg'%tstep)
    return
def Plotting(var_cls,nt,path,PlotFoldername,parstr,consts_dict,vargs):
    ''' Plot an animation of a tricontour plot for the variable `var_cls`.
    Save plot at path+PlotFoldername+var_cls.name+'.mp4'
    :param var_cls: `SpatialVariable` object with results in.
    :param path: Location for the output
    :param PlotFoldername: directory for the results.
    :param consts_dict: dictionary of the parameters.'''
    var_list=var_cls.var_hist
    var_init=var_cls.init_var
    v0,v1=vargs
    fig, axes = plt.subplots()
    axes.set_aspect('equal')
    colors = tripcolor(var_init, num_sample_points=1, axes=axes,cmap='inferno')
    #,vmin=0,vmax=5)
    fig.colorbar(colors)
    fig.savefig(path+PlotFoldername+parstr+'/'+var_cls.name+ff.numfix(0)+'.jpg')
    fig.clear()
    plt.close()
    print('length of t',nt)
    for tstep in range(nt-1):
        fig, axes = plt.subplots()
        axes.set_aspect('equal')
        colors = tripcolor(var_list[tstep], num_sample_points=1, 
        axes=axes,cmap='inferno')#,vmin=0,vmax=5)
        fig.colorbar(colors)
        fig.savefig(path+PlotFoldername+parstr+'/'+var_cls.name+ff.numfix(tstep)+'.jpg')
        fig.clear()
        plt.close()
    return
        #interval = 1e3 * dt
    #def animate(q):
    #    colors.set_array(fn_plotter(q))
    #animation = FuncAnimation(fig, animate, frames=var_list, interval=interval)
    #savename=path+PlotFoldername+var_cls.name+'.mp4'
    #try:
    #    animation.save(savename, writer="ffmpeg")
    #except:
    #    print("Failed to write movie! Try installing `ffmpeg`.")
    #print(var_cls.var_hist)

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
    if simon: eta_cls,the_cls,parstr,consts_dict,vargs=Simulations(filename,path,MeshFolder,ResultsFolder,varnames,test,ind)
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
    varr=((vargs[0],vargs[1]),(vargs[2],vargs[3]))
    var_clslist=(eta_cls,the_cls)
    for i,var in enumerate(var_clslist):
        Plotting(var,nt,path,ResultsFolder,parstr,consts_dict,varr[i])
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