#!/usr/bin/env python
# coding: utf-8
import firedrake
from firedrake import *
# Packages for loading data
import pandas as pd
import FileFuncs as ff

# Packages for managing data
import numpy as np
from itertools import product
import SpatialVariable as sv


# Packages for runningfiredrake
#import petsc4py as PETSc
import ufl
import mpi4py as MPI


def check_steps(meshpars,dt):
    '''Check that :math:'dt<h**2/2'
    :param meshpars: Array for the axis limits and number of steps on each axis [nx,ny,xm,ym,x0,y0]
    :param dt: time step, float
    return True if valid False if not'''
    nx,ny,xm,ym,x0,y0=meshpars
    h=min((xm-x0)/nx,(ym-y0)/ny)
    print(meshpars)
    #assert h<1
    check=bool(dt<h**2/2)
    return check


def f_func(r_mesh,eta,theta,consts,test=0):
    _,eps,_,_,_,_,_,_,_=consts
    #Eps=fem.Constant(r_mesh, PETSc.ScalarType(eps))
    #return - eta*theta+eps*eta**2/(1+eta) 
    # Pure Diffusion (Heat Eqn)
    if test:
        return 0
    else:
        return - eta*theta+eps*eta**2/(1+eta) 
def g_func(r_mesh,eta,theta,consts,test=0):
    _,eps,_,_,a0,b0,A0,_,_=consts
    #Eps  =fem.Constant(r_mesh, PETSc.ScalarType(eps))
    #a0cst=fem.Constant(r_mesh, PETSc.ScalarType(a0))
    #b0cst=fem.Constant(r_mesh, PETSc.ScalarType(b0))
    #A0cst=fem.Constant(r_mesh, PETSc.ScalarType(A0))
    if test:
        return 0 #fem.Constant(r_mesh, PETSc.ScalarType(0))
    else:
        return A0*ufl.exp(a0*eta+b0*theta)*(1-eta*(1+(1/eps)*((1-eps*eta)**2)))

def linear_form_func(varkey,nsteps,var_cls,othervar_cls,testvar,consts,test=0):
    '''The function `linear_form_func` defines the linear form for the variational problem. 
    By defining this outside the main simulation the same main program can be used for 
    different variational problems.
    :param varkey: either 'e' or 't' this gives which variable to take the form form.
    :param nsteps: is an integer giving the number of timesteps featuring in the linear form.
    :param var: `if nsteps=1:dolfinx.fem` Function. 
                `if nsteps>1:
                    var=(var0,var1,...,var_{nsteps-1})
                    type(var_i)=`dolfinx.fem` Function`
    :param testvar: `dolfinx.ufl` Test Function
    :param consts: The constants required for the function declaration.
                   consts= [eps,rho,sig,a0,b0,A0,d,dt]
    :rtype: `ufl` expression
    :return: L=linear form
    '''
    _,_,rho,sig,_,_,_,d,dt=consts
    othervar=othervar_cls.var
    if nsteps==1:
        initvar=var_cls.var
        r_mesh=var_cls.mesh
    elif nsteps==2:
        var0_cls,var1_cls=var_cls
        var0=var0_cls.var
        var1=var1_cls.var
        r_mesh=var0_cls.mesh
    #Rho=fem.Constant(r_mesh, PETSc.ScalarType(rho))
    #Sig=fem.Constant(r_mesh, PETSc.ScalarType(sig))
    #Dt =fem.Constant(r_mesh, PETSc.ScalarType(dt))
    #D  =fem.Constant(r_mesh, PETSc.ScalarType(d))
    print('Heateqn test',test)
    if nsteps==1:
        divterm=initvar*testvar
        if varkey=='e':
            L=(dt*rho*f_func(r_mesh,initvar,othervar,consts,test)*testvar
             +divterm)*ufl.dx
        if varkey=='t':
            L=(dt*sig*g_func(r_mesh,initvar,othervar,consts,test)*testvar
             +divterm)*ufl.dx
    elif nsteps==2:
        diffterm=2*dt*ufl.dot(ufl.grad(var0),ufl.grad(testvar))
        divterm=var1*testvar
        if varkey=='e':
            L=(diffterm
             +2*dt*rho*f_func(r_mesh,var0,othervar,consts,test)*testvar
             +divterm)*ufl.dx
        if varkey=='t':
            L=(d*diffterm
             +2*dt*sig*g_func(r_mesh,var0,othervar,consts,test)*testvar
             +divterm)*ufl.dx
    return L
            


def bilinear_form_func(varkey,tmpvar,testvar,consts,nsteps,r_mesh,test=0):
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
    _,_,_,_,_,_,_,d,dt=consts
    #Dt =fem.Constant(r_mesh, PETSc.ScalarType(dt))
    if varkey=='e':
        D=1.0#fem.Constant(r_mesh, PETSc.ScalarType(1.0))
    elif varkey=='t':
        D=d#D  =fem.Constant(r_mesh, PETSc.ScalarType(d))
    if nsteps==1:
        a=D*dt*ufl.dot(ufl.grad(tmpvar),ufl.grad(testvar))*ufl.dx+tmpvar*testvar*ufl.dx
    elif nsteps==2:
        a=2*D*dt*ufl.dot(ufl.grad(tmpvar),ufl.grad(testvar))*ufl.dx+tmpvar*testvar*ufl.dx
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

def Simulations(filename,path,MeshFolder,ResultsFolder,varnames,test=0,simstep=1,ind=0):
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
    pars,pdes_pars=ff.read_inputs(path,filename,ind)
    etaname,thetaname,tname,domname,etaresname,theresname=varnames
    # nx,ny are number of elements on x and y axis
    # xm, ym are maximum x and y values
    # x0,y0 are minimum x and y values
    # hx, hy are meshwidth on the x and y axis.
    nx,ny,xm,ym,x0,y0=pars
    hx=(xm-x0)/nx
    hy=(ym-y0)/ny # mesh size
    boundaries=np.array([x0,y0,xm,ym])
    # Parameters for the PDEs
    global StSt, perm
    StSt,_,_,_,_,_,_,_,_,dt,T,perm=pdes_pars #StSt,tol,eps,rho,sig,a0,b0,A0,d,dt,T,perm
    parstr=ff.par_string(ind,dt,T,perm,hx,hy)
    MeshFolder=MeshFolder+parstr
    ResultsFolder=ResultsFolder+parstr
    ff.check_folder(path,ResultsFolder)
    ff.check_folder(path,MeshFolder)
    consts=pdes_pars[1:10] # These parameters are required for g_func, f_func and the variational form

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
    eta0_cls.initialise('o')
    the0_cls.initialise('o')
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
    Lvec,avec=first_step(vars_cls,consts,test)
    Le,Lt=Lvec
    ae,at=avec
    eta1_cls=sv.spatial_variable(V,r_mesh,etaname+'1')
    eta1_cls.set_char_key('e')
    the1_cls=sv.spatial_variable(V,r_mesh,thetaname+'1')
    the1_cls.set_char_key('t')
   #Solver_e=NonlinearVariationalProblem(Le-ae,eta1_cls.var)
   # Solver_t=NonlinearVariationalProblem(Lt-at,the1_cls.var)
    Ae=assemble(ae)
    At=assemble(at)
    be=assemble(Le)
    bt=assemble(Lt)
    #Solvers=(Solver_e,Solver_t)
    Outfiles=(etaout,theout)
    if simstep==2:
        #Solver_e,Solver_t=Solvers

        # Run the one-step solver once to input into the two-step
        # Each solve gives solutions to the trial variable in the solver. 
        # This is assigned to the second term in solve.
        solve(Ae,eta1_cls.var,be)
        solve(At,the1_cls.var,bt)
        eta0_cls.update_var(eta1_cls.var)
        the0_cls.update_var(the1_cls.var)
        etaout.write(eta0_cls.var,time=dt)
        theout.write(the0_cls.var,time=dt)
        vars_cls=(eta0_cls,the0_cls,eta1_cls,the1_cls)
        vars_cls, eNorm, tNorm=two_step_loop(vars_cls,consts,tvec,ResultsFolder,Outfiles)
        eta2_cls,the2_cls=vars_cls
    elif simstep==1:
        vars_cls, eNorm, tNorm=one_step_loop(vars_cls,Lvec,avec,consts,tvec,ResultsFolder,Outfiles)
        eta2_cls,the2_cls=vars_cls
    #plotting.plot(eta1)
    # Save the outputs from the simulation
    #eta_xdmf.close()
    #the_xdmf.close()
    ff.save_var(tvec,tname,MeshFolder)
    ff.save_var(boundaries,domname,MeshFolder)
    ff.save_var(eNorm,etaresname,MeshFolder)
    ff.save_var(tNorm,theresname,MeshFolder)
    print('Results saved in',MeshFolder)
    return eta2_cls,the2_cls,parstr
    
def first_step(vars_cls,consts,test=1):
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
    tmpv0   = ufl.TrialFunction(V)
    tmpv1   = ufl.TrialFunction(V)
    v0      = ufl.TestFunction(V)
    v1      = ufl.TestFunction(V)
    simsteps=1
    L0=linear_form_func(var0_cls.char_key,simsteps,var0_cls,var1_cls,v0,consts,test) # Linear form for eta with time step 1
    L1=linear_form_func(var1_cls.char_key,simsteps,var1_cls,var0_cls,v1,consts,test)
    a0=bilinear_form_func(var0_cls.char_key,tmpv0,v0,consts,simsteps,r_mesh,test=0)
    a1=bilinear_form_func(var1_cls.char_key,tmpv0,v0,consts,simsteps,r_mesh,test=0)
    
    Lvec=(L0,L1)
    avec=(a0,a1)
    return Lvec,avec


def one_step_loop(vars_cls,Solvers,Lvec,avec,consts,tvec,path,MeshFolder="",Outfiles=()):
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
    Solver_0,Solver_1=Solvers
    L0,L1            =Lvec
    a0,a1            =avec
    Norm0=np.zeros(len(tvec))
    Norm1=np.zeros(len(tvec))
    Norm_t=1
    tol=consts[0]
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
    dt=tvec[1]-tvec[0]
    for tstep,t in enumerate(tvec):
        if check_out: # When the residual is reached a steady state has been reached.
            print('Nans in output ')
            print('Final time %.3f'%(t-dt))
            Norm0=Norm0[1:tstep]
            Norm1=Norm1[1:tstep]
            tvec=tvec[2:tstep+2]
            break
        elif Norm_t <=tol: # When the residual is reached a steady state has been reached.
            print('tolerance met %.6f'%Norm_t)
            print('Final time %.3f'%(t-dt))
            Norm0=Norm0[1:tstep]
            Norm1=Norm1[1:tstep]
            tvec=tvec[2:tstep+2]
            break
        elif tstep+2>=len(tvec): 
            print('Time steps exceeded maximum')
            print('Final time %.3f'%(t-dt))
            Norm0=Norm0[1:tstep]
            Norm1=Norm1[1:tstep]
            tvec=tvec[2:tstep+2]
            break
        else:
            #print('Solving for time step',t)
            # Update the right hand side reusing the initial vector
            A0=assemble(a0)
            A1=assemble(a1)
            b0=assemble(L0)
            b1=assemble(L1)
            solve(A0,var01_cls.var,b0)
            solve(A1,var11_cls.var,b1)
            var01_cls.var.x.scatter_forward()
            var11_cls.var.x.scatter_forward()
        
            # Write the solution to the xdmf file
            #eta_xdmf.write_function(eta2_cls.var, dt*(tstep+2))
            #the_xdmf.write_function(the2_cls.var,dt*(tstep+2))
            varlist=(var01_cls.var_vals(),var11_cls.var_vals())
            check_out=check_output(varlist)
        
            # Calculate the residual
            Norm0[tstep+1]=norm_res(var01_cls.var_vals(),var00_cls.var_vals()) # Sqrt of the sum of the sqs
            Norm1[tstep+1]=norm_res(var11_cls.var_vals(),var10_cls.var_vals()) # Sqrt of the sum of the sqs
            Norm_t=max(Norm0[tstep+1],Norm1[tstep+1])
            #DEBUGSTART
            if DEBUG:
                #for var_cls in (var00_cls,var01_cls,var10_cls,var11_cls):
                #    c=check_zero(var_cls.var_vals())
                #    print('before update zero check on',var_cls.name,' is ',c)
                print('solved for tstep',tstep)
            #DEBUGEND
            # Reassign the variables before the next step
            var00_cls.update_var(var01_cls)
            var10_cls.update_var(var11_cls)
            # RESET the solution variables
            var01_cls.reset_var()
            var11_cls.reset_var()
            # Plots the estimation, this is done on the 0th variable instead of the first since the varname is then missing the index. 
            Out0.write(var00_cls.var,time=t+dt)
            Out1.write(var10_cls.var,time=t+dt)
    vars_cls=(var01_cls,var11_cls)
    return vars_cls, Norm0, Norm1

def two_step_loop(vars_cls,consts,tvec,test=1,path='',MeshFolder="",Outfiles=()):
    '''Iterate through the time steps estimating solutions using the two step explicit solver.
    :param vars_cls: The spatial variables for variable i at time step j varij_cls. 
    $(varij_cls| i \in [0,1],j \in [0,1])$
    :param consts: consts for forming the variational functions and the simulation tolerance.
    :param tvec: vector with the time at desired timesteps :math:`tvec=(0,dt,...,nt*dt)`
    :param test: indicator, if 1 then the simulation is run for the Heat Eqn
    
    .. code-block::
        Create 2-step solver `Solve`
        for tstep,t in tvec:
            vari(j+1)=Solve(varij,vari(j-1))
            Norm[tstep]=residual(vari(j+1),varij)
            plot(t,vari(j+1))
            Update_Solver
            Update_vars(vari(j-1)=varij,varij=vari(j+1))
        return vari(j+1) Norm
        
    Returns the variable object with the estimated at the final time step and the residual vector from all the time steps.
    :returns: vari(j+1), Norm_vec,... for all variables i
    '''            
    # Retrieve the variables from the inputs
    var00_cls,var10_cls,var11_cls,var01_cls=vars_cls
    V            =var00_cls.function_space
    r_mesh       =var00_cls.mesh
    var02_cls=sv.spatial_variable(V,r_mesh,var00_cls.name+'2')
    var02_cls.set_char_key(var00_cls.char_key)
    var12_cls=sv.spatial_variable(V,r_mesh,var10_cls.name+'2')
    var12_cls.set_char_key(var10_cls.char_key)
    Norm0=np.zeros(len(tvec))
    Norm1=np.zeros(len(tvec))
    Norm_t=1
    tol=consts[0]
    simsteps=2
    
    v0=ufl.TestFunction(V)
    v1=ufl.TestFunction(V)
    tmpv0=ufl.TrialFunction(V)
    tmpv1=ufl.TrialFunction(V)

    if not len(Outfiles)==2:
        out0name=path+MeshFolder+'/'+var00_cls.name
        Out0=File(out0name+'.pvd')
        out1name=path+MeshFolder+'/'+var10_cls.name
        Out1=File(out1name+'.pvd')
    else:
        Out0,Out1=Outfiles
    
    
    # Calculate change between timesteps
    Norm0[0]=norm_res(var00_cls.var_vals(),var01_cls.var_vals()) # Sqrt of the sum of the squares.
    Norm1[0]=norm_res(var10_cls.var_vals(),var11_cls.var_vals()) # Sqrt of the sum of the squares
    Norm_t=max(Norm0[0],Norm1[0])


    # Improve solver for two step
    # linear form - inner must be used instead of * for the solvers to work
    L0=linear_form_func(var00_cls.char_key,simsteps,(var00_cls,var01_cls),var10_cls,v0,consts,test)
    L1=linear_form_func(var10_cls.char_key,simsteps,(var10_cls,var11_cls),var00_cls,v1,consts,test)
    # bilinear form
    a0=bilinear_form_func(var00_cls.char_key,tmpv0,v0,consts,simsteps,r_mesh,test)
    a1=bilinear_form_func(var10_cls.char_key,tmpv1,v1,consts,simsteps,r_mesh,test)

    #Solver_2step_0=NonlinearVariationalProblem(L0-a0,var02_cls.var)
    #Solver_2step_1=NonlinearVariationalProblem(L1-a1,var12_cls.var)
    A0=assemble(a0)
    A1=assemble(a1)
    b0=assemble(L0)
    b1=assemble(L1)
    # CTIME STEPPING COMPUTATIONS------------------------------------------------------

    # Begin Iterative Simulations --------------------------------------
    check_out=False
    dt=tvec[1]-tvec[0]
    for tstep,t in enumerate(tvec):
        if check_out: # When the residual is reached a steady state has been reached.
            print('Nans in output ')
            print('Final time %.3f'%(t-dt))
            Norm0=Norm0[1:tstep]
            Norm1=Norm1[1:tstep]
            tvec=tvec[2:tstep+2]
            break
        elif Norm_t <=tol: # When the residual is reached a steady state has been reached.
            print('tolerance met %.6f'%Norm_t)
            print('Final time %.3f'%(t-dt))
            Norm0=Norm0[1:tstep]
            Norm1=Norm1[1:tstep]
            tvec=tvec[2:tstep+2]
            break
        elif tstep+2>=len(tvec): 
            print('Time steps exceeded maximum')
            print('Final time %.3f'%(t-dt))
            Norm0=Norm0[1:tstep]
            Norm1=Norm1[1:tstep]
            tvec=tvec[2:tstep+2]
            break
        else:
            #print('Solving for time step',t)
            solve(A0,var02_cls.var,b0)
            solve(A1,var12_cls.var,b1)
            var02_cls.var.x.scatter_forward()
            var12_cls.var.x.scatter_forward()
            # Reassign the variables before the next step
            var00_cls.update_var(var01_cls)
            var10_cls.update_var(var11_cls)
            var01_cls.update_var(var02_cls)
            var11_cls.update_var(var12_cls)
        
            # Write the solution to the xdmf file
            varlist=(var02_cls.var_vals(),var12_cls.var_vals())
            check_out=check_output(varlist)
        
            # Calculate the residual
            Norm0[tstep+1]=norm_res(var00_cls.var_vals(),var01_cls.var_vals()) # Sqrt of the sum of the sqs
            Norm1[tstep+1]=norm_res(var10_cls.var_vals(),var11_cls.var_vals()) # Sqrt of the sum of the sqs
            Norm_t=max(Norm0[tstep+1],Norm1[tstep+1])
            # Update the right hand side reusing the initial vector
            #DEBUGSTART
            if DEBUG:
                for var_cls in (var00_cls,var01_cls,var02_cls,var10_cls,var11_cls,var12_cls):
                 c=check_zero(var_cls.var_vals())
                 print('before update zero check on',var_cls.name,' is ',c)
            #DEBUGEND
            # RESET eta2,the2
            var02_cls.reset_var()
            var12_cls.reset_var()
            Out0.write(var00_cls.var,time=t+2*dt)
            Out1.write(var10_cls.var,time=t+2*dt)
    var00_cls.update_var(var02_cls)
    var10_cls.update_var(var12_cls)
    vars_cls=(var02_cls,var12_cls)
    return vars_cls, Norm0, Norm1


def main(path,filename,simon=0,test=0,simstep=1,ind=0):
    '''Runs the simulations and plotting functions.
    Ouputs from the simulation are saved as numpy variables. These are then loaded before plotting.
    :param filename: The name of the file containing the input parameters
    :param simon: 0 or 1, if 1 then run simulations, if 0 then only plotting required. default 0.
    :param test: 0 or 1 if 1 then the simulation will run as heat equation
    :param simstep: 1 or 2 number of steps in spatial solver
    Uses functions :py:func:`Simulations(filename,MeshFolder,ResultsFolder,varnames,test)' and :py:func:`plotting(var1,var2,time,foldername)'
    :returns: nothing
    '''
    ResultsFolder='Plots'
    MeshFolder  ='Mesh'
    if test==1:
        ResultsFolder='./DiffusionOnly/'+ResultsFolder
        MeshFolder   ='./DiffusionOnly/'+MeshFolder
    dirlist=(ResultsFolder,MeshFolder)
    etaname     ='eta'
    thetaname   ='theta'
    tname       ='tvec'
    domname     ='boundaries'
    etaresname  ='eta_res'
    theresname  ='the_res'
    varnames=[etaname,thetaname,tname,domname,etaresname,theresname]
    if simon: eta_cls,the_cls,parstr=Simulations(filename,path,MeshFolder,ResultsFolder,varnames,test,simstep,ind)
    print('Simulations completed')
    #MeshFolder=MeshFolder+parstring
    #ResultsFolder=ResultsFolder+parstring
    # tvec=ff.load_file(MeshFolder+parstr,tname)
    # eta_res=ff.load_file(MeshFolder+parstr,etaresname)
    # pf.line_plot(eta_res,tvec,etaresname,tname,ResultsFolder+parstr)
    # del eta_res
    # theta_res=ff.load_file(MeshFolder+parstr,theresname)
    # pf.line_plot(theta_res,tvec,theresname,tname,ResultsFolder+parstr)
    # del theta_res
    # varnames=[etaname,thetaname]
    return

if __name__=='__main__':
    ff.change_owner()
    global DEBUG
    DEBUG=True
    path='~/Code/SussexPython/SussexFiredrake/'
    filename='ParameterSets.csv'
    simon=1
    heateqntest=0
    simstep=1
    main(path,filename,simon,heateqntest,simstep)
    exit()