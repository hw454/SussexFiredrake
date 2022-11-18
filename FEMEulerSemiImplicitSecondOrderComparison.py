#!/usr/bin/env python
# coding: utf-8
# Packages for runningfiredrake
import os
import sys

#Plotting
import PlottingFunctions as pf
# Packages for managing data
import numpy as np
# Packages for loading data
import FileFuncs as ff
# FEM solvers
from firedrake import *
from irksome import GaussLegendre, Dt, TimeStepper


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

def save_outs(vardict,path,savedir):
    for k in vardict.keys():
        ff.save_var(vardict[k],k,path,savedir)
    print('Results saved in',path+savedir)
    return 0

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
    ResultsFolder='Plots'+ff.simstr(simstep)+''
    MeshFolder  ='Mesh'+ff.simstr(simstep)+''
    etaname     ='eta'
    thetaname   ='theta'
    tname       ='tvec'
    etaresname  ='eta_res'
    theresname  ='the_res'
    mesh_par_dict,pdes_par_dict=ff.read_inputs(path,filename,ind)
    ff.check_folder(path,ff.teststr(0))
    ResultsFolder=ff.teststr(0)+ResultsFolder
    ff.check_folder(path,ResultsFolder)
    parstr=ff.par_string(ind,pdes_par_dict)
    ResultsFolder+='/'+parstr+'/'
    ff.check_folder(path,ResultsFolder)
    txtfile=path+ResultsFolder+'ParameterValues.txt'
    with open(txtfile, 'w') as f:
        for k in pdes_par_dict.keys():
            f.write(str(k)+' '+str(pdes_par_dict[k])+'\n')
        for k in mesh_par_dict.keys():
            f.write(str(k)+' '+str(mesh_par_dict[k])+'\n')
    nx=int(mesh_par_dict['nx'])
    ny=int(mesh_par_dict['ny'])
    x0 = mesh_par_dict['x0']
    x1 = mesh_par_dict['xm']
    y0 = mesh_par_dict['y0']
    y1 = mesh_par_dict['ym']
    dtau=pdes_par_dict['dt']
    
    xm=x1-x0
    ym=y1-y0
    msh = RectangleMesh(nx,ny,xm,ym)
    V = FunctionSpace(msh, "CG", 1)
    x, y = SpatialCoordinate(msh)

    D_e=Constant(pdes_par_dict['d_e'])
    D_t=Constant(pdes_par_dict['d_t'])
    StSt=pdes_par_dict['StSt']
    pe,pt=pdes_par_dict['perm']
    Pe=Constant(pe)
    Pt=Constant(pt)
    e0=Constant(StSt[0])
    t0=Constant(StSt[1])
    Rho=Constant(pdes_par_dict['rho'])
    Sig=Constant(pdes_par_dict['sig'])
    dt = Constant(dtau)
    t = Constant(0.0)
    tol=pdes_par_dict['tol']

    eta_i = interpolate(e0+0.5*Pe*(cos(x)+cos(y)), V)
    the_i=interpolate(t0+0.5*Pt*(cos(x)+cos(y)), V)
    vi1 = TestFunction(V)
    vi2=TestFunction(V)
    # Variational form for irksome stepper
    F_e = inner(Dt(eta_i), vi1)*dx +D_e*inner(grad(eta_i), grad(vi1))*dx -Rho*f_func(eta_i,the_i,pdes_par_dict)*vi1*dx
    F_t = inner(Dt(the_i), vi2)*dx + D_t*inner(grad(the_i), grad(vi2))*dx -Sig*g_func(eta_i,the_i,pdes_par_dict)*vi2*dx#
    butcher_tableau = GaussLegendre(2)
    luparams = {'snes_max_it': 100,
                'snes_type': 'newtonls',
                'ksp_type': 'gmres',
                'pc_type': 'lu', 
                'mat_type': 'aij',
                'pc_factor_mat_solver_type': 'mumps'}
    stepper_e= TimeStepper(F_e, butcher_tableau, t, dt, eta_i,
                      solver_parameters=luparams)
    stepper_t = TimeStepper(F_t, butcher_tableau, t, dt, the_i,
                      solver_parameters=luparams)
    # Variational form for manual
    eta_m0 = interpolate(e0+0.5*pe*(cos(x)+cos(y)), V)
    the_m0=interpolate(t0+0.5*pt*(cos(x)+cos(y)), V)
    vm1  =   TestFunction(V)
    vm2  =   TestFunction(V)
    # Create outfiles
    eta_m_hist=[]
    eta_m_hist.append(eta_m0.copy(deepcopy=True))
    the_m_hist=[]
    the_m_hist.append(the_m0.copy(deepcopy=True))
    eta_i_hist=[]
    the_i_hist=[]
    eta_i_hist.append(eta_i.copy(deepcopy=True))
    the_i_hist.append(the_i.copy(deepcopy=True))
    
    # Define two-step variational problem ---------------------------------------
    eta_m1=Function(V)
    the_m1=Function(V)
    eta_m2=Function(V)
    the_m2=Function(V)
    tmpet  = TrialFunction(V)
    tmpth  = TrialFunction(V)
    ae0=inner(tmpet,vm1)*dx+dt*D_e*inner(grad(tmpet),grad(vm1))*dx
    at0=inner(tmpth,vm2)*dx+dt*D_t*inner(grad(tmpth),grad(vm2))*dx
    Le0=inner(eta_m0,vm1)*dx+Rho*dt*f_func(eta_m0,the_m0,pdes_par_dict)*vm1*dx
    Lt0=inner(the_m0,vm2)*dx+Sig*dt*g_func(eta_m0,the_m0,pdes_par_dict)*vm2*dx
    ae1=3*inner(tmpet,vm1)*dx+2*dtau*D_e*inner(grad(tmpet),grad(vm1))*dx
    at1=3*inner(tmpth,vm2)*dx+2*dtau*D_t*inner(grad(tmpth),grad(vm2))*dx
    Le1=4*inner(eta_m1,vm1)*dx-inner(eta_m0,vm1)*dx+4*Rho*dtau*f_func(eta_m1,the_m1,pdes_par_dict)*vm1*dx-2*Rho*dtau*f_func(eta_m0,the_m0,pdes_par_dict)*vm1*dx
    Lt1=4*inner(the_m1,vm2)*dx-inner(the_m0,vm2)*dx+4*Sig*dtau*g_func(eta_m1,the_m1,pdes_par_dict)*vm2*dx-2*Sig*dtau*g_func(eta_m0,the_m0,pdes_par_dict)*vm2*dx
    nt=0
    tmax=pdes_par_dict['T']
    # Initialise the bounds for the variable 
    # values for plotting instances on the same range later.
    tsteps=int(tmax/dtau)
    Norm_res=np.zeros((tsteps+1,2))
    tvec=np.arange(0,tmax,dtau)
    res_max=0
    outvars={tname:tvec,
            etaresname:Norm_res[:,0],theresname:Norm_res[:,1]}
    try:
        # irksome step 1 solver
        stepper_e.advance()
        stepper_t.advance()
        t.assign(float(t) + float(dt))
        eta_i_hist.append(eta_i.copy(deepcopy=True))
        the_i_hist.append(the_i.copy(deepcopy=True))
        # Manual first order solver
        solve(ae0==Le0,eta_m1,solver_parameters=luparams)
        solve(at0==Lt0,the_m1,solver_parameters=luparams)
        eta_m_hist.append(eta_m1.copy(deepcopy=True))
        the_m_hist.append(the_m1.copy(deepcopy=True))
        Norm_res[nt,0]=norm_res(eta_i_hist[nt].vector(),eta_m_hist[nt].vector())
        Norm_res[nt,1]=norm_res(the_i_hist[nt].vector(),the_m_hist[nt].vector())
        res_max=max(Norm_res[nt,:])        
        nt+=1
        if DEBUG:
            print('solved for tstep',nt)
    except: 
        print('Solving at step 1 failed')
        zeroarr=np.array([0])
        outvars={tname:tvec[0],
            etaresname:zeroarr,theresname:zeroarr}
        save_outs(outvars,path,MeshFolder)
        return eta_m_hist,the_m_hist,eta_i_hist,the_i_hist,parstr,pdes_par_dict
    # TIME STEPPING COMPUTATIONS------------------------------------------------------

    # Begin Iterative Simulations --------------------------------------
    # Loop for second order solver and irksome
    while (float(t) < tmax):
        if (float(t) + float(dt) > tmax):
            dt.assign(tmax - float(t))
        if res_max>tol:
            print('Residual has blown up')
            print(Norm_res[nt-1])
            break
        try: 
            stepper_e.advance()
            stepper_t.advance()
            solve(ae1==Le1,eta_m2,solver_parameters=luparams)
            solve(at1==Lt1,the_m2,solver_parameters=luparams)
        except:
            print('time stepper failed at step %d'%nt)
            break
        if DEBUG:
            print('solved for tstep',nt+1)
        #DEBUGEND
        # Reassign the variables before the next step
        eta_m0.assign(eta_m1.copy(deepcopy=True))
        eta_m1.assign(eta_m2.copy(deepcopy=True))
        the_m0.assign(the_m1.copy(deepcopy=True))
        the_m1.assign(the_m2.copy(deepcopy=True))
        # RESET the solution variables
        eta_m2=Function(V)
        the_m2=Function(V)

        t.assign(float(t) + float(dt))
        eta_m_hist.append(eta_m1.copy(deepcopy=True))
        the_m_hist.append(the_m1.copy(deepcopy=True))
        eta_i_hist.append(eta_i.copy(deepcopy=True))
        the_i_hist.append(the_i.copy(deepcopy=True))
        Norm_res[nt,0]=norm_res(eta_i_hist[nt].vector(),eta_m_hist[nt].vector())
        Norm_res[nt,1]=norm_res(the_i_hist[nt].vector(),the_m_hist[nt].vector())
        res_max=max(Norm_res[nt,:])
        nt+=1
    etaname='eta'
    thetaname='theta'
    varname=(etaname,thetaname)
    tvec=np.arange(0,float(t),dtau)
    tname='t'
    for i in range(2):
        pf.line_plot(Norm_res[0:nt,i],tvec,varname[i]+'comparison_res',tname,ResultsFolder)
    #plt.show()
    outvars[etaresname]=Norm_res[0:nt,0]
    outvars[theresname]=Norm_res[0:nt,1]
    outvars[tname]=tvec
    save_outs(outvars,path,MeshFolder)
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