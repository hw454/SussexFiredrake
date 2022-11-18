from firedrake import *
from irksome import GaussLegendre, Dt, TimeStepper
from ufl.algorithms.ad import expand_derivatives
import FileFuncs as ff
import PlottingFunctions as pf
import os
import sys
import math as ma
import numpy as np

def f_func(eta,theta,consts_dict):
    eps=Constant(consts_dict['eps'])
    #Eps=fem.Constant(r_mesh, PETSc.ScalarType(eps))
    #return - eta*theta+eps*eta**2/(1+eta) 
    # Pure Diffusion (Heat Eqn)
    ec=conditional(eta>0.0,eta,0.0)
    tc=conditional(theta>0.0,theta,0.0)
    return - ec*tc+(eps*ec**2/(1+ec))
def g_func(eta,theta,consts_dict):
    eps=Constant(consts_dict['eps'])
    a0=Constant(consts_dict['a0'])
    A0=Constant(consts_dict['A0'])
    b0=Constant(consts_dict['b0'])
    A1=A0*2/eps
    a1=a0-eps*ma.log(2)
    ec=conditional(eta>0.0,eta,0.0)
    tc=conditional(theta>0.0,theta,0.0)
    return A0*exp(a0*ec+b0*tc)*(1-tc)-A1*exp(a1*ec+b0*tc)*tc
def norm_res(v,w):
    diff=np.linalg.norm(v-w)
    bot=np.linalg.norm(v)
    return diff/bot

def time_div(v1,v0,dt):
    diff=np.linalg.norm(v1-v0)
    return diff/dt
def simulations(path,filename,ind,plotdir):
    mesh_par_dict,pdes_par_dict=ff.read_inputs(path,filename,ind)
    ff.check_folder(path,plotdir)
    plotdir+=ff.teststr(0)
    ff.check_folder(path,plotdir)
    parstr=ff.par_string(ind,pdes_par_dict)
    plotdir+=parstr+'/'
    ff.check_folder(path,plotdir)
    txtfile=path+plotdir+'ParameterValues.txt'
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

    eta = interpolate(e0+0.5*Pe*(cos(x)+cos(y)), V)
    theta=interpolate(t0+0.5*Pt*(cos(x)+cos(y)), V)
    v1 = TestFunction(V)
    v2=TestFunction(V)
    F_e = inner(Dt(eta), v1)*dx +D_e*inner(grad(eta), grad(v1))*dx -Rho*f_func(eta,theta,pdes_par_dict)*v1*dx
    F_t = inner(Dt(theta), v2)*dx + D_t*inner(grad(theta), grad(v2))*dx -Sig*g_func(eta,theta,pdes_par_dict)*v2*dx#
    butcher_tableau = GaussLegendre(2)
    luparams = {'snes_max_it': 100,
                'snes_type': 'newtonls',
                'ksp_type': 'gmres',
                'pc_type': 'lu', 
                'mat_type': 'aij',
                'pc_factor_mat_solver_type': 'mumps'}
    stepper_e= TimeStepper(F_e, butcher_tableau, t, dt, eta,
                      solver_parameters=luparams)
    stepper_t = TimeStepper(F_t, butcher_tableau, t, dt, theta,
                      solver_parameters=luparams)
    # Set up lists for storing the solution for each time step.
    eta_hist=[]
    theta_hist=[]
    eta_hist.append(eta.copy(deepcopy=True))
    theta_hist.append(theta.copy(deepcopy=True))
    nt=0
    tmax=pdes_par_dict['T']
    # Initialise the bounds for the variable 
    # values for plotting instances on the same range later.
    vmin_e=min(eta.vector())
    vmin_t=min(theta.vector())
    vmax_e=max(eta.vector())
    vmax_t=max(theta.vector())
    vargs={'eta':{'min':vmin_e,'max':vmax_e},'theta':{'min':vmin_t,'max':vmax_t}}
    tsteps=int(tmax/dtau)
    Norm_res=np.zeros((tsteps+1,2))
    Time_div=np.zeros((tsteps+1,2))
    res_max=0
    rescount=0
    while (float(t) < tmax):
        if (float(t) + float(dt) > tmax):
            print('Reached final time %.3f'%(float(t)+float(dt)))
        if res_max>tol:
            print('Residual has blown up')
            print(Norm_res[nt-1])
            break
        elif abs(res_max)<10*macheps:
            rescount+=1
            if rescount>10:
                print('Residual has set at 0')
                print('Final time %.3f on step %d'%(float(t)+float(dt),nt))
                break
        try: 
            stepper_e.advance()
            stepper_t.advance()
        except:
            print('time stepper failed at step %d'%nt)
            break
        umin_e=min(eta.vector())
        umax_e=max(eta.vector())
        if umin_e<vargs['eta']['max']: vargs['eta']['min']=umin_e
        if umax_e>vargs['eta']['max']: vargs['eta']['max']=umax_e
        umin_t=min(theta.vector())
        umax_t=max(theta.vector())
        if umin_t<vargs['theta']['min']: vargs['theta']['min']=umin_t
        if umax_t>vargs['theta']['max']: vargs['theta']['max']=umax_t
        print('time = %.3f, timestep=%d'%(float(t)+float(dt),nt))
        t.assign(float(t) + float(dt))
        eta_hist.append(eta.copy(deepcopy=True))
        theta_hist.append(theta.copy(deepcopy=True))
        Norm_res[nt,0]=norm_res(eta_hist[-1].vector(),eta_hist[-2].vector())
        Norm_res[nt,1]=norm_res(theta_hist[-1].vector(),theta_hist[-2].vector())
        res_max=max(Norm_res[nt,:])
        Time_div[nt,0]=time_div(eta_hist[-1].vector(),eta_hist[-2].vector(),dtau)
        Time_div[nt,1]=time_div(theta_hist[-1].vector(),theta_hist[-2].vector(),dtau)
        nt+=1
    etaname='eta'
    thetaname='theta'
    varname=(etaname,thetaname)
    tvec=np.arange(0,t,dtau)
    tname='t'
    variance_residual(Norm_res,path,pdes_par_dict)
    for i in range(2):
        pf.line_plot(Norm_res[:,i],tvec,varname[i]+'res',tname,path+plotdir)
        pf.line_plot(Time_div[:,i],tvec,varname[i]+'res',tname,path+plotdir)
    #plt.show()
    pf.Plotting(eta_hist,etaname,path,plotdir,parstr,pdes_par_dict,vargs)
    pf.Plotting(theta_hist,thetaname,path,plotdir,parstr,pdes_par_dict,vargs)
    return
def variance_residual(res_arr,path,consts_dict):
    samp_var_eta=np.var(res_arr[:,0],ddof=1)
    samp_var_the=np.var(res_arr[:,1],ddof=1)
    d=consts_dict['d_t']/consts_dict['d_e']
    A0=consts_dict['A0']
    ff.add_var_to_d(d,A0,samp_var_eta,samp_var_the,path)
    return 0
def get_ind(argv):
    job=0 # default jobHeateqn t
    if len(argv)>1:
        job=int(argv[1])
    return job
if __name__=='__main__':
    #global DEBUG,
    global macheps
    DEBUG=True
    macheps=sys.float_info.epsilon
    ind=get_ind(sys.argv)
    path=os.path.abspath(os.getcwd())
    filename='/ParameterSets.csv'
    PlotFolder='/irksomePlots/'
    simulations(path,filename,ind,PlotFolder)
    exit()