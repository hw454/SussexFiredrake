
#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import os
import numpy as np
import GenerateParameterSets as gp
import sys
import num2words as nw
macheps=sys.float_info.epsilon

def check_folder(path,dirname):
    if not os.path.isdir(path+dirname):
        os.makedirs(path+dirname)
    return

def check_folder_nest(path,dirlist):
    initpath=''
    for d in dirlist:
        check_folder(path,initpath+d)
        initpath=initpath+d
    return

def save_var(var,varname,path,foldername,typ='.npy'):
    ''' Saves the variable `var' with the name `varname' inside the folder `foldername'
    :param var: numpy variable
    :param varname: string for the variable name
    :param path: string indicating the path to the directory.
    :param foldername: string indicating where the variable should be saved.
    :param typ: variable extension. Default to .npy
    :returns: nothing
    '''
    # Check the folder exists at path and create it if not
    check_folder(path,foldername)
    # Save variable
    filename=path+foldername+'/'+varname+typ
    print('Save var at'+filename)
    np.save(filename,var)
    return
def teststr(test):
    if test==0:return '/Both/'
    elif test==1: return '/DiffusionOnly/'
    elif test==2: return '/SourceOnly/'
    else: return ''

def load_file(path,dirname,fname,typ='.npy'):
    filesave=path+dirname+'/'+fname+typ
    if not os.path.exists(filesave):
        if not os.path.exists(path+dirname): 
            print('Directory does not exist, check path and location')
            print('Path'+path)
            print('Directory'+dirname)
            return
        else: 
            print('File'+filesave+' does not exist, could not load')
            return
    else:
        var=np.load(filesave)
        return var



def load_files(dirname,file_list,typ='.npy'):
    out=list()
    for fname in file_list:
        if not isinstance(fname, str):
            sublist=load_files(dirname,fname,typ)
            out.append(sublist)
        else:
            if not os.path.exists(dirname+'/'+fname+typ):
                if not os.path.exists(dirname): 
                    raise ValueError('directory '+dirname+' does not exist, check path')
                else: 
                    raise ValueError('unable to load file '+fname+' in directory '+dirname)
            else:
                f=load_file(dirname,fname)
                out.append(f) 
    return out

def read_inputs(path,filename,ind=0,sims=0,parfile='/parameters'):
    '''Reads the inputs from csv. The input variables correspond to #FIXME-InsertModelType
    :param filename: filename where inputs are located includes path and file type
    * mesh parameters
    :rtype: [[],[]]
    :returns: [mesh parameters, PDE parameters]
    '''
    inputdata = pd.read_csv(path+filename,header=0)
    #inputdata=dict(zip(list(inputdata.label), list(inputdata.value)))
    # Dicretisation Parameters
    nx=inputdata['nspace'][ind]
    ny=inputdata['nspace'][ind]
    xm=inputdata['xm'][ind]
    ym=inputdata['ym'][ind]
    x0=inputdata['x0'][ind]
    y0=inputdata['y0'][ind]
    meshdict={'nx':nx,'ny':ny,'xm':xm,'ym':ym,'x0':x0,'y0':y0}
    # PDE parameters
    tol =inputdata['tol'][ind]
    eps =inputdata['epsilon'][ind]
    #eps=0.65
    rho =inputdata['rho'][ind]
    a0  =inputdata['a0'][ind]
    b0  =inputdata['b0'][ind]
    T   =inputdata['T'][ind]
    perm=inputdata['perm'][ind]
    A0f=inputdata['A0f'][ind]
    df=inputdata['df'][ind]
    dtf=inputdata['dtf'][ind]
    sig=inputdata['sig'][ind]
    #A0=inputdata['A0'][ind]
    StSt=[1/eps,eps/(eps+1)]
    #d=['d'][ind]
    #StSt=[0,1/(eps+2)]
    #sig=calc_sig(eps,b0)
    A0=calc_A0(rho,sig,a0,b0,eps,A0f)
    print('A0f: ',A0f,' A0: ',A0)
    d=calc_d(rho,sig,a0,b0,eps,A0,df)
    print('df: ',df,' d:',d)
    hx=(xm-x0)/nx
    hy=(ym-y0)/ny
    h=min(hx,hy)
    dt=dtf*h**2/3
    pde_dict={'StSt': StSt, 'tol':tol,'eps':eps,'rho':rho,
    'sig':sig,'a0':a0,'b0':b0,'A0':A0,'d':d,'dt':dt,'T':T,'perm':perm}
    gp.check_instab_cond(StSt,A0,a0,b0,eps,d)
    check_folder(path,parfile)
    parfile+='/'+par_string(ind,dt,T,perm,hx,hy,A0,d,eps)
    check_folder(path,parfile)
    if sims:
        with open(parfile+"/parameters.txt", 'w') as f: 
            f.write('PDE Parameters: \n')
            for key, value in pde_dict.items(): 
                f.write('%s:%s\n' % (key, value))
            f.write('Mesh Parameters: \n')
            for key, value in meshdict.items(): 
                f.write('%s:%s\n' % (key, value))
            print('parameters written to',parfile+"/parameters.txt")
    return [meshdict,pde_dict]
def calc_alpha(a0,b0,eps):
    return  (a0*(eps+1)+b0*eps**2)/(eps*(eps+1))
def calc_A0(rho,sig,a0,b0,eps,f):
    alp=calc_alpha(a0,b0,eps)
    mul=rho*(eps**2)/(sig*np.exp(alp)*((eps+1)**3))
    ln2=np.log(2)
    Ac=mul*eps+eps/(sig*(eps+1)*np.exp(alp))
    Ap=mul*(2*ln2-eps+2*np.sqrt(ln2*(ln2-eps)))
    Am=mul*(2*ln2-eps-2*np.sqrt(ln2*(ln2-eps)))
    Ac=((rho/sig)*np.exp(-alp)*(eps/(eps+1))**3)*(1/(5-4*ln2))
    Ap=max(Ac,Ap)
    Am=min(Ac,Am)
    A0=Ac*(1-f)+Ap*f
    return A0
def calc_sig(eps,b0):
    t=eps*(1-b0)+2
    b=eps*b0
    return 0.9*t/b
def calc_d(rho,sig,a0,b0,eps,A0,df):
    e=1/eps
    t=eps/(eps+1)
    re=t/(e+1)
    p1=-t*A0*np.exp(a0*e+b0*t)
    p2=-np.log(2)*p1
    d1=2*e*p2-re*p1+np.sqrt(4*e*p2*(e*p2-re*p1))
    d2=sig/(rho*(re**2))
    d=df*d1*d2
    return d
def K(e,t,A,a,b):
    return A*np.exp(a*e+b*t)
def f_theta(e,t,eps):
    return -e
def f_eta(e,t,eps):
    t1=eps*e*(2+e)/((1+e)**2)
    return t1-t
def g_eta(e,t,consts):
    A0,a0,b0,A1,a1,b1=consts
    Ka=K(e,t,A0,a0,b0)
    Kd=K(e,t,A1,a1,b1)
    return a0*Ka-t*(a0*Ka+a1*Kd)
def g_theta(e,t,consts):
    A0,a0,b0,A1,a1,b1=consts
    Ka=K(e,t,A0,a0,b0)
    Kd=K(e,t,A1,a1,b1)
    return -t*b0*Ka-(1+t)*b1*Kd
def numfix(j):
    if j>=1:    return '%04d'%j
    else:    return '%04d'%(1000*j)

def par_string(ind,dt,T,perm,hx,hy,A0,d,eps):
    '''Generate string for appending to folder name. '''
    return 'ind'+numfix(ind)+'dt'+numfix(dt)+'_T'+numfix(T)+'_perm'+numfix(perm)+'_h'+numfix(min(hx,hy))+'_A0'+numfix(A0)+'_d'+numfix(d)+'_eps'+numfix(eps)
def change_owner():
    gid=os.getgid()
    uid=1000
    dirname=os.getcwd()
    os.chown(dirname,uid,gid)
    return

if __name__=='__main__':
    path=os.path.abspath(os.getcwd())
    sheetname='/ParameterSets.csv'
    ind=0
    pars=read_inputs(path,sheetname,ind)
    print(pars)
    exit() 
