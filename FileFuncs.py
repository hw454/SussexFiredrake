
#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import os
import numpy as np

def par_string(ind,dt,T,perm,hx,hy):
    '''Generate string for appending to folder name. '''
    return 'ind'+numfix(ind)+'dt'+numfix(dt)+'_T'+numfix(T)+'_perm'+numfix(perm)+'_h'+numfix(min(hx,hy))

def check_folder(path,dirname):
    if not os.path.isdir(path+dirname):
        os.makedirs(path+dirname)
    return

def check_folder_nest(path,dirlist):
    initpath=path
    for d in dirlist:
        check_folder(initpath+d)
        initpath=initpath+d
    return


def load_file(path,dirname,fname,typ='.npy'):
    filesave=path+dirname+'/'+fname+typ
    if not os.path.exists(filesave):
        if not os.path.exists(path+dirname): 
            print('Directory does not exist, check path and location')
            return
        else: 
            print('File does not exist, could not load')
            return
    else:
        var=np.load(filesave)
        return var


def read_inputs(path,filename,ind=0):
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
    meshpars=[nx,ny,xm,ym,x0,y0]
    # PDE parameters
    tol =inputdata['tol'][ind]
    eps =inputdata['epsilon'][ind]
    rho =inputdata['rho'][ind]
    sig =inputdata['sigma'][ind]
    a0  =inputdata['a0'][ind]
    b0  =inputdata['b0'][ind]
    T   =inputdata['T'][ind]
    perm=inputdata['perm'][ind]
    alp=(a0*(eps+1)+b0*eps**2)/(eps*(eps+1))
    mul=rho*(eps**2)/(sig*np.exp(alp)*((eps+1)**3))
    ln2=np.log(2)
    if eps<2:
        StSt=[1/eps,eps/(eps+1)]
        Ac=mul*eps
        Am=mul*(2*ln2-eps-2*np.sqrt(ln2*(ln2-eps)))
        A0=(Am+Ac)/2
    else:
        StSt=[0,eps/(eps+1)]
        Am=mul*(2*ln2-eps-2*np.sqrt(ln2*(ln2-eps)))
        A0=0.9*Am
    a=rho**eps**2/((eps+1)**2)
    b=sig*A0*np.exp(alp)*(eps+1)/eps
    c=sig*A0*np.exp(alp)*rho*(ln2-eps)/(eps+1)
    d1=1.1*(2*c-a*b+2*np.sqrt(c*(c-a*b)))/(a**2)
    d2=1.1*(eps+1)**3*A0*np.exp(alp)*sig/(rho*eps**3)
    d=max(d1,d2)
    h=min((xm-x0)/nx,(ym-y0)/ny)
    dt=h**2/3
    pdes_pars=[StSt,tol,eps,rho,sig,a0,b0,A0,d,dt,T,perm]
    return [meshpars,pdes_pars]
def numfix(j):
    if j>=1:    return '%04d'%j
    else:    return '%04d'%(1000*j)
def change_owner():
    gid=os.getgid()
    uid=1000
    dirname=os.getcwd()
    os.chown(dirname,uid,gid)
    return
