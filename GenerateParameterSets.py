#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import os
import numpy as np
import csv
from itertools import permutations

import FileFuncs as ff
def create_par_set(path,filename):
    '''Reads the inputs from csv. The input variables correspond to #FIXME-InsertModelType
    :param filename: filename where inputs are located includes path and file type
    * mesh parameters
    :rtype: [[],[]]
    :returns: [mesh parameters, PDE parameters]
    '''
    inputdata = pd.read_csv(path+filename)
    datadict_low=dict(zip(list(inputdata.label), list(inputdata.value_low)))
    datadict_high=dict(zip(list(inputdata.label), list(inputdata.value_high)))
    Nopts=int(datadict_low['N_opts'])
    # Remove the fixed parameters from the dictionary
    keylist={k for k in datadict_low.keys() if datadict_low[k]==datadict_high[k]}
    dictfixed=dict()
    for k in keylist:
        dictfixed[k]=datadict_low[k]
        datadict_low.pop(k)
        datadict_high.pop(k)
    datadict_range=dict()
    stepvec=np.arange(0,Nopts,1)/Nopts
    for k in datadict_low.keys():
        a=datadict_low[k]
        b=datadict_high[k]
        vals=a+(b-a)*stepvec
        datadict_range[k]=vals
    # Create csv with headers = datadict_range.keys()
    headerlabels=list(k for k in datadict_range.keys())
    dictrow={key: datadict_low[key] for key in headerlabels}
    df=pd.DataFrame(columns=headerlabels)
    fullheader=headerlabels.copy()
    for k in dictfixed.keys():
        fullheader.append(k)
    nk=len(headerlabels)
    for i in range(nk):
        # Term in key list
        k0=headerlabels[i]
        for j in range(Nopts):
            # All options for that key
            dictrow[k0]=datadict_range[k0][j]
            if i==0:
                #checkvec=check_pars_from_row(dictrow,dictfixed)
                #if not all(checkvec): 
                #    print('Conditions not satified',checkvec)
                #    print(dictrow)
                #else:
                #    print('Conditions satisfied')
                    print(dictrow)
                    vallist=list(val for val in dictrow.values())
                    for val in dictfixed.values():
                        vallist.append(val)
                    df2=pd.DataFrame([vallist],columns=fullheader)
                    df=pd.concat([df,df2],names=fullheader,ignore_index=True)
            for ik in range(i):
                # All keys before the current key
                k1=headerlabels[ik]
                for jk in range(Nopts):
                    # All options for each previous key
                        dictrow[k1]=datadict_range[k1][jk]
                    #checkvec=check_pars_from_row(dictrow,dictfixed)
                    #if not all(checkvec): 
                    #    print('Conditions not satified',checkvec)
                    #    print(dictrow)
                    #else:
                    #    print('Conditions satisfied')
                        print(dictrow)
                        vallist=list(val for val in dictrow.values())
                        for val in dictfixed.values():
                            vallist.append(val)
                        df2=pd.DataFrame([vallist],columns=fullheader)
                        df=pd.concat([df,df2],names=fullheader,ignore_index=True)
    df.to_csv(path+'/ParameterSets.csv',sep=',', header=fullheader,mode='w')
    return 
def check_pars_from_row(dictrow,dictfixed):
    # PDE parameters
    eps =dictrow['epsilon']
    if eps>=np.log(2) or eps<0:
        print('invalid eps',eps)
        return [False,False,False,False]     
    rho =dictfixed['rho']
    a0  =dictfixed['a0']
    b0  =dictfixed['b0']    
    #d=1.1*((2*(fe*gt+gt*ge)/fe)-gt/(fe**2))
    #d=-gt/fe
    #d1=(2*c-a*b+2*np.sqrt(c*(c-a*b)))/(a**2)
    #d2=(-b/a)
    #d=1.1*max(d1,d2)
    #d=max(d1,d2)
    f   =dictrow['A0f']
    #A0=dictfixed['A0']
    sig=dictrow['sig']
    #sig=ff.calc_sig(eps,b0)
    alph=ff.calc_alpha(a0,b0,eps)
    A0=ff.calc_A0(rho,sig,a0,b0,eps,f)
    d=ff.calc_d(rho,sig,a0,b0,eps,A0)
    # alp=(a0*(eps+1)+b0*eps**2)/(eps*(eps+1))
    # mul=rho*(eps**2)/(sig*np.exp(alp)*((eps+1)**3))
    # ln2=np.log(2)
    e=1/eps
    t=eps/(eps+1)
    StSt=[e,t]
    if d<0:
        print('negative diffusion',d)
        return [False,False,False,False]   
    if a0<0:
        print('negative a0',a0)
        return [False,False,False,False] 
    if a0-eps*np.log(2)<0:
        print('negative a1',a0-eps*np.log(2))
        return [False,False,False,False] 
    if A0<0:
        print('negative A0',A0)
        return [False,False,False,False]  
    if b0<0:
        print('negative b0',b0)
        return [False,False,False,False]  
    fe=ff.f_eta(e,t,eps)
    ft=ff.f_theta(e,t,eps)
    A1=A0*2/eps
    a1=a0-eps*np.log(2)
    b1=b0
    gt=ff.g_theta(e,t,(A0,a0,b0,A1,a1,b1)) 
    ge=ff.g_eta(e,t,(A0,a0,b0,A1,a1,b1))  
    J=np.array([[rho*fe,rho*ft],[sig*ge,sig*gt]])
    D=np.array([[1,0],[0,d]])
    print('Eigen vals',np.linalg.eig(J+D))
    cond=check_instab_cond(StSt,A0,a0,b0,eps,d)
    return cond

def check_instab_cond(StSt,A0,a0,b0,eps,d):
    e,t=StSt
    A1=A0*2/eps
    a1=a0-eps*np.log(2)
    b1=b0
    fe=ff.f_eta(e,t,eps)
    ft=ff.f_theta(e,t,eps)
    ge=ff.g_eta(e,t,(A0,a0,b0,A1,a1,b1))
    gt=ff.g_theta(e,t,(A0,a0,b0,A1,a1,b1))
    c0=fe+gt
    c1=fe*gt-ft*ge
    c2=d*fe+gt
    c3=(d*fe+gt)**2-4*d*(fe*gt-ft*ge)
    conds=[c0<0,c1>0,True,True]#c2>=0,c3>0]
    return conds


if __name__=='__main__':
    path=os.path.abspath(os.getcwd())
    sheetname='/InputsRange.csv'
    create_par_set(path,sheetname)
    exit()