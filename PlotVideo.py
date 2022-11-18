#!/usr/bin/env python
# coding: utf-8


# Packages for loading data
import sys
import pandas as pd
import cv2
import os
import FileFuncs as ff
import imageio
import num2words as nw


# Packages for managing data
import numpy as np
from itertools import product

# Packages for plotting
import pyvista as pv
import matplotlib.pyplot as plt


def plot_animation(nt,varname='var',ResultsFolder='',tp=0.25):
    ''' Plotting sequence of heatmaps for a variable `var' and create animation.
    :param domain: numpy array containing the measurements for the envrionment [xmin,ymin,xmax,ymax]
    :param var: numpy array containing variable values
    :param tvec: numpy array containing the time values which correspond with the var results.
    :param varname: string corresponding to the name of var
    :param tp: Time pause for video
    :param ResultsFolder: folder where the plots should be save. default ''
    '''
    #_,_,T=var.shape
 
    if not os.path.exists(ResultsFolder):
        os.makedirs(ResultsFolder)
    filename=ResultsFolder+'/'+varname
    with imageio.get_writer(filename+'.gif', mode='I',duration=0.25) as writer:
        for j in range(nt):
            filesave=filename+ff.numfix(j)+'.png'
            if not os.path.exists(filesave):
                print('No file',filesave)
                img=None
            else: img = imageio.imread(filesave)
            if isinstance(img,type(None)):
                pass
            else:
                writer.append_data(img)
    print('Video at',filename+'.gif')
    return


def line_plot(f,t,varname,tname,ResultsFolder=''):
    ''' Plot f as a function of t. label x axis with `tname` 
    and y axis with `varname` save in ResultsFolder'''
    if not os.path.exists(ResultsFolder):
        os.makedirs(ResultsFolder)
    if len(t)>len(f):
        t=t[-len(f)-1:-1]
    elif len(f)>len(t):
        f=f[-len(t)-1:-1]
    plt.figure()
    plt.plot(t,f)
    plt.xlabel(tname)
    plt.ylabel(varname)
    plt.savefig(ResultsFolder+'/'+varname+'_vs_'+tname+'.png')
    plt.close()
    return

def plotting(xvar,varnames,ResultsFolder='',tp=1):
    '''Plots output variables from simulation and errors over time
    :param var 3: numpy variable for x-axos(or similar dimensional variable)
    :param varnames: list of names corresponding to variables to plot
    :param ResultsFolder: string indicating the folder path where the plots should be save
    '''
    for vname in varnames:
        plot_animation(len(xvar),vname,ResultsFolder,tp)
    return


def main(path,filename,test=0,tp=1,ind=0,simstep=1):
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
    etaname     ='eta'
    thetaname   ='theta'
    tname       ='tvec'
    domname     ='boundaries'
    etaresname  ='eta_res'
    theresname  ='the_res'
    etadivname  ='eta_div'
    thedivname  ='the_div'
    varnames=[etaname,thetaname,tname,domname,etaresname,theresname,etadivname,thedivname]
    mesh_par_dict,pdes_par_dict=ff.read_inputs(path,filename,ind,0)
    nx=mesh_par_dict['nx']
    ny=mesh_par_dict['ny']
    xm=mesh_par_dict['xm']
    ym=mesh_par_dict['ym']
    x0=mesh_par_dict['x0']
    y0=mesh_par_dict['y0']
    hx=(xm-x0)/nx
    hy=(ym-y0)/ny # mesh size
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
    ff.check_folder(path,ResultsFolder)
    ff.check_folder(path,MeshFolder)
    MeshFolder=MeshFolder+'/'+parstr
    ResultsFolder=ResultsFolder+'/'+parstr
    ff.check_folder(path,ResultsFolder)
    ff.check_folder(path,MeshFolder)
    print('Mesh folder',MeshFolder)
    tvec=ff.load_file(path,MeshFolder,tname)
    varnames=[etaname,thetaname]
    plotting(tvec,varnames,ResultsFolder,tp)
    print('Plots saved in',ResultsFolder)
    return

def get_ind(argv):
    job=0 # default job
    if len(argv)>1:
        job=int(argv[1])
    return job


if __name__=='__main__':
    global DEBUG
    DEBUG=True
    path=os.path.abspath(os.getcwd())
    filename='ParameterSets.csv'
    testlist=(0,1,2)
    tp=4
    path='./'
    simstep=1
    ind=get_ind(sys.argv)  
    for test in testlist:
        main(path,filename,test,tp,ind,simstep)
    exit()





