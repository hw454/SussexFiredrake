#!/usr/bin/env python
# coding: utf-8


# Packages for loading data
import pandas as pd
import cv2
import os
import FileFuncs as ff


# Packages for managing data
import numpy as np
from itertools import product

# Packages for plotting
import pyvista as pv
import matplotlib.pyplot as plt


def plot_animation(nt,varname='var',ResultsFolder='',tp=1):
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
    img_array=[]
    for j in range(nt):
        if j>0:
            filesave=filename+ff.numfix(j)+'.png'
            #tstring=varname+'(x,y), time {:.2f}'.format(tstep)
            #sns.heatmap(var[:,:,j],vmin=varmin,vmax=varmax).set(title=tstring)
            #im = plt.imshow(var[:,:,j],vmin=varmin,vmax=varmax)
            #im.title(tstring)
            #plt.savefig(filesave)
            #plt.close('all')
            if not os.path.exists(filesave):
                print('No file',filesave)
                img=None
            else: img = cv2.imread(filesave)
            if isinstance(img,type(None)):
                pass
            else:
                height, width, layers = img.shape
                size = (width,height)
                img_array.append(img)
    out = cv2.VideoWriter(filename+'00.avi',cv2.VideoWriter_fourcc(*'DIVX'), tp, size)
    for j in range(len(img_array)):
        out.write(img_array[j])
    out.release()
    print('Video at',filename+'00.avi')
    del img_array
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
    #FIXME-time plots need to be added
    :param var 3: numpy variable for x-axos(or similar dimensional variable)
    :param varnames: list of names corresponding to variables to plot
    :param ResultsFolder: string indicating the folder path where the plots should be save
    '''
    for vname in varnames:
        plot_animation(len(xvar),vname,ResultsFolder,tp)
    return


def main(filename,test=0,tp=1):
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
    pars,pdes_pars=ff.read_inputs(filename)
    nx,ny,xm,ym,x0,y0=pars
    hx=(xm-x0)/nx
    hy=(ym-y0)/ny # mesh size
    # Parameters for the PDEs
    _,_,_,_,_,_,_,_,_,dt,T,perm=pdes_pars
    parstr=ff.par_string(dt,T,perm,hx,hy)
    print('Mesh folder',MeshFolder+'/'+parstr)
    tvec=ff.load_file(MeshFolder+'/'+parstr,tname)
    eta_res=ff.load_file(MeshFolder+'/'+parstr,etaresname)
    line_plot(eta_res,tvec,etaresname,tname,ResultsFolder+parstr)
    del eta_res
    theta_res=ff.load_file(MeshFolder+'/'+parstr,theresname)
    line_plot(theta_res,tvec,theresname,tname,ResultsFolder+parstr)
    del theta_res
    varnames=[etaname,thetaname]
    plotting(tvec,varnames,ResultsFolder+'/'+parstr,tp)
    print('Plots saved in',ResultsFolder+'/'+parstr)
    return



if __name__=='__main__':
    global DEBUG
    DEBUG=True
    filename='Inputs.csv'
    heateqn=0
    tp=0.25
    main(filename,heateqn,tp)
    exit()





