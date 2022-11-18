#!/usr/bin/env python
# coding: utf-8
from firedrake import *
import cv2
import os
import sys
import FileFuncs as ff
# Packages for plotting lines and heatmaps
import matplotlib.pyplot as plt
from mpi4py import MPI

import pandas as pd

def line_plot(f,t,varname,tname,path='',ResultsFolder=''):
    ''' Plot f as a function of t. label x axis with `tname` 
    and y axis with `varname` save in ResultsFolder'''
    ff.check_folder(path,ResultsFolder)
    if len(t)>len(f):
        t=t[-len(f)-1:-1]
    elif len(f)>len(t):
        f=f[-len(t)-1:-1]
    plt.figure()
    plt.plot(t,f,'*',label=varname)
    plt.xlabel(tname)
    plt.ylabel(varname)
    plt.savefig(path+ResultsFolder+'/'+varname+'_vs_'+tname+'.png')
    plt.close()
    return

def plot_var_videos(xvar,varnames,path='',MeshFolder='/',ResultsFolder='',tp=1):
    '''PCombine plots of variable into video
    :param var 3: numpy variable for x-axos(or similar dimensional variable)
    :param varnames: list of names corresponding to variables to plot
    :param ResultsFolder: string indicating the folder path where the plots should be save
    :param tp: Time pause for video default 1
    '''
    for vname in varnames:
        create_video(vname,path=path,PlotsFolder=ResultsFolder,nt=len(xvar),tp=tp)
    return


def create_video(varname,path='',PlotsFolder='',nt=0,tp=1):
    ''' Combine sequence of heatmaps for a variable `var' and create animation.
    :param varname:       string corresponding to the name of var
    :param PlotsFolder: folder where the plots should be save. default ''
    :param nt:            number of time steps. default `nt=0`
                          if nt is zero then an if exists check will need to be run 
                          to determine the exit of the video.
    :param tp:            time pause, real number. default `tp=1`.
    Check the desired directory exists. If it doesn't then plots can not be loaded. 
    Use :func:`create_img_array` to create the object for the video. 
    Iterate through to store to a video.
    :rtype: 0,1
    :return: 0 for unsuccessful creation, 1 for successful video.
    '''
    print('Making video')
    if not os.path.exists(path+PlotsFolder):
        print("Directory doesn't exist so can not create video")
        return 0
    else:
        img_array,nt=create_img_array(path,PlotsFolder,varname,nt)
        img=img_array[-1]
        height, width, _ = img.shape
        size = (width,height)
        vidname=path+PlotsFolder+'/'+varname+'00.avi'
        fourcc_avi=cv2.VideoWriter_fourcc(*'DIVX')
        out = cv2.VideoWriter(vidname,fourcc_avi, tp, size)
        for j in range(nt):
            out.write(img_array[j])
        out.release()
        vidname=path+PlotsFolder+'/'+varname+'00.mp4'
        fourcc_mp = cv2.VideoWriter_fourcc(*'MP4V')
        out = cv2.VideoWriter(vidname,fourcc_mp, tp, size)
        for j in range(nt):
            out.write(img_array[j])
        out.release()
    return 1


def create_img_array(path,PlotsFolder,varname,nt=0):
    ''' Load heatmap plots and combine them into an img_array.
    :param ResultsFolder: string for the directory path
    :param varname: string for the preindex part of the variable saved name.
    :param nt: number of time steps to iterate through. default `nt=0`
    
    ```if nt==0:
         while fileexists:
            loadfile and append to img array
       else:
         for j in range(nt):
             loadfile and append to img array
             if not img exists:
                 reassign nt and exit
    ```
    :rtype: array
    :returns: img_array,nt
    '''
    print('Making img_array')
    filename=PlotsFolder+'/'+varname
    img_array=[]
    if nt==0:
        # While loop with if exists exit condition
        filesave=path+filename+ff.numfix(0)+'.jpg'
        while os.path.exists(filesave):
            nt=j
            img = cv2.imread(filesave)
            img_array.append(img)
            j+=1
            filesave=filename+ff.numfix(j)+'.jpg'
    else:
        # Create img_array
        for j in range (nt):
            filesave=path+filename+ff.numfix(j)+'.jpg'
            print(filesave)
            if not os.path.exists(filesave):
                nt=j
                break
            else:
                img = cv2.imread(filesave)
                img_array.append(img)
    return img_array,nt

def Plot_tstep(var,tstep,dt,varname,savedir,vargs):
    v0=vargs['min']
    v1=vargs['max']
    fig, axes = plt.subplots()
    axes.set_aspect('equal')
    colors = tripcolor(var, num_sample_points=1, axes=axes,cmap='inferno',vmin=v0,vmax=v1)
    plt.title('Time = %.3f'%(tstep*dt))
    fig.colorbar(colors)
    fig.savefig(savedir+'/'+varname+ff.numfix(tstep)+'.jpg')
    fig.clear()
    plt.close()
    return
def Plotting(var_list,varname,path,PlotFoldername,parstr,consts_dict,vargs):
    ''' Plot an animation of a tricontour plot for the variable `var_cls`.
    Save plot at path+PlotFoldername+var_cls.name+'.mp4'
    :param var_cls: `SpatialVariable` object with results in.
    :param path: Location for the output
    :param PlotFoldername: directory for the results.
    :param consts_dict: dictionary of the parameters.'''
    dt=consts_dict['dt']
    tstep=0.0
    savedir=path+PlotFoldername+parstr
    Num_Plots=100
    interval=int(len(var_list)/Num_Plots)
    for var in var_list:
        if tstep%interval==0:
                Plot_tstep(var,tstep,dt,varname,savedir,vargs[varname])
        tstep+=1.0
    print('final t',tstep*dt)
    return
def iteration_plot(path,filename,test=0,tp=1,ind=0,simstep=1):
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
    varnames=[etaname,thetaname,tname,domname,etaresname,theresname]
    pars_dict,pdes_pars_dict=ff.read_inputs(path,filename,ind)
    nx=pars_dict['nx']
    ny=pars_dict['ny']
    xm=pars_dict['xm']
    ym=pars_dict['ym']
    x0=pars_dict['x0']
    y0=pars_dict['y0']
    hx=(xm-x0)/nx
    hy=(ym-y0)/ny # mesh size
    # Parameters for the PDEs
    dt=pdes_pars_dict['dt']
    T=pdes_pars_dict['T']
    perm=pdes_pars_dict['perm']
    A0=pdes_pars_dict['A0']
    d=pdes_pars_dict['d']
    eps=pdes_pars_dict['eps']
    parstr=ff.par_string(ind,dt,T,perm,hx,hy,A0,d,eps)
    tvec=ff.load_file(path,MeshFolder+parstr,tname)
    varnames=[thetaname,etaname]
    plot_var_videos(tvec,varnames,path,MeshFolder+parstr,ResultsFolder+parstr,tp)
    print('Plots saved in',path+ResultsFolder+parstr)
    return 
def main(path):
    plot_diff_var(path)
    return
def get_ind(argv):
    job=0 # default jobHeateqn t
    if len(argv)>1:
        job=int(argv[1])
    return job

def plot_diff_var(path):
    filename=path+'/diffusion_range.csv'
    plotname=path+'/ResVariance_vs_Diff.png'
    df=pd.read_csv(filename)
    d=np.array([df['d']])
    var_e=np.array([df['var_e']])
    var_t=np.array([df['var_t']])
    plt.plot(d,var_e,'*',label='eta')
    plt.plot(d,var_t,'*',label='theta')
    plt.title('variance of residual')
    plt.savefig(plotname)
    return
    

if __name__=='__main__':
    #global DEBUG
    DEBUG=True
    path=os.path.abspath(os.getcwd())
    main(path)
    sheetname    ='/ParameterSets.csv'
    test=0
    tp  =5
    #ind =get_ind(sys.argv)
    simstep=1
    for ind in range(0,20):
        iteration_plot(path,sheetname,test,tp,ind,simstep)


