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

import SpatialVariable as SV

def plot_animation(nt,varname='var',path='',ResultsFolder='',tp=1):
    ''' Plotting sequence of heatmaps for a variable `var' and create animation.
    :param domain: numpy array containing the measurements for the envrionment [xmin,ymin,xmax,ymax]
    :param var: numpy array containing variable values
    :param tvec: numpy array containing the time values which correspond with the var results.
    :param varname: string corresponding to the name of var
    :param ResultsFolder: folder where the plots should be save. default \'\'
    :param tp: Time pause for video default 1
    '''
    ff.check_folder(path,ResultsFolder)
    filename=path+ResultsFolder+'/'+varname
    img_array=[]
    for j in range(nt):
        if j%10==0:
            filesave=filename+ff.numfix(j)+'.png'
            #tstring=varname+'(x,y), time {:.2f}'.format(tstep)
            #sns.heatmap(var[:,:,j],vmin=varmin,vmax=varmax).set(title=tstring)
            #im = plt.imshow(var[:,:,j],vmin=varmin,vmax=varmax)
            #im.title(tstring)
            #plt.savefig(filesave)
            #plt.close('all')
            img = cv2.imread(filesave)
            if isinstance(img,type(None)):
                pass
            else:
                height, width, layers = img.shape
                size = (width,height)
                img_array.append(img)
    out = cv2.VideoWriter(filename+'.avi',cv2.VideoWriter_fourcc(*'DIVX'), tp, size)
    for j in range(len(img_array)):
        out.write(img_array[j])
    out.release()
    del img_array
    return

def line_plot(f,t,varname,tname,path='',ResultsFolder=''):
    ''' Plot f as a function of t. label x axis with `tname` 
    and y axis with `varname` save in ResultsFolder'''
    ff.check_folder(path,ResultsFolder)
    if len(t)>len(f):
        t=t[-len(f)-1:-1]
    elif len(f)>len(t):
        f=f[-len(t)-1:-1]
    plt.figure()
    plt.plot(t,f)
    plt.xlabel(tname)
    plt.ylabel(varname)
    plt.savefig(path+ResultsFolder+'/'+varname+'_vs_'+tname+'.png')
    plt.close()
    return

def plot_function(t, varname,j=-1,path='',MeshFolder='/',ResultsFolder='/'):
    """
    Create a figure of the concentration uh warped visualized in 3D at timet step t.
    :param t: time corresponding to the snapshot in the plot
    :param uh_cls: Spatial variable
    :param j: The count of the time step. 
    :param path: location for the Folder to store results.
    :param ResultFolder: The folder for the plots to be stored in
    
    uh_cls is a class object with attributes
        - uh_cls.function_space: The function space V
        - uh_cls.var: dolfinx function on the function space(V)
        - uh_cls.mesh: dolfinx mesh used to create the function space.
        - uh_cls.name: The string name for the variable. 
        - s.char_key: Character indicating the category of variable.
        
    Plots are saved in: path+ResultsFolder+uh_cls.name+numfix(j)+'.png'"""
    if j==-1:
        j=t
    ff.check_folder(path,ResultsFolder)
    filename=path+MeshFolder+'/'+varname
    # Warp mesh by point values
    fileload=filename+ff.numfix(j)+'.pvd'
    plotsave=path+ResultsFolder+'/'+varname+ff.numfix(j)+'.pvd'
    pvd=PVDReader(FileName=fileload)
    tstring=varname+'(x,y), time {:.2f}'.format(t)

    if DEBUG:
        print('Plot saved at',plotsave)
    return


def plot_var_videos(xvar,varnames,path='',MeshFolder='/',ResultsFolder='',tp=1):
    '''Plots output variables from simulation and errors over time
    :param var 3: numpy variable for x-axos(or similar dimensional variable)
    :param varnames: list of names corresponding to variables to plot
    :param ResultsFolder: string indicating the folder path where the plots should be save
    :param tp: Time pause for video default 1
    '''
    for vname in varnames:
        for t,j in enumerate(xvar):
            plot_function(t,j,vname,path,MeshFolder,ResultsFolder)
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
        height, width, layers = img.shape
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
    filename=path+PlotsFolder+'/'+varname
    img_array=[]
    if nt==0:
        # While loop with if exists exit condition
        filesave=path+filename+ff.numfix(0)+'.png'
        while os.path.exists(filesave):
            nt=j
            img = cv2.imread(filesave)
            img_array.append(img)
            j+=1
            filesave=filename+ff.numfix(j)+'.png'
    else:
        # Create img_array
        for j in range (nt):
            filesave=path+filename+ff.numfix(j)+'.png'
            if not os.path.exists(filesave):
                nt=j
                break
            else:
                img = cv2.imread(filesave)
                img_array.append(img)
    return img_array,nt

def plotting_residuals(resvars,resnames,xvar,xvarname,path='',PlotsFolder=''):
    '''Plots output variables from simulation and errors over time
    :param vars: list of numpy variables over space and time
    :param varnames: list of strings matching variable names
    :param xvar: numpy variable matching thee third index of of the variables
    :param xvarnam: The name of the dependence variable
    :param domain: numpy array of the bounds for space [xmin,ymin,xmax,ymax]
    :param PlotsFolder: string indicating the folder path where the plots should be save
    '''
    nt=len(xvar)
    for j,va in enumerate(resvars):
        vname=resnames[j]
        line_plot(va,xvar,vname,xvarname,path,PlotsFolder)
    return 1

def main(path,filename,test=0,tp=1,ind=0):
    ResultsFolder='/Plots'
    MeshFolder  ='/Mesh'
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
    pars,pdes_pars=ff.read_inputs(path,filename,ind)
    nx,ny,xm,ym,x0,y0=pars
    hx=(xm-x0)/nx
    hy=(ym-y0)/ny # mesh size
    # Parameters for the PDEs
    _,_,_,_,_,_,_,_,_,dt,T,perm=pdes_pars
    parstr=ff.par_string(ind,dt,T,perm,hx,hy)
    print('Simulations completed, plotting results')
    tvec=ff.load_file(path,MeshFolder+'/'+parstr,tname)
    eta_res=ff.load_file(path,MeshFolder+'/'+parstr,etaresname)
    line_plot(eta_res,tvec,etaresname,tname,path,ResultsFolder+'/'+parstr)
    del eta_res
    theta_res=ff.load_file(path,MeshFolder+'/'+parstr,theresname)
    line_plot(theta_res,tvec,theresname,tname,path,ResultsFolder+'/'+parstr)
    del theta_res
    varnames=[etaname,thetaname]
    plot_var_videos(tvec,varnames,path,MeshFolder+'/'+parstr,ResultsFolder+'/'+parstr,tp)
    print('Plots saved in',path+ResultsFolder+'/'+parstr)
    return 


if __name__=='__main__':
    global DEBUG
    DEBUG=True
    path=os.path.abspath(os.getcwd())
    sheetname    ='/ParameterSets.csv'
    test=0
    tp  =0.5
    ind =0
    main(path,sheetname,test,tp,ind)


