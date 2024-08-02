#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 16:16:12 2023

@author: asharma
"""
import os
import sys
import time
import numpy as np

import math
import linecache

import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 20
plt.rcParams['axes.linewidth'] = 3
plt.rcParams['font.weight'] = 'bold'# 'normal' #'bold'
#plt.rcParams['axes.labelweight']='bold'
plt.rcParams['axes.labelsize']=25
plt.rcParams['legend.fontsize']=20
plt.rc('grid', linestyle="--", alpha = 0.3)

import dam.mlp_data as MLD
#import concurrent.futures
import time
from itertools import repeat

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

from scipy.stats import gaussian_kde
#-----------------------------------------------------------------------------

def plot_Eparity(fname,xlabel1,ylabel1,title1,pfname1):#Plot liner fit comarison
    f = open(fname,'r')
    d = f.readlines();
    x = [];
    y = [];
    e = [];
    for i in d:
        k = i.split()
        if(len(k)==3):
            x.append(float(k[0]))
            y.append(float(k[1]))
            e.append(float(k[2]))
    f.close()
    del d;
    
    plt.figure(figsize=(10, 10))
    #plt.plot(x,y,'go',markersize=5)
    plt.plot(x,y,'o',markersize=9,alpha=0.4,markeredgewidth=0.0,markerfacecolor='green') # crimson purple teal

    #xy = np.vstack([x,y])
    #z = gaussian_kde(xy)(xy)
    #plt.scatter(x, y, c=z, s=100)

    dmin=min([min(x),min(y)]);
    dmax=max([max(x),max(y)]);
    dtext=(dmax-dmin)/20;        
    #Calculating important parameters and writing them to plot-------------
    rmse= np.sqrt(mean_squared_error(x,y))
    r2  = r2_score(x,y)
    mae = mean_absolute_error(x,y)
    maxd=0;
    for i in range(len(x)):
        if(abs(x[i]-y[i])>maxd):maxd = abs(x[i]-y[i]);
    plt.text(dmin,dmax-1*dtext,'RMSE: '+str(round(rmse,6)))
    plt.text(dmin,dmax-2*dtext,'MAE: '+str(round(mae,6)))
    plt.text(dmin,dmax-3*dtext,'Max error: '+str(round(maxd,6)))
    plt.text(dmin,dmax-4*dtext,'$R^2$: '+str(round(r2,6)))
    #---------------------------------------------
    plt.plot([dmin,dmax],[dmin,dmax],'k',linewidth=3)
    #plt.plot(x,y,'go',markersize=5)
    #plt.xlim([dmin, dmax])
    plt.tick_params(direction='in')
    plt.grid()
    plt.xticks(rotation=45)
    plt.xlabel(xlabel1);plt.ylabel(ylabel1);plt.title(title1)
        
    plt.savefig(pfname1, dpi=300,bbox_inches="tight")
    plt.close()
    
    
def plot_Eparity_details(fname,xlabel1,ylabel1,title1,pfname1):#Plot liner fit comarison
    f = open(fname,'r')
    d = f.readlines();
    x = [];
    y = [];
    for i in d:
        k = i.split()
        if(len(k)==3):
            x.append(float(k[0]))
            y.append(float(k[1]))
    f.close()
    del d;
    
    error = (np.array(x) - np.array(y)) 
    
    fig=plt.figure(figsize=(10, 10),dpi=300)        
    gs = fig.add_gridspec(2,1,height_ratios=(7,2),bottom=0.08,top=0.99,wspace=0.02,hspace=0.02,)
           
    ax = fig.add_subplot(gs[0, 0])
    ax_err = fig.add_subplot(gs[1, 0], sharex=ax)

    dmin=min([min(x),min(y)]);
    dmax=max([max(x),max(y)]);  
    dtext=(dmax-dmin)/15;
    ax.plot([dmin,dmax], [dmin,dmax], alpha=0.8, color="k", linestyle=":",linewidth=3)        

    #colors = ["r", "g", "b", "y", "black", "orange", "black", "purple"]
    #ax.scatter(x,y,color='r',marker="o",alpha=0.5,s=40,) 
    ax.scatter(x,y,marker='o',alpha=0.4,color='green',s=70) # crimson purple teal
    ax_err.scatter(x, error, color='green', marker="o", alpha=0.5, s=30)
    
    #Calculating important parameters and writing them to plot-------------
    rmse= np.sqrt(mean_squared_error(x,y)) 
    r2  = r2_score(x,y)
    mae = mean_absolute_error(x,y) 
    ax.text(dmin,dmax-1*dtext,'RMSE: '+str(round(rmse,6)))
    ax.text(dmin,dmax-2*dtext,'MAE: '+str(round(mae,6)))
    ax.text(dmin,dmax-3*dtext,'Max error: '+str(round(max(abs(error)),6)))
    ax.text(dmin,dmax-4*dtext,'$R^2$: '+str(round(r2,6)))
    
    #---------------------------------------------
    ax.tick_params(bottom=False, labelbottom=False, left=True, labelleft=True)
         
    ax_err.axhline(0.0, color="k", linestyle=":", alpha=0.5)
    ax_err.set_xlabel(xlabel1)
    ax_err.tick_params(axis='x', labelrotation=45)
    ax.set_ylabel(ylabel1)
    ax_err.set_ylabel("$Error$")#("$\Delta$ ")
    ax.tick_params(direction='in')
    ax.grid()
    ax_err.tick_params(direction='in')
    ax_err.grid()
    #ax.legend(loc="center left")
    fig.savefig(pfname1, bbox_inches="tight")
    plt.close(fig);


'''        
    def plot_LFitCompare(self,x,y,xlabel1,ylabel1,title1,pfname1):#Plot liner fit comarison
        plt.figure(figsize=(10, 10))
        plt.plot(x,y,'go',markersize=5)
        #plt.plot(x,y,'g.',markersize=6,alpha=0.6,markeredgewidth=0.0)
        dmin=min([min(x),min(y)]);
        dmax=max([max(x),max(y)]);
        dtext=(dmax-dmin)/20;        
        #Calculating important parameters and writing them to plot-------------
        rmse= np.sqrt(mean_squared_error(x,y))
        #r2  = r2_score(x,y)
        mae = mean_absolute_error(x,y)
        maxd=0;
        for i in range(len(x)):
            if(abs(x[i]-y[i])>maxd):maxd = abs(x[i]-y[i]);
        plt.text(dmin,dmax-1*dtext,'RMSE: '+str(round(rmse,6)))
        plt.text(dmin,dmax-2*dtext,'MAE: '+str(round(mae,6)))
        plt.text(dmin,dmax-4*dtext,'Max error: '+str(round(maxd,6)))
        #plt.text(dmin,dmax-3*dtext,'$R^2$: '+str(round(r2,6)))
        #---------------------------------------------
        plt.plot([dmin,dmax],[dmin,dmax],'k',linewidth=3)
        #plt.plot(x,y,'go',markersize=5)
        plt.xticks(rotation=45)
        plt.xlabel(xlabel1);plt.ylabel(ylabel1);plt.title(title1)
        
        plt.savefig(pfname1, dpi=300,bbox_inches="tight")
        plt.close()
        
    def plot_StressLFitCompare(self,x,y,xlabel1,ylabel1,title1,pfname1):#For stress, show liner fit comarison
        sdft_xx=[];sfit_xx=[];
        sdft_yy=[];sfit_yy=[];
        sdft_zz=[];sfit_zz=[];
        
        sdft_yz=[];sfit_yz=[];
        sdft_xz=[];sfit_xz=[];
        sdft_xy=[];sfit_xy=[];
        for i in range(0,len(x),6):
            sdft_xx.append(x[i]);sfit_xx.append(y[i]);
            sdft_yy.append(x[i+1]);sfit_yy.append(y[i+1]);
            sdft_zz.append(x[i+2]);sfit_zz.append(y[i+2]);
            sdft_yz.append(x[i+3]);sfit_yz.append(y[i+3]);
            sdft_xz.append(x[i+4]);sfit_xz.append(y[i+4]);
            sdft_xy.append(x[i+5]);sfit_xy.append(y[i+5]);
            
        dmin=min([min(x),min(y)])-0.2;
        dmax=max([max(x),max(y)])+0.2;
        dtext=(dmax-dmin)/20;        
        #Calculating important parameters and writing them to plot-------------
        rmse=np.sqrt(mean_squared_error(x,y))
        r2=r2_score(x,y)
        mae = mean_absolute_error(x,y)
        maxd=0;
        for i in range(len(x)):
            if(abs(x[i]-y[i])>maxd):maxd = abs(x[i]-y[i]);
 
        plt.figure(figsize=(10, 10))
        plt.plot(sdft_xx,sfit_xx,'bs',markersize=11,label='xx',alpha=0.8,markeredgewidth=0.0)
        plt.plot(sdft_yy,sfit_yy,'r^',markersize=10,label='yy',alpha=0.6,markeredgewidth=0.0)
        plt.plot(sdft_zz,sfit_zz,'g<',markersize=9,label='zz',alpha=0.8,markeredgewidth=0.0)
        plt.plot(sdft_yz,sfit_yz,'cv',markersize=8,label='yz',alpha=0.7,markeredgewidth=0.0)
        plt.plot(sdft_xz,sfit_xz,'m>',markersize=7,label='xz',alpha=0.6,markeredgewidth=0.0)
        plt.plot(sdft_xy,sfit_xy,'yo',markersize=6,label='xy',alpha=0.6,markeredgewidth=0.0)
        
        plt.legend(loc='lower right')
        
        plt.text(dmin,dmax-1*dtext,'RMSE: '+str(round(rmse,6)))
        plt.text(dmin,dmax-2*dtext,'MAE: '+str(round(mae,6)))
        plt.text(dmin,dmax-3*dtext,'$R^2$: '+str(round(r2,6)))
        plt.text(dmin,dmax-4*dtext,'Max error: '+str(round(maxd,3)))
        #---------------------------------------------
        plt.plot([dmin,dmax],[dmin,dmax],'k',linewidth=3)
        plt.xticks(rotation=45)
        plt.xlabel(xlabel1);plt.ylabel(ylabel1);plt.title(title1)
        plt.savefig(pfname1, dpi=300,bbox_inches="tight")
        plt.close()    
    #
    
    def plot_StressLFitCompare_separate(self,x,y,xlabel1,ylabel1,title1,pfname1):#For stress, show liner fit comarison
        sdft_xx=[];sfit_xx=[];
        sdft_yy=[];sfit_yy=[];
        sdft_zz=[];sfit_zz=[];
        
        sdft_yz=[];sfit_yz=[];
        sdft_xz=[];sfit_xz=[];
        sdft_xy=[];sfit_xy=[];
        for i in range(0,len(x),6):
            sdft_xx.append(x[i]);sfit_xx.append(y[i]);
            sdft_yy.append(x[i+1]);sfit_yy.append(y[i+1]);
            sdft_zz.append(x[i+2]);sfit_zz.append(y[i+2]);
            sdft_yz.append(x[i+3]);sfit_yz.append(y[i+3]);
            sdft_xz.append(x[i+4]);sfit_xz.append(y[i+4]);
            sdft_xy.append(x[i+5]);sfit_xy.append(y[i+5]);
            
        dmin=min([min(x),min(y)]);
        dmax=max([max(x),max(y)]);       
        #Calculating important parameters and writing them to plot-------------

        maxd=0;
        for i in range(len(x)):
            if(abs(x[i]-y[i])>maxd):maxd = abs(x[i]-y[i]);
                    
        fig=plt.figure(dpi=300)        
        gs = fig.add_gridspec(3,2,wspace=0.05,hspace=0.05)
                       
        xx = fig.add_subplot(gs[0, 0])
        xx.plot(sdft_xx,sfit_xx,'b.',markersize=6,label='xx',alpha=0.6,markeredgewidth=0.0)
        xx.plot([dmin,dmax],[dmin,dmax],'k--',linewidth=1)
        
        yy = fig.add_subplot(gs[1, 0],sharex=xx,sharey=xx)
        yy.plot(sdft_yy,sfit_yy,'b.',markersize=6,label='yy',alpha=0.6,markeredgewidth=0.0)
        yy.plot([dmin,dmax],[dmin,dmax],'k--',linewidth=1)
        
        zz = fig.add_subplot(gs[2, 0],sharex=xx,sharey=xx)
        zz.plot(sdft_zz,sfit_zz,'b.',markersize=6,label='zz',alpha=0.6,markeredgewidth=0.0)
        zz.plot([dmin,dmax],[dmin,dmax],'k--',linewidth=1)
        
        xy = fig.add_subplot(gs[0, 1],sharex=xx,sharey=xx)
        xy.plot(sdft_xy,sfit_xy,'b.',markersize=6,label='xy',alpha=0.6,markeredgewidth=0.0)
        xy.plot([dmin,dmax],[dmin,dmax],'k--',linewidth=1)
        
        xz = fig.add_subplot(gs[1, 1],sharex=xx,sharey=xx)
        xz.plot(sdft_xz,sfit_xz,'b.',markersize=6,label='xz',alpha=0.6,markeredgewidth=0.0)
        xz.plot([dmin,dmax],[dmin,dmax],'k--',linewidth=1)
        
        yz = fig.add_subplot(gs[2, 1],sharex=xx,sharey=xx)
        yz.plot(sdft_yz,sfit_yz,'b.',markersize=6,label='yz',alpha=0.6,markeredgewidth=0.0)
        yz.plot([dmin,dmax],[dmin,dmax],'k--',linewidth=1)

        for ax in fig.get_axes():
            ax.label_outer()
            
        for ax in [xx,yy,zz, xy,xz, yz]:
            ax.legend(loc = 'lower right',fontsize=8)
            
            for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(10) 
            for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(10) 
                
            ax.locator_params(axis='both', nbins=6)
            ax.tick_params(axis="x",direction="in")
            ax.tick_params(axis="y",direction="in")
            for axis in ['top','bottom','left','right']:
                ax.spines[axis].set_linewidth(1)
        

        fig.text(0.5, 0.02, xlabel1, ha='center',fontsize=14)
        fig.text(0.02, 0.5, ylabel1, va='center', rotation='vertical',fontsize=14)
        fig.savefig(pfname1, bbox_inches="tight")
    
    #
    def plot_LFitCompare_details(self,x,y,xlabel1,ylabel1,pfname1):#Plot liner fit comarison
        fig=plt.figure(figsize=(10, 10),dpi=300)        
        gs = fig.add_gridspec(3,2,width_ratios=(7, 2),height_ratios=(2, 7, 2),left=0.15,right=0.99,bottom=0.08,top=0.99,wspace=0.02,hspace=0.02,)
               
        ax = fig.add_subplot(gs[1, 0])
        ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
        ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
        ax_err = fig.add_subplot(gs[2, 0], sharex=ax)
        ax_hist_err= fig.add_subplot(gs[2, 1], sharey=ax_err)

        dmin=min([min(x),min(y)]);
        dmax=max([max(x),max(y)]);  
        dtext=(dmax-dmin)/15;
        ax.plot([dmin,dmax], [dmin,dmax], alpha=0.8, color="k", linestyle=":",linewidth=3)        
        #ax.set_xlim([dmin-0.01,dmax+0.01])
        #ax.set_ylim([dmin-0.01,dmax+0.01])

        nbins = 200
        bins = np.linspace(dmin, dmax, nbins + 1)
        histx, _ = np.histogram(x, bins=bins, density=False)
        ax_histx.hist(bins[:-1], bins, weights=histx, color="r", histtype="step", linewidth=2.0,)

        #colors = ["r", "g", "b", "y", "black", "orange", "black", "purple"]
        error = x - y
        histy, _ = np.histogram(y, bins=bins, density=False)
        ax_histy.hist(bins[:-1],bins,weights=histy,color='r',histtype="step",linewidth=2.0,orientation="horizontal",)
        ax.scatter(x,y,color='r',marker="o",alpha=0.5,s=40,)
        ax_err.scatter(x, error, color='r', marker="o", alpha=0.5, s=40)
        
        #Calculating important parameters and writing them to plot-------------
        rmse= np.sqrt(mean_squared_error(x,y))
        r2  = r2_score(x,y)
        mae = mean_absolute_error(x,y)
        ax.text(dmin,dmax-1*dtext,'RMSE: '+str(round(rmse,6)))
        ax.text(dmin,dmax-2*dtext,'MAE: '+str(round(mae,6)))
        ax.text(dmin,dmax-3*dtext,'$R^2$: '+str(round(r2,6)))
        ax.text(dmin,dmax-4*dtext,'Max error: '+str(round(max(abs(error)),6)))
        #---------------------------------------------
        
        bins_err = np.linspace(min(error), max(error), nbins + 1)
        histerr, _ = np.histogram(error, bins=bins_err, density=False)
        ax_hist_err.hist(bins_err[:-1], bins_err, weights=histerr, color="r", histtype="step", linewidth=2.0,orientation="horizontal",)

        ax.tick_params(bottom=False, labelbottom=False, left=True, labelleft=True)
        ax_histx.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
        ax_histy.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
        ax_hist_err.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
        
        for ob in [ax_histx,ax_histy,ax_hist_err]:
            ob.spines['top'].set_visible(False)
            ob.spines['bottom'].set_visible(False)
            ob.spines['left'].set_visible(False)
            ob.spines['right'].set_visible(False)
            
        # ax_histx.spines['top'].set_visible(False)
        # ax_histx.spines['right'].set_visible(False)
        # ax_histy.spines['bottom'].set_visible(False)
        # ax_histy.spines['right'].set_visible(False)
        # ax_hist_err.spines['bottom'].set_visible(False)
        # ax_hist_err.spines['right'].set_visible(False)
        
        ax_err.axhline(0.0, color="k", linestyle=":", alpha=0.5)
        ax_err.set_xlabel(xlabel1)
        ax_err.tick_params(axis='x', labelrotation=45)
        ax.set_ylabel(ylabel1)
        ax_err.set_ylabel("$Error$")#("$\Delta$ ")
        #ax.legend(loc="center left")
        fig.savefig(pfname1, bbox_inches="tight")
        plt.close(fig);
        
    def plot_LFitCompare_file(self,x,y):
        plt.figure(figsize=(10, 10))
        plt.plot(x,y,'r*',markersize=10)
        dmin=min([min(x),min(y)]);
        dmax=max([max(x),max(y)]);
        plt.plot([dmin,dmax],[dmin,dmax],'k',linewidth=3)
        plt.xticks(rotation=45)
        #plt.xlabel(xlabel1);plt.ylabel(ylabel1);plt.title(title1)
        #plt.savefig(pfname1, dpi=300,bbox_inches="tight")
        plt.show()
    def plot_LearningCurve(self,x,y,ylab,ylog,pfname1):
        plt.figure(figsize=(10, 10))
        plt.plot(x,y,'r',marker='*')#,markersize=10)
        plt.xticks(rotation=45)
        plt.xlabel('N Configurations');plt.ylabel(ylab);#plt.title(title1)
        if(ylog):plt.yscale('log')
        plt.savefig(pfname1, dpi=300,bbox_inches="tight")
        #plt.show()
    def plot_ICdist(self,data,xlabel,id_val,pfname1):#Plot distribution of internal coordinates
        plt.figure(figsize=(10, 10))
        for i in range(data.shape[0]):
            plt.plot(data[i,:],'r.',Markersize=0.8);
        for i in id_val:
            plt.plot(data[i,:],'b*');
        plt.xlabel(xlabel);plt.ylabel('#');
        plt.savefig(pfname1, dpi=300,bbox_inches="tight")
        #plt.show()
    def plot_IChist(self,data,n1,xlabel,pfname1):#Plot historgam of an internal coordinate (bond, angle, or dihedral)
        plt.figure(figsize=(10, 10))
        n1,bin1,_=plt.hist(data,bins=np.linspace(min(data),max(data),n1),alpha=0.5,color='steelblue',edgecolor='none',density=True);
        bin_centers = 0.5*(bin1[1:]+bin1[:-1]);
        #dmp=bin1[n1.argmax()];
        plt.plot(bin_centers,n1); ## using bin_centers rather than edges
        plt.xlabel(xlabel);plt.ylabel('#');
        plt.savefig(pfname1, dpi=300,bbox_inches="tight")
        #plt.show()

'''
