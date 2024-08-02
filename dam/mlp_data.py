#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 16:16:12 2022

@author: asharma
"""
import os
import sys
import time
import numpy as np
from ase import Atoms
from ase import io
from ase.geometry.analysis import Analysis
from lammps import lammps;
import math
import linecache
import random

from sklearn.linear_model import Ridge
#from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import scipy.optimize as optimize

import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 20
plt.rcParams['axes.linewidth'] = 3
plt.rcParams['font.weight'] = 'bold'# 'normal' #'bold'
#plt.rcParams['axes.labelweight']='bold'
plt.rcParams['axes.labelsize']=25
plt.rcParams['legend.fontsize']=20

#sys.path.append(os.getcwd())
#from aSNAP_v36 import *

import dam.aSNAP as AS
#-----------------------------------------------------------------------------
#This can create total data set from both 
#->1. DFT of individual snapshots
#->2. AIMD runs

def make_path(path1):    # To check presence and make a directory 
    dir1 = os.path.dirname(path1)
    if not os.path.exists(dir1):
        os.makedirs(dir1)
#

def random_displaced_configurations(fin,fout,dr,Nconfigs,config_type):
    #fin
    #fout
    #config_type= tri_npt, npt, nvt
    fi = open(fin,'r')
    Na = int(fi.readline())
    val = fi.readline().split()
    A = np.asarray([val[2], val[3], val[4]], dtype = np.float64)
    B = np.asarray([val[5], val[6], val[7] ], dtype = np.float64)
    C = np.asarray([val[8], val[9], val[10] ], dtype = np.float64)
    
    F2C=np.asarray([A,B,C]).T
    C2F=np.linalg.inv(F2C)
    
    Rc = np.zeros([Na,3], dtype=np.float64) # Cartesian coordinate
    Rf = np.zeros([Na,3], dtype=np.float64) # Fractional coordinate
    e = [];
    for i in range(Na):
        l = fi.readline().split()
        e.append(l[0])
        Rc[i] = l[1:4] 
        Rf[i] = C2F.dot(Rc[i].T)
        #print(Rc[i],Rf[i],Rf[i][0]*A+Rf[i][2]*B+Rf[i][2]*C)
        #Rf[i]=Rf[i]*(1+0.001)
        #print(Rf[i][0]*A+Rf[i][2]*B+Rf[i][2]*C,Rf[i])
    fi.close()
    Rf = Rf%1
    
    #Creating configurations with random displacements and writing the in files
    fo = open(fout,'w')
    ft = open('cell.tcl','w')
    
    #Writing the first configuration without displacement
    fo.write(f'{Na}\n');
    fo.write(f'snap energy {A[0]} {A[1]} {A[2]} {B[0]} {B[1]} {B[2]} {C[0]} {C[1]} {C[2]}\n')
    cc = AS.Cell()
    a,b,c,alpha,beta,gamma = cc.vec2para(A,B,C)
    ft.write('animate goto 0\n pbc set {'+str(a)+' '+str(b)+' '+str(c)+' '+str(alpha*180/math.pi)+' '+str(beta*180/math.pi)+' '+str(gamma*180/math.pi)+'}\n')
    for i in range(Na):
        r = F2C.dot(Rf[i].T)
        fo.write(f'{e[i]} {r[0]} {r[1]} {r[2]}\n')
    
    dc = 0.005
    #da = 0.005
    for i in range(Nconfigs-1):
        fo.write(f'{Na}\n');
        #Creating random distortion to the cell vectors
        if('npt' in config_type):
            if(config_type=='tri_npt'):
                A = A*np.array([1+random.uniform(-dc,dc) for ii in range(3)])
                B = B*np.array([1+random.uniform(-dc,dc) for ii in range(3)])#B = B*(1+sign*0.0025)
                C = C*np.array([1+random.uniform(-dc,dc) for ii in range(3)])
            else:
                A = A*(1+random.uniform(-dc,dc))
                B = B*(1+random.uniform(-dc,dc)) #pow(-1,random.randrange(2))*0.0025)
                C = C*(1+random.uniform(-dc,dc))

        fo.write(f'snap energy {A[0]} {A[1]} {A[2]} {B[0]} {B[1]} {B[2]} {C[0]} {C[1]} {C[2]}\n')
        a,b,c,alpha,beta,gamma = cc.vec2para(A,B,C)
        ft.write('animate goto '+str(i+1)+'\n pbc set {'+str(a)+' '+str(b)+' '+str(c)+' '+str(alpha*180/math.pi)+' '+str(beta*180/math.pi)+' '+str(gamma*180/math.pi)+'}\n')
        F2C=np.asarray([A,B,C]).T
        C2F=np.linalg.inv(F2C)
        
        for i in range(Na):
            sign = pow(-1,random.randrange(2))
            r = F2C.dot(Rf[i].T)+np.array([random.uniform(-dr,dr) for ii in range(3)])
            rf = C2F.dot(r.T)%1
            r  = F2C.dot(rf.T)
            Rc[i] = r
            #Rf[i] = Rf[i]*np.array([1+random.uniform(-da,da) for ii in range(3)]) #(1+sign*0.002)
            #r = F2C.dot(Rf[i].T)
            fo.write(f'{e[i]} {r[0]} {r[1]} {r[2]}\n')
    
    fo.close();
    ft.close();
#

def random_displaced_configurations_eq(fin,fout,dr,Nconfigs,config_type):
    #It will make randomly displaced configurations around the given structure
    #fin
    #fout
    #config_type= tri_npt, npt, nvt
    
    #Opening the file and reading the first configuration
    fi = open(fin,'r')
    Na = int(fi.readline())
    val = fi.readline().split()
    A = np.asarray([val[2], val[3], val[4]], dtype = np.float64)
    B = np.asarray([val[5], val[6], val[7] ], dtype = np.float64)
    C = np.asarray([val[8], val[9], val[10] ], dtype = np.float64)
    
    F2C=np.asarray([A,B,C]).T
    C2F=np.linalg.inv(F2C)
    
    Rc = np.zeros([Na,3], dtype=np.float64) # Cartesian coordinate
    Rf = np.zeros([Na,3], dtype=np.float64) # Fractional coordinate
    e = []; #For list of elements
    for i in range(Na):
        l = fi.readline().split()
        e.append(l[0])
        Rc[i] = l[1:4] 
        Rf[i] = C2F.dot(Rc[i].T)
        #print(Rc[i],Rf[i],Rf[i][0]*A+Rf[i][2]*B+Rf[i][2]*C)
        #Rf[i]=Rf[i]*(1+0.001)
        #print(Rf[i][0]*A+Rf[i][2]*B+Rf[i][2]*C,Rf[i])
    fi.close()
    Rf = Rf%1
    
    #Creating configurations with random displacements and writing the in files
    fo = open(fout,'w')
    ft = open('cell.tcl','w')
    
    #Writing the first configuration without displacement
    fo.write(f'{Na}\n');
    fo.write(f'snap energy {A[0]} {A[1]} {A[2]} {B[0]} {B[1]} {B[2]} {C[0]} {C[1]} {C[2]}\n')
    cc = AS.Cell()
    a,b,c,alpha,beta,gamma = cc.vec2para(A,B,C)
    ft.write('animate goto 0\n pbc set {'+str(a)+' '+str(b)+' '+str(c)+' '+str(alpha*180/math.pi)+' '+str(beta*180/math.pi)+' '+str(gamma*180/math.pi)+'}\n')
    for i in range(Na):
        r = F2C.dot(Rf[i].T)
        fo.write(f'{e[i]} {r[0]} {r[1]} {r[2]}\n')
    
    dc = 0.005
    #da = 0.005
    for i in range(Nconfigs-1):
        fo.write(f'{Na}\n');
        #Creating random distortion to the cell vectors
        if('npt' in config_type):
            if(config_type=='tri_npt'):
                A1 = A*np.array([1+random.uniform(-dc,dc) for ii in range(3)])
                B1 = B*np.array([1+random.uniform(-dc,dc) for ii in range(3)])#B = B*(1+sign*0.0025)
                C1 = C*np.array([1+random.uniform(-dc,dc) for ii in range(3)])
            else:
                A1 = A*(1+random.uniform(-dc,dc))
                B1 = B*(1+random.uniform(-dc,dc)) #pow(-1,random.randrange(2))*0.0025)
                C1 = C*(1+random.uniform(-dc,dc))

        fo.write(f'snap energy {A1[0]} {A1[1]} {A1[2]} {B1[0]} {B1[1]} {B1[2]} {C1[0]} {C1[1]} {C1[2]}\n')
        a,b,c,alpha,beta,gamma = cc.vec2para(A1,B1,C1)
        ft.write('animate goto '+str(i+1)+'\n pbc set {'+str(a)+' '+str(b)+' '+str(c)+' '+str(alpha*180/math.pi)+' '+str(beta*180/math.pi)+' '+str(gamma*180/math.pi)+'}\n')
        F2C=np.asarray([A1,B1,C1]).T
        C2F=np.linalg.inv(F2C)
        
        for i in range(Na):
            sign = pow(-1,random.randrange(2))
            #r = F2C.dot(Rf[i].T)+np.array([random.uniform(-dr,dr) for ii in range(3)])
            r = Rc[i]+np.array([random.uniform(-dr,dr) for ii in range(3)])
            rf = C2F.dot(r.T)%1
            r  = F2C.dot(rf.T)
            #Rc[i] = r
            #Rf[i] = Rf[i]*np.array([1+random.uniform(-da,da) for ii in range(3)]) #(1+sign*0.002)
            #r = F2C.dot(Rf[i].T)
            fo.write(f'{e[i]} {r[0]} {r[1]} {r[2]}\n')
    
    fo.close();
    ft.close();
#

def axyz2cp2kinp(faxyz,run_dir,cp2k_inp_dir,rm_at):
    #rm_at: T/F: weather to remove atom types from element name or not
    Nc = int(os.popen('cat '+faxyz+' | grep snap | wc -l').read())
    print('Creating ', Nc, ' Configurations')

    fa = open(faxyz,'r')
    make_path(run_dir)
    for i in range(Nc):
        p = run_dir+'snap'+str(i)+'/'
        make_path(p)
        f = open(p+'snap.xyz','w')
        Na = int(fa.readline())
        f.write(f'{Na}\n\n')
        l = fa.readline().split()
        A = [l[2],l[3],l[4]];
        B = [l[5],l[6],l[7]];
        C = [l[8],l[9],l[10]];

        for j in range(Na):
            l = fa.readline().split()
            if(rm_at):
                if(l[0][1] in '0123456789'):el = l[0][0]
                if(l[0][2] in '0123456789'):el = l[0][0:2]
            else:
                el = l[0]
            f.write(f'{el} {l[1]} {l[2]} {l[3]}\n')
        f.close()

        f1=open(p+'cell_top_coor.inc','w')
        f1.write(f"\n&CELL\n A\t\t\t {A[0]} {A[1]} {A[2]} \n B\t\t\t {B[0]} {B[1]} {B[2]} \n C\t\t\t {C[0]} {C[1]} {C[2]} \n PERIODIC\t\t xyz \n MULTIPLE_UNIT_CELL\t 1 1 1 \n&END CELL\n")
        f1.write("\n&TOPOLOGY\n MULTIPLE_UNIT_CELL\t 1 1 1 \n COORD_FILE_NAME \t snap.xyz\n COORD_FILE_FORMAT \t XYZ\n&END TOPOLOGY\n")
        f1.close()
        os.system('cp -r '+cp2k_inp_dir+'* '+p)
    fa.close()


class ReadCp2kFiles():#This class has methods to read different cp2k files and return the values
    #
    def Et(self,fi):#Read total energy
        e=os.popen("cat "+fi+" | grep 'Total energy:' | tail -1 | cut -d ':' -f2 | tr -d ' '");
        et=float(e.read().strip())*27.2113838565563;#>>Total energy in eV (27.211 is to convert hartree (atomic unit) to eV)
        #Eh2=627.509468713739;  # Energy hartree (atomic unit) to kcal/mol
        return et;  # in eV
    #
    def Evdw(self,fi):      
        ev=os.popen("cat "+fi+" | grep 'Dispersion energy:' | tail -1 | cut -d ':' -f2 | tr -d ' '");
        evdw=float(ev.read().strip())*27.2113838565563;#>>VDW energy
        return evdw; #in eV
    #
    def Cell(self,fi,li):
        f=open(fi,'r')
        cell='';cx=1;
        lines_to_read = range(li,li+3);#@^^^^^^^^^^^^^^^^^^^^^^^^ Might Need change
        for pos, l in enumerate(f):
            if pos in lines_to_read:
                cp=l.split()[1:4];
                if(cx==1):cell=cell+cp[0]+' '+cp[1]+' '+cp[2];
                else:cell=cell+' '+cp[0]+' '+cp[1]+' '+cp[2];
                cx+=1;
        f.close();
        return cell
    #
    def Coord(self,fi,Na):
        f = open(fi,'r')
        element=[];
        coord=np.zeros([Na,3],dtype=float);#>>Atomic coordinates
        lines_to_read = range(2,2+Na);
        i=0;
        for pos, l in enumerate(f):
            if pos in lines_to_read:
                element.append(l.split()[0]);
                coord[i]=l.split()[1:4];
                i+=1;
        f.close();
        return element,coord;
    #
    def Force(self,fi,Na):
        f = open(fi,'r')
        fdft=np.zeros([Na,3],dtype=float);#>>Total force
        lines_to_read = range(4,4+Na)
        i=0;
        for pos, l in enumerate(f):
            if pos in lines_to_read:
                fdft[i]=l.split()[3:6];
                #print(pos,fdft[i]);
                i+=1;
        f.close();
        fdft=fdft*51.42208619083232 # To convert force in hartree/bohr to eV/Angstrom
        #Fhb2=1185.821041661538; # Force hartree/bohr to kcal/mol/Angstrom
        return fdft; # in eV/Angstrom
    #
    def Stress(self,fi,volume):
        f = open(fi,'r')
        sdft=np.zeros([3,3],dtype=float);#>>Total stress 
        lines_s = range(3,6)
        i=0;
        for pos, l in enumerate(f):
            if pos in lines_s:
                sdft[i]=l.split()[2:5];
                i+=1;
        f.close();
        sdft=sdft*(volume/160.21766) #GPa to ev
        return sdft; #in eV
    #
    def Charges(self,fi,Na):
        f = open(fi,'r')
        qa=np.zeros([Na],dtype=float);#>>Charges
        lines_to_read = range(2,2+Na)
        i=0;
        for pos, l in enumerate(f):
            if pos in lines_to_read:
                qa[i]=l.split()[4];
                i+=1;
        f.close();
        #print('sum_charges = ',sum(qa))
        return qa;
    #
    def FSvdw(self,fi,Na,volume):
        f = open(fi,'r')
        fvdw=np.zeros([Na,3],dtype=float);#>>VDW forces
        svdw=np.zeros([3,3],dtype=float);#>>VDW stress
        lines_f = range(23+Na,23+2*Na)
        lines_s = range(27+2*Na,27+2*Na+3)
        i1=0;i2=0;
        for pos, l in enumerate(f):
            if pos in lines_f:
                fvdw[i1]=l.split()[2:5];
                #print(pos,fvdw[i1]);
                i1+=1;
            if pos in lines_s:
                svdw[i2]=l.split();
                i2+=1;
        f.close();
        fvdw=fvdw*51.42208619083232 # To convert force in hartree/bohr to eV/Angstrom
        #svdw=svdw*(volume/160.21766) #GPa to ev
        svdw=svdw*27.2113838565563 #a.u. to ev
        return fvdw,svdw;
    #

class CreateDataSet:
    def __init__(self,dft_cal_dir,local_fname,cp2k_files_dir):
        self.dft_cal_dir = dft_cal_dir  #Directory to put input folders for DFT calculations over different snapshots
        make_path(self.dft_cal_dir)
        self.local_fname = local_fname; #Prefix of local directories inside the dft_cal_dir for better identification 
        self.cp2k_files_dir=cp2k_files_dir
        
    def from_MDConfig(self,fname,fcst,lcst,element_list):
        #fcst: fist configuration step
        #lcst: last configuration step
        f=open(fname,'r')
        for i in range(fcst):#Ignoring initial frames
            for j in range(3):f.readline()
            Na=int(f.readline().split()[0]);
            for j in range(Na+5):f.readline();
            
        for i in range(fcst,lcst+1):
            for j in range(3):f.readline();#Ignoring lines
            Na=int(f.readline().split()[0]);
            f.readline();#Ignoring line
            l=f.readline().split();
            xlob=float(l[0]);xhib=float(l[1]);xy=float(l[2]);
            l=f.readline().split();
            ylob=float(l[0]);yhib=float(l[1]);xz=float(l[2]);
            l=f.readline().split();
            zlob=float(l[0]);zhib=float(l[1]);yz=float(l[2]);
            f.readline();#Ignoring line
            
            xlo = xlob - min([0.0,xy,xz,xy+xz])
            xhi = xhib - max([0.0,xy,xz,xy+xz])
            ylo = ylob - min([0.0, yz])
            yhi = yhib + max([0.0, yz])
            zlo = zlob
            zhi = zhib
            
            A=[xhi-xlo,0,0];
            B=[xy,yhi-ylo,0];
            C=[xz,yz,zhi-zlo];
                        
            dir1=self.dft_cal_dir+self.local_fname+'_'+str(i)+'/'
            make_path(dir1)
            
            f1=open(dir1+'cell_top_coor.inc','w')
            f1.write(f"\n&CELL\n A\t\t\t {A[0]} {A[1]} {A[2]} \n B\t\t\t {B[0]} {B[1]} {B[2]} \n C\t\t\t {C[0]} {C[1]} {C[2]} \n PERIODIC\t\t xyz \n MULTIPLE_UNIT_CELL\t 1 1 1 \n&END CELL\n")
            f1.write("\n&TOPOLOGY\n MULTIPLE_UNIT_CELL\t 1 1 1 \n COORD_FILE_NAME \t snap.xyz\n COORD_FILE_FORMAT \t XYZ\n&END TOPOLOGY\n")
            f1.close()
            
            f1=open(dir1+'snap.xyz','w')
            f1.write(f'{Na}\nSnap\n')
            for j in range(Na):
                l=f.readline().split();
                #f1.write(f'{element_list[int(l[1])-1]} {float(l[2])-xlo} {float(l[3])-ylo} {float(l[4])-zlo}\n')
                f1.write(f'{element_list[int(l[1])-1]} {float(l[2])} {float(l[3])} {float(l[4])}\n')
            f1.close()
            os.system('cp '+self.cp2k_files_dir+'main.inp '+dir1)
            os.system('cp '+self.cp2k_files_dir+'sel_basis_pot.inc '+dir1)
            os.system('cp '+self.cp2k_files_dir+'job_control.txt '+dir1)
        
        
#
class MLPDataSet:#To create data set for machine learning potentials
    def __init__(self,dft_data_dir):
        self.dft_data_dir = dft_data_dir
    #
    def fetch_icp2k(self,Na,do_vdw,fname1):#Fetch data from individual cp2k calculations over different snapshots
        print('Fetching data of individual DFT calculations from ',self.dft_data_dir)       
        fa = open(fname1,'w');
        e = os.popen('ls '+self.dft_data_dir);
        fi = e.read().split();
        Nsnap = len(fi);
        readc = ReadCp2kFiles();        
        for s in range(Nsnap):
            snap = fi[s];
            print(snap)
            et = readc.Et(self.dft_data_dir+snap+"/output.txt");
            if(do_vdw):
                evdw = readc.Evdw(self.dft_data_dir+snap+"/output.txt");
            #Reading cell------------------
            cell = readc.Cell(self.dft_data_dir+snap+'/cell_top_coor.inc',2);#Will read cell string from line 2 onwards
            volume = AS.Cell.cellstring2vol(cell);
            #Reading coordinates------------
            element,coord = readc.Coord(self.dft_data_dir+snap+'/snap.xyz',Na)
            #Reading total forces--------
            fdft = readc.Force(self.dft_data_dir+snap+'/forces',Na)
            #Reading stress --------------
            sdft = readc.Stress(self.dft_data_dir+snap+'/stress',volume)
            #Reading charges------------
            if(os.path.exists(self.dft_data_dir+snap+'/DDEC6_even_tempered_net_atomic_charges.xyz')):
                qa = readc.Charges(self.dft_data_dir+snap+'/DDEC6_even_tempered_net_atomic_charges.xyz',Na)
            else:
                qa = np.zeros([Na],dtype=float);#>>Charges
                print('Warning charges not found for snapshot ',snap, ', assigned zero charges')
            #---------------------------
            fa.write(str(Na)+'\n');
            if(do_vdw):
                #Reading VDW forces and stress --------------
                fvdw,svdw = readc.FSvdw(self.dft_data_dir+snap+'/VDWForces',Na,volume)
                stress=sdft-svdw;#print(stress)
                ewrite=et-evdw
            else:               
                stress=sdft;#print(stress)
                ewrite=et
            st=str(stress[0][0])+' '+str(stress[1][1])+' '+str(stress[2][2])+' '+str(stress[1][2])+' '+str(stress[0][2])+' '+str(stress[0][1]);
            fa.write(snap+' '+str(format(ewrite,'.10f'))+' '+cell+' '+st+'\n');
            for i in range(Na):
                if(do_vdw):force=fdft[i]+fvdw[i];
                else:force=fdft[i];
                fa.write(element[i]+' \t'+str(format(coord[i][0],'.7f'))+' \t'+str(format(coord[i][1],'.7f'))+' \t'+str(format(coord[i][2],'.7f'))+'\t'+str(format(force[0],'.10f'))+' \t'+str(format(force[1],'.10f'))+' \t'+str(format(force[2],'.10f'))+' \t'+str(qa[i])+'\n');
        fa.close();
#

    def fetch_aimdcp2k(self,Na,do_vdw,fname1,ef,cellf,coordf,fposcar): #Fetch data from files of AIMD simulations using cp2k v7.1
        #ef: name of energy file
        print('Fetching data of AIMD simulation from ',self.dft_data_dir)
        fa = open(fname1,'w');
        Nsnap=int(os.popen('cat '+self.dft_data_dir+'/'+ef+' | wc -l').read().strip())-1;
        fe = open(self.dft_data_dir+'/'+ef,'r')
        fce = open(self.dft_data_dir+'/'+cellf,'r')
        fco = open(self.dft_data_dir+'/'+coordf,'r')
        ff = open(self.dft_data_dir+'/forces','r')
        fs = open(self.dft_data_dir+'/stress','r')
        fp = open(self.dft_data_dir+'/'+fposcar,'r')
        fv = open(self.dft_data_dir+'/VDWForces','r')
        
        fe.readline();
        fce.readline();
        print('hii its here',Nsnap)
        #for s in range(2):
        for s in range(Nsnap):
            print(s)
            et=float(fe.readline().split()[4])*27.2113838565563;#>>Total energy in eV (27.211 is to convert hartree (atomic unit) to eV)
            #Reading cell------------------
            l = fce.readline().split()
            cell=l[2]+' '+l[3]+' '+l[4]+' '+l[5]+' '+l[6]+' '+l[7]+' '+l[8]+' '+l[9]+' '+l[10]+' ';
            volume = float(l[11]);
               
            #Use these few lines if the entries in cellf files are not reliable due to use of dcd file
            cell='';
            for i in range(2):fp.readline();
            for i in range(3):
                l=fp.readline().split();
                cell=cell+' '+l[0]+' '+l[1]+' '+l[2];
            for i in range(2+Na):fp.readline();
            
            #Reading coordinates------------
            element=[];
            coord=np.zeros([Na,3],dtype=float);#>>Atomic coordinates
            for i in range(2):fco.readline();
            for i in range(Na):
                l=fco.readline().split();
                #print(l)
                element.append(l[0]);
                coord[i]=l[1:4];
                
            #Reading total forces--------
            fdft=np.zeros([Na,3],dtype=float);#>>Total force
            for i in range(4):ff.readline();
            for i in range(Na):
                l=ff.readline();
                #print(l)
                fdft[i]=l.split()[3:6];
            fdft=fdft*51.42208619083232 # To convert force in hartree/bohr to eV/Angstrom
            ff.readline();#Ignoring line mentioning sum
            
            #Reading stress --------------
            sdft=np.zeros([3,3],dtype=float);#>>Total stress 
            for i in range(4):fs.readline();
            for i in range(3):
                l=fs.readline();
                sdft[i]=l.split()[1:4];
            sdft=sdft*(volume/160.21766);#GPa to ev
            for i in range(27):fs.readline();#Ignoring other lines having details
            
            #Reading charges------------
            qa=np.zeros([Na],dtype=float);#>>Charges
            #---------------------------
            fa.write(str(Na)+'\n');
            
            
            if(do_vdw):
                #Reading VDW forces and stress --------------
                for i in range(18+Na):l=fv.readline();
                evdw = float(fv.readline().split()[5])*27.2113838565563;
                for i in range(4):l=fv.readline();
                
                fvdw=np.zeros([Na,3],dtype=float);#>>VDW forces
                svdw=np.zeros([3,3],dtype=float);#>>VDW stress
                for i in range(Na):
                    l=fv.readline();
                    fvdw[i]=l.split()[2:5];
                for i in range(4):l=fv.readline();
                for i in range(3):
                    l=fv.readline();
                    svdw[i]=l.split();
                    #print('>',l,l.split()[2:5],'.....')
                for i in range(2):l=fv.readline();
                
                print(svdw[0])
                fvdw=fvdw*51.42208619083232 # To convert force in hartree/bohr to eV/Angstrom
                svdw=svdw*27.2113838565563 #a.u. to ev
                
                stress=sdft-svdw;#print(stress)
                ewrite=et-evdw
                print(svdw[0],'---')
            else:               
                stress=sdft;#print(stress)
                ewrite=et
            
            
            st=str(stress[0][0])+' '+str(stress[1][1])+' '+str(stress[2][2])+' '+str(stress[1][2])+' '+str(stress[0][2])+' '+str(stress[0][1]);
            fa.write('snap_aimd'+str(s)+' '+str(format(ewrite,'.10f'))+' '+cell+' '+st+'\n');
            for i in range(Na):
                if(do_vdw):force = fdft[i] + fvdw[i];
                else:force = fdft[i];
                #force = fdft[i]
                fa.write(element[i]+' \t'+str(format(coord[i][0],'.7f'))+' \t'+str(format(coord[i][1],'.7f'))+' \t'+str(format(coord[i][2],'.7f'))+'\t'+str(format(force[0],'.10f'))+' \t'+str(format(force[1],'.10f'))+' \t'+str(format(force[2],'.10f'))+' \t'+str(qa[i])+'\n');
        
        fe.close();
        fce.close();
        ff.close();
        fs.close();
        fa.close();
        fp.close();
        fv.close();
#
    def fetch_aimdcp2k_noposcar(self,Na,do_vdw,fname1,ef,cellf,coordf): #Fetch data from files of AIMD simulations using cp2k v7.1
        #ef: name of energy file
        print('Fetching data of AIMD simulation from ',self.dft_data_dir)
        fa = open(fname1,'w');
        Nsnap=int(os.popen('cat '+self.dft_data_dir+'/'+ef+' | wc -l').read().strip())-1;
        fe = open(self.dft_data_dir+'/'+ef,'r')
        fce = open(self.dft_data_dir+'/'+cellf,'r')
        fco = open(self.dft_data_dir+'/'+coordf,'r')
        ff = open(self.dft_data_dir+'/forces','r')
        fs = open(self.dft_data_dir+'/stress','r')
        if(do_vdw):fv = open(self.dft_data_dir+'/VDWForces','r')
        
        fe.readline();
        fce.readline();
        print('hii its here',Nsnap)
        #for s in range(2):
        for s in range(Nsnap):
            print(s)
            et=float(fe.readline().split()[4])*27.2113838565563;#>>Total energy in eV (27.211 is to convert hartree (atomic unit) to eV)
            #Reading cell------------------
            l = fce.readline().split()
            
            #Transforming cell (and corresponding coordinates,forces, and stress) such that A vector is along x axis
            #This is necessary to make data compatible with LAMMPS
            if(float(l[3])==0.0 and float(l[4])==0.0):
                An=np.asarray([l[2], l[3], l[4]], dtype = np.float64)
                Bn=np.asarray([l[5], l[6], l[7] ], dtype = np.float64)
                Cn=np.asarray([l[8], l[9], l[10] ], dtype = np.float64)
                T = np.identity(3);                
            else:
                A=np.asarray([l[2], l[3], l[4]], dtype = np.float64)
                B=np.asarray([l[5], l[6], l[7] ], dtype = np.float64)
                C=np.asarray([l[8], l[9], l[10] ], dtype = np.float64)
                
                a=np.sqrt(A.dot(A));
                b=np.sqrt(B.dot(B));
                c=np.sqrt(C.dot(C));
                alpha=math.acos(B.dot(C)/b/c);
                beta=math.acos(A.dot(C)/a/c);
                gamma=math.acos(B.dot(A)/a/b);
            
                F2C=np.asarray([A,B,C]).T
                C2F=np.linalg.inv(F2C)
            
                An=np.asarray([a ,0.,0.])
                bx = b*math.cos(gamma)
                by = b*math.sin(gamma)
                Bn = np.asarray([bx ,by,0.])
                cx = c*math.cos(beta)
                cy = (np.dot(B,C)-bx*cx)/by
            
                Cn=np.asarray([cx, cy, math.sqrt(c*c - cx*cx -cy*cy)])
                F2Cn = np.asarray([An,Bn,Cn]).T
                T = np.matmul(F2Cn,C2F)
            #--------------------------------------------------------------------------------------------------------           
            
            #cell=l[2]+' '+l[3]+' '+l[4]+' '+l[5]+' '+l[6]+' '+l[7]+' '+l[8]+' '+l[9]+' '+l[10]+' ';
            cell=f'{An[0]} {An[1]} {An[2]} {Bn[0]} {Bn[1]} {Bn[2]} {Cn[0]} {Cn[1]} {Cn[2]} ';
            volume = float(l[11]);
            
            #Reading coordinates------------
            element=[];
            coord=np.zeros([Na,3],dtype=float);#>>Atomic coordinates
            for i in range(2):fco.readline();
            for i in range(Na):
                l=fco.readline().split();
                #print(l)
                element.append(l[0]);
                c_na = np.asarray(l[1:4],dtype=np.float64)#
                coord[i] = T.dot(c_na.T); #Transforming coordinate to aligned coordinate system
                
            #Reading total forces--------
            fdft=np.zeros([Na,3],dtype=float);#>>Total force
            for i in range(4):ff.readline();
            for i in range(Na):
                l=ff.readline();
                #print(l)
                f_na = np.asarray(l.split()[3:6],dtype=np.float64)#
                fdft[i] = T.dot(f_na.T);  #Transforming forces to aligned coordinate system
            fdft=fdft*51.42208619083232 # To convert force in hartree/bohr to eV/Angstrom
            ff.readline();#Ignoring line mentioning sum
            
            #Reading stress --------------
            sdft=np.zeros([3,3],dtype=float);#>>Total stress 
            for i in range(4):fs.readline();
            for i in range(3):
                l=fs.readline();
                sdft[i]=l.split()[1:4];
            sdft = np.matmul(T,np.matmul(sdft,T.T))#Transforming stress to aligned coordinate system
            sdft=sdft*(volume/160.21766);#GPa to ev
            
            for i in range(27):fs.readline();#Ignoring other lines having details
            
            #Reading charges------------
            qa=np.zeros([Na],dtype=float);#>>Charges
            #---------------------------
            fa.write(str(Na)+'\n');
            
            
            if(do_vdw):
                #Reading VDW forces and stress --------------
                for i in range(18+Na):l=fv.readline();
                evdw = float(fv.readline().split()[5])*27.2113838565563;
                for i in range(4):l=fv.readline();
                
                fvdw=np.zeros([Na,3],dtype=float);#>>VDW forces
                svdw=np.zeros([3,3],dtype=float);#>>VDW stress
                for i in range(Na):
                    l=fv.readline();
                    f_na = np.asarray(l.split()[2:5],dtype=np.float64)#
                    fvdw[i]=T.dot(f_na.T);  #Transforming forces to aligned coordinate system
                    
                for i in range(4):l=fv.readline();
                for i in range(3):
                    l=fv.readline();
                    svdw[i]=l.split();
                svdw = np.matmul(T,np.matmul(svdw,T.T))#Transforming stress to aligned coordinate system
                    #print('>',l,l.split()[2:5],'.....')
                for i in range(2):l=fv.readline();
                
                print(svdw[0])
                fvdw=fvdw*51.42208619083232 # To convert force in hartree/bohr to eV/Angstrom
                svdw=svdw*27.2113838565563 #a.u. to ev
                
                stress=sdft-svdw;#print(stress)
                ewrite=et-evdw
                print(svdw[0],'---')
            else:               
                stress=sdft;#print(stress)
                ewrite=et
            #-------------------------------------------------------------------------------------------------------
            
            st=str(stress[0][0])+' '+str(stress[1][1])+' '+str(stress[2][2])+' '+str(stress[1][2])+' '+str(stress[0][2])+' '+str(stress[0][1]);
            fa.write('snap_aimd'+str(s)+' '+str(format(ewrite,'.10f'))+' '+cell+' '+st+'\n');
            for i in range(Na):
                if(do_vdw):force = fdft[i] + fvdw[i];
                else:force = fdft[i];
                #force = fdft[i]
                fa.write(element[i]+' \t'+str(format(coord[i][0],'.7f'))+' \t'+str(format(coord[i][1],'.7f'))+' \t'+str(format(coord[i][2],'.7f'))+'\t'+str(format(force[0],'.10f'))+' \t'+str(format(force[1],'.10f'))+' \t'+str(format(force[2],'.10f'))+' \t'+str(qa[i])+'\n');
        
        fe.close();
        fce.close();
        ff.close();
        fs.close();
        fa.close();
        if(do_vdw):fv.close();
#
        
    def axyz2poscar(self,faxyz,fposcar):
        Nconfigs=int(os.popen('cat '+faxyz+' | grep snap | wc -l').read().strip());
        fx=open(faxyz,'r');
        fpp=open(fposcar,'w');
        
        A=np.zeros(3,dtype=float);#>>A vector
        B=np.zeros(3,dtype=float);#>>B vector
        C=np.zeros(3,dtype=float);#>>C vector        
        for s in range(Nconfigs):
            print('Reading training snapshot #',s)
            Na=int(fx.readline().split()[0]);            
            l2=fx.readline().split();
            A=[float(k) for k in l2[2:5]];B=[float(k) for k in l2[5:8]];C=[float(k) for k in l2[8:11]];
            
            elements=[];
            coord=np.zeros([Na,3],dtype=float);#>>Atomic coordinates            
            for i in range(Na):
                l=fx.readline().split();
                elements.append(l[0]);
                coord[i]=l[1:4];
        
            e=np.unique(elements);
            for i in e:fpp.write(i+' ')
            fpp.write('\n1.00\n')
            fpp.write(f'  {A[0]}  {A[1]}  {A[2]}\n  {B[0]}  {B[1]}  {B[2]}\n  {C[0]}  {C[1]}  {C[2]}\n')
            for i in e:fpp.write(f' {elements.count(i)}')
            fpp.write('\nCartesian\n')
            for i in range(Na):
                fpp.write(f'\t {coord[i][0]} \t{coord[i][1]} \t{coord[i][2]}\n')
        fx.close()
        fpp.close();
        
    def axyz2cell(self,faxyz,fcell):
        Nconfigs=int(os.popen('cat '+self.dft_data_dir+'/'+faxyz+' | grep snap | wc -l').read().strip());
        fx=open(self.dft_data_dir+'/'+faxyz,'r');
        fpp=open(self.dft_data_dir+'/'+fcell,'w');
        
        cc=AS.Cell();
        A=np.zeros(3,dtype=float);#>>A vector
        B=np.zeros(3,dtype=float);#>>B vector
        C=np.zeros(3,dtype=float);#>>C vector        
        for s in range(Nconfigs):
            print('Reading training snapshot #',s)
            Na=int(fx.readline().split()[0]);            
            l2=fx.readline().split();
            A=[float(k) for k in l2[2:5]];B=[float(k) for k in l2[5:8]];C=[float(k) for k in l2[8:11]];
                      
            for i in range(Na):
                l=fx.readline().split();
            a,b,c,alpha,beta,gamma=cc.vec2para(A, B, C)
            fpp.write(f'animate goto {s}\n')
            #fpp.write('{a}  {b}  {c} {alpha*180/math.pi}  {beta*180/math.pi}  {gamma*180/math.pi}\n')
            fpp.write('pbc set {'+str(a)+' '+str(b)+' '+str(c)+' '+str(alpha*180/math.pi)+' '+str(beta*180/math.pi)+' '+str(gamma*180/math.pi)+' }'+'\n')
        fx.close()
        fpp.close();
        
    #
#

class MLSetAnalysis:#To analyze the data set and split it in training and test sets
    def __init__(self,fname, special_indexes):
        self.fname=fname; #File contining details of all the snapshots
        
        self.Ndset = int(os.popen('cat '+self.fname+' | grep snap | wc -l').read().strip()); #Total number of snapshots 
        #Reading cell vectors of first configuration-------------------------
        f = open(self.fname,'r')
        f.readline()
        l=f.readline().split()
        self.Aexp=[float(l[2]), float(l[3]), float(l[4])];
        self.Bexp=[float(l[5]), float(l[6]), float(l[7])];
        self.Cexp=[float(l[8]), float(l[9]), float(l[10])];
        f.close()
        #--------------------------------------------------------------------
        self.S=io.read(fname,format='xyz',index=0)
        self.S.set_cell([(self.Aexp[0],self.Aexp[1],self.Aexp[2]), (self.Bexp[0],self.Bexp[1],self.Bexp[2]), (self.Cexp[0],self.Cexp[1],self.Cexp[2])])
        self.S.set_pbc(True)
        self.special_indexes=special_indexes
        print(self.Aexp, self.Bexp, self.Cexp)
    #
    def get_atmtyp(self,fsorted):
        ubi,uai,udi,ana=self.get_BADindex()
        ubv,uav,udv=self.get_BADval(ana,ubi,uai,udi)        
        atom_sym=self.S.get_chemical_symbols()
        Natoms=len(atom_sym)
        atom_name=[];
        nblist=[[] for i in range(Natoms)]#*Natoms;
        nbv=[[] for i in range(Natoms)]#*Natoms;
        
        #for x in range(len(udi)):
        #    print(udi[x],udv[x])
        
        #Finding the connectivity of each element
        freq=[0]*len(atom_sym);
        for b in ubi:
            for b1 in b:
                freq[b1] += 1;
        
        for c in range(len(atom_sym)):#Modifying chemical symbol to symbol+connectivity for better understanding
            atom_name.append(atom_sym[c]+str(freq[c]))
        
        for b in range(len(ubi)):
            b0=ubi[b];
            nblist[b0[0]].append(atom_name[b0[1]])
            nblist[b0[1]].append(atom_name[b0[0]])
            nbv[b0[0]].append(ubv[b])
            nbv[b0[1]].append(ubv[b])
         
        for c in range(len(nblist)):#Sorting alphabatically
            sv=sorted(zip(nblist[c],nbv[c]))
            nblist[c]=[s for s, v in sv]
            nbv[c]=[v for s, v in sv]
            #nblist[c].sort();
         
        #Finding atom types--------------------------------------
        atm_typ=[];
        for c in range(len(atom_name)):#Loop over all atoms
            #print(c,atom_name[c],freq[c],nblist[c],nbv[c])
            present=0;
            for a in atm_typ:                                                          
                if(a[0]==atom_name[c] and a[1]==freq[c] and a[2]==nblist[c]):
                    present=1;
                    if(np.linalg.norm(np.asarray(a[3])-np.asarray(nbv[c]))<0.01):
                        present=1;                        
            if(present==0):
                atm_typ.append([atom_name[c],freq[c],nblist[c],nbv[c]])
                #    print('-- > ',atm_typ)
        
        #Cleaning name of atom types---------------------------       
        atm_typ_log=[];
        for p in range(len(atm_typ)):
            atm_typ_log.append(atm_typ[p][0])
            atm_typ[p].append(chr(96+atm_typ_log.count(atm_typ[p][0])))
            #p[0]+=chr(96+atm_typ_log.count(p[0])) #Adding a,b,.. at the end of name 
            print(atm_typ[p],'///')
        
        #Cleaning name of all atoms with correct atom types-----------------
        for c in range(len(atom_name)):#Loop over all atoms
            for a in atm_typ:
                if(a[0]==atom_name[c] and a[1]==freq[c] and a[2]==nblist[c]):
                    if(np.linalg.norm(np.asarray(a[3])-np.asarray(nbv[c]))<0.03): #Note: Sometimes you need it 0.01
                        atom_name[c]+=a[4]
                #if(a[0]==atom_name[c] and a[1]==freq[c]):print('->',a[0],a[1],a[2],nblist[c])
        print(atom_name)                
        unique_atom_types = sorted(list(set(atom_name)))
        
        #Writing the sorted xyz file
        Coord = self.S.get_positions()
        fx = open(fsorted,'w')
        f1 = open('order_change.txt','w')
        cell=self.S.get_cell()
        fx.write(f'{Natoms} \nsnap energy {cell[0][0]} {cell[0][1]} {cell[0][2]} {cell[1][0]} {cell[1][1]} {cell[1][2]} {cell[2][0]} {cell[2][1]} {cell[2][2]}\n')
        for at in unique_atom_types:
            for pat in range(Natoms):
                if(at == atom_name[pat]):
                    #fx.write(f'{atom_sym[pat]} {Coord[pat][0]} {Coord[pat][1]} {Coord[pat][2]}\n')
                    fx.write(f'{at} {Coord[pat][0]} {Coord[pat][1]} {Coord[pat][2]}\n')
                    f1.write(f'{at} {pat}\n')
        fx.close()
        f1.close()
    #
    
        
    def get_BADindex(self):
        #AO: ASE atoms object
        ana=Analysis(self.S)
        ub=ana.unique_bonds
        ua=ana.unique_angles
        ud=ana.unique_dihedrals    
        ubi_list=[]; uai_list=[]; udi_list=[];
        for i in range(len(ub[0])):  
            for j in ub[0][i]:ubi_list.append([i,j]) # get_bond_value(0,[i,j])
            for j in ua[0][i]:uai_list.append([i,j[0],j[1]]) #ana.get_angle_value(0,[i,j[0],j[1]])
            for j in ud[0][i]:udi_list.append([i,j[0],j[1],j[2]]) #ana.get_dihedral_value(0,[i,j[0],j[1],j[2]])
        return ubi_list,uai_list,udi_list,ana;
    #
    # =============================================================================
    
    def get_BADval(self,GAO,bl,al,dl):
         #GAO: ASE geometry analysis object
         #bl,al,dl: List of indexes for bonds, angles, and dihedrals
         bval=np.zeros(len(bl));
         aval=np.zeros(len(al));
         dval=np.zeros(len(dl));
         for j in range(len(bl)):bval[j]=GAO.get_bond_value(0,bl[j][0:2],mic=True)
         for j in range(len(al)):aval[j]=GAO.get_angle_value(0,al[j][0:3],mic=True)
         for j in range(len(dl)):dval[j]=GAO.get_dihedral_value(0,dl[j][0:4],mic=True)
         return bval,aval,dval;
    ##Alternative way
    #def get_BADval(AO,bl,al,dl):
    #    #GAO: ASE atom object
    #    #bl,al,dl: List of indexes for bonds, angles, and dihedrals
    #    bval=np.zeros(len(bl));
    #    for j in range(len(bl)):bval[j]=AO.get_distance(bl[j][0],bl[j][1],mic=True)
    #    aval=AO.get_angles(al,mic=True)
    #    dval=AO.get_dihedrals(dl,mic=True)
    #    return bval,aval,dval;
    # =============================================================================



    def get_unique_BADindex(self):
        ubi,uai,udi,ana=self.get_BADindex()
        ubv,uav,udv=self.get_BADval(ana,ubi,uai,udi)        
        chem_sym=self.S.get_chemical_symbols()
        
        #for x in range(len(udi)):
        #    print(udi[x],udv[x])
        #Finding the connectivity of each element
        freq=[0]*len(chem_sym);
        for b in ubi:
            for b1 in b:
                freq[b1] += 1;
        
        for c in range(len(chem_sym)):#Modifying chemical symbol to symbol+connectivity for better understanding
            chem_sym[c] += str(freq[c])
 
        #----------------------------------------------
        usym_bonds=[];#List of unique and symmetric bonds
        ub_abl=[];#Unique bond atoms and bond length
        for b0 in range(len(ubi)):
            b=ubi[b0];
            au=True;
            for b1 in range(len(ub_abl)):
                #print('xx ',b1,len(ub_abl),ub_abl)
                bl=ub_abl[b1]
                if( ((chem_sym[b[0]]==bl[0] and chem_sym[b[1]]==bl[1]) or (chem_sym[b[0]]==bl[1] and chem_sym[b[1]]==bl[0]) ) and abs(ubv[b0]-bl[2])<0.01):
                    #print(b,bl,chem_sym[b[0]],chem_sym[b[1]],b[0],b[1],abs(ubv[b0]-bl[2]),ubv[b0],bl[2],ana.get_bond_value(0,b, mic=True))
                    usym_bonds[b1].append(b)
                    au=False;                    
            if(au):
                ub_abl.append([chem_sym[b[0]], chem_sym[b[1]], ana.get_bond_value(0,b, mic=True)])
                usym_bonds.append([b])                
        #----------------------------------------------
        
        usym_angles=[];#List of unique and symmetric bonds
        ua_aav=[];#Unique angle atoms and angle value
        for a0 in range(len(uai)):
            a=uai[a0];
            au=True;
            for a1 in range(len(ua_aav)):
                #print('xx ',b1,len(ub_abl),ub_abl)
                av=ua_aav[a1]
                if(chem_sym[a[1]]==av[1]):
                    #print(chem_sym[a[1]],av[1],a,av)
                    if( ((chem_sym[a[0]]==av[0] and chem_sym[a[2]]==av[2]) or (chem_sym[a[0]]==av[2] and chem_sym[a[2]]==av[0]) ) and abs(uav[a0]-av[3])<1):
                        #print(b,bl,chem_sym[b[0]],chem_sym[b[1]],b[0],b[1],abs(ubv[b0]-bl[2]),ubv[b0],bl[2],ana.get_bond_value(0,b, mic=True))
                        usym_angles[a1].append(a)
                        au=False;                    
            if(au):
                ua_aav.append([chem_sym[a[0]], chem_sym[a[1]], chem_sym[a[2]], ana.get_angle_value(0,a, mic=True)])
                #print(ua_aav)
                usym_angles.append([a])                
         #---------------------------------------------- 

        usym_dih=[];#List of unique and symmetric bonds
        ud_adv=[];#Unique dihedral atoms and dihedral angle value
        for d0 in range(len(udi)):
            d=udi[d0];
            au=True;
            for d1 in range(len(ud_adv)):
                #print('xx ',b1,len(ub_abl),ub_abl)
                dv=ud_adv[d1]
                if( (chem_sym[d[1]]==dv[1] and chem_sym[d[2]]==dv[2]) or (chem_sym[d[1]]==dv[2] and chem_sym[d[2]]==dv[1])):
                    #print(chem_sym[a[1]],av[1],a,av)
                    #if( ((chem_sym[d[0]]==dv[0] and chem_sym[d[2]]==dv[2]) or (chem_sym[d[0]]==dv[2] and chem_sym[d[2]]==dv[0]) ) and abs(udv[d0]-dv[4])<1):
                    if( ((chem_sym[d[0]]==dv[0] and chem_sym[d[3]]==dv[3]) or (chem_sym[d[0]]==dv[3] and chem_sym[d[3]]==dv[0]) ) and abs(udv[d0]-dv[4])<1):
                        #print(b,bl,chem_sym[b[0]],chem_sym[b[1]],b[0],b[1],abs(ubv[b0]-bl[2]),ubv[b0],bl[2],ana.get_bond_value(0,b, mic=True))
                        usym_dih[d1].append(d)
                        au=False;                    
            if(au):
                ud_adv.append([chem_sym[d[0]], chem_sym[d[1]], chem_sym[d[2]], chem_sym[d[3]], ana.get_dihedral_value(0,d, mic=True)])
                #print(ua_aav)
                usym_dih.append([d])                
         #---------------------------------------------- 
         
        for x in range(len(ub_abl)):
            print(x,ub_abl[x],len(usym_bonds[x]),len(ubi))
            #print(usym_bonds[x])
        print('\n')
        for x in range(len(ua_aav)):
            print(x,ua_aav[x],len(usym_angles[x]),len(uai))
        print('\n')
        for x in range(len(ud_adv)):
            print(x,ud_adv[x],len(usym_dih[x]),len(udi))  
            
        #self.get_unique_BADval(ana, usym_bonds, usym_angles, usym_dih)
        return usym_bonds,usym_angles,usym_dih,ana;
    #
    def get_unique_BADval(self,GAO,bl,al,dl):
         #GAO: ASE geometry analysis object
         #bl,al,dl: List of list of indexes for equivalent bonds, angles, and dihedrals
         bval=[];
         for i in range(len(bl)):
             bval.append([])
             for b1 in bl[i]:
                 bval[i].append(GAO.get_bond_value(0,b1,mic=True))
         aval=[];
         for i in range(len(al)):
             aval.append([])
             for a1 in al[i]:
                 aval[i].append(GAO.get_angle_value(0,a1,mic=True))
         dval=[];
         for i in range(len(dl)):
             dval.append([])
             for d1 in dl[i]:
                 dval[i].append(GAO.get_dihedral_value(0,d1,mic=True))
                 
         return bval,aval,dval;
    #
    
    def get_environ_BADindex(self):
        ubi,uai,udi,ana=self.get_BADindex()
        ubv,uav,udv=self.get_BADval(ana,ubi,uai,udi)        
        atom_sym=self.S.get_chemical_symbols()
        Natoms=len(atom_sym)
        atom_name=[];
        nblist=[[] for i in range(Natoms)]#*Natoms;
        nbv=[[] for i in range(Natoms)]#*Natoms;
        
        #for x in range(len(udi)):
        #    print(udi[x],udv[x])
        
        #Finding the connectivity of each element
        freq=[0]*len(atom_sym);
        for b in ubi:
            for b1 in b:
                freq[b1] += 1;
        
        for c in range(len(atom_sym)):#Modifying chemical symbol to symbol+connectivity for better understanding
            atom_name.append(atom_sym[c]+str(freq[c]))
        
        for b in range(len(ubi)):
            b0=ubi[b];
            nblist[b0[0]].append(atom_name[b0[1]])
            nblist[b0[1]].append(atom_name[b0[0]])
            nbv[b0[0]].append(ubv[b])
            nbv[b0[1]].append(ubv[b])
         
        for c in range(len(nblist)):#Sorting alphabatically
            sv=sorted(zip(nblist[c],nbv[c]))
            nblist[c]=[s for s, v in sv]
            nbv[c]=[v for s, v in sv]
            #nblist[c].sort();
         
        #Finding atom types--------------------------------------
        atm_typ=[];
        for c in range(len(atom_name)):#Loop over all atoms
            #print(c,atom_name[c],freq[c],nblist[c],nbv[c])
            present=0;
            for a in atm_typ:                                                          
                if(a[0]==atom_name[c] and a[1]==freq[c] and a[2]==nblist[c]):
                    present=1;
                    if(np.linalg.norm(np.asarray(a[3])-np.asarray(nbv[c]))<0.01):
                        present=1;                        
            if(present==0):
                atm_typ.append([atom_name[c],freq[c],nblist[c],nbv[c]])
                #    print('-- > ',atm_typ)
        
        #Cleaning name of atom types---------------------------       
        atm_typ_log=[];
        for p in range(len(atm_typ)):
            atm_typ_log.append(atm_typ[p][0])
            atm_typ[p].append(chr(96+atm_typ_log.count(atm_typ[p][0])))
            #p[0]+=chr(96+atm_typ_log.count(p[0])) #Adding a,b,.. at the end of name 
            print(atm_typ[p])
        
        #Cleaning name of all atoms with correct atom types-----------------
        for c in range(len(atom_name)):#Loop over all atoms
            for a in atm_typ:
                if(a[0]==atom_name[c] and a[1]==freq[c] and a[2]==nblist[c]):
                    if(np.linalg.norm(np.asarray(a[3])-np.asarray(nbv[c]))<0.01):
                        atom_name[c]+=a[4]
        
        #Finding bonds/angles/dihedrals associated with one type of atom
        all_pairs=[[[],[],[],[]] for _ in range(Natoms)]# Atom_name, bond, angle, dihedrals
        all_pairs_val=[[[],[],[],[]] for _ in range(Natoms)]
        all_pairs_name=[[[],[],[],[]] for _ in range(Natoms)]
        
        for at in range(Natoms):#Loop over all atoms
            all_pairs[at][0]=atom_name[at]
            all_pairs_val[at][0]=atom_name[at]
            all_pairs_name[at][0]=atom_name[at]
            #print(at,atom_name[at])
            for b0 in range(len(ubi)):
                if(at in ubi[b0]):
                    all_pairs[at][1].append(ubi[b0])
                    all_pairs_val[at][1].append(ubv[b0])
                    all_pairs_name[at][1].append([atom_name[ubi[b0][0]],atom_name[ubi[b0][1]]])
            for a0 in range(len(uai)):
                if(at in uai[a0]):
                    all_pairs[at][2].append(uai[a0])
                    all_pairs_val[at][2].append(round(uav[a0],2))
                    all_pairs_name[at][2].append([atom_name[uai[a0][0]],atom_name[uai[a0][1]],atom_name[uai[a0][2]]])
            #for d0 in range(len(udi)):
            #    if(at in udi[d0]):
            #        all_pairs[at][3].append(udi[d0])
            #        all_pairs_val[at][3].append(round(udv[d0],2))
            #        all_pairs_name[at][3].append([atom_name[udi[d0][0]],atom_name[udi[d0][1]],atom_name[udi[d0][2]],atom_name[udi[d0][3]]])
                    
            #        if(at>96 and at <103):print('>>> ',at,d0,udi[d0],all_pairs[at][0],all_pairs_name[at][3])
                    
                    
        atm_typ_details=[];
        for at in range(Natoms):#To sort everything
            present=0;
            for a in atm_typ_details:
                if(all_pairs[at][0]==a[0]):#Checking if atom type is present 
                    present=1;
                    #a: atom_name, bond atoms names, bond values, angle atoms names, angle values, dihedral atoms names, dihedral angle values
                    #Sorting bonds
                    for b1 in range(len(a[1])):
                        for b2 in range(len(all_pairs_name[at][1])):
                            if(a[1][b1]==all_pairs_name[at][1][b2]): # and a[2][b1]==all_pairs_val[at][1][b2]):                                
                                all_pairs[at][1][b1],      all_pairs[at][1][b2]      = all_pairs[at][1][b2]     ,all_pairs[at][1][b1]
                                all_pairs_name[at][1][b1], all_pairs_name[at][1][b2] = all_pairs_name[at][1][b2],all_pairs_name[at][1][b1]
                                all_pairs_val[at][1][b1],  all_pairs_val[at][1][b2]  = all_pairs_val[at][1][b2] ,all_pairs_val[at][1][b1]
                    #Sorting angles
                    #print('\n',a[1],'\n',a[2],'\n',a[3],'\n',a[4])
                    #print('\n',all_pairs_name[at][2],'\n',all_pairs_val[at][2])
                    for a1 in range(len(a[3])):
                        for a2 in range(len(all_pairs_name[at][2])):
                            if(a[3][a1]==all_pairs_name[at][2][a2]): # and a[4][a1]==all_pairs_val[at][2][a2]):   
                                
                                all_pairs[at][2][a1],      all_pairs[at][2][a2]      = all_pairs[at][2][a2]     ,all_pairs[at][2][a1]
                                all_pairs_name[at][2][a1], all_pairs_name[at][2][a2] = all_pairs_name[at][2][a2],all_pairs_name[at][2][a1]
                                all_pairs_val[at][2][a1],  all_pairs_val[at][2][a2]  = all_pairs_val[at][2][a2] ,all_pairs_val[at][2][a1]
                    
                    #Sorting dihedral angles
                    for a1 in range(len(a[5])):#Over atom types or atom names
                        for a2 in range(len(all_pairs_name[at][3])):
                            #if(a[5][a1]==all_pairs_name[at][3][a2]):
                            if( (a[5][a1]==all_pairs_name[at][3][a2] or a[5][a1]==all_pairs_name[at][3][a2][::-1])  and a[6][a1]==all_pairs_val[at][3][a2]):
                                if(a[5][a1]!=all_pairs_name[at][3][a2]):#If names are not in same order i.e. reverse order
                                    all_pairs[at][3][a2]=all_pairs[at][3][a2][::-1]
                                all_pairs[at][3][a1],      all_pairs[at][3][a2]      = all_pairs[at][3][a2]     ,all_pairs[at][3][a1]
                                all_pairs_name[at][3][a1], all_pairs_name[at][3][a2] = all_pairs_name[at][3][a2],all_pairs_name[at][3][a1]
                                all_pairs_val[at][3][a1],  all_pairs_val[at][3][a2]  = all_pairs_val[at][3][a2] ,all_pairs_val[at][3][a1]

                    
            if(present==0):
                atm_typ_details.append([all_pairs[at][0],all_pairs_name[at][1],all_pairs_val[at][1],all_pairs_name[at][2],all_pairs_val[at][2],all_pairs_name[at][3],all_pairs_val[at][3]])       

        atm_typ_names=[];
        print('--->')
        for f in atm_typ_details:
            print(f)
            atm_typ_names.append(f[0])
        print('<----')  
        
        all_pairs.sort(key=lambda y: y[0])
        all_pairs_val.sort(key=lambda y: y[0])
        all_pairs_name.sort(key=lambda y: y[0])
        
        for at in range(Natoms):#Loop over all atoms 
            #print(len(all_pairs[at][1]),len(all_pairs[at][2]),all_pairs[at])       
            #print(len(all_pairs_val[at][1]),len(all_pairs_val[at][2]),all_pairs_val[at][0],all_pairs_val[at][2])#,all_pairs_name[at][2]) 
            print(at,len(all_pairs_val[at][1]),len(all_pairs_val[at][2]),len(all_pairs_val[at][3]),all_pairs[at][0],all_pairs_name[at][0],all_pairs_name[at][3],'--')#,all_pairs_name[at][2]) 
        
        atomic_env=[[] for _ in range(len(atm_typ_names))];#Atom_name, bond pairs, angle atoms, dihedral atoms
        for an in range(len(atm_typ_names)):
            for at in range(Natoms):
                if(atm_typ_names[an]==all_pairs[at][0]):
                    #Adding indexes of bonded atoms
                    atomic_env[an].append(all_pairs[at][1:4])
                    print('\t',len(all_pairs[at][1]),len(all_pairs[at][2]),len(all_pairs[at][3]))
            print(len(atomic_env[an]))
        #print(atm_typ_names, atomic_env[0][0][0][0], atomic_env[0][0][1][0], atomic_env[0][0][2][0])
                    
                    
        return atomic_env,ana;
    #
    
    def get_environ_BADval(self,GAO,atomic_env):
        atomic_env_val=[[] for _ in range(len(atomic_env)) ]
        for an in range(len(atomic_env)):#Over atom types
            env_val=[];
            for en in range(len(atomic_env[an])):#Over environment of each atom of given atom type
                bad=[[],[],[]]
                for b in atomic_env[an][en][0]:
                    bad[0].append(GAO.get_bond_value(0,b,mic=True))
                for a in atomic_env[an][en][1]:
                    bad[1].append(GAO.get_angle_value(0,a,mic=True))
                for d in atomic_env[an][en][2]:
                    bad[2].append(GAO.get_dihedral_value(0,d,mic=True))
                atomic_env_val[an].append(bad)
        return atomic_env_val
    #

    def get_bond_environ_BADindex(self): #To fetch environment indexes corresponding to similar bonds
        ubi,uai,udi,ana=self.get_BADindex()
        ubv,uav,udv=self.get_BADval(ana,ubi,uai,udi)        
        atom_sym=self.S.get_chemical_symbols()
        Natoms=len(atom_sym)
        atom_name=[];
        nblist=[[] for i in range(Natoms)]#*Natoms;
        nbv=[[] for i in range(Natoms)]#*Natoms;
        
        #Assinging names to atoms based on connectivity >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        #Finding the connectivity of each element
        freq=[0]*len(atom_sym);
        for b in ubi:
            for b1 in b:
                freq[b1] += 1;
        
        for c in range(len(atom_sym)):#Adding atom name as symbol+connections for better understanding
            atom_name.append(atom_sym[c]+str(freq[c]))
        
        for b in range(len(ubi)):#Loop over all bonds to find the bonded atom name and bond distance for each atom
            b0=ubi[b];
            nblist[b0[0]].append(atom_name[b0[1]]) #Names
            nblist[b0[1]].append(atom_name[b0[0]]) 
            nbv[b0[0]].append(ubv[b]) #Bond values
            nbv[b0[1]].append(ubv[b])
        
        for c in range(len(nblist)):#Sorting alphabatically
            sv=sorted(zip(nblist[c],nbv[c]))
            nblist[c]=[s for s, v in sv]
            nbv[c]=[v for s, v in sv]
            #nblist[c].sort();
        
        #Finding atom types based on connectivity------------------------------
        atm_typ=[];
        for c in range(len(atom_name)):#Loop over all atoms
            #print(c,atom_name[c],freq[c],nblist[c],nbv[c])
            present=0;
            for a in atm_typ:                                                          
                if(a[0]==atom_name[c] and a[1]==freq[c] and a[2]==nblist[c]):
                    present=1;
                    if(np.linalg.norm(np.asarray(a[3])-np.asarray(nbv[c]))<0.01):
                        present=1;                        
            if(present==0):
                atm_typ.append([atom_name[c],freq[c],nblist[c],nbv[c]])
                #    print('-- > ',atm_typ)
        
        #Cleaning name of atom types---------------------------       
        atm_typ_log=[];
        for p in range(len(atm_typ)):
            atm_typ_log.append(atm_typ[p][0])
            atm_typ[p].append(chr(96+atm_typ_log.count(atm_typ[p][0]))) #Adding a,b,.. at the end of name 
            print(atm_typ[p])
        
        #Cleaning name of all atoms with correct atom types-----------------
        for c in range(len(atom_name)):#Loop over all atoms
            for a in atm_typ:
                if(a[0]==atom_name[c] and a[1]==freq[c] and a[2]==nblist[c]):
                    if(np.linalg.norm(np.asarray(a[3])-np.asarray(nbv[c]))<0.01):
                        atom_name[c]+=a[4] # Adding a,b,... in the end of name
        #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        #----------------------------------------------
        usym_bonds=[];#List of unique and symmetric bonds
        ub_abl=[];#Unique bond atoms and bond length
        for b0 in range(len(ubi)):
            b=ubi[b0];
            au=True;
            for b1 in range(len(ub_abl)):
                #print('xx ',b1,len(ub_abl),ub_abl)
                bl=ub_abl[b1]
                if( ((atom_name[b[0]]==bl[0] and atom_name[b[1]]==bl[1]) or (atom_name[b[0]]==bl[1] and atom_name[b[1]]==bl[0]) ) and abs(ubv[b0]-bl[2])<0.01):
                    #print(b,bl,chem_sym[b[0]],chem_sym[b[1]],b[0],b[1],abs(ubv[b0]-bl[2]),ubv[b0],bl[2],ana.get_bond_value(0,b, mic=True))
                    usym_bonds[b1].append(b)
                    au=False;                    
            if(au):
                ub_abl.append([atom_name[b[0]], atom_name[b[1]], ana.get_bond_value(0,b, mic=True)])
                usym_bonds.append([b]) 
        #print(usym_bonds,len(usym_bonds),len(ubi),ub_abl,'=====')
        #----------------------------------------------
        
        #Finding bonds/angles/dihedrals associated with one type of bond        
        bond_env=[[] for _ in range(len(usym_bonds))];#[[Bond atoms, angle atoms, dihedral atoms]]
        for bt in range(len(usym_bonds)): #Loop over bond types
            #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            #Finding proper order to fill the pairs of B/A/D based on first pair of each type         
            ##bpair_order[bt].append([atom_name[bond_start[0]], atom_name[bond_start[1]]])
            
            bond_start = usym_bonds[bt][0] 
            blist_name=ub_abl[bt][0:2]
            #blist_val=ub_abl[bt][2]
            
            alist_name=[];
            alist_val=[];
            for a0 in range(len(uai)):#Loop over list of all angles
                if(bond_start[0] in uai[a0] and bond_start[1] in uai[a0]): #Checking presence of given bond atoms in angle list
                    alist_tmp=[ atom_name[uai[a0][0]], atom_name[uai[a0][1]], atom_name[uai[a0][2]] ]
                    
                    if(alist_tmp[::-1] in alist_name):
                        alist_name.append(alist_tmp[::-1])
                    else:
                        alist_name.append(alist_tmp)
                    alist_val.append(uav[a0])

            dlist_name=[];
            dlist_val=[];                                                         
            for d0 in range(len(udi)):
                #if(bond_start[0] in udi[d0] and bond_start[1] in udi[d0]): #Checking presence of given bond atoms in dihedral list
                if(bond_start[0] in udi[d0][1:3] and bond_start[1] in udi[d0][1:3]): #Checking presence of given bond atoms in dihedral list
                    dlist_tmp=[ atom_name[udi[d0][0]], atom_name[udi[d0][1]], atom_name[udi[d0][2]], atom_name[udi[d0][3]] ]
                    if(dlist_tmp[::-1] in dlist_name):
                        dlist_name.append(dlist_tmp[::-1])
                    else:
                        dlist_name.append(dlist_tmp)
                    dlist_val.append(udv[d0])
            
            print(ub_abl[bt],len(alist_name), len(dlist_name))
            for pz in range(len(alist_name)):print('  >',bt,alist_name[pz],alist_val[pz])
            for pz in range(len(dlist_name)):print('  >',bt,dlist_name[pz],dlist_val[pz])
            print('\n')
            #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            for cbi in usym_bonds[bt]:#Loop over all the bonds of current bond type 
                #Collecting and correcting bond pair indices
                cb_name=[atom_name[cbi[0]],atom_name[cbi[1]]];                
                if(cb_name!=blist_name):
                    bindex=cbi[::-1]
                else:
                    bindex=cbi                   
                #Collecting and correcting angle pair indices
                alist_name_tmp=alist_name.copy()
                fill_id = [fi for fi in range(len(alist_name_tmp))]
                aindex = [[] for _ in range(len(alist_name))];
                for a0 in range(len(uai)):#Loop over list of all angles
                    if(cbi[0] in uai[a0] and cbi[1] in uai[a0]): #Checking presence of given bond atoms in angle list                        
                        alist_tmp = [ atom_name[uai[a0][0]], atom_name[uai[a0][1]], atom_name[uai[a0][2]] ]
                        
                        #print(cbi,uai[a0])                        
                        if(alist_tmp in alist_name_tmp):
                            rm_id = alist_name_tmp.index(alist_tmp)
                            aindex[fill_id[rm_id]] = uai[a0];
                            fill_id.remove(fill_id[rm_id])
                            alist_name_tmp.remove(alist_name_tmp[rm_id])
                            
                        elif(alist_tmp[::-1] in alist_name_tmp):
                            rm_id = alist_name_tmp.index(alist_tmp[::-1])
                            aindex[fill_id[rm_id]] = uai[a0][::-1];
                            fill_id.remove(fill_id[rm_id])
                            alist_name_tmp.remove(alist_name_tmp[rm_id]) 
                          
                #Collecting and correcting dihedral pair indices
                dlist_name_tmp=dlist_name.copy()
                fill_id = [fi for fi in range(len(dlist_name_tmp))]
                dindex = [[] for _ in range(len(dlist_name))];
                for d0 in range(len(udi)):#Loop over list of all dihedrals
                    if(cbi[0] in udi[d0] and cbi[1] in udi[d0]): #Checking presence of given bond atoms in dihedral list                        
                        dlist_tmp = [ atom_name[udi[d0][0]], atom_name[udi[d0][1]], atom_name[udi[d0][2]], atom_name[udi[d0][3]] ]
                        #print(cbi,udi[d0],dlist_tmp)
                        if(dlist_tmp in dlist_name_tmp):
                            rm_id = dlist_name_tmp.index(dlist_tmp)
                            dindex[fill_id[rm_id]] = udi[d0];
                            fill_id.remove(fill_id[rm_id])
                            dlist_name_tmp.remove(dlist_name_tmp[rm_id])
                            
                        elif(dlist_tmp[::-1] in dlist_name_tmp):
                            rm_id = dlist_name_tmp.index(dlist_tmp[::-1])
                            dindex[fill_id[rm_id]] = udi[d0][::-1];
                            fill_id.remove(fill_id[rm_id])
                            dlist_name_tmp.remove(dlist_name_tmp[rm_id]) 
                
                bond_env[bt].append([[bindex],aindex,dindex])                
                                      
                #print(blist_name,bindex,aindex,dindex,'$$')
                #print([atom_name[i] for i in bindex],[[atom_name[ai] for ai in aid] for aid in aindex])
                #print([atom_name[i] for i in bindex],[ana.get_angle_value(0,aid,mic=True) for aid in aindex])                
                #print([atom_name[i] for i in bindex],[[atom_name[ai] for ai in aid] for aid in dindex])
                #print([atom_name[i] for i in bindex],[ana.get_dihedral_value(0,aid,mic=True) for aid in dindex])  
                #print('\n')
                #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<            
        return bond_env,ana;
    #

    def analysis_environ(self,db,da,dda,dscell,dacell,env_type):
        if(env_type=='atomic'):at_env,ana = self.get_environ_BADindex() 
        if(env_type=='bond'):at_env,ana = self.get_bond_environ_BADindex()
        
        natom_name=len(at_env);
        bad_no=[];
        bad_vec_count=0;
        for an in range(natom_name):
            #Filling [Nbonds,Nangles,Ndihedrals,No of such atoms in a snapshot]
            bad_no.append([len(at_env[an][0][0]),len(at_env[an][0][1]),len(at_env[an][0][2]),len(at_env[an])])
            bad_vec_count += len(at_env[an]);
        
        check_log=[[] for _ in range(natom_name)]#It is going to store BAD coordinate for each atom
        celllog=[[],[],[],[],[],[]];#For cell vectors
        ##bind=np.floor_divide(ubv,db)#List of indexes      

        nbonds=sum([i[0] for i in bad_no]);
        nangles=sum([i[1] for i in bad_no]);
        ndihedrals=sum([i[2] for i in bad_no]);
        print(f'{nbonds} Bonds; {nangles} Angles; {ndihedrals} Dihedrals')
        #---------------------------------------------------    
        ntest_al=0;ntrain_al=0;
        fs=open(self.fname,'r')   
        os.system('rm -rf Train_AL.xyz Test_AL.xyz')
        ftr=open('Train_AL_env.xyz','w');
        fte=open('Test_AL_env.xyz','w');       
        for s in range(self.Ndset):
            #print(s)
            Na=int(fs.readline().split()[0]);
            l2=fs.readline().split();
            sno=l2[0];
            Esnap=float(l2[1]);
            A=[float(k) for k in l2[2:5]];B=[float(k) for k in l2[5:8]];C=[float(k) for k in l2[8:11]];  
            #volume=np.dot(A,np.cross(B,C))
            stress=np.array(l2[11:17],dtype=float);#print(Na,sno,Esnap,A,B,C,stress);
            stress=stress #in ev-Ang
            elements=[];
            coord=np.zeros([Na,3],dtype=float);#>>Atomic coordinates
            force=np.zeros([Na,3],dtype=float);#>>Atomic forces
            qd=np.zeros(Na,dtype=float);#>>Dynamic charges
            for i in range(Na):
                l=fs.readline().split();
                elements.append(l[0]);
                coord[i]=l[1:4];
                force[i]=l[4:7];
                qd[i]=l[7];
                
            self.S.set_cell([A,B,C])
            self.S.set_positions(coord,apply_constraint=False)
            #-----------------------------------------------------------
            atomic_env_v=self.get_environ_BADval(ana,at_env)
            
            cc=AS.Cell();
            a_cell,b_cell,c_cell,al,be,ga=cc.vec2para(A,B,C)
            cell=[a_cell,b_cell,c_cell,al,be,ga]
            del cc
            #-------------------------------------
            testbad=0;testcell=0;
            testb=0;testa=0;testd=0;
            test_index=[];
            for an in range(len(atomic_env_v)):
                check_v=0
                for vv in atomic_env_v[an]:
                    b_index=[]
                    for b1 in vv[0]:
                        b_index.append( int(np.floor(b1/db)) )                    
                    a_index=[]
                    for a1 in vv[1]:
                        a_index.append( int(np.floor(a1/da)) )                     
                    d_index=[]
                    for d1 in vv[2]:
                        d_index.append( int(np.floor(d1/dda)) )
                    
                    bad_index=[b_index,a_index,d_index]
                    if(bad_index in check_log[an]):
                        testbad += 1;
                    else:
                        check_log[an].append(bad_index)
                        if(b_index in [bpair[0] for bpair in check_log[an]]):testb += 1
                        if(a_index in [bpair[1] for bpair in check_log[an]]):testa += 1
                        if(d_index in [bpair[2] for bpair in check_log[an]]):testd += 1
                        #if(an==0):
                        #if(an==0 and check_v==0):
                            #print(vv)
                            #print(check_v,bad_index)
                        test_index.append(an)
                    #print(check_v,bad_index)
                    check_v+=1;
                    
            for l1 in range(3):
                idex=int(np.floor(cell[l1]/dscell))
                if idex not in celllog[l1]:
                    celllog[l1].append(idex)
                else:
                    testcell+=1;  
                    
                idex=int(np.floor(cell[l1+3]/dacell))
                if idex not in celllog[l1+3]:
                    celllog[l1+3].append(idex)
                else:
                    testcell+=1;
            
            #print(blog)
            if( testbad == bad_vec_count and testcell == 6):
                ntest_al +=1;
                fte.write(f"{Na}\n")
                fte.write(f"{sno} {Esnap} {A[0]} {A[1]} {A[2]} {B[0]} {B[1]} {B[2]} {C[0]} {C[1]} {C[2]} {stress[0]} {stress[1]} {stress[2]} {stress[3]} {stress[4]} {stress[5]}\n")
                for ii in range(Na):
                    fte.write(f"{elements[ii]} \t{coord[ii][0]} \t{coord[ii][1]} \t{coord[ii][2]} \t{force[ii][0]} \t{force[ii][1]} \t{force[ii][2]} \t{qd[ii]}\n")
            else:
                ntrain_al +=1;
                ftr.write(f"{Na}\n")
                ftr.write(f"{sno} {Esnap} {A[0]} {A[1]} {A[2]} {B[0]} {B[1]} {B[2]} {C[0]} {C[1]} {C[2]} {stress[0]} {stress[1]} {stress[2]} {stress[3]} {stress[4]} {stress[5]}\n")
                for ii in range(Na):
                    ftr.write(f"{elements[ii]} \t{coord[ii][0]} \t{coord[ii][1]} \t{coord[ii][2]} \t{force[ii][0]} \t{force[ii][1]} \t{force[ii][2]} \t{qd[ii]}\n")
            print(s,'BAD:',bad_vec_count - testbad,bad_vec_count, testbad,[len(clg) for clg in check_log],set(test_index),[test_index.count(cz) for cz in set(test_index)],'Cell',6-testcell,round(a_cell,2),round(b_cell,2),round(c_cell,2),round(al,3),round(be,3),round(ga,3),Esnap,'[ntrain = ',s-ntest_al,'ntest = ',ntest_al,']')
            print(testb,testa,testd)
            #--------------------------------------
        print('Out of',self.Ndset,'configurations',ntrain_al,'are for training and',ntest_al,'are for test',db,da,dda,dscell,dacell)
        fs.close()
        ftr.close()
        fte.close()
#

    def analysis_sym(self,db,da,dda,dscell,dacell):
        usb,usa,usd,ana=self.get_unique_BADindex()
        
        abdist=np.zeros([self.Ndset,len(usb)]);
        aadist=np.zeros([self.Ndset,len(usa)]);
        addist=np.zeros([self.Ndset,len(usd)]);

        blog=[[] for _ in usb]; #For bonds; Lists are mutable, so [[]]*len(ubi) will generate a same copy
        alog=[[] for _ in usa]; #For angle
        dlog=[[] for _ in usd]; #For dihedral
        celllog=[[],[],[],[],[],[]];#For cell
        #bind=np.floor_divide(ubv,db)#List of indexes      

        nbonds=sum([len(i) for i in usb]);
        nangles=sum([len(i) for i in usa]);
        ndihedrals=sum([len(i) for i in usd]);
        print(f'{nbonds} Bonds; {nangles} Angles; {ndihedrals} Dihedrals')
        #---------------------------------------------------    
        ntest_al=0;ntrain_al=0;
        fs=open(self.fname,'r')   
        os.system('rm -rf Train_AL.xyz Test_AL.xyz')
        ftr=open('Train_AL.xyz','w');
        fte=open('Test_AL.xyz','w');       
        for s in range(self.Ndset):
            #print(s)
            Na=int(fs.readline().split()[0]);
            l2=fs.readline().split();
            sno=l2[0];
            Esnap=float(l2[1]);
            A=[float(k) for k in l2[2:5]];B=[float(k) for k in l2[5:8]];C=[float(k) for k in l2[8:11]];  
            #volume=np.dot(A,np.cross(B,C))
            stress=np.array(l2[11:17],dtype=float);#print(Na,sno,Esnap,A,B,C,stress);
            stress=stress #in ev-Ang
            elements=[];
            coord=np.zeros([Na,3],dtype=float);#>>Atomic coordinates
            force=np.zeros([Na,3],dtype=float);#>>Atomic forces
            qd=np.zeros(Na,dtype=float);#>>Dynamic charges
            for i in range(Na):
                l=fs.readline().split();
                elements.append(l[0]);
                coord[i]=l[1:4];
                force[i]=l[4:7];
                qd[i]=l[7];
                
            self.S.set_cell([A,B,C])
            self.S.set_positions(coord,apply_constraint=False)
            #-----------------------------------------------------------
            ubv,uav,udv=self.get_unique_BADval(ana,usb,usa,usd)
            #print(ubv[0])
            
            cc=AS.Cell();
            a1,b1,c1,al,be,ga=cc.vec2para(A,B,C)
            cell=[a1,b1,c1,al,be,ga]
            del cc
            #---------------------------------------
            testb=0;testa=0;testd=0;testcell=0;
            for l1 in range(len(usb)):
                #print(ubv[l1])
                for l2 in range(len(usb[l1])):
                    idex=int(np.floor(ubv[l1][l2]/db))                  
                    if idex not in blog[l1]:
                        blog[l1].append(idex)
                    else:
                        testb+=1;
                    abdist[s][l1]=ubv[l1][l2];

            for l1 in range(len(usa)):
                for l2 in range(len(usa[l1])):
                    idex=int(np.floor(uav[l1][l2]/da))                  
                    if idex not in alog[l1]:
                        alog[l1].append(idex)
                    else:
                        testa+=1;
                    aadist[s][l1]=uav[l1][l2];

            for l1 in range(len(usd)):
                for l2 in range(len(usd[l1])):
                    idex=int(np.floor(udv[l1][l2]/dda))                  
                    if idex not in dlog[l1]:
                        dlog[l1].append(idex)
                    else:
                        testd+=1;
                    addist[s][l1]=udv[l1][l2];                    
                                       
            for l1 in range(3):
                idex=int(np.floor(cell[l1]/dscell))
                if idex not in celllog[l1]:
                    celllog[l1].append(idex)
                else:
                    testcell+=1;  
                    
                idex=int(np.floor(cell[l1+3]/dacell))
                if idex not in celllog[l1+3]:
                    celllog[l1+3].append(idex)
                else:
                    testcell+=1;
            
            #print(blog)
            if( testb==nbonds and testa==nangles and testd==ndihedrals and testcell==6):
                ntest_al +=1;
                fte.write(f"{Na}\n")
                fte.write(f"{sno} {Esnap} {A[0]} {A[1]} {A[2]} {B[0]} {B[1]} {B[2]} {C[0]} {C[1]} {C[2]} {stress[0]} {stress[1]} {stress[2]} {stress[3]} {stress[4]} {stress[5]}\n")
                for ii in range(Na):
                    fte.write(f"{elements[ii]} \t{coord[ii][0]} \t{coord[ii][1]} \t{coord[ii][2]} \t{force[ii][0]} \t{force[ii][1]} \t{force[ii][2]} \t{qd[ii]}\n")
            else:
                ntrain_al +=1;
                ftr.write(f"{Na}\n")
                ftr.write(f"{sno} {Esnap} {A[0]} {A[1]} {A[2]} {B[0]} {B[1]} {B[2]} {C[0]} {C[1]} {C[2]} {stress[0]} {stress[1]} {stress[2]} {stress[3]} {stress[4]} {stress[5]}\n")
                for ii in range(Na):
                    ftr.write(f"{elements[ii]} \t{coord[ii][0]} \t{coord[ii][1]} \t{coord[ii][2]} \t{force[ii][0]} \t{force[ii][1]} \t{force[ii][2]} \t{qd[ii]}\n")
            print(s,'Bo:',nbonds-testb,'An:',nangles-testa,'Dh:',ndihedrals-testd,'Cell',6-testcell,round(a1,2),round(b1,2),round(c1,2),round(al,3),round(be,3),round(ga,3),Esnap,'[ntrain = ',s-ntest_al,'ntest = ',ntest_al,']')
            #---------------------------------------
        print('Out of',self.Ndset,'configurations',ntrain_al,'are for training and',ntest_al,'are for test',db,da,dda,dscell,dacell)
        fs.close()
        ftr.close()
        fte.close()
        
        dlen=[len(x) for x in blog];
        print('Atoms with maximum bond fluctuation',usb[dlen.index(max(dlen))][0], 'with value of ', max(dlen))
        print('Atoms with minimum bond fluctuation',usb[dlen.index(min(dlen))][0], 'with value of ', min(dlen))
        dlen=[len(x) for x in alog];
        print('Atoms with maximum angle fluctuation',usa[dlen.index(max(dlen))][0],'with value of ', max(dlen))
        print('Atoms with minimum angle fluctuation',usa[dlen.index(min(dlen))][0], 'with value of ', min(dlen))
        dlen=[len(x) for x in dlog];
        print('Atoms with maximum DHA fluctuation',usd[dlen.index(max(dlen))][0],'with value of ', max(dlen))
        print('Atoms with minimum DHA fluctuation',usd[dlen.index(min(dlen))][0], 'with value of ', min(dlen))
        
        pp=AS.PlotD();        
        pp.plot_ICdist(abdist,'Bonds',self.special_indexes,'Bond_dist')
        pp.plot_ICdist(aadist,'Angles',self.special_indexes,'Angle_dist')
        pp.plot_ICdist(addist,'Dihedrals',self.special_indexes,'Dihedral_dist')
        
        #------------------------------------------------------------------------------       
    # 

    def analysis_sym_small_axyz(self,db,da,dda,dscell,dacell,fnpz,cell_type):
        usb,usa,usd,ana=self.get_unique_BADindex()
        
        abdist=np.zeros([self.Ndset,len(usb)]);
        aadist=np.zeros([self.Ndset,len(usa)]);
        addist=np.zeros([self.Ndset,len(usd)]);

        blog=[[] for _ in usb]; #For bonds; Lists are mutable, so [[]]*len(ubi) will generate a same copy
        alog=[[] for _ in usa]; #For angle
        dlog=[[] for _ in usd]; #For dihedral
        if(cell_type=='cubic'):
            celllog=[[],[]]
        else:
            celllog=[[],[],[],[],[],[]];#For cell a,b,c,alpha,beta,gamma 
        #bind=np.floor_divide(ubv,db)#List of indexes      

        nbonds=sum([len(i) for i in usb]);
        nangles=sum([len(i) for i in usa]);
        ndihedrals=sum([len(i) for i in usd]);
        print(f'{nbonds} Bonds; {nangles} Angles; {ndihedrals} Dihedrals')
        #---------------------------------------------------    
        ntest_al=0;ntrain_al=0;
        fs=open(self.fname,'r')   
        os.system('rm -rf Train_AL.xyz Test_AL.xyz')
        ftr=open('Train_AL.xyz','w');
        fte=open('Test_AL.xyz','w');       
        for s in range(self.Ndset):
            #print(s)
            Na=int(fs.readline().split()[0]);
            l2=fs.readline().split();
            A=[float(k) for k in l2[2:5]];B=[float(k) for k in l2[5:8]];C=[float(k) for k in l2[8:11]];  
            #volume=np.dot(A,np.cross(B,C))
            stress=np.array(l2[11:17],dtype=float);#print(Na,sno,Esnap,A,B,C,stress);
            stress=stress #in ev-Ang
            elements=[];
            coord=np.zeros([Na,3],dtype=float);#>>Atomic coordinates
            force=np.zeros([Na,3],dtype=float);#>>Atomic forces
            qd=np.zeros(Na,dtype=float);#>>Dynamic charges
            for i in range(Na):
                l=fs.readline().split();
                elements.append(l[0]);
                coord[i]=l[1:4];
                
            self.S.set_cell([A,B,C])
            self.S.set_positions(coord,apply_constraint=False)
            #-----------------------------------------------------------
            ubv,uav,udv=self.get_unique_BADval(ana,usb,usa,usd)
            #print(ubv[0])
            
            cc=AS.Cell();
            a1,b1,c1,al,be,ga=cc.vec2para(A,B,C)
            cell=[a1,b1,c1,al,be,ga]
            del cc
            #---------------------------------------
            testb=0;testa=0;testd=0;testcell=0;
            for l1 in range(len(usb)):
                #print(ubv[l1])
                for l2 in range(len(usb[l1])):
                    idex=int(np.floor(ubv[l1][l2]/db))                  
                    if idex not in blog[l1]:
                        blog[l1].append(idex)
                    else:
                        testb+=1;
                    abdist[s][l1]=ubv[l1][l2];

            for l1 in range(len(usa)):
                for l2 in range(len(usa[l1])):
                    idex=int(np.floor(uav[l1][l2]/da))                  
                    if idex not in alog[l1]:
                        alog[l1].append(idex)
                    else:
                        testa+=1;
                    aadist[s][l1]=uav[l1][l2];

            for l1 in range(len(usd)):
                for l2 in range(len(usd[l1])):
                    idex=int(np.floor(udv[l1][l2]/dda))                  
                    if idex not in dlog[l1]:
                        dlog[l1].append(idex)
                    else:
                        testd+=1;
                    addist[s][l1]=udv[l1][l2]; 
            
            if(cell_type=='cubic'):
                for l1 in range(3):
                    idex=int(np.floor(cell[l1]/dscell))
                    if idex not in celllog[0]:
                        celllog[0].append(idex)
                    else:
                        testcell+=1;
                        
                    idex=int(np.floor(cell[l1+3]/dacell))
                    if idex not in celllog[1]:
                        celllog[1].append(idex)
                    else:
                        testcell+=1;
            else:
                for l1 in range(3):
                    idex=int(np.floor(cell[l1]/dscell))
                    if idex not in celllog[l1]:
                        celllog[l1].append(idex)
                    else:
                        testcell+=1;
                        
                    idex=int(np.floor(cell[l1+3]/dacell))
                    if idex not in celllog[l1+3]:
                        celllog[l1+3].append(idex)
                    else:
                        testcell+=1;
                                       
            
            #print(blog)
            if( testb==nbonds and testa==nangles and testd==ndihedrals and testcell==6):
                ntest_al +=1;
                fte.write(f"{Na}\n")
                fte.write(f"snap energy {A[0]} {A[1]} {A[2]} {B[0]} {B[1]} {B[2]} {C[0]} {C[1]} {C[2]} \n")
                for ii in range(Na):
                    fte.write(f"{elements[ii]} \t{coord[ii][0]} \t{coord[ii][1]} \t{coord[ii][2]} \n")
            else:
                ntrain_al +=1;
                ftr.write(f"{Na}\n")
                ftr.write(f"snap energy {A[0]} {A[1]} {A[2]} {B[0]} {B[1]} {B[2]} {C[0]} {C[1]} {C[2]} \n")
                for ii in range(Na):
                    ftr.write(f"{elements[ii]} \t{coord[ii][0]} \t{coord[ii][1]} \t{coord[ii][2]} \n")
            print(s,'Bo:',nbonds-testb,'An:',nangles-testa,'Dh:',ndihedrals-testd,'Cell',6-testcell,round(a1,2),round(b1,2),round(c1,2),round(al,3),round(be,3),round(ga,3),'[ntrain = ',s-ntest_al,'ntest = ',ntest_al,']')
            #---------------------------------------
        print('Out of',self.Ndset,'configurations',ntrain_al,'are for training and',ntest_al,'are for test',db,da,dda,dscell,dacell)
        fs.close()
        ftr.close()
        fte.close()
                
        np.savez(fnpz, cell_type=cell_type, db=db, da=da, dda=dda, dscell=dscell, dacell=dacell, usb=usb, usa=usa, usd=usd, blog=blog, alog=alog, dlog=dlog, celllog=celllog)
        #------------------------------------------------------------------------------       
    # 
    def save_first_TS_analysis_sym(self,db,da,dda,dscell,dacell,cell_type):
        #It will not split in training and test set
        usb,usa,usd,ana=self.get_unique_BADindex()        
        blog=[[] for _ in usb]; #For bonds; Lists are mutable, so [[]]*len(ubi) will generate a same copy
        alog=[[] for _ in usa]; #For angle
        dlog=[[] for _ in usd]; #For dihedral

        if(cell_type=='cubic'):
            celllog=[[],[]]
        else:
            celllog=[[],[],[],[],[],[]];#For cell a,b,c,alpha,beta,gamma     
        
        nbonds=sum([len(i) for i in usb]);
        nangles=sum([len(i) for i in usa]);
        ndihedrals=sum([len(i) for i in usd]);
        print(f'{nbonds} Bonds; {nangles} Angles; {ndihedrals} Dihedrals')
        #---------------------------------------------------    
        fs=open(self.fname,'r')        
        for s in range(self.Ndset):
            Na=int(fs.readline().split()[0]);
            l2=fs.readline().split();
            sno=l2[0];
            A=[float(k) for k in l2[2:5]];B=[float(k) for k in l2[5:8]];C=[float(k) for k in l2[8:11]];  
            #volume=np.dot(A,np.cross(B,C))
            elements=[];
            coord=np.zeros([Na,3],dtype=float);#>>Atomic coordinates
            for i in range(Na):
                l=fs.readline().split();
                elements.append(l[0]);
                coord[i]=l[1:4];

            self.S.set_cell([A,B,C])
            self.S.set_positions(coord,apply_constraint=False)
            #-----------------------------------------------------------
            ubv,uav,udv=self.get_unique_BADval(ana,usb,usa,usd)
            
            cc=AS.Cell();
            a1,b1,c1,al,be,ga=cc.vec2para(A,B,C)
            cell=[a1,b1,c1,al,be,ga]
            del cc
            #---------------------------------------
            for l1 in range(len(usb)):
                for l2 in range(len(usb[l1])):
                    idex=int(np.floor(ubv[l1][l2]/db))                  
                    if idex not in blog[l1]:
                        blog[l1].append(idex)

            for l1 in range(len(usa)):
                for l2 in range(len(usa[l1])):
                    idex=int(np.floor(uav[l1][l2]/da))                  
                    if idex not in alog[l1]:
                        alog[l1].append(idex)

            for l1 in range(len(usd)):
                for l2 in range(len(usd[l1])):
                    idex=int(np.floor(udv[l1][l2]/dda))                  
                    if idex not in dlog[l1]:
                        dlog[l1].append(idex)
                         
            if(cell_type=='cubic'):
                for l1 in range(3):
                    idex=int(np.floor(cell[l1]/dscell))
                    if idex not in celllog[0]:
                        celllog[0].append(idex)
                    idex=int(np.floor(cell[l1+3]/dacell))
                    if idex not in celllog[1]:
                        celllog[1].append(idex)
            else:
                for l1 in range(3):
                    idex=int(np.floor(cell[l1]/dscell))
                    if idex not in celllog[l1]:
                        celllog[l1].append(idex)
                    idex=int(np.floor(cell[l1+3]/dacell))
                    if idex not in celllog[l1+3]:
                        celllog[l1+3].append(idex)
            print(s)
        #------------------------------------------------------------------------------
        print(blog,alog,dlog,celllog)
        np.savez('nf.npz', cell_type=cell_type, db=db, da=da, dda=dda, dscell=dscell, dacell=dacell, usb=usb, usa=usa, usd=usd, blog=blog, alog=alog, dlog=dlog, celllog=celllog)
    #

    def append_TS_analysis_sym(self,fnpz,ftrain_append,nconfig_append):
        #nconfig_append: Number of new configurations to append as per BADC algorithm
        #It will be appended in the training set
        dd = dict(np.load(fnpz,allow_pickle=True))
        usb = dd['usb']
        usa = dd['usa']
        usd = dd['usd']
        ana = Analysis(self.S)
        cell_type = dd['cell_type']
        
        nbonds=sum([len(i) for i in usb]);
        nangles=sum([len(i) for i in usa]);
        ndihedrals=sum([len(i) for i in usd]);
        #---------------------------------------------------    
        fs=open(self.fname,'r')   
        ftr=open(ftrain_append,'a');
        found_train = 0
        for s in range(self.Ndset):
            #print(s)
            Na=int(fs.readline().split()[0]);
            l2=fs.readline().split();
            A=[float(k) for k in l2[2:5]];B=[float(k) for k in l2[5:8]];C=[float(k) for k in l2[8:11]];  
            #volume=np.dot(A,np.cross(B,C))
            elements=[];
            coord=np.zeros([Na,3],dtype=float);#>>Atomic coordinates
            for i in range(Na):
                l=fs.readline().split();
                elements.append(l[0]);
                coord[i]=l[1:4];                
            self.S.set_cell([A,B,C])
            self.S.set_positions(coord,apply_constraint=False)
            #-----------------------------------------------------------
            ubv,uav,udv=self.get_unique_BADval(ana,usb,usa,usd)
            #print(ubv[0])
            
            cc=AS.Cell();
            a1,b1,c1,al,be,ga=cc.vec2para(A,B,C)
            cell=[a1,b1,c1,al,be,ga]
            del cc
            #---------------------------------------
            testb=0;testa=0;testd=0;testcell=0;
            for l1 in range(len(usb)):
                #print(ubv[l1])
                for l2 in range(len(usb[l1])):
                    idex=int(np.floor(ubv[l1][l2]/dd['db']))                  
                    if idex not in dd['blog'][l1]:
                        dd['blog'][l1].append(idex)
                    else:
                        testb+=1;

            for l1 in range(len(usa)):
                for l2 in range(len(usa[l1])):
                    idex=int(np.floor(uav[l1][l2]/dd['da']))                  
                    if idex not in dd['alog'][l1]:
                        dd['alog'][l1].append(idex)
                    else:
                        testa+=1;

            for l1 in range(len(usd)):
                for l2 in range(len(usd[l1])):
                    idex=int(np.floor(udv[l1][l2]/dd['dda']))                  
                    if idex not in dd['dlog'][l1]:
                        dd['dlog'][l1].append(idex)
                    else:
                        testd+=1;                   
                                       
            if(cell_type=='cubic'):
                for l1 in range(3):
                    idex=int(np.floor(cell[l1]/dd['dscell']))
                    if idex not in dd['celllog'][0]:
                        dd['celllog'][0].append(idex)
                    else:
                        testcell+=1;
                    idex=int(np.floor(cell[l1+3]/dd['dacell']))
                    if idex not in dd['celllog'][1]:
                        dd['celllog'][1].append(idex)
                    else:
                        testcell+=1;
            else:
                for l1 in range(3):
                    idex=int(np.floor(cell[l1]/dd['dscell']))
                    if idex not in dd['celllog'][l1]:
                        dd['celllog'][l1].append(idex)
                    else:
                        testcell+=1;

                    idex=int(np.floor(cell[l1+3]/dd['dacell']))
                    if idex not in dd['celllog'][l1+3]:
                        dd['celllog'][l1+3].append(idex)
                    else:
                        testcell+=1;
            
            #print(blog)
            if( testb==nbonds and testa==nangles and testd==ndihedrals and testcell==6):
                print(f'{s} is not in training set')
            else:
                found_train += 1;
                ftr.write(f"{Na}\n")
                ftr.write(f"snap energy {A[0]} {A[1]} {A[2]} {B[0]} {B[1]} {B[2]} {C[0]} {C[1]} {C[2]}\n")
                for ii in range(Na):
                    ftr.write(f"{elements[ii]} \t{coord[ii][0]} \t{coord[ii][1]} \t{coord[ii][2]}\n")
            if(found_train > nconfig_append):
                break;
            #-----------------------------------

        fs.close()
        ftr.close()
        np.savez(fnpz, **dd)
    #

    def analysis(self,db,da,dda,dscell,dacell):
        ubi,uai,udi,ana=self.get_BADindex()
        #ubv,uav,udv=get_BADval(S,ubi,uai,udi)
        #ubv,uav,udv=self.get_BADval(ana,ubi,uai,udi)
        
        abdist=np.zeros([self.Ndset,len(ubi)]);
        aadist=np.zeros([self.Ndset,len(uai)]);
        addist=np.zeros([self.Ndset,len(udi)]);

        blog=[[] for _ in ubi]; #For bonds; Lists are mutable, so [[]]*len(ubi) will generate a same copy
        alog=[[] for _ in uai]; #For angle
        dlog=[[] for _ in udi]; #For dihedral
        celllog=[[],[],[],[],[],[]];#For cell
        #bind=np.floor_divide(ubv,db)#List of indexes      
    
        ntest_al=0;ntrain_al=0;
        fs=open(self.fname,'r')   
        os.system('rm -rf Train_AL.xyz Test_AL.xyz')
        ftr=open('Train_AL.xyz','w');
        fte=open('Test_AL.xyz','w');
        bdist=[];adist=[];ddist=[];        
        for s in range(self.Ndset):
            #print(s)
            Na=int(fs.readline().split()[0]);
            l2=fs.readline().split();
            sno=l2[0];
            Esnap=float(l2[1]);
            A=[float(k) for k in l2[2:5]];B=[float(k) for k in l2[5:8]];C=[float(k) for k in l2[8:11]];  
            #volume=np.dot(A,np.cross(B,C))
            stress=np.array(l2[11:17],dtype=float);#print(Na,sno,Esnap,A,B,C,stress);
            stress=stress #in ev-Ang
            elements=[];
            coord=np.zeros([Na,3],dtype=float);#>>Atomic coordinates
            force=np.zeros([Na,3],dtype=float);#>>Atomic forces
            qd=np.zeros(Na,dtype=float);#>>Dynamic charges
            for i in range(Na):
                l=fs.readline().split();
                elements.append(l[0]);
                coord[i]=l[1:4];
                force[i]=l[4:7];
                qd[i]=l[7];
                
            self.S.set_cell([A,B,C])
            self.S.set_positions(coord,apply_constraint=False)
            ubv,uav,udv=self.get_BADval(ana,ubi,uai,udi)
            #print(ubv[0])
            bdist.append(ubv[280])
            adist.append(udv[498])
            ddist.append(udv[585])
            
            cc=AS.Cell();
            a1,b1,c1,al,be,ga=cc.vec2para(A,B,C)
            cell=[a1,b1,c1,al,be,ga]
            del cc
            #---------------------------------------
            testb=0;testa=0;testd=0;testcell=0;
            for l1 in range(len(ubi)):
                idex=int(np.floor(ubv[l1]/db))
                if idex not in blog[l1]:
                    blog[l1].append(idex)
                else:
                    testb+=1;
                abdist[s][l1]=ubv[l1];
                
            for l1 in range(len(uai)):
                idex=int(np.floor(uav[l1]/da))
                if idex not in alog[l1]:
                    alog[l1].append(idex)
                else:
                    testa+=1;
                aadist[s][l1]=uav[l1];
                
            for l1 in range(len(udi)):
                idex=int(np.floor(udv[l1]/dda))
                if idex not in dlog[l1]:
                    dlog[l1].append(idex)
                else:
                    testd+=1;
                addist[s][l1]=udv[l1];
                    
            for l1 in range(3):
                idex=int(np.floor(cell[l1]/dscell))
                if idex not in celllog[l1]:
                    celllog[l1].append(idex)
                else:
                    testcell+=1;  
                    
                idex=int(np.floor(cell[l1+3]/dacell))
                if idex not in celllog[l1+3]:
                    celllog[l1+3].append(idex)
                else:
                    testcell+=1;
            
            print(s,'Bond:',len(ubi)-testb,'An:',len(uai)-testa,'DH:',len(udi)-testd,'Cell',6-testcell,round(a1,2),round(b1,2),round(c1,2),round(al,3),round(be,3),round(ga,3),Esnap)
            if( testb==len(ubi) and testa==len(uai) and testd==len(udi) and testcell==6):
                ntest_al +=1;
                print(s,' Its test set and ntest = ',ntest_al)
                fte.write(f"{Na}\n")
                fte.write(f"{sno} {Esnap} {A[0]} {A[1]} {A[2]} {B[0]} {B[1]} {B[2]} {C[0]} {C[1]} {C[2]} {stress[0]} {stress[1]} {stress[2]} {stress[3]} {stress[4]} {stress[5]}\n")
                for ii in range(Na):
                    fte.write(f"{elements[ii]} \t{coord[ii][0]} \t{coord[ii][1]} \t{coord[ii][2]} \t{force[ii][0]} \t{force[ii][1]} \t{force[ii][2]} \t{qd[ii]}\n")
            else:
                ntrain_al +=1;
                ftr.write(f"{Na}\n")
                ftr.write(f"{sno} {Esnap} {A[0]} {A[1]} {A[2]} {B[0]} {B[1]} {B[2]} {C[0]} {C[1]} {C[2]} {stress[0]} {stress[1]} {stress[2]} {stress[3]} {stress[4]} {stress[5]}\n")
                for ii in range(Na):
                    ftr.write(f"{elements[ii]} \t{coord[ii][0]} \t{coord[ii][1]} \t{coord[ii][2]} \t{force[ii][0]} \t{force[ii][1]} \t{force[ii][2]} \t{qd[ii]}\n")
            #---------------------------------------
        print('Out of',self.Ndset,'configurations',ntrain_al,'are for training and',ntest_al,'are for test',db,da,dda,dscell,dacell)
        fs.close()
        ftr.close()
        fte.close()
        
        dlen=[len(x) for x in blog];
        print('Atoms with maximum bond fluctuation',ubi[dlen.index(max(dlen))], 'with value of ', max(dlen))
        print('Atoms with minimum bond fluctuation',ubi[dlen.index(min(dlen))], 'with value of ', min(dlen))
        dlen=[len(x) for x in alog];
        print('Atoms with maximum angle fluctuation',uai[dlen.index(max(dlen))],'with value of ', max(dlen))
        print('Atoms with minimum angle fluctuation',uai[dlen.index(min(dlen))], 'with value of ', min(dlen))
        dlen=[len(x) for x in dlog];
        print('Atoms with maximum DHA fluctuation',udi[dlen.index(max(dlen))],'with value of ', max(dlen))
        print('Atoms with minimum DHA fluctuation',udi[dlen.index(min(dlen))], 'with value of ', min(dlen))
        
        pp=AS.PlotD();
        pp.plot_IChist(bdist,250,'$Bond length (\AA)$','Bond_hist')
        pp.plot_IChist(adist,180,'$Angle (^o)$','Angle_hist')
        pp.plot_IChist(ddist,180,'$Dihedral Angle (^o)$','Dihedral_hist')
        
        pp.plot_ICdist(abdist,'Bonds',self.special_indexes,'Bond_dist')
        pp.plot_ICdist(aadist,'Angles',self.special_indexes,'Angle_dist')
        pp.plot_ICdist(addist,'Dihedrals',self.special_indexes,'Dihedral_dist')
        #------------------------------------------------------------------------------       


    def modify_CBAD_index(self,club_bonds, club_angles, club_dihedrals, club_cell):
        usb,usa,usd,ana=self.get_unique_BADindex()         
        
        #-------------------------------------------
        blist = [i for i in range(len(usb))]
        for i in club_bonds:
            for j in i:
                blist.remove(j)
        for i in blist:
            club_bonds.append([i])
        club_bonds.sort()
        
        usb_n = [[] for _ in range(len(club_bonds))]
        for i in range(len(club_bonds)):
            for j in club_bonds[i]:
                usb_n[i] += usb[j]
        #------------------------------------------ 
        #-------------------------------------------
        alist = [i for i in range(len(usa))]
        for i in club_angles:
            for j in i:
                alist.remove(j)
        for i in alist:
            club_angles.append([i])
        club_angles.sort()

        usa_n = [[] for _ in range(len(club_angles))]
        for i in range(len(club_angles)):
            for j in club_angles[i]:
                usa_n[i] += usa[j]
        #------------------------------------------
        #-------------------------------------------
        dlist = [i for i in range(len(usd))]
        for i in club_dihedrals:
            for j in i:
                dlist.remove(j)
        for i in dlist:
            club_dihedrals.append([i])
        club_dihedrals.sort()

        usd_n = [[] for _ in range(len(club_dihedrals))]
        for i in range(len(club_dihedrals)):
            for j in club_dihedrals[i]:
                usd_n[i] += usd[j]
        #------------------------------------------
        print(club_bonds)
        for i in range(len(usb_n)):
            print(len(usb_n[i]),usb_n[i][0])
        print(club_angles)
        for i in range(len(usa_n)):
            print(len(usa_n[i]),usa_n[i][0])
        print(club_dihedrals)
        for i in range(len(usd_n)):
            print(i,len(usd_n[i]),usd_n[i][0])
        for i in range(len(usd)):
            print(i,len(usd[i]),usd[i][0])

        np.savez('dist.npz',  bindex=usb_n, aindex=usa_n, dindex=usd_n)
        return usb_n,usa_n,usd_n,ana;


    def dist_CBAD(self,club_bonds, club_angles, club_dihedrals, club_cell,bticks,aticks,dticks):
        usb,usa,usd,ana=self.modify_CBAD_index(club_bonds, club_angles, club_dihedrals, club_cell)

        abdist=[[] for _ in range(len(usb))];
        aadist=[[] for _ in range(len(usa))];
        addist=[[] for _ in range(len(usd))];

        nbonds=sum([len(i) for i in usb]);
        nangles=sum([len(i) for i in usa]);
        ndihedrals=sum([len(i) for i in usd]);
        print(f'{nbonds} Bonds; {nangles} Angles; {ndihedrals} Dihedrals')
        fs=open(self.fname,'r')
        #---------------------------------------------------
        for s in range(self.Ndset):
            print(s)
            Na=int(fs.readline().split()[0]);
            l2=fs.readline().split();
            sno=l2[0];
            Esnap=float(l2[1]);
            A=[float(k) for k in l2[2:5]];B=[float(k) for k in l2[5:8]];C=[float(k) for k in l2[8:11]];
            #volume=np.dot(A,np.cross(B,C))

            elements=[];
            coord=np.zeros([Na,3],dtype=float);#>>Atomic coordinates
            for i in range(Na):
                l=fs.readline().split();
                elements.append(l[0]);
                coord[i]=l[1:4];

            self.S.set_cell([A,B,C])
            self.S.set_positions(coord,apply_constraint=False)
            #-----------------------------------------------------------
            ubv,uav,udv=self.get_unique_BADval(ana,usb,usa,usd)
            #for _ in udv:
            #    print('**   ',len(_),_)
            #print(' ')
            cc=AS.Cell();
            a1,b1,c1,al,be,ga=cc.vec2para(A,B,C)
            cell=[a1,b1,c1,al,be,ga]
            del cc
            #---------------------------------------
            for l1 in range(len(usb)):
                for l2 in range(len(usb[l1])):
                    abdist[l1].append(ubv[l1][l2]);

            for l1 in range(len(usa)):
                for l2 in range(len(usa[l1])):
                    aadist[l1].append(uav[l1][l2]);

            for l1 in range(len(usd)):
                for l2 in range(len(usd[l1])):
                    addist[l1].append(udv[l1][l2]);
        fs.close()

        plt.figure(figsize=(10, 10))
        plt.violinplot(abdist)
        for l1 in range(len(abdist)):
            for l2 in abdist[l1]:
                plt.plot(l1+1,l2,'b.',markersize=3.0);
        plt.xticks([i+1 for i in range(len(abdist))], bticks, rotation=80)
        plt.ylabel('$Bond\ distance\ (\\AA)$');
        plt.savefig('Bond_dist.png', dpi=300,bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(15, 10))
        plt.violinplot(aadist)
        for l1 in range(len(aadist)):
            for l2 in aadist[l1]:
                plt.plot(l1+1,l2,'b.',markersize=3.0);
        plt.xticks([i+1 for i in range(len(aadist))], aticks, rotation=80)
        plt.ylabel('$Bond\ angle\ (^o)$');
        plt.savefig('Angle_dist.png', dpi=300,bbox_inches="tight")
        plt.close()

        print('\n ---',len(addist[7]))
        plt.figure(figsize=(20, 10))
        plt.violinplot(addist)
        for l1 in range(len(addist)):
            for l2 in addist[l1]:
                plt.plot(l1+1,l2,'b.',markersize=3.0);
        plt.xticks([i+1 for i in range(len(addist))], dticks, rotation=80)
        plt.ylabel('$Dihedral\ angle\ (^o)$');
        plt.savefig('Dihedral_dist.png', dpi=300,bbox_inches="tight")
        plt.close()

        #pp.plot_ICdist(aadist,'Angles',self.special_indexes,'Angle_dist')
        #pp.plot_ICdist(addist,'Dihedrals',self.special_indexes,'Dihedral_dist')

        #------------------------------------------------------------------------------
    #
#
class Configuration:
    def __init__(self,A,B,C,elem,coord,E,F,stress,qd):
        self.A=A
        self.B=B
        self.C=C
        self.elem=elem                                      #List of all elements serially
        self.uelem=list(set(elem))                          #Unique list of elements
        self.uelem_freq=[elem.count(i) for i in self.uelem] #Number of each elements in configuration
        self.Na=len(elem)
        self.coord=coord        
        self.E=E
        self.F=F
        self.stress=stress
        self.qd=qd
        self.volume=A.dot(np.cross(B,C))
        
    def vec2para(self):# Function to convert lattice vectors to lattice parameters a,b,c,alpha,beta,gamma
        a=np.sqrt(self.A.dot(self.A));
        b=np.sqrt(self.B.dot(self.B));
        c=np.sqrt(self.C.dot(self.C));
        alpha=math.acos(self.B.dot(self.C)/b/c);
        beta=math.acos(self.A.dot(self.C)/a/c);
        gamma=math.acos(self.B.dot(self.A)/a/b);
        return a,b,c,alpha,beta,gamma
    
    def vec2lmp(self):# Function to convert lattice vectors to lattice parameters a,b,c,alpha,beta,gamma
        [a,b,c,alpha,beta,gamma]=self.vec2para();
        lx=a;
        xy=b*math.cos(gamma);
        xz=c*math.cos(beta);
        ly=math.sqrt(b*b-xy*xy);
        yz=(b*c*math.cos(alpha)-xy*xz)/ly;
        lz=math.sqrt(c*c-xz*xz-yz*yz);        
        return lx,ly,lz,xy,xz,yz;
    
    def write_lmpdata_snap(self,bs_dict,path): # Replace it to create lammps object directly with atoms, to avoid file creation
        ele=[bs_dict['e'+str(i+1)] for i in range(bs_dict['nat'])];#List of atom types
        #Writing lammps data file to generate bispectrum components
        os.system('rm '+path+'data.snap')
        fd=open(path+'data.snap','w');
        fd.write('LAMMPS data file to generate snap fitting equations\n\n');
        fd.write(f"{self.Na} atoms\n0 bonds\n0 angles\n0 dihedrals\n0 impropers\n\n{bs_dict['nat']} atom types\n0 bond types\n0 angle types\n0 dihedral types\n0 improper types\n\n");
        
        #volume=a*b*c*math.sqrt(1+2*math.cos(alpha)*math.cos(beta)*math.cos(gamma) -math.cos(alpha)**2 - math.cos(beta)**2 -math.cos(gamma)**2)
        [lx,ly,lz,xy,xz,yz]=self.vec2lmp()
        xmin=0.;xmax=lx;
        ymin=0.;ymax=ly;
        zmin=0.;zmax=lz;    
        fd.write('   '+str(xmin)+' '+str(xmax)+' xlo xhi \n');### Change here to control box dimension
        fd.write('   '+str(ymin)+' '+str(ymax)+' ylo yhi \n');
        fd.write('   '+str(zmin)+' '+str(zmax)+' zlo zhi \n');
        fd.write('   '+str(round(xy,3))+' '+str(round(xz,3))+' '+str(round(yz,3))+' xy xz yz \n');### Change here to control box tilt
    
        fd.write('\nMasses\n\n')
        for i in range(bs_dict['nat']):fd.write(f"{i+1} {bs_dict['m'+str(i+1)]}\n");
        fd.write('\nAtoms\n\n')
        for i in range(self.Na):
            at=str(ele.index(self.elem[i])+1);
            cw=str(self.qd[i])+'\t'+str(self.coord[i][0])+'\t'+str(self.coord[i][1])+'\t'+str(self.coord[i][2]);
            fd.write(str(i+1)+'\t1\t'+at+'\t'+cw+'\t0\t0\t0\n');
        fd.close();
        
    def write_lmpdata_wbonds(self,bs_dict,path,ubi_list): # Replace it to create lammps object directly with atoms, to avoid file creation
        ele=[bs_dict['e'+str(i+1)] for i in range(bs_dict['nat'])];#List of atom types
        #Writing lammps data file to generate bispectrum components
        os.system('rm '+path+'data.snap')
        fd=open(path+'data.snap','w');
        fd.write('LAMMPS data file to generate snap fitting equations\n\n');
        nbonds=len(ubi_list)
        
        
        fd.write(f"{self.Na} atoms\n{nbonds} bonds\n0 angles\n0 dihedrals\n0 impropers\n\n{bs_dict['nat']} atom types\n1 bond types\n0 angle types\n0 dihedral types\n0 improper types\n\n");
        
        #volume=a*b*c*math.sqrt(1+2*math.cos(alpha)*math.cos(beta)*math.cos(gamma) -math.cos(alpha)**2 - math.cos(beta)**2 -math.cos(gamma)**2)
        [lx,ly,lz,xy,xz,yz]=self.vec2lmp()
        xmin=0.;xmax=lx;
        ymin=0.;ymax=ly;
        zmin=0.;zmax=lz;    
        fd.write('   '+str(xmin)+' '+str(xmax)+' xlo xhi \n');### Change here to control box dimension
        fd.write('   '+str(ymin)+' '+str(ymax)+' ylo yhi \n');
        fd.write('   '+str(zmin)+' '+str(zmax)+' zlo zhi \n');
        fd.write('   '+str(round(xy,3))+' '+str(round(xz,3))+' '+str(round(yz,3))+' xy xz yz \n');### Change here to control box tilt
    
        fd.write('\nMasses\n\n')
        for i in range(bs_dict['nat']):fd.write(f"{i+1} {bs_dict['m'+str(i+1)]}\n");
        fd.write('\nAtoms\n\n')
        for i in range(self.Na):
            at=str(ele.index(self.elem[i])+1);
            cw=str(self.qd[i])+'\t'+str(self.coord[i][0])+'\t'+str(self.coord[i][1])+'\t'+str(self.coord[i][2]);
            fd.write(str(i+1)+'\t1\t'+at+'\t'+cw+'\t0\t0\t0\n');
            
        fd.write('\nBonds\n\n')
        for i in range(len(ubi_list)):
            fd.write(f'{i+1} 1 {ubi_list[i][0]+1} {ubi_list[i][1]+1}\n')
        fd.close();
            

