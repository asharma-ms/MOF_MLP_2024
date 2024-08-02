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

from sklearn.linear_model import Ridge
#from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

import scipy.optimize as optimize

import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 20
plt.rcParams['axes.linewidth'] = 3
plt.rcParams['font.weight'] = 'bold'# 'normal' #'bold'
#plt.rcParams['axes.labelweight']='bold'
plt.rcParams['axes.labelsize']=25
plt.rcParams['legend.fontsize']=20

import dam.mlp_data as MLD
#import concurrent.futures
import time
import multiprocessing
from itertools import repeat
#-----------------------------------------------------------------------------

class Cell:
    def lmpbox2vec(self,box):# Function to convert lattice parameters a,b,c,alpha,beta,gamma to lammps box parameters lx ly lz xy xz yz
        xy=box[2];yz=box[3];xz=box[4];
        lx=box[1][0]-box[0][0];ly=box[1][1]-box[0][1];lz=box[1][2]-box[0][2];
        A=[lx,0,0];
        B=[xy,ly,0];
        C=[xz,yz,lz]
        return A,B,C;
    #
    def cellstring2vol(st):
        x=np.array(st.split(),dtype=float).reshape(3,3);
        #A=x[0].copy();B=x[1].copy();C=x[2].copy()
        return np.dot(x[0],np.cross(x[1],x[2]))
    #
    def vec2para(self,A,B,C):#Function to convert lattice vectors to lattice parameters a,b,c,alpha,beta,gamma
        a=math.sqrt(A[0]*A[0]+A[1]*A[1]+A[2]*A[2]);
        b=math.sqrt(B[0]*B[0]+B[1]*B[1]+B[2]*B[2]);
        c=math.sqrt(C[0]*C[0]+C[1]*C[1]+C[2]*C[2]);
        alpha=math.acos((B[0]*C[0]+B[1]*C[1]+B[2]*C[2])/self.modvec(B)/self.modvec(C));
        beta=math.acos((A[0]*C[0]+A[1]*C[1]+A[2]*C[2])/self.modvec(A)/self.modvec(C));
        gamma=math.acos((B[0]*A[0]+B[1]*A[1]+B[2]*A[2])/self.modvec(A)/self.modvec(B));
        return a,b,c,alpha,beta,gamma;

    def modvec(self,X):
        return math.sqrt(X[0]*X[0]+X[1]*X[1]+X[2]*X[2])
#    
 
class LampCmd:#This class have functions to execute useful lammps commands for different purposes
    def basic_cmd(self,lmp):
        lmp.command("units metal");
        lmp.command("dimension 3");
        lmp.command("boundary p p p");
        lmp.command("atom_style full");
        lmp.command("read_data data.snap");
        #lmp.command("thermo_style custom step temp press etotal ke pe evdwl ecoul elong cella cellb cellc cellalpha cellbeta cellgamma  vol density pxx pyy pzz")
        #lmp.command("thermo 1")

    def get_bs_cmd(self,lmp,bs_dict):
        st=" ";
        for i in range(bs_dict['nat']):st=st+" "+str(bs_dict['r'+str(i+1)])+" ";
        for i in range(bs_dict['nat']):st=st+" "+str(bs_dict['w'+str(i+1)])+" ";
        cmd_st="variable  snap_options string '"+str(bs_dict['rcutfac'])+" "+str(bs_dict['rfac0'])+" "+  str(bs_dict['twojmax'])+" "+st+" rmin0 0.0 quadraticflag 0 bzeroflag 1 switchflag 1'"
        lmp.command(cmd_st)
        lmp.command('compute snap_p all snap    ${snap_options}')
        #lmp.command('fix     snap1 all ave/time 1 1 1 c_snap_p[*] file snap.eq mode vector')
        #lmp.command('compute b  all sna/atom  ${snap_options}')
        #lmp.command('dump ef all custom 1 dump.myforce id type x y z')
        lmp.command("run 0");

    def bseq_zeropp(self,lmp,bs_dict):
        self.basic_cmd(lmp)
        ##For zero pair potential
        lmp.command("pair_style      zero 10.0");
        lmp.command("pair_coeff      * *");        
        self.get_bs_cmd(lmp,bs_dict)
        
    def bseq_mcoulomb(self,lmp,bs_dict):#Be careful before using it and check what it is doing
        self.basic_cmd(lmp)
        #For electrostatic energy calculation using pppm method
        lmp.command("pair_style lj/cut/coul/long 8.0")
        lmp.command("pair_coeff      * * 0 0");
        lmp.command("kspace_style pppm 1.0e-6")
        lmp.command("dielectric 1.0")
        lmp.command("special_bonds lj 0.0 0.0 0.0 coul 0.0 1.0 1.0")        
        self.get_bs_cmd(lmp,bs_dict)
        
    def snaptest(self,lmp,estring):
        #estring: string of elements in order of data file
        self.basic_cmd(lmp)
        lmp.command('pair_style snap')
        lmp.command('pair_coeff * * snapcoeff snapparam '+estring)
        
        
    def bseq_charge(self,lmp,bs_dict): #Bispectrum equations for charge prediction
        self.basic_cmd(lmp)
        lmp.command("pair_style      zero 10.0");
        lmp.command("pair_coeff      * *");
        
        st=" ";
        for i in range(bs_dict['nat']):st=st+" "+str(bs_dict['r'+str(i+1)])+" ";
        for i in range(bs_dict['nat']):st=st+" "+str(bs_dict['w'+str(i+1)])+" ";
        cmd_st="variable  snap_options string '"+str(bs_dict['rcutfac'])+" "+str(bs_dict['rfac0'])+" "+  str(bs_dict['twojmax'])+" "+st+" rmin0 0.0 quadraticflag 0 bzeroflag 1 switchflag 1'"
        lmp.command(cmd_st)
        lmp.command('compute snap_p all snap    ${snap_options}') 
        #lmp.command('fix     snap1 all ave/time 1 1 1 c_snap_p[*] file snap.eq mode vector')
        lmp.command('compute snap_c  all sna/atom  ${snap_options}')
        #lmp.command('dump ef all custom 1 dump.myforce id type x y z')
        lmp.command("run 0");
        
        
        
    def bseq_zeropp_nodata(self,lmp,bs_dict,Config): #No need for data file, it will assign coordinates and cell by itself
            ele=[bs_dict['e'+str(i+1)] for i in range(bs_dict['nat'])];
            lmp.command("clear")
            lmp.command("units metal");
            lmp.command("dimension 3");
            lmp.command("boundary p p p");
            lmp.command("atom_style full");
            [lx,ly,lz,xy,xz,yz]=Config.vec2lmp();
            #lmp.command(f"region simbox prism  0 {lx} 0 {ly} 0 {lz} {round(xy,3)} {round(xz,3)} {round(yz,3)}")
            lmp.command(f"region simbox prism  0 {lx} 0 {ly} 0 {lz} {xy} {xz} {yz}")
            lmp.command("create_box 4  simbox")

            aid=[];atype=[];
            for at in range(Config.Na):
                aid.append(at+1)
                atype.append(int(ele.index(Config.elem[at])+1))            
            lmp.create_atoms(Config.Na, aid, atype,Config.coord.ravel().ctypes,None)   
            
            for i in range(bs_dict['nat']):
                lmp.command(f"mass {i+1} {bs_dict['m'+str(i+1)]}"); 
            
            lmp.command("thermo_style custom step temp press etotal ke pe evdwl ecoul elong cella cellb cellc cellalpha cellbeta cellgamma  vol density pxx pyy pzz")
            lmp.command("thermo 1")        
            lmp.command("pair_style      zero 10.0");
            lmp.command("pair_coeff      * *"); 
            
            self.get_bs_cmd(lmp,bs_dict)
        
    def md(self,lmp,estring):
        #estring: string of elements in order of data file
        self.basic_cmd(lmp)
        lmp.command('pair_style snap')
        lmp.command('pair_coeff * * snapcoeff snapparam '+estring)
        lmp.command('thermo 1')
        lmp.command('thermo_style custom  step temp etotal epair etail ke pe ecoul elong evdwl emol')
        lmp.command('thermo_modify   norm no')
        lmp.command('dump ef all custom 1 dump.myforce id type x y z vx vy vz')
        lmp.command('dump tj1 all dcd 1 MD_traj.dcd')

        block="""
        variable s equal step
        variable t equal temp
        variable p equal pe
        variable k equal ke
        variable e equal etotal
        variable v equal vol
        fix extra all print 1 "$s $t $p $k $e $v" file "sim_details.txt" screen no
        """
        lmp.commands_string(block)
        block = """
        fix 1 all box/relax aniso 1.0
        min_style cg
        minimize 0.0 1.0e-6 100 1000
        unfix 1
        """
        lmp.commands_string(block)
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
        fd.write('   '+str(xy)+' '+str(xz)+' '+str(yz)+' xy xz yz \n');### Change here to control box tilt round(xy,3)
    
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
        fd.write('   '+str(xy)+' '+str(xz)+' '+str(yz)+' xy xz yz \n');### Change here to control box tilt
    
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
            

#    
class SnapEQ:
    #Objects of this class can read data from xyz file and create corresponding
    #bispectrum equations for fitting
    def __init__(self,fname,ntrain,ftype,bs_dict,flr_el):
        self.fname=fname             #Name of XYZ file with details of E, F, and S
        self.ntrain=ntrain           #Number of training sets to read from the file
        self.ftype=ftype             #Type of fit to read from this file E,EF,EFS
        self.bs_dict=bs_dict.copy(); #Dictionary with details about computation of bispectrum components
        self.flr_el=flr_el;          #flag T/F 1/0 to consider long range electrostatic interaction in snap equations or not
        self.etype = None
        
        if('E' in self.ftype):
            self.nfpara=(bs_dict['nbspa']+1)*bs_dict['nat'];#Number of parameters to fit=(Number of bispectum components+1)*Number of atom types
        else:
            self.nfpara=(bs_dict['nbspa'])*bs_dict['nat'];#Number of parameters to fit=(Number of bispectum components)*Number of atom types
        
        self.Natoms=int(linecache.getline(self.fname,1).split()[0])
        self.ntotatoms=0;
        for ii in range(ntrain):
            self.ntotatoms+=int(linecache.getline(self.fname,self.ntotatoms+2*ii+1).split()[0])
        linecache.clearcache();
        #print('total number of lines',self.ntotatoms+self.ntrain*2)
        
        ##List of atom types present in the each of all configurations--
        self.current_at=[];
        self.at_index=[];
        for at in range(self.bs_dict['nat']): 
            self.current_at.append(self.bs_dict['e'+str(at+1)]);
            self.at_index.append(at)
            
        #Getting detail of bonds-----------
        if(self.flr_el):self.ubi_list=self.get_bonds_details(1);#lno i 1: fetching bond details from the first configuration
        
        
        self.neq=0;                  # Number of equations to fit 
        if('E' in self.ftype):
            self.neq = self.neq + self.ntrain
        if('F' in self.ftype):   
            self.neq = self.neq + 3*self.ntotatoms
        if('S' in self.ftype): 
            self.neq = self.neq + 6*self.ntrain
        if('Q' in self.ftype): 
            self.nfpara=(bs_dict['nbspa']+1)*bs_dict['nat'];#Number of parameters to fit=(Number of bispectum components+1)*Number of atom types
            ##self.neq = self.neq + self.ntrain + self.ntotatoms # Inculding charge neutrality conditions + charge on each atom
            self.neq = self.neq + self.ntotatoms # Inculding charge neutrality conditions + charge on each atom
            
        self.lno = 0;#Initial line number
        self.bscoef = [];#Initially no fitted coefficients
        
        self.Aeq  = np.zeros([self.neq, self.nfpara], dtype=float);#np.empty((0,self.nfpara));
        self.Beq  = np.zeros(self.neq,dtype=float);#np.empty((0,1));
        self.Weq  = np.zeros(self.neq,dtype=float);#np.empty((0,1));
        self.Pred = np.zeros(self.neq,dtype=float);#np.empty((0,1));
        
        self.config_vols = np.zeros(self.ntrain,dtype=float) # To store list of volumes (A**3) for each configuration
    #
    
    def read_axyz(self,lno):#Function to read one configuration from Abhishek's style of xyz file
        #lno:Line number of first line (where Natoms are given) of configuration, it starts from 1
        Natoms=int(linecache.getline(self.fname,lno).split()[0]);
        l2=linecache.getline(self.fname,lno+1).split();
        #sno=l2[0];
        Esnap=float(l2[1]);
        A=np.array(l2[2:5],dtype=float) #; l2[2:5].copy()
        B=np.array(l2[5:8],dtype=float)
        C=np.array(l2[8:11],dtype=float)        
        stress=np.array(l2[11:17],dtype=float);#print(Na,sno,Esnap,A,B,C,stress);
        
        elements=[];
        coord=np.zeros([Natoms,3],dtype=float);#>>Atomic coordinates
        force=np.zeros([Natoms,3],dtype=float);#>>Atomic forces
        qd=np.zeros(Natoms,dtype=float);#>>Dynamic charges
        for i in range(Natoms):
            l=linecache.getline(self.fname,lno+2+i).split();
            elements.append(l[0]);
            coord[i]=l[1:4];
            force[i]=l[4:7];
            ##**qd[i]=l[7];
            
        Config=Configuration(A,B,C,elements,coord,Esnap,force,stress,qd)
        linecache.clearcache();
        return Config;
    #
    
    def get_bonds_details(self,lno):
        li=self.lno;
        Ci=self.read_axyz(lno) #Reading the sanpshot based on the value of self.lno
        self.lno=li         
        ts=MLD.MLSetAnalysis(self.fname,1,Ci.A.tolist(),Ci.B.tolist(),Ci.C.tolist(),[0])
        ubi,uai,udi,ana=ts.get_BADindex()
        return ubi;
    #

    def get_BSEQ(self):#For fix number of atoms in all snapshots within the order of bs_dict       
        for i in range(self.ntrain): # This need to be parallel
           self.fill_BSEQ(i)
           
        #p = multiprocessing.Pool(4)
        #res = p.map(self.fill_BSEQ, range(self.ntrain))
    #
    
    def fill_BSEQ(self,config_id):#For fix number of atoms in all snapshots within the order of bs_dict
        #config_id: ID of configuration written in the file, starts from 0
        nbs=self.bs_dict['nbspa']
        ncf=nbs+1;
        Natoms=int(linecache.getline(self.fname,1).split()[0]);#Assuming constant number of atoms in all files and reading it from first line
        rof=0;
        rows=0;
        #print ('Start reading snapshots',time.strftime("%H:%M:%S"))
        cwd=os.getcwd();                
        #----------------------------------------------------------------------
        lno=config_id*(2+Natoms)+1;
        print('reading configuration',config_id)
        Ci=self.read_axyz(lno)   
        
        self.config_vols[config_id]=Ci.volume;
        
        #if(self.flr_el):Ci.write_lmpdata_wbonds(self.bs_dict,cwd+'/',ubi_list)
        #else:Ci.write_lmpdata_snap(self.bs_dict,cwd+'/')
        
        #Natoms=Ci.Na;
        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        #lmp=lammps(cmdargs=['-log','log.lmp','-screen','none']);
        lmp = lammps(cmdargs=['-log','none','-screen','none']);
        almpc = LampCmd()                           
        if(self.flr_el):
            almpc.bseq_mcoulomb(lmp,self.bs_dict)
        else:
            Ci.write_lmpdata_snap(self.bs_dict,cwd+'/')
            almpc.bseq_zeropp(lmp,self.bs_dict)           
            #almpc.bseq_zeropp_nodata(lmp,self.bs_dict,Ci)
        
        sbs = lmp.extract_compute("snap_p", 0, 2)
        #print(sbs[0][0].__sizeof__())
        #b=lmp.extract_compute("b",1,2)
        #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<       
        #---------------------------------------------------------------------
        #Reading snap equations and values to minus from quantum data (m_val) 
        if('E' in self.ftype):
            #First filling energy equations------------------------------------------
            row=0;
            rowe=config_id;
            rowf=self.ntrain;
            rows=self.ntrain;
            
            for at in self.at_index:#over present atom types
                self.Aeq[rowe][at*ncf]=Ci.elem.count(self.current_at[at]);#Filling number of atoms
                self.Aeq[rowe][at*ncf+1:at*ncf+1+nbs]=sbs[row][at*nbs:at*nbs+nbs];
            self.Beq[rowe]=Ci.E-sbs[row][-1];#=np.vstack((self.Eval,Ci.E-sbs[row][-1]));#np.append(self.Eval,Ci.E-sbs[row][-1])
            self.Weq[rowe]=self.we_fac/Natoms;#=np.append(self.We,6*1./Natoms)            
            #print(self.Beq[rowe],rowe,config_id,Ci.E)
            rowe+=1;           
            if('F' in self.ftype):#Filling force equations-------------------------------------------
                row=1;
                rowf=rowf+config_id*(3*Natoms);
                rows=rows+3*self.ntotatoms;
                for ft in range(Natoms):
                    for cc in range(3): 
                        for at in self.at_index:#over present atom types 
                            self.Aeq[rowf][at*ncf+1:at*ncf+1+nbs]=sbs[row][at*nbs:at*nbs+nbs];  
                        self.Beq[rowf]=Ci.F[ft][cc]-sbs[row][-1];
                        self.Weq[rowf]=1.;
                        row+=1;rowf+=1;                   
            if('S' in self.ftype):#Filling stress equations-------------------------------------------   
                row=3*Natoms+1;
                rows=rows+config_id*6;                         
                for st in range(6):
                    for at in self.at_index:#over present atom types
                        self.Aeq[rows][at*ncf+1:at*ncf+1+nbs]=sbs[row][at*nbs:at*nbs+nbs];
                    self.Beq[rows]=Ci.stress[st] - sbs[row][-1]*Ci.volume/160.21766/10000.;
                    self.Weq[rows]=self.we_fac/Ci.volume;
                    row+=1;rows+=1;
        else:
            if('F' in self.ftype):#Filling force equations-------------------------------------------
                row=1;
                rowf=config_id*(3*Natoms);
                rows=3*self.ntotatoms;
                for ft in range(Natoms):
                    for cc in range(3):   
                        for at in self.at_index:#over present atom types 
                            self.Aeq[rowf][at*nbs:at*nbs+nbs]=sbs[row][at*nbs:at*nbs+nbs];  
                        self.Beq[rowf]=Ci.F[ft][cc]-sbs[row][-1];
                        self.Weq[rowf]=1.;
                        row+=1;rowf+=1;                   
            if('S' in self.ftype):#Filling stress equations-------------------------------------------   
                row=3*Natoms+1;
                rows=rows+config_id*6;                         
                for st in range(6):
                    for at in self.at_index:#over present atom types
                        self.Aeq[rows][at*nbs:at*nbs+nbs]=sbs[row][at*nbs:at*nbs+nbs];
                    self.Beq[rows]=Ci.stress[st] - sbs[row][-1]*Ci.volume/160.21766/10000.;
                    self.Weq[rows]=self.we_fac/Ci.volume;
                    row+=1;rows+=1;
        lmp.close(); 
        #print(self.Aeq[rowe-1][0:5])
        del sbs,Ci
    #
    
    
    def get_qBSEQ(self):#For fix number of atoms in all snapshots within the order of bs_dict       
        for i in range(self.ntrain): # This need to be parallel
           self.fill_BSEQ_charge(i)
    #
    
    def fill_BSEQ_charge(self,config_id):#For fix number of atoms in all snapshots within the order of bs_dict
        #config_id: ID of configuration written in the file, starts from 0
        nbs=self.bs_dict['nbspa']
        ncf=nbs+1;
        Natoms=int(linecache.getline(self.fname,1).split()[0]);#Assuming constant number of atoms in all files and reading it from first line
        #print ('Start reading snapshots',time.strftime("%H:%M:%S"))
        cwd=os.getcwd();                
        #----------------------------------------------------------------------
        lno=config_id*(2+Natoms)+1;
        print('reading configuration',config_id)
        Ci=self.read_axyz(lno)   
        
        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        #lmp=lammps(cmdargs=['-log','log.lmp','-screen','none']);
        lmp = lammps(cmdargs=['-log','none','-screen','none']);
        almpc = LampCmd()                           
        Ci.write_lmpdata_snap(self.bs_dict,cwd+'/')
        almpc.bseq_charge(lmp,self.bs_dict) 
        
        sbs_tot = lmp.extract_compute("snap_p", 0, 2)
        sbs_i = lmp.extract_compute("snap_c", 1, 2)
        #print(sbs[0][0].__sizeof__())
        #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<       
        #---------------------------------------------------------------------
        rowq=0;
        row=0;
        #for i in range(56):
        #    print(sbs[0][i],sbs[30][i],sbs[31][i])
        
        ele = [self.bs_dict['e'+str(i+1)] for i in range(self.bs_dict['nat'])];
        if('Q' in self.ftype):
            row=0;
            rowq=config_id*(Natoms);
            for ft in range(Natoms):
                at = ele.index(Ci.elem[ft]);# Finding atom types                
                self.Aeq[rowq][at*ncf]=1;#w
                
                self.Aeq[rowq][at*ncf+1:at*ncf+1+nbs]=[sbs_i[row][ii] for ii in range(nbs)] #sbs[row][at*nbs:at*nbs+nbs]; 
                self.Beq[rowq]=Ci.qd[ft]
                self.Weq[rowq]=1;
                row+=1;rowq+=1;
                
            '''
            row=0;
            rowq=config_id;            
            for at in self.at_index:#over present atom types
                self.Aeq[rowq][at*ncf]=Ci.elem.count(self.current_at[at]);#Filling number of atoms
                self.Aeq[rowq][at*ncf+1:at*ncf+1+nbs]=sbs_tot[row][at*nbs:at*nbs+nbs];
            self.Beq[rowq]=0; #Sum of total charges
            self.Weq[rowq]=self.we_fac/Natoms;#=np.append(self.We,6*1./Natoms) 
 
            rowq=self.ntrain;
            rowq=rowq+config_id*(Natoms);
            for ft in range(Natoms):
                at = ele.index(Ci.elem[ft]);# Finding atom types                
                self.Aeq[rowq][at*ncf]=1;#w
                
                self.Aeq[rowq][at*ncf+1:at*ncf+1+nbs]=[sbs_i[row][ii] for ii in range(nbs)] #sbs[row][at*nbs:at*nbs+nbs]; 
                self.Beq[rowq]=Ci.qd[ft]
                self.Weq[rowq]=1;
                row+=1;rowq+=1;
            '''
        #print(rowq, self.Aeq[rowq-1],self.Beq[rowq-1],[sbs[275][i] for i in range(60)])
        #print(rowq, self.Aeq[rowq-2],self.Beq[rowq-2],[sbs[275][i] for i in range(56)])
            
        lmp.close(); 
        #print(self.Aeq[rowe-1][0:5])
        del sbs_tot, sbs_i, Ci
    #
    
    def get_BSEQ_parallel(self):#For fix number of atoms in all snapshots within the order of bs_dict       
        #for i in range(self.ntrain): # This need to be parallel
        #   self.fill_BSEQ(i)
              
        x = self.ntrain
        group_l = 500;
        
        bounds = [[(i-1)*group_l,i*group_l] for i in range(1,int(x/group_l)+1)]+[[int(x/group_l)*group_l,int(x/group_l)*group_l+x%group_l]]
        print(bounds)
        for b1 in bounds:
            print('Reading configurations from ', b1[0], ' to ', b1[1])
            p = multiprocessing.Pool(4)
            results = p.map(self.fill_BSEQ_parallel, range(b1[0],b1[1]), chunksize=4)
            p.close()
            Natoms=self.Natoms;
            for config_id in range(b1[0],b1[1]):
                rowf=0; 
                rows=0;
                #E, EF, F, FS, EFS: Possible ftypes
                if('E' in self.ftype):#Filling energy equations------------------------------------------
                    rowe=config_id;
                    rowf=self.ntrain;
                    rows=self.ntrain;
                    row=0;
                    #[configuration][eq/value/volume][eqno/value_id]
                    self.Aeq[rowe]=results[0][0][row] # Its 0 because always reading first element and then deleting it in end
                    self.Beq[rowe]=results[0][1][row]
                    self.Weq[rowe]=self.we_fac/Natoms;                
                if('F' in self.ftype):#Filling force equations-------------------------------------------
                    rowf=rowf+config_id*(3*Natoms);
                    rows=rows+3*self.ntotatoms;
                    row=1;
                    for ft in range(Natoms):
                        for cc in range(3):
                            self.Aeq[rowf]=results[0][0][row] # Its 0 because always reading first element and then deleting it in end
                            self.Beq[rowf]=results[0][1][row]
                            self.Weq[rowf]=1.; 
                            row+=1;rowf+=1;                        
                if('S' in self.ftype):#Filling stress equations-------------------------------------------   
                    rows=rows+config_id*6;                         
                    row=3*Natoms+1;
                    for st in range(6):
                        self.Aeq[rows]=results[0][0][row] # Its 0 because always reading first element and then deleting it in end
                        self.Beq[rows]=results[0][1][row]
                        self.Weq[rows]=self.we_fac/results[0][2];
                        row+=1;rows+=1;  
                        
                del results[0]  
            #del p
        #res = p.map(self.fill_BSEQ, [x for x in range(self.ntrain)])
        #res = p.starmap(self.fill_BSEQ, zip(repeat(self), range(self.ntrain)))
        
        
        #self.Weq[rowe]=self.we_fac/Natoms;#=np.append(self.We,6*1./Natoms)
        #self.Weq[rowf]=1.;
    #
    
    def fill_BSEQ_parallel(self,config_id):#For fix number of atoms in all snapshots within the order of bs_dict
        #config_id: ID of configuration written in the file, starts from 0
        nbs=self.bs_dict['nbspa']
        ncf=nbs+1;
        Natoms=int(linecache.getline(self.fname,1).split()[0]);#Assuming constant number of atoms in all files and reading it from first line
        cAeq=np.zeros([7+3*Natoms, self.nfpara], dtype=float);
        cBeq=np.zeros(7+3*Natoms, dtype=float);
        ##print ('Start reading snapshots',time.strftime("%H:%M:%S"))
        cwd=os.getcwd();                
        #----------------------------------------------------------------------
        lno=config_id*(2+Natoms)+1;
        print('reading config',config_id)
        Ci=self.read_axyz(lno)
        
        #if(self.flr_el):Ci.write_lmpdata_wbonds(self.bs_dict,cwd+'/',self.ubi_list)
        #else:Ci.write_lmpdata_snap(self.bs_dict,cwd+'/')
        #Natoms=Ci.Na;
        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        #lmp=lammps(cmdargs=['-log','log.lmp','-screen','none']);
        lmp = lammps(cmdargs=['-log','none','-screen','none']);
        almpc = LampCmd()                           
        if(self.flr_el):
            almpc.bseq_mcoulomb(lmp,self.bs_dict)
        else:
            #Ci.write_lmpdata_snap(self.bs_dict,cwd+'/')
            #almpc.bseq_zeropp(lmp,self.bs_dict)           
            almpc.bseq_zeropp_nodata(lmp,self.bs_dict,Ci)
        
        sbs = lmp.extract_compute("snap_p", 0, 2)
        #print(sbs[0][0].__sizeof__())
        #b=lmp.extract_compute("b",1,2)
        #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<       
        #---------------------------------------------------------------------
        #Reading snap equations and values to minus from quantum data (m_val) 
        row=0;
        for at in self.at_index:#over present atom types
            cAeq[row][at*ncf]=Ci.elem.count(self.current_at[at]);#Filling number of atoms
            cAeq[row][at*ncf+1:at*ncf+1+nbs]=sbs[row][at*nbs:at*nbs+nbs];
        cBeq[row]=Ci.E-sbs[row][-1];#=np.vstack((self.Eval,Ci.E-sbs[row][-1]));#np.append(self.Eval,Ci.E-sbs[row][-1])
                
        row=1;
        for ft in range(Natoms):
            for cc in range(3):   
                for at in self.at_index:#over present atom types 
                    cAeq[row][at*ncf+1:at*ncf+1+nbs]=sbs[row][at*nbs:at*nbs+nbs];  
                cBeq[row]=Ci.F[ft][cc]-sbs[row][-1];
                row+=1;       

        row=3*Natoms+1;                        
        for st in range(6):
            for at in self.at_index:#over present atom types
                cAeq[row][at*ncf+1:at*ncf+1+nbs]=sbs[row][at*nbs:at*nbs+nbs];
            cBeq[row]=Ci.stress[st] - sbs[row][-1]*Ci.volume/160.21766/10000.;
            row+=1; 
            
        volume=Ci.volume;
        lmp.close(); 
        #print(self.Aeq[rowe-1][0:5])
        del sbs,Ci
        return cAeq,cBeq,volume;
        #print ('End reading snapshots',time.strftime("%H:%M:%S"))
    #
    
    def get_NextBSEQ(self):#For fix number of atoms in all snapshots within the order of bs_dict
        nbs=self.bs_dict['nbspa']
        ncf=nbs+1;
        cwd=os.getcwd();
        ##Making list of atom types present in the each of all configurations--
        current_at=[];at_index=[];
        for at in range(self.bs_dict['nat']): 
            current_at.append(self.bs_dict['e'+str(at+1)]);
            at_index.append(at)
        #----------------------------------------------------------------------
        #print('-----****----- >>> Current line number i s',self.lno,self.fname)
        #print(os.popen('cat '+self.fname+' | wc -l').read(), self.lno)
        Ci=self.read_axyz()
        Ci.write_lmpdata_snap(self.bs_dict,cwd+'/')
        Natoms=Ci.Na;
        #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        #lmp=lammps(cmdargs=['-log','log.lmp','-screen','none']);
        lmp = lammps(cmdargs=['-log','none','-screen','none']);
        almpc = LampCmd()
        almpc.bseq_zeropp(lmp,self.bs_dict)
        sbs = lmp.extract_compute("snap_p", 0, 2)
        #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<       
        #---------------------------------------------------------------------
        #Reading snap equations and value to minus from quantum data (m_val)        
        if('E' in self.ftype):#Filling energy equations------------------------------------------
            row=0;
            cEeq = np.zeros([1, self.nfpara], dtype=float);#
            for at in at_index:#over present atom types
                cEeq[0][at*ncf] = Ci.elem.count(current_at[at]);#Filling number of atoms
                cEeq[0][at*ncf+1:at*ncf+1+nbs] = sbs[row][at*nbs:at*nbs+nbs];
            np.append(self.Eval,Ci.E-sbs[row][-1])
            np.append(self.We,self.we_fac/Natoms)
            np.append(self.Eeq,cEeq,axis=0)
           
        if('F' in self.ftype):#Filling force equations-------------------------------------------
            row=1;rowf=0;
            cFeq = np.zeros([3*Natoms, self.nfpara], dtype=float);#
            for ft in range(Natoms):
                for cc in range(3):   
                    for at in at_index:#over present atom types
                        cFeq[rowf][at*ncf+1:at*ncf+1+nbs]=sbs[row][at*nbs:at*nbs+nbs];  
                    np.append(self.Fval,Ci.F[ft][cc]-sbs[row][-1])
                    np.append(self.Wf,1.)
                    row+=1;rowf+=1; 
            np.append(self.Feq,cFeq,axis=0)
            
        if('S' in self.ftype):#Filling stress equations-------------------------------------------                            
            row=3*Natoms+1;rows=0;
            cSeq = np.zeros([6,  self.nfpara], dtype=float);#np.empty((0,self.nfpara));
            for st in range(6):
                for at in at_index:#over present atom types
                    cSeq[rows][at*ncf+1:at*ncf+1+nbs] = sbs[row][at*nbs:at*nbs+nbs];
                np.append(self.Sval, Ci.stress[st] - sbs[row][-1]*Ci.volume/160.21766/10000.)
                np.append(self.Ws,   self.we_fac/Ci.volume)
                row+=1;rows+=1;
            np.append(self.Seq,cSeq,axis=0)
        lmp.close(); 
        del Ci
    #
        
    def get_BSEQ_variableN(self):#For variable number of atoms in the snapshots
        nbs=self.bs_dict['nbspa']
        ncf=nbs+1;
        rowe=0;
        rowf=0;
        rows=0;
        print ('Start reading snapshots',time.strftime("%H:%M:%S"))
        for i in range(self.ntrain):
            #print('reading snap',i)
            Ci=self.read_axyz()
            Ci.write_lmpdata_snap(self.bs_dict,'/home/asharma/apycodes/')
            Natoms=Ci.Na;
        
            #lmp=lammps(cmdargs=['-log','log.lmp','-screen','none']);
            lmp=lammps(cmdargs=['-log','none','-screen','none']);
            almpc=LampCmd()
            almpc.bseq_zeropp(lmp,self.bs_dict)
            #almpc.bseq_mcoulomb(lmp,self.bs_dict)
            sbs=lmp.extract_compute("snap_p", 0, 2)
            #b=lmp.extract_compute("b",1,2) 
        
            ##Finding which atom type exists in the given configuration; **There must be atleast one**
            current_at=[];at_index=[];
            for at in range(self.bs_dict['nat']): 
                if(self.bs_dict['e'+str(at+1)] in Ci.uelem):#If atom type exists in the given configuration 
                    current_at.append(self.bs_dict['e'+str(at+1)]);
                    at_index.append(at)
            if(len(current_at)==0):
                print('No atom type found from BS_dictionary in the given configuration');
                sys.exit("Error")
            #---------------------------------------------------------------------
            #Reading snap equations and value to minus from quantum data (m_val) 
            if('E' in self.ftype):#Filling energy equations------------------------------------------
                row=0;
                for at in at_index:#over present atom types
                    self.Eeq[rowe][at*ncf]=Ci.elem.count(current_at[at]);#Filling number of atoms
                    self.Eeq[rowe][at*ncf+1:at*ncf+1+nbs]=sbs[row][at*nbs:at*nbs+nbs];
                self.Eval[rowe]=Ci.E-sbs[row][-1];#=np.vstack((self.Eval,Ci.E-sbs[row][-1]));#np.append(self.Eval,Ci.E-sbs[row][-1])
                self.We[rowe]=self.we_fac/Natoms;#=np.append(self.We,6*1./Natoms)
                rowe+=1;
                #print('\n',self.Eeq)            
            if('F' in self.ftype):#Filling force equations-------------------------------------------
                row=1;
                for ft in range(Natoms):
                    for cc in range(3):   
                        for at in at_index:#over present atom types 
                            self.Feq[rowf][at*ncf+1:at*ncf+1+nbs]=sbs[row][at*nbs:at*nbs+nbs];  
                        self.Fval[rowf]=Ci.F[ft][cc]-sbs[row][-1];
                        self.Wf[rowf]=1.;
                        row+=1;rowf+=1;                   
            if('S' in self.ftype):#Filling stress equations-------------------------------------------                            
                row=3*Natoms+1;
                for st in range(6):
                    for at in at_index:#over present atom types
                        self.Seq[rows][at*ncf+1:at*ncf+1+nbs]=sbs[row][at*nbs:at*nbs+nbs];
                    self.Sval[rows]=Ci.stress[st] - sbs[row][-1]*Ci.volume/160.21766/10000.;
                    self.Ws[rows]=self.we_fac/Ci.volume
                    row+=1;rows+=1;
            lmp.close(); 
            del Ci
        print ('End reading snapshots',time.strftime("%H:%M:%S"))
        #
  
    def write_bsPara(self):
        nat=self.bs_dict['nat'];
        ncf=self.bs_dict['nbspa']+1
        fc=open('snapcoeff','w');
        fc.write('#This line is just a comment\n\n'+str(nat)+'  '+str(ncf)+'\n')
        for i in range(nat):
            fc.write(f"{self.bs_dict['e'+str(i+1)]}   {self.bs_dict['r'+str(i+1)]}   {self.bs_dict['w'+str(i+1)]}\n")
            if('E' in self.ftype):
                for j in range(ncf):
                    fc.write(str(self.bscoef[ncf*i+j])+'\n')
            else:
                fc.write('0.0\n')
                ncf=ncf-1;
                for j in range(ncf):
                    fc.write(str(self.bscoef[ncf*i+j])+'\n')
        fc.close();
        fc=open('snapparam','w');
        fc.write('#This line is just a comment: File wrote by Abhishek\n\n#Required\n\nrcutfac '+str(self.bs_dict['rcutfac'])+'\ntwojmax '+str(self.bs_dict['twojmax'])+'\n\n#optional');
        fc.write('\n\nrfac0 '+str(self.bs_dict['rfac0'])+'\nrmin0  0 \nquadraticflag  0 \nbzeroflag  1 \nswitchflag 1');
        fc.close();
        
    def fit_bsPara(self):
        rr=Ridge(alpha=self.bs_dict['Alpha'],fit_intercept=False)      
        rr.fit(self.Aeq,self.Beq,self.Weq)
        self.Pred=rr.predict(self.Aeq)
        self.bscoef=rr.coef_; 
        del rr
        #Am=np.vstack((self.Eeq,self.Feq,self.Seq))
        #Bm=np.concatenate((self.Eval,self.Fval,self.Sval))
        #rr.fit(np.asarray(self.Eeq+self.Feq+self.Seq),Bm,wt)
        # #r2=r2_score(Bm, pred_train_rr);
        # #print(rr.score(Am,Bm),rr.best_score_)
    #
    
    def fit_bsPara_charge(self):
        rr=Ridge(alpha=self.bs_dict['Alpha'],fit_intercept=False)      
        rr.fit(self.Aeq,self.Beq,self.Weq)
        self.Pred=rr.predict(self.Aeq)
        self.bscoef=rr.coef_; 
        del rr
    #
    
    def fit_bsPara_varyAlpha(self):
        #rr=RidgeCV(alphas=[1e-15, 1e-10, 1e-5, 1e-2, 1e-1, 0.2,0.5, 1],fit_intercept=False,store_cv_values=True)      
        #Alphas=[1e-12,1e-8,1e-6,1e-4,1e-2,1e-1,1,10,100]
        Alphas=[0.0,1e-6,1e-4,1e-2,1e-1,1,10,100]
        rmse=1e6;
        
        for alpha1 in Alphas:
            rr=Ridge(alpha=alpha1,fit_intercept=False)      
            rr.fit(self.Aeq,self.Beq,self.Weq)
            self.Pred=rr.predict(self.Aeq)
            if(self.ftype=='E'):
                rmse_c=self.get_rmse();
            if(self.ftype=='F'):
                rmse_c=self.get_rmse(); 
                
            if(self.ftype=='EF'):
                rmse_e,rmse_f = self.get_rmse();
                rmse_c = rmse_e+rmse_f;
                
            if(self.ftype=='EFS'):
                rmse_e,rmse_f,rmse_s = self.get_rmse();
                rmse_c = rmse_e+rmse_f+rmse_s;
                
            if(rmse_c<rmse):
                rmse=rmse_c;
                good_alpha=alpha1; #
                print(alpha1,rmse_c)
            del rr
            
        self.bs_dict['Alpha'] = good_alpha;
        rr = Ridge(alpha = good_alpha, fit_intercept=False)      
        rr.fit(self.Aeq,self.Beq,self.Weq)
        self.Pred=rr.predict(self.Aeq)
        self.bscoef=rr.coef_;
        del rr       
           
    def get_rmse(self):
        if(self.ftype=='E'):
            e_rmse=np.sqrt(mean_squared_error(self.Beq,self.Pred));
            return e_rmse;
        if(self.ftype=='F'):
            f_rmse=np.sqrt(mean_squared_error(self.Beq,self.Pred));
            return f_rmse;
        if(self.ftype=='EF'):           
            ei=0;ef=self.ntrain;
            fi=ef;ff=fi+3*self.ntotatoms;
            e_rmse=np.sqrt(mean_squared_error(self.Beq[ei:ef],self.Pred[ei:ef]));
            f_rmse=np.sqrt(mean_squared_error(self.Beq[fi:ff],self.Pred[fi:ff]));
            return e_rmse,f_rmse;
        if(self.ftype=='EFS'):           
            ei=0;ef=self.ntrain;
            fi=ef;ff=fi+3*self.ntotatoms;
            si=ff;sf=ff+6*ef;
            e_rmse=np.sqrt(mean_squared_error(self.Beq[ei:ef],self.Pred[ei:ef]));
            f_rmse=np.sqrt(mean_squared_error(self.Beq[fi:ff],self.Pred[fi:ff]));
            vol4stress = np.repeat(self.config_vols,6)
            Sdft = 160.217662*self.Beq[si:sf]/vol4stress; Spred = 160.217662*self.Pred[si:sf]/vol4stress; # To convert eV to GPa
            s_rmse=np.sqrt(mean_squared_error(Sdft,Spred));
            return e_rmse,f_rmse,s_rmse;
        if(self.ftype=='Q'):
            q_rmse=np.sqrt(mean_squared_error(self.Beq,self.Pred));
            return q_rmse;
        #

    def get_MAE(self):
        if(self.ftype=='E'):
            e_mae = mean_absolute_error(self.Beq,self.Pred);
            return e_mae;
        if(self.ftype=='F'):
            f_mae = mean_absolute_error(self.Beq,self.Pred);
            return f_mae;
        if(self.ftype=='EF'):           
            ei=0;ef=self.ntrain;
            fi=ef;ff=fi+3*self.ntotatoms;
            e_mae = mean_absolute_error(self.Beq[ei:ef],self.Pred[ei:ef]);
            f_mae = mean_absolute_error(self.Beq[fi:ff],self.Pred[fi:ff]);
            return e_mae,f_mae;
        if(self.ftype=='EFS'):           
            ei=0;ef=self.ntrain;
            fi=ef;ff=fi+3*self.ntotatoms;
            si=ff;sf=ff+6*ef;
            e_mae = mean_absolute_error(self.Beq[ei:ef],self.Pred[ei:ef]);
            f_mae = mean_absolute_error(self.Beq[fi:ff],self.Pred[fi:ff]);
            s_mae = mean_absolute_error(self.Beq[si:sf],self.Pred[si:sf]);
            return e_mae,f_mae,s_mae;
        
    def get_max_err(self):
        if(self.ftype=='E'):
            maxerr_e=max(abs(self.Beq - self.Pred));
            
            return maxerr_e;
        if(self.ftype=='EFS'):
            ei=0;ef=self.ntrain;
            fi=ef;ff=fi+3*self.ntotatoms;
            si=ff;sf=ff+6*ef;
            maxerr_e=max(abs(self.Beq[ei:ef] - self.Pred[ei:ef]));
            maxerr_f=max(abs(self.Beq[fi:ff] - self.Pred[fi:ff]));
            
            vol4stress = np.repeat(self.config_vols,6)
            Sdft = 160.217662*self.Beq[si:sf]/vol4stress; Spred = 160.217662*self.Pred[si:sf]/vol4stress; # To convert eV to GPa
            maxerr_s=max(abs(Sdft - Spred));
            return maxerr_e,maxerr_f,maxerr_s;
        
        if(self.ftype=='Q'):
            maxerr_q=max(abs(self.Beq - self.Pred));
            return maxerr_q;
        
        
    def plot_fit(self):
        pp=PlotD();
        if(self.ftype=='E'):
            e_rmse = self.get_rmse();
            te = f"{self.ntrain} Config, RMSE = {round(e_rmse,5)} ev"
            pp.plot_LFitCompare(self.Beq,self.Pred,'$E_{DFT}\ (eV)$','$E_{SNAP}\ (eV)$',te,'etfit3.png')
            pp.plot_LFitCompare_details(self.Beq,self.Pred,'$E_{DFT}\ (eV)$','$E_{SNAP}\ (eV)$','etfit3_details.png')
            f=open('E_compare.txt','w')
            for i in range(len(self.Beq)):f.write(f'{self.Beq[i]} \t {self.Pred[i]} \t{self.Beq[i]-self.Pred[i]}\n')
            f.close();

        if(self.ftype=='F'):
            f_rmse = self.get_rmse();
            tf = f"{self.ntrain} Config, RMSE = {round(f_rmse,5)} $ev/\AA$"
            pp.plot_LFitCompare(self.Beq,self.Pred,'$F_{DFT}\ (eV/\AA)$','$F_{SNAP}\ (eV/\AA)$',tf,'ftfit3.png')
            pp.plot_LFitCompare_details(self.Beq,self.Pred,'$F_{DFT}\ (eV/\AA)$','$F_{SNAP}\ (eV/\AA)$','ftfit3_details.png')
            f=open('F_compare.txt','w')
            for i in range(len(self.Beq)):f.write(f'{self.Beq[i]} \t {self.Pred[i]} \t{self.Beq[i]-self.Pred[i]}\n')
            f.close();
            
        if(self.ftype=='EF'):
            ei=0;ef=self.ntrain;
            fi=ef;ff=fi+3*self.ntotatoms;
            
            Edft=self.Beq[ei:ef];Epred=self.Pred[ei:ef]
            Fdft=self.Beq[fi:ff];Fpred=self.Pred[fi:ff]
            
            e_rmse,f_rmse= self.get_rmse();
            
            te = f"{self.ntrain} Config, RMSE = {round(e_rmse,5)} ev"
            tf = f"{self.ntrain} Config, RMSE = {round(f_rmse,5)} $ev/\AA$"
            
            pp.plot_LFitCompare(Edft,Epred,'$E_{DFT}\ (eV)$','$E_{SNAP}\ (eV)$',te,'etfit3.png')
            pp.plot_LFitCompare_details(Edft,Epred,'$E_{DFT}\ (eV)$','$E_{SNAP}\ (eV)$','etfit3_details.png')
            
            pp.plot_LFitCompare(Fdft,Fpred,'$F_{DFT}\ (eV/\AA)$','$F_{SNAP}\ (eV/\AA)$',tf,'ftfit3.png') 
            pp.plot_LFitCompare_details(Fdft,Fpred,'$F_{DFT}\ (eV/\AA)$','$F_{SNAP}\ (eV/\AA)$','ftfit3_details.png')
                       
            f=open('E_compare.txt','w')
            for i in range(len(Edft)):f.write(f'{Edft[i]} \t {Epred[i]} \t{Edft[i]-Epred[i]}\n')
            f.close();
            f=open('F_compare.txt','w')
            for i in range(len(Fdft)):f.write(f'{Fdft[i]} \t {Fpred[i]} \t{Fdft[i]-Fpred[i]}\n')
            f.close();
            
        if(self.ftype=='EFS'):
            ei=0;ef=self.ntrain;
            fi=ef;ff=fi+3*self.ntotatoms;
            si=ff;sf=ff+6*ef;
            
            Edft = self.Beq[ei:ef]/self.Natoms; Epred = self.Pred[ei:ef]/self.Natoms
            print('Total no of atoms ',self.Natoms)
            Fdft = self.Beq[fi:ff]; Fpred = self.Pred[fi:ff];
            
            vol4stress = np.repeat(self.config_vols,6)
            Sdft = 160.217662*self.Beq[si:sf]/vol4stress; Spred = 160.217662*self.Pred[si:sf]/vol4stress; # To convert eV to GPa
            
            #e_rmse,f_rmse,s_rmse = self.get_rmse();
            #e_mae,f_mae,s_mae = self.get_MAE();
            
            ti = f"{self.ntrain} Configurations" # Title
                        
            pp.plot_LFitCompare(Edft,Epred,'$E_{DFT}\ (eV/atom)$','$E_{SNAP}\ (eV/atom)$',ti,'etfit3.png')
            pp.plot_LFitCompare_details(Edft,Epred,'$E_{DFT}\ (eV/atom)$','$E_{SNAP}\ (eV/atom)$','etfit3_details.png')
            
            pp.plot_LFitCompare(Fdft,Fpred,'$F_{DFT}\ (eV/\AA)$','$F_{SNAP}\ (eV/\AA)$',ti,'ftfit3.png') 
            pp.plot_LFitCompare_details(Fdft,Fpred,'$F_{DFT}\ (eV/\AA)$','$F_{SNAP}\ (eV/\AA)$','ftfit3_details.png')
            
            #pp.plot_LFitCompare(Sdft,Spred,'$W_{DFT}\ (eV)$','$W_{SNAP}\ (eV)$',ts,'stfit3.png')
            pp.plot_StressLFitCompare(Sdft,Spred,'$W_{DFT}\ (GPa)$','$W_{SNAP}\ (GPa)$',ti,'stfit3.png')
            pp.plot_StressLFitCompare_separate(Sdft,Spred,'$W_{DFT}\ (GPa)$','$W_{SNAP}\ (GPa)$',ti,'stfit3_sep.png')
            pp.plot_LFitCompare_details(Sdft,Spred,'$W_{DFT}\ (GPa)$','$W_{SNAP}\ (GPa)$','stfit3_details.png')
            
            
            f=open('E_compare.txt','w')
            for i in range(len(Edft)):f.write(f'{Edft[i]} \t {Epred[i]} \t{Edft[i]-Epred[i]}\n')
            f.close();
            f=open('F_compare.txt','w')
            for i in range(len(Fdft)):f.write(f'{Fdft[i]} \t {Fpred[i]} \t{Fdft[i]-Fpred[i]}\n')
            f.close();
            f=open('S_compare.txt','w')
            for i in range(len(Sdft)):f.write(f'{Sdft[i]} \t {Spred[i]} \t{Sdft[i]-Spred[i]}\n')
            f.close();
            
            
        if(self.ftype=='Q'):
            q_rmse = self.get_rmse();
            tf = f"{self.ntrain} Config, RMSE = {round(q_rmse,5)} $e$"
            pp.plot_LFitCompare(self.Beq,self.Pred,'$q_{DDEC}$','$q_{SNAP}$',tf,'Qfit.png')
            pp.plot_LFitCompare_details(self.Beq,self.Pred,'$q_{DDEC}$','$q_{SNAP}$','Qfit_details.png')
            f=open('Q_compare.txt','w')
            for i in range(len(self.Beq)):f.write(f'{self.Beq[i]} \t {self.Pred[i]} \t{self.Beq[i]-self.Pred[i]}\n')
            f.close();
    #

    def get_error(self,x):
        #err_type='rmse';#'rmse', 'max_err'
        #err_type='max_err'
        err_type = self.etype; #'rmse', 'max_err','intercept'

        self.lno=0
        nat=self.bs_dict['nat']
        #self.bs_dict['rcutfac']=x[0];
        self.bs_dict['rfac0']=x[0];
        self.we_fac=x[1]; #Energy weight factor 
        self.ws_fac=x[2]; #Stress weight factor       
        ii=3;#** Change it based on above line.
        for i in range(nat):
            self.bs_dict['r'+str(i+1)] =x[ii+i];
            self.bs_dict['w'+str(i+1)] =x[ii+nat+i];
        
        if(self.ftype=='Q'):
            self.get_qBSEQ()
            self.fit_bsPara_charge()
        else:
            self.get_BSEQ()
            self.fit_bsPara()
            
        if(self.ftype=='E'):
            e_rmse=self.get_rmse();
            max_err_e=self.get_max_err();
            print(x,e_rmse,max_err_e)
            with open('optimization_run_e.txt','a') as f:
                for f1 in x:
                    f.write("%s\t"%f1)
                f.write(" %s %s\n"%(e_rmse,max_err_e))
                
            if(err_type=='rmse'):error=e_rmse;
            if(err_type=='max_err'):error=max_err_e;
            
        if(self.ftype=='EFS'):
            e_rmse,f_rmse,s_rmse=self.get_rmse();
            rmse=e_rmse+f_rmse+s_rmse
            max_err_e,max_err_f,max_err_s=self.get_max_err();
            max_err=max_err_e+max_err_f+max_err_s;
            
            print(x,e_rmse,max_err_e,f_rmse,max_err_e,s_rmse,max_err_e,rmse,max_err)
            with open('optimization_run_efs.txt','a') as f:
                for f1 in x:
                    f.write("%s\t"%f1)
                f.write(" %s %s %s %s %s %s %s %s\n"%(e_rmse,max_err_e,f_rmse,max_err_f,s_rmse,max_err_s,rmse,max_err))
                
            if(err_type=='rmse'):error=rmse;
            if(err_type=='max_err'):error=max_err;
            
        if(self.ftype=='Q'):
            e_rmse=self.get_rmse();
            max_err_e=self.get_max_err();
            print(x,e_rmse,max_err_e)
            with open('optimization_run_q.txt','a') as f:
                for f1 in x:
                    f.write("%s\t"%f1)
                f.write(" %s %s\n"%(e_rmse,max_err_e))
                
            if(err_type=='rmse'):error=e_rmse;
            if(err_type=='max_err'):error=max_err_e;
            
        return error;
    #
    
    
    def opt_para(self,algo):
        #algo: Shgo','DE','DA','Direct'
        #guess=[bs_dict['rcutfac'],bs_dict['rfac0']];bounds1=((0.1,4.),(0.1,1)); #For rcutfac and rmin0 respectively
        guess=[self.bs_dict['rfac0'],self.bs_dict['Ew'],self.bs_dict['Sw']]
        bounds1=((0.1,1),(0.01,2000),(0.01,10000))
        for i in range(self.bs_dict['nat']):#For radius
            guess.append(self.bs_dict['r'+str(i+1)])
            bounds1 += ((0.1,0.82),)
        for i in range(self.bs_dict['nat']):#For weights
            guess.append(self.bs_dict['w'+str(i+1)])
            bounds1 += ((0,3),)
        #print(guess,bounds1)
        if(algo=='DA'):best_para = optimize.dual_annealing(self.get_error,maxiter=50000,bounds=bounds1)
        if(algo=='DE'):best_para = optimize.differential_evolution(self.get_error,maxiter=5000,tol=0.0001,bounds=bounds1,updating='immediate')
        if(algo=='Direct'):best_para = optimize.direct(self.get_error,maxiter=5000,bounds=bounds1)
        if(algo=='Shgo'):best_para = optimize.shgo(self.get_error,bounds=bounds1)
        if(algo=='Local'):best_para = optimize.minimize(self.get_error, guess, method='L-BFGS-B', tol=1e-5, bounds=bounds1)
        if(algo=='LocalCG'):best_para = optimize.minimize(self.get_error, guess, method='CG', tol=1e-5)
        
        #best_para = optimize.minimize(rmse_FitSnap,guess,args=(fname,train_dir,bs_dict,Nnat),method='L-BFGS-B',bounds=bounds1)
        #best_para = optimize.differential_evolution(self.get_error(),args=(fname,train_dir,bs_dict,Nnat),maxiter=5000,tol=0.0001,bounds=bounds1)
        #best_para = optimize.basinhopping(rmse_FitSnap,guess,niter=500,T=1.0,stepsize=0.001,minimizer_kwargs={'method':'BFGS','args':(fname,train_dir,bs_dict,Nnat)})
        #best_para=optimize.brute(rmse_FitSnap,bounds1,args=(fname,train_dir,bs_dict,Nnat),finish=optimize.fmin,full_output=False)
        print('optimization finished',best_para)
        return best_para;
    #
    
#
class SnapTest:
    def __init__(self, fname, ntest, ftype, bs_dict):
        self.fname=fname   #Name of XYZ file with details of E, F, and S
        self.ntest=ntest #Number of test sets to read from the file
        self.lno=0;        #Initial line number
        self.bs_dict=bs_dict.copy();
        self.ftype=ftype
        
        #self.tmp=[];
        
    def read_axyz(self):#Function to read one configuration from Abhishek's style of xyz file
        Natoms=int(linecache.getline(self.fname,self.lno+1).split()[0]);
        l2=linecache.getline(self.fname,self.lno+2).split();
        #sno=l2[0];
        Esnap=float(l2[1]);
        A=np.array(l2[2:5],dtype=float) #; l2[2:5].copy()
        B=np.array(l2[5:8],dtype=float)
        C=np.array(l2[8:11],dtype=float)        
        stress=np.array(l2[11:17],dtype=float);#print(Na,sno,Esnap,A,B,C,stress);
        
        elements=[];
        coord=np.zeros([Natoms,3],dtype=float);#>>Atomic coordinates
        force=np.zeros([Natoms,3],dtype=float);#>>Atomic forces
        qd=np.zeros(Natoms,dtype=float);#>>Dynamic charges
        for i in range(Natoms):
            l=linecache.getline(self.fname,self.lno+3+i).split();
            elements.append(l[0]);
            coord[i]=l[1:4];
            force[i]=l[4:7];
            ##**qd[i]=l[7];
            
        self.lno=self.lno+2+Natoms;
        Config=Configuration(A,B,C,elements,coord,Esnap,force,stress,qd)
        linecache.clearcache();
        return Config;
    
    def test_fit(self):
        cwd=os.getcwd();
        edft=[];fdft=[];sdft=[];
        etest=[];ftest=[];stest=[];
        
        es=' ';
        estring=es.join([' '+self.bs_dict['e'+str(i+1)] for i in range(self.bs_dict['nat'])]) #List of elements
        for j in range(self.ntest):
            print('reading snap test',j)
            Ci=self.read_axyz()
            Ci.write_lmpdata_snap(self.bs_dict,cwd+'/')
            #Natoms=Ci.Na;            
            #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            lmp=lammps(cmdargs=['-log','log.lmp','-screen','none']);
            #lmp = lammps(cmdargs=['-log','none','-screen','none']);
            almpc = LampCmd()
            almpc.snaptest(lmp,estring)
            
            st=" ";
            for i in range(self.bs_dict['nat']):st=st+" "+str(self.bs_dict['r'+str(i+1)])+" ";
            for i in range(self.bs_dict['nat']):st=st+" "+str(self.bs_dict['w'+str(i+1)])+" ";
            cmd_st="variable  snap_options string '"+str(self.bs_dict['rcutfac'])+" "+str(self.bs_dict['rfac0'])+" "+  str(self.bs_dict['twojmax'])+" "+st+" rmin0 0.0 quadraticflag 0 bzeroflag 1 switchflag 1'"
            lmp.command(cmd_st)
            lmp.command('compute snap_p all snap    ${snap_options}')
            lmp.command('fix     snap1 all ave/time 1 1 1 c_snap_p[*] file snap.eq mode vector')
            #lmp.command('atom_modify sort 1 0')
            #lmp.command('dump ef all custom 1 dump.myforce id type x y z fx fy fz')
            if('F' in self.ftype):
                lmp.command('compute f all property/atom fx fy fz')
            lmp.command('atom_modify sort 0 0')
            #lmp.command('dump_modify 1 sort id')
            lmp.command("thermo_style   custom step etotal pe pxx")#This line is necessary, without writing pxx lammps won't compute it
            lmp.command("thermo		    1")
            lmp.command('run 0')
            
            if('E' in self.ftype):
                etest.append(lmp.get_thermo("pe") / Ci.Na) # Energy per atom
                edft.append(Ci.E / Ci.Na) # Energy per atom
            if('S' in self.ftype):
                #Order is important
                for i in range(6):sdft.append(Ci.stress[i]*160.217662/Ci.volume)
                #Order is important
                stest.append(lmp.get_thermo("pxx")/10000.)#*Ci.volume/160.21766/10000.)
                stest.append(lmp.get_thermo("pyy")/10000.)#*Ci.volume/160.21766/10000.)
                stest.append(lmp.get_thermo("pzz")/10000.)#*Ci.volume/160.21766/10000.)
                stest.append(lmp.get_thermo("pyz")/10000.)#*Ci.volume/160.21766/10000.)
                stest.append(lmp.get_thermo("pxz")/10000.)#*Ci.volume/160.21766/10000.)
                stest.append(lmp.get_thermo("pxy")/10000.)#*Ci.volume/160.21766/10000.)
                #print(stest,Ci.stress)            
            #sbs = lmp.extract_compute("snap_p", 0, 2)
            if('F' in self.ftype):
                f_lmp=lmp.extract_compute("f",1,2)
                fi=0;
                for c1 in range(Ci.Na):
                    for c2 in range(3):
                        ftest.append(f_lmp[0][fi]);
                        fi+=1;
                        fdft.append(Ci.F[c1][c2])
            lmp.close()
            #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        pp=PlotD();
        ti = f"{self.ntest} Configurations"
        if('E' in self.ftype):
            f=open('E_compare_test.txt','w')
            for i in range(len(edft)):f.write(f'{edft[i]} \t {etest[i]} \t{edft[i]-etest[i]}\n')
            f.close();
            
            e_rmse=np.sqrt(mean_squared_error(edft,etest))
            pp.plot_LFitCompare(edft,etest,'$E_{DFT}\ (eV/atom)$','$E_{SNAP}\ (eV/atom)$',ti,'etest.png')
            
        if('F' in self.ftype):
            f=open('F_compare_test.txt','w')
            for i in range(len(fdft)):f.write(f'{fdft[i]} \t {ftest[i]} \t{fdft[i]-ftest[i]}\n')
            f.close();
            
            f_rmse=np.sqrt(mean_squared_error(fdft,ftest))                
            pp.plot_LFitCompare(fdft,ftest,'$F_{DFT}\ (eV/\AA)$','$F_{SNAP}\ (eV/\AA)$',ti,'ftest.png') 
            
        if('S' in self.ftype):
            f=open('S_compare_test.txt','w')
            for i in range(len(sdft)):f.write(f'{sdft[i]} \t {stest[i]} \t{sdft[i]-stest[i]}\n')
            f.close();
            
            s_rmse=np.sqrt(mean_squared_error(sdft,stest))
            pp.plot_StressLFitCompare(sdft,stest,'$W_{DFT}\ (GPa)$','$W_{SNAP}\ (GPa)$',ti,'stest.png')
            pp.plot_StressLFitCompare_separate(sdft,stest,'$W_{DFT}\ (GPa)$','$W_{SNAP}\ (GPa)$',ti,'stest_sep.png')

        #print(min(edft),min(etest))
#     
class PlotD:
    '''
    def __init__(self, x, y, val_type, xlabel1, ylabel1, title1, pfname1):
        self.x = x;
        self.y = y;
        self.val_type = val_type;
        self.xlabel1 = xlabel1;
        self.ylabel1 = ylabel1;
        self.title1  = title1;
        self.pfname1 = pfname1;

        self.rmse = np.sqrt(mean_squared_error(x,y))
        self.mae  = mean_absolute_error(x,y);
        self.maxd = 0;
        for i in range(len(x)):
            if(abs(x[i]-y[i])>maxd):maxd = abs(x[i]-y[i]);

        if(self.val_type == 'E'):
            self.rmse = self.rmse
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
            plt.plot(data[i,:],'r.',markersize=3.2);
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
#
class cp2k:
    def __init__(self):
        self.au2kcal=27.2114; #Hartree to ev
        self.Fhb2kca=self.au2kcal/0.529177; #Hartree/bohr to ev/Ang
        
    #Function to perform CP2K calculation and get energy and forces---------------------------------------------------------
    def get_EFSQcp2k(self,A,B,C,elements,coord,dftdir):
        #This function will take cell vectors (A,B,C), list of elements, and coordinates and will return enegy and forces of the configuration using cp2k
        cwd1=os.getcwd(); #Getting current working directory
        os.chdir(dftdir); #Changing the directory
        os.system('rm cell_top_coor.inc snap.xyz output.txt *RESTART*  *orces DDEC*');
        f=open('cell_top_coor.inc','w')
        f.write(f"\n&CELL\n A\t\t\t {A[0]} {A[1]} {A[2]} \n B\t\t\t {B[0]} {B[1]} {B[2]} \n C\t\t\t {C[0]} {C[1]} {C[2]} \n PERIODIC\t\t xyz \n MULTIPLE_UNIT_CELL\t 1 1 1 \n&END CELL\n")
        f.write("\n&TOPOLOGY\n MULTIPLE_UNIT_CELL\t 1 1 1 \n COORD_FILE_NAME \t snap.xyz\n COORD_FILE_FORMAT \t XYZ\n&END TOPOLOGY\n")
        f.close()
    
        Na=len(elements);
        f=open('snap.xyz','w')
        f.write(f"{Na}\nConfiguration\n")
        for i in range(Na):
            f.write(f"{elements[i]} {coord[i][0]}  {coord[i][1]}  {coord[i][2]}\n")
        f.close();
        #Running cp2k calculation
        os.system('mpirun -n 72 /projects/pi-ssanvito/parsons/SHARED/cp2k/parsons8.2/cp2k-8.2/exe/parsonsmkl/cp2k.popt -in main.inp > output.txt')
        os.system('mv Snap-ELECTRON_DENSITY-1_0.cube valence_density.cube')
        os.system('/projects/pi-ssanvito/parsons/abhishek/packages/chargemol_09_26_2017/chargemol_FORTRAN_09_26_2017/compiled_binaries/linux/Chargemol_09_26_2017_linux_parallel > out_ddec.txt')
        
        readc=MLD.ReadCp2kFiles();
        
        #Reading energy and total forces-------     
        et=readc.Et("output.txt");#Total energy in eV
        #Reading total forces--------
        fdft=readc.Force('forces',Na); #Forve in ev/Angstrom
        #Reading stress --------------
        volume=np.dot(A,np.cross(B,C))
        stress=readc.Stress('stress',volume); #Stress in ev-Angstrom
        #Reading charges------------
        qa=readc.Charges('DDEC6_even_tempered_net_atomic_charges.xyz',Na)
        
        #Moving calculation data to a directory-------------------
        if(os.path.exists('snap_AL1/')):
            n_al=os.popen('ls ').read().strip().count('L')
            nd='snap_AL'+str(n_al+1)
            os.system('mkdir '+nd)
            os.system('cp sel_basis_pot.inc main.inp '+nd+'/')
            os.system('mv cell_top_coor.inc snap.xyz DDEC6_even_tempered_net_atomic_charges.xyz DDEC6_even_tempered_bond_orders.xyz VDWForces forces stress output.txt '+nd+'/')
            os.system('rm DDEC* *bak* valence* overlap* out* *wfn')
        else:
            os.system('mkdir snap_AL1')
            os.system('cp sel_basis_pot.inc main.inp '+nd+'/')
            os.system('mv cell_top_coor.inc snap.xyz DDEC6_even_tempered_net_atomic_charges.xyz DDEC6_even_tempered_bond_orders.xyz VDWForces forces stress output.txt '+nd+'/')
            os.system('rm DDEC* *bak* valence* overlap* out* *wfn')
        #---------------------------------------------------------
        os.chdir(cwd1); #Rechanging the directory back to original place
        return et,fdft,stress,qa
    #------------------------------------------------------------------------------------------------------------------------
#        
class mdNPT_AL:
    def __init__(self,S,elements,Nmd,dt,T,ensemble,Nofl,md_dir,dft_dir,train_dir):
        self.S=S;
        self.elements=elements #List of elements
        self.Nmd=Nmd;          #Number of MD steps
        self.dt=dt;            #Timestep in ps
        self.T=T;              #Temperature in kelvin
        self.ensemble=ensemble # Ensemble min,nve,nvt,npt
        self.Nofl=Nofl;        #Number of on the fly learning steps
        self.md_dir=md_dir;    #Path of directory where MD simulation is going to run
        self.dft_dir=dft_dir;
        self.train_dir=train_dir;
    
    def write_data(self,lmp,fname):
        Natoms=lmp.get_natoms();
        box=lmp.extract_box();
        coord = lmp.extract_atom("x",3)
        velo = lmp.extract_atom("v",3)
        id1 = lmp.extract_atom("id",0)
        t1 = lmp.extract_atom("type",0)
        q = lmp.extract_atom("q",2)
        t = lmp.extract_global("ntypes",0)

        f=open(fname,'w')
        f.write(f"LAMMPS Data File\n\n{Natoms} atoms\n{t} atom types\n");
        f.write(f"{box[0][0]}  {box[1][0]} xlo xhi\n{box[0][1]}  {box[1][1]} ylo yhi\n{box[0][2]}  {box[1][2]} zlo zhi\n{box[2]} {box[4]} {box[3]} xy xz yz");
        f.write('\nMasses\n\n')
        for i in range(self.S.bs_dict['nat']):f.write(f"{i+1} {self.S.bs_dict['m'+str(i+1)]}\n");
        f.write('\nAtoms\n\n') 
        for i in range(Natoms):f.write(f"{id1[i]} 1 {t1[i]} {q[i]} {coord[i][0]} {coord[i][1]} {coord[i][2]}  0 0 0\n")
        f.write("\nVelocities\n\n")
        for i in range(Natoms):f.write(f"{id1[i]} {velo[i][0]} {velo[i][1]} {velo[i][2]}\n") 
        f.close();
        
    def runmd(self):
        n_al=0;# Number of active learning steps
        es=' ';
        estring=es.join([' '+self.S.bs_dict['e'+str(i+1)] for i in range(self.S.bs_dict['nat'])]) #List of elements
        
        lmp=lammps(cmdargs=['-log','log_AL.lmp'])#,'-screen','none']);
        almpc = LampCmd()
        almpc.md(lmp,estring)
        lmp.command('run_style verlet')
        lmp.command("timestep "+str(self.dt))
        lmp.command("velocity all create "+str(self.T-150)+" 666 rot yes mom yes dist gaussian")
        lmp.command("fix aNPT all npt temp "+str(self.T-150)+" "+str(self.T)+ " "+str(self.dt*100)+" tchain 5 aniso 1.0 1.0 "+str(self.dt*100))
        lmp.command('run 0') # Its necessary, otherwise it will get stuck
        #lmp.command("fix 1 all nve")
        
        for i in range(self.Nmd):
            print('Step',i,' = ',self.dt*i,' ps')
            lmp.command("run 1 pre no post no")
            #step=lmp.get_thermo("step")
            #pe = lmp.get_thermo("pe")
            temp=lmp.get_thermo("temp")

            if(i%50==0):self.write_data(lmp,'data.restart');#Writing data file for restart purpose after few time steps
    
            if(temp>2*self.T or temp < 0.1*self.T):#A condition to check wheather extrapolation is wrong
                n_al+=1;#Increasing the n_al
                if(n_al>self.Nofl):break;
                Natoms=lmp.get_natoms(); #lmp.extract_global("nlocal",0)
                box=lmp.extract_box()
                
                cc=Cell();
                A,B,C=cc.lmpbox2vec(box)
                del cc;
                coord = lmp.extract_atom("x",3)    
                dd=cp2k();                
                et,force,stress,qd=dd.get_EFSQcp2k(A,B,C,self.elements,coord,self.dft_dir) 

                print(lmp.gather_atoms("x",type,12))
                cell=str(A[0])+' '+str(A[1])+' '+str(A[2])+' '+str(B[0])+' '+str(B[1])+' '+str(B[2])+' '+str(C[0])+' '+str(C[1])+' '+str(C[2]);
                ##st=str(stress[0])+' '+str(stress[1])+' '+str(stress[2])+' '+str(stress[3])+' '+str(stress[4])+' '+str(stress[5]);
                st=str(stress[0][0])+' '+str(stress[1][1])+' '+str(stress[2][2])+' '+str(stress[1][2])+' '+str(stress[0][2])+' '+str(stress[0][1]);
    
                #Writing the snapshot
                fc = open(self.train_dir+'Train_AL.xyz','a')
                fc.write(str(Natoms)+'\n');
                fc.write('snap '+str(format(et,'.10f'))+' '+cell+' '+st+'\n');
                for i in range(Natoms):
                    fc.write(f"{self.elements[i]} \t{coord[i][0]}  \t{coord[i][1]}  \t{coord[i][2]} \t{force[i][0]} \t{force[i][1]} \t{force[i][2]} \t{qd[i]} \n")
                fc.close()
                
                #self.S.fname=self.train_dir+'Train_AL.xyz'
                self.S.get_NextBSEQ()
                self.S.fit_bsPara()
                er,fr,sr=self.S.get_rmse()
                self.S.write_bsPara()
                print('Error is ', er,fr,sr,sr+fr+sr)
                   
                #Stopping old lammps simulaiton and re-running it with modified snap parameters
                lmp.close();
                os.system('mv MD_traj.xyz mdrun_details/MD_traj'+str(n_al)+'.xyz; mv sim_details.txt mdrun_details/sim_details'+str(n_al)+'.txt;')
            
                lmp=lammps(cmdargs=['-log','mdrun_details/md_SNAPAL_log'+str(n_al)+'.lmp','-screen','none']);
                almpc = LampCmd()
                almpc.md(lmp,estring)
                
                lmp.command('run_style verlet')
                lmp.command("timestep "+str(self.dt)) 
                lmp.command("velocity all create "+str(self.T-150)+" 666 rot yes mom yes dist gaussian")
                lmp.command("fix aNPT all npt temp "+str(self.T-150)+" "+str(self.T)+ " "+str(self.dt*100)+" tchain 5 aniso 1.0 1.0 "+str(self.dt*100))
                lmp.command('run 0') 
        lmp.close()
#------------------------------------------------------------------------------             
