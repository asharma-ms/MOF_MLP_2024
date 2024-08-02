
import numpy as np
import math



def vec2para(A,B,C):# Function to convert lattice vectors to lattice parameters a,b,c,alpha,beta,gamma
    a=np.sqrt(A.dot(A));
    b=np.sqrt(B.dot(B));
    c=np.sqrt(C.dot(C));
    alpha=math.acos(B.dot(C)/b/c);
    beta=math.acos(A.dot(C)/a/c);
    gamma=math.acos(B.dot(A)/a/b);
    return a,b,c,alpha,beta,gamma

def vec2lmp(A,B,C):# Function to convert lattice vectors to lattice parameters a,b,c,alpha,beta,gamma
    [a,b,c,alpha,beta,gamma]=vec2para(A,B,C);
    lx=a;
    xy=b*math.cos(gamma);
    xz=c*math.cos(beta);
    ly=math.sqrt(b*b-xy*xy);
    yz=(b*c*math.cos(alpha)-xy*xz)/ly;
    lz=math.sqrt(c*c-xz*xz-yz*yz);        
    return lx,ly,lz,xy,xz,yz;


def cellData2supercellData(fdata_in,fdata_out,ra,rb,rc):
    fi = open(fdata_in,'r')   
    x=fi.readlines();
    fi.close()
    
    Na  = int(x[2].split()[0])
    Nat = int(x[8].split()[0])
    print(Na,Nat)
    xlo = float(x[14].split()[0])
    xhi = float(x[14].split()[1])
    
    ylo = float(x[15].split()[0])
    yhi = float(x[15].split()[1])
    
    zlo = float(x[16].split()[0])
    zhi = float(x[16].split()[1])
    
    xy  = float(x[17].split()[0])
    xz  = float(x[17].split()[1])
    yz  = float(x[17].split()[2])
    
    lx = xhi - xlo;
    ly = yhi - ylo;
    lz = zhi - zlo;
    
    A  = np.array([lx,0,0])
    B  = np.array([xy,ly,0])
    C  = np.array([xz,yz,lz])
    
    An,Bn,Cn = ra*A,rb*B,rc*C;
    
    print(A,An,B,Bn,C,Cn)
    [lxn,lyn,lzn,xyn,xzn,yzn] = vec2lmp(An,Bn,Cn)
    
    
    fo = open(fdata_out,'w')
    for i in range(0,2):fo.write(x[i])
    fo.write(f'{Na*ra*rb*rc} atoms\n')
    for i in range(3,8):fo.write(x[i])
    fo.write(f'{Nat} atom types\n')
    for i in range(9,14):fo.write(x[i])
    
    fo.write(f'  {xlo} {xlo+lxn} xlo xhi\n')
    fo.write(f'  {ylo} {ylo+lyn} ylo yhi\n')
    fo.write(f'  {zlo} {zlo+lzn} zlo zhi\n')
    fo.write(f'  {xyn} {xzn} {yzn} xy xz yz\n')
    for i in range(18,18+Nat+6):fo.write(x[i])
    
    atid=1;
    for i in range(Nat+24,Nat+Na+24):
        a = x[i].split()
        r = np.array(a[4:7],dtype=float)
        for na in range(ra):
            for nb in range(rb):
                for nc in range(rc):
                    print(r+na*A+nb*B+nc*C,na*A,nb*B,nc*C)
                    R = r+na*A+nb*B+nc*C
                    fo.write(f'{atid}\t{a[1]}\t{a[2]}\t{a[3]}\t{R[0]}\t{R[1]}\t{R[2]}\t{a[7]}\t{a[8]}\t{a[9]}\n')
                    atid += 1;
        print(x[i])
    fo.close()    
    
cellData2supercellData('data.snap','data_sc.snap',2,2,2)