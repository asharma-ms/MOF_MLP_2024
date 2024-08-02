from ase import Atoms
from ase.calculators.dftd3 import DFTD3
import numpy as np

import os
from lammps import lammps
#from lammps import LMP_STYLE_GLOBAL,LMP_TYPE_ARRAY

    

if __name__ == "__main__":     
    lmp = lammps()
    lmp.file("in_md.snap")
    lmp.command('atom_modify sort 0 0') 
    lmp.command("run 100")
