@SET v_basis TZV2P-MOLOPT-GTH
@SET v_potential GTH-PBE

&KIND H
  BASIS_SET ${v_basis}
  POTENTIAL ${v_potential}-q1
&END KIND

&KIND C
  BASIS_SET ${v_basis}
  POTENTIAL ${v_potential}-q4
&END KIND

&KIND O
  BASIS_SET ${v_basis}
  POTENTIAL ${v_potential}-q6
&END KIND

&KIND Zn
  BASIS_SET TZV2P-MOLOPT-SR-GTH
  POTENTIAL GTH-PBE-q12
&END KIND
