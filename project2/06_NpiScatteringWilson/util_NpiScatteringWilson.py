import util as yu
from util import *

ens2full={'milc':'MILC_a09m130W'}
# ens2label={'a24':'A24','a':'A48','b':'B64','c':'C80','d':'D96','e':'E112'}
ens2a={'milc':0.0871} # fm
ens2NL={'milc':64}
ens2NT={'milc':96}

ens2aInv={ens:1/(ens2a[ens]*yu.hbarc) for ens in ens2a.keys()} # MeV


# scattering length <-> energy shift
def sl2es(mu, L, a0):
    c1=-2.837297; c2=6.375183
    return - (2*np.pi)/(mu*L) * a0/L * ( 1 + c1*a0/L + c2*(a0/L)**2 ) / L
def es2sl(mu, L, dE):
    return yu.fsolve2(lambda a0:sl2es(mu, L, a0)-dE, 0)

iso2a0mpi_phy_me={
    '32':(-86.3*1e-3,1.8*1e-3),
    '12':(169.8*1e-3,2.0*1e-3),
} # a32mpi_phy=(-77.5*1e-3,3.5*1e-3); a12mpi_phy=(178.8*1e-3,3.8*1e-3) # pheLat @ 135 MeV
iso2a0mpi_pol_me={
    '32':(-0.104,0.018),
    '12':(0.157,0.031),
} # a32mpi_pol=(-0.128,0.015); a12mpi_pol=(0.093,0.025) # old

iso2a0mpi_delta_me={
    '32':(-0.13,0.04),
}