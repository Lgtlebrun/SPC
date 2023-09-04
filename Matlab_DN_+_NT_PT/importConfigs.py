# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 15:09:48 2023

@author: lgtle
"""
# %% Import necessary modules

import os
from pickle import load
from pickle import dump
from scipy.optimize import minimize, fmin
import matplotlib.pyplot as plt
import numpy as np
import time
import h5py as h5
import scipy.special as sc

startTime = time.time()

plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({
    "font.family": "Helvetica"
})
plt.rcParams.update({'font.size': 22})

# %% Initial parameters

# TODO : adapt initconfig depending on Xlb Xup

kappaTarg = 1.7
deltaTargs = [-0.5]
Xlb = 140
Xub = 170
tolerance = 0.01
respath = "resultsPy/"
hp5path = "hp5_GBS/"
GBS = False

Lx = 600
Ly = 800

R = 700
nx = 244        # Take even int
ny = 324        # Take even int

dx = Lx/(nx-4)
dy = Ly/(ny-4)

x = np.arange(-3/2*dx, Lx+5*dx/2, step=dx)
y = np.arange(-3/2*dy, Ly+5*dy/2, step=dy)

X, Y = np.meshgrid(x, y)

I0 = 2000*Ly/400*Lx/300         # Reference current amplitude
s = 60*Ly/400

# %% Initial symmetric profile


def coordsAux(nb, Lx, Ly, x0, y0):
    """Gives coordinates of auxiliary currents depending on their number.
    Auxiliary currents are dispatched on a circle of ray R = max(Lx, Ly)*0.7, 
    at equal angulary distance"""

    R = max(Lx, Ly)*0.7

    xs = [x0+R*np.cos(i*(2*np.pi)/nb) for i in range(nb)]
    ys = [y0+R*np.sin(i*(2*np.pi)/nb) for i in range(nb)]

    return xs, ys


n = 3  # nb of auxiliary currents is 2^n, with n > 2

if n < 2:
    raise ValueError("n cannot be inferior to 2")

nbaux = 2**n

# X-coord. of the main plasma current : shifted by 0.1Lx to avoid superimposition with mesh
x0 = 0.51*Lx
y0 = 0.5*Ly           # Y-coord. of the main plasma current

xaux, yaux = coordsAux(nbaux, Lx, Ly, x0, y0)


xcurrents = [x0] + xaux
ycurrents = [y0] + yaux


# Yxpt_init = 0.18*Ly   # Yxpt_low


# y1 = y0 - np.sqrt(3)*(y0-Yxpt_init)
# y2 = y0 + np.sqrt(3)*(y0-Yxpt_init)
# y3 = y0 - np.sqrt(3)*(y0-Yxpt_init)  # Upper shaping current
# y4 = y0 + np.sqrt(3)*(y0-Yxpt_init)  # Lower shaping current

# y5 = y0
# y6 = y0

# Fine-tuned values for optimal init
prop = max(Lx, Ly)/800
c0 = 2*prop
caux = [3*prop if (i+nbaux/4) % (nbaux/2) == 0 else 0 for i in range(nbaux)]


ccurrents = np.array([c0] + caux)

C0 = [xcurrents, ycurrents, ccurrents]
C = C0


def Psi(C):
    """Computes the magnetic flux out of a given configuration C"""

    xcurrents, ycurrents, ccurrents = C

    global X, Y, I0

    # Psi0 : normal current + gaussian profile
    Psi = I0/2*ccurrents[0]*(np.log((X-xcurrents[0]) **
                             2+((Y-ycurrents[0]) ** 2))+sc.expn(1, ((X-xcurrents[0]) ** 2+((Y-ycurrents[0]) ** 2))/(s ** 2)))  # + gaussian profile

    # Additional currents
    for i in range(len(xcurrents)-1):
        Psi += I0/2 * \
            ccurrents[i+1]*(np.log((X-xcurrents[i+1]) **
                            2+((Y-ycurrents[i+1]) ** 2)))

    return Psi


def gradPsi(C):
    """Computes the gradient of magnetic flux out of a given configuration C"""

    xcurrents, ycurrents, ccurrents = C

    global X, Y, I0, s

    # F = 1-np.exp(-((X-x0) ** 2+(Y-y0) ** 2)/(s ** 2)) * \
    # ((X-x0)**2 + (Y-y0) ** 2)/(s**2)

    r0 = ((X-xcurrents[0]) ** 2) + ((Y-ycurrents[0]) ** 2)

    #
   # bx = ccurrents[0]*I0*(Y-ycurrents[0]) * \
   #     (1/r0 - np.exp(- r0/(s ** 2))/(s**2))
    #
   # by = -ccurrents[0]*I0*(X-xcurrents[0]) * \
    #    (1/r0 - np.exp(- r0/(s ** 2))/(s**2))

    # for i in range(len(xcurrents)-1):

    #     ri = ((X-xcurrents[i+1]) ** 2) + ((Y-ycurrents[i+1]) ** 2)
    #     bx += ccurrents[i+1]*I0*(Y-ycurrents[i+1]) * \
    #         (1/ri)
    #     by += -ccurrents[i+1]*I0*(X-xcurrents[i+1]) * \
    #         (1/ri)

    # dpsidx = -by
    # dpsidy = bx

    psi = Psi(C)
    dpsidx = np.gradient(psi, axis=1)
    dpsidy = np.gradient(psi, axis=0)

    return dpsidx, dpsidy


def grad2Psicheat(C):

    dpdx, dpdy = gradPsi(C)
    d2pdx2 = np.gradient(dpdx, axis=1)
    d2pdy2 = np.gradient(dpdy, axis=0)
    d2pdxy = np.gradient(dpdx, axis=0)

    return d2pdx2, d2pdy2, d2pdxy


def XptCoordsIdx(gradPsi):
    """Computes the coordinates of lower Xpoint /!\ assumes symmetry"""

    global x, y

    dpdx, dpdy = gradPsi

    Bp2 = dpdx**2 + dpdy ** 2

    iY, iX = divmod(Bp2.argmin(), Bp2.shape[1])

    if iY > len(y)/2:
        iY = len(y) - iY

    return iX, iY


def XptCoords(C):

    iX, iY = XptCoordsIdx(gradPsi(C))

    return x[iX], y[iY]


def isXptInBnds(C, lb, ub):

    Xxpt, Yxpt = XptCoords(C)

    if (Yxpt < ub) and (Yxpt > lb):
        return True
    return False


def plotMagField(C):
    """Plots magnetic field"""

    global X, Y
    xcurrents, ycurrents, ccurrents = C

    currents = [xcurrents,
                ycurrents]

    psi = Psi(C)
    k, d = specs(C)
    Xxpt, Yxpt = XptCoords(C)

    x0 = xcurrents[0]
    y0 = ycurrents[0]

    sgn = -1 if x0 < 0.51*Lx else 1

    # Value close to index of x0. Nice not to fall exactly for division by zero purposes
    x0i = int(np.round(x0/dx))
    y0i = int(np.round(y0/dy))

    iX, iY = XptCoordsIdx(gradPsi(C))
    ixlowlvl = iX - sgn*2

    levels = np.arange(psi[y0i][x0i], psi[int(ny/2)]
                       [nx - 5 if sgn == -1 else 6], step=psi[int(ny/2)][int(nx/2)]/200)

    # Plot level lines
    plt.figure()
    plt.axis("equal")
    plt.contour(X, Y, psi, levels=levels, colors='k')

    for i in range(len(currents[0])):
        plt.plot(currents[0][i], currents[1][i], marker='o')

    plt.contour(X, Y, psi, levels=[psi[iY, iX]], colors='r')
    plt.vlines(0, 0, Ly, colors='grey', linestyles='dashed')
    plt.vlines(Lx, 0, Ly, colors='grey', linestyles='dashed')
    plt.xlabel(r'$R/\rho_{s0}$')
    plt.ylabel(r'$Z/\rho_{s0}$')
    plt.title(r'$\Psi$')
    textstr = fr"$\delta = {d:.4}$""\n"fr"$\kappa = {k:.4}$""\n"r"$Z_{Xpt} = $"f"{Yxpt:.1f}"
    props = dict(boxstyle='round',
                 facecolor='lightsteelblue', alpha=0.5)
    plt.text(0.02, 0.98, textstr, fontsize=14,
             verticalalignment='top', bbox=props, transform=plt.gca().transAxes)


def importC(delta, kappa, Xl, Xu):

    with open(respath+f"C_delta_{delta}_kappa_{kappa}_Yl_{Xl}_Yu_{Xu}.pkl", "rb") as file:
        C = load(file)

    return C


def exportC(delta, kappa, Xl, Xu):

    with open(respath+f"Ccentered_delta_{delta}_kappa_{kappa}_Yl_{Xl}_Yu_{Xu}.pkl", "wb") as file:
        dump(C, file)


def plotSavedC(delta, kappa):

    with open(respath+f"C_delta_{delta}_kappa_{kappa}.pkl", "rb") as file:
        C = load(file)
        plt.figure()
        plotMagField(C)
        plt.savefig(respath+f"C_delta_{delta}_kappa_{kappa}.png", format="png")
        plt.show()


def specs(C):
    """Computes and returns : elongation kappa, triangularity delta
    """

    global X, Y, x, y

    psi = Psi(C)

    iX, iY = XptCoordsIdx(gradPsi(C))
    Xxpt = x[iX]
    Yxptlow = y[iY]
    Yxptup = y[-1] - Yxptlow

    level = psi[iY, iX]

    plt.figure(1)
    cs = plt.contour(X, Y, psi, levels=[level])
    plt.close()

    separatrix = [x.vertices for x in cs.collections[0].get_paths()]
    separatrix = np.concatenate(separatrix)

    sepX = separatrix[:, 0]
    sepY = separatrix[:, 1]

    # Rs = [separatrix[i][0] for i in range(separatrix.shape[0]) if (
    #   separatrix[i][1] > Yxptlow and separatrix[i][1] < Yxptup)]

    Rs = sepX[[i for i in range(len(sepX)) if (
        sepY[i] > Yxptlow and sepY[i] < Yxptup)]]
    Zs = sepY[[i for i in range(len(sepX)) if (
        sepY[i] > Yxptlow and sepY[i] < Yxptup)]]

    Rmin = min(Rs)
    Rmax = max(Rs)

    R0 = (Rmin+Rmax)/2
    a = (Rmax - Rmin)/2

    kappa = (Yxptup-Yxptlow)/(Rmax-Rmin)
    delta = (R0 - Xxpt)/a   # Assumed same X for both Xpoints

    return kappa, delta


# %% IMPORT nt

Cs = {}
ds = [-0.4, - 0.5]
k = 1.7
Xl = 125
Xu = 155

for d in ds:
    C = importC(d, k, Xl, Xu)
    print(C)

    xc, yc, cc = C
    xc = [x - 120 for x in xc]
    C = [xc, yc, cc]

    plotMagField(C)
    plt.show()

    Cs[d] = C

# %% Cqse -0.3

C = importC(-0.3, k, Xl, Xu)
xc, yc, cc = C
xc = [x - 100 for x in xc]
C = [xc, yc, cc]
plotMagField(C)
plt.show()
Cs[-0.3] = C

C = importC(0.3, k, Xl, Xu)
xc, yc, cc = C
xc = [x + 100 for x in xc]
C = [xc, yc, cc]
plotMagField(C)
plt.show()
Cs[0.3] = C

# %% Import PT

ds = [0.4, 0.5]
for d in ds:
    C = importC(d, k, Xl, Xu)

    xc, yc, cc = C
    xc = [x + 120 for x in xc]
    C = [xc, yc, cc]

    plotMagField(C)
    plt.show()

    Cs[d] = C


# %% kyungtak

pathK = respath+"kyungtak/"


def exportH5(deltaTarg, C, path):

    Xxpt, Yxpt = XptCoords(C)

    psi = Psi(C)
    dpdx, dpdy = gradPsi(C)

    iX, iY = XptCoordsIdx(gradPsi(C))
    iYup = len(y) - iY
    Yxptup = y[iYup]

    d2pdx2, d2pdy2, d2pdxy = grad2Psicheat(C)
    triang = ("_NT_" if deltaTarg < 0 else "_PT_") + \
        "d" + (str(deltaTarg).replace('.', 'p'))

    with h5.File(path+"Equil_DN"+triang+".h5", "w") as f:
        psi_eq = f.create_dataset("psi_eq", (324, 244), dtype='float64')
        psi_eq[...] = psi
        dpdx_v = f.create_dataset("dpsidx_v", (324, 244), dtype='float64')
        dpdx_v[...] = dpdx
        dpdy_v = f.create_dataset("dpsidy_v", (324, 244), dtype='float64')
        dpdy_v[...] = dpdy
        d2psidx2_v = f.create_dataset(
            "d2psidx2_v", (324, 244), dtype='float64')
        d2psidx2_v[...] = d2pdx2
        d2psidy2_v = f.create_dataset(
            "d2psidy2_v", (324, 244), dtype='float64')
        d2psidy2_v[...] = d2pdy2
        xmain = f.create_dataset("xmag1", (1, 1), dtype='float64')
        xmain[...] = C[0][0]    # x0
        ymain = f.create_dataset("y0_source", (1, 1), dtype='float64')
        ymain[...] = C[1][0]
        yxpt = f.create_dataset("Yxpt_low", (1, 1), dtype='float64')
        yxpt[...] = Yxpt
        yxptup = f.create_dataset("Yxpt_up", (1, 1), dtype='float64')
        yxptup[...] = Yxptup


with open(pathK+"CONFIGS.txt", 'w') as f:
    f.write("CONFIGURATIONS OF CURRENTS FOR DN PROFILES\n-----------------------\n")
    for d, C in Cs.items():
        f.write(f"CONFIGURATION FOR DELTA = {d}\n --------------- \n")
        xc, yc, cc = C
        for i, xi in enumerate(xc):
            f.write(
                f"current {i} : X = {xi}, Y = {yc[i]}, amplitude = {cc[i]}\n")
        f.write('--------------------------------------\n')

        exportH5(d, C, pathK)
        exportH5(d, C, pathK)
        exportC(d, k, Xl, Xu)
