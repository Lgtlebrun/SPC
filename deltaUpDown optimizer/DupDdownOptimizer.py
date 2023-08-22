# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 10:37:28 2023
Script aimed at shaping a tokamak plasma magnetic profile for given 
elongation and up/down triangularity

@author: LL
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
# deltaTargs = [0.5]
dupTargs = [-0.3, 0.3]
ddownTargs = [0.3]

Xlb = 120
Xub = 185
tolerance = 0.01
respath = "resultsPy/"
hp5path = "hp5_GBS/"
GBS = True

Lx = 800
Ly = 930

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


prop = max(Lx, Ly)/800
c0 = 0.5*prop
caux = [0.9*prop if (i == int(0.75*nbaux)) else 0 for i in range(nbaux)]
caux[0] = 0.2*prop
caux[int(nbaux/2)] = 0.2*prop
caux[int(nbaux/4)] = 0.8*prop

ccurrents = np.array([c0] + caux)

C0 = [xcurrents, ycurrents, ccurrents]
C = C0


# %% FUNCTIONS


def Psi(C):
    """Computes the magnetic flux out of a given configuration C"""

    xcurrents, ycurrents, ccurrents = C

    global X, Y, I0

    # Psi0 : normal current + gaussian profile
    Psi = I0/2*ccurrents[0]*(np.log((X-xcurrents[0]) ** 2+((Y-ycurrents[0]) ** 2)) +
                             np.exp(-((X-xcurrents[0]) ** 2+((Y-ycurrents[0]) ** 2))/(s ** 2)))  # + gaussian profile

    # Additional currents
    for i in range(len(xcurrents)-1):
        Psi += I0/2 * \
            ccurrents[i+1]*(np.log((X-xcurrents[i+1]) **
                            2+(Y-ycurrents[i+1]) ** 2))

    return Psi


def gradPsi(C):
    """Computes the gradient of magnetic flux out of a given configuration C"""

    xcurrents, ycurrents, ccurrents = C

    global X, Y, I0, s

    # F = 1-np.exp(-((X-x0) ** 2+(Y-y0) ** 2)/(s ** 2)) * \
    # ((X-x0)**2 + (Y-y0) ** 2)/(s**2)

    r0 = ((X-xcurrents[0]) ** 2) + ((Y-ycurrents[0]) ** 2)

    bx = ccurrents[0]*I0*(Y-ycurrents[0])*(1/r0 - np.exp(-r0/(s ** 2))/(s**2))
    by = -ccurrents[0]*I0*(X-xcurrents[0])*(1/r0 - np.exp(-r0/(s ** 2))/(s**2))

    for i in range(len(xcurrents)-1):

        ri = ((X-xcurrents[i+1]) ** 2) + ((Y-ycurrents[i+1]) ** 2)
        bx += ccurrents[i+1]*I0*(Y-ycurrents[i+1])/ri
        by += -ccurrents[i+1]*I0*(X-xcurrents[i+1])/ri

    dpsidx = -by
    dpsidy = bx

    return dpsidx, dpsidy


def grad2Psicheat(C):

    dpdx, dpdy = gradPsi(C)
    d2pdx2 = np.gradient(dpdx, axis=1)
    d2pdy2 = np.gradient(dpdy, axis=0)
    d2pdxy = np.gradient(dpdx, axis=0)

    return d2pdx2, d2pdy2, d2pdxy


# def grad2Psi(C):

#     xcurrents, ycurrents, ccurrents = C

#     global X, Y, I0, s

#     x0 = xcurrents[0]
#     y0 = ycurrents[0]
#     r0 = ((X-x0) ** 2) + ((Y-y0) ** 2)

#     # Compute separations
#     r = [r0]
#     for i in range(len(xcurrents)-1):
#         r.append(((X-xcurrents[i+1]) ** 2) + ((Y-ycurrents[i+1]) ** 2))

#     drdx = 2*(X - xcurrents[i])
#     drdy = 2*(Y - ycurrents[i])

#     # d/dx (1/r)
#     d1rdx = [-(1/r[i]**2)*drdx for i in range(len(r))]
#     d1rdy = [-(1/r[i]**2)*drdy for i in range(len(r))]

#     F = 1 - r[0]*np.exp(-r0/(s**2))/(s**2)
#     dFdx = drdx*((F-1)/r[0] + (1-F)/(s**2))
#     dFdy = drdy*((F-1)/r[0] + (1-F)/(s**2))

#     c0 = ccurrents[0]
#     d2psidx2 = c0*(dFdx * (X - x0)/r[0] + F *
#                    ((1/r[0]) + (X-x0)*d1rdx[0]))
#     d2psidy2 = c0*(dFdy * (Y - y0)/r[0] + F *
#                    ((1/r[0]) + (Y-y0)*d1rdy[0]))
#     d2psidxdy = c0 * (dFdy * (X - x0)/r[0] +
#                       F*(1/r[0] + (X-x0)*d1rdy[0]))

#     for i in range(len(xcurrents)-1):
#         d2psidx2 += ccurrents[i+1]*(1/r[i+1] + (X - xcurrents[i+1])*d1rdx[i+1])
#         d2psidy2 += ccurrents[i+1]*(1/r[i+1] + (Y - ycurrents[i+1])*d1rdy[i+1])
#         d2psidxdy += ccurrents[i+1] * \
#             (1/r[i+1] + (X - xcurrents[i+1])*d1rdy[i+1])

#     d2psidx2 *= I0
#     d2psidy2 *= I0
#     d2psidxdy *= I0

#     return d2psidx2, d2psidy2, d2psidxdy


def XptCoordsIdx(gradPsi):
    """Computes the coordinates of lower Xpoint /!\ assumes symmetry"""

    global x, y

    dpdx, dpdy = gradPsi

    Bp2 = dpdx**2 + dpdy ** 2

    iY, iX = divmod(Bp2.argmin(), Bp2.shape[1])

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
    dup, ddown = d

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

    _, _, ZminPt, ZmaxPt = specs(C, True)

    # Plot level lines
    plt.figure()
    plt.axis("equal")
    plt.contour(X, Y, psi, levels=levels, colors='k')

    for i in range(len(currents[0])):
        plt.plot(currents[0][i], currents[1][i], marker='o')

    plt.contour(X, Y, psi, levels=[psi[iY, iX]], colors='r')
    plt.plot(ZminPt[0], ZminPt[1], marker='+', color='magenta')
    plt.plot(ZmaxPt[0], ZmaxPt[1], marker='+', color='cyan')
    plt.vlines(0, 0, Ly, colors='grey', linestyles='dashed')
    plt.vlines(Lx, 0, Ly, colors='grey', linestyles='dashed')
    plt.xlabel(r'$R/\rho_{s0}$')
    plt.ylabel(r'$Z/\rho_{s0}$')
    plt.title(r'$\Psi$')
    textstr = r"$\delta_{up} = $"f"{dup:.4}""\n"r"$\delta_{down} = $"f"{ddown:.4}""\n"r"$\kappa = $"f"{k:.4}""\n"r"$Z_{Xpt} = $"f"{Yxpt:.1f}"
    props = dict(boxstyle='round',
                 facecolor='lightsteelblue', alpha=0.5)
    plt.text(0.02, 0.98, textstr, fontsize=14,
             verticalalignment='top', bbox=props, transform=plt.gca().transAxes)


def importC(delta, kappa, Xl, Xu):

    dup, ddown = delta

    with open(respath+f"C_dup_{dup}_ddown_{ddown}_kappa_{kappa}_Yl_{Xl}_Yu_{Xu}.pkl", "rb") as file:
        C = load(file)

    return C


def plotSavedC(delta, kappa, Xl, Xu):

    dup, ddown = delta

    with open(respath+f"C_dup_{dup}_ddown_{ddown}_kappa_{kappa}_Yl_{Xl}_Yu_{Xu}.pkl", "rb") as file:
        C = load(file)
        plt.figure()
        plotMagField(C)
        plt.savefig(
            respath+f"C_dup_{dup}_ddown_{ddown}_kappa_{kappa}.png", format="png")
        plt.show()


def specs(C, full=False):
    """Computes and returns : elongation kappa, triangularity delta
    """

    global X, Y, x, y

    psi = Psi(C)

    iX, iY = XptCoordsIdx(gradPsi(C))
    Xxpt = x[iX]
    Yxptlow = y[iY]
    Yxptup = y[-1] - Yxptlow    # Symmetric to x point about x axis

    level = psi[iY, iX]

    plt.figure(1)
    cs = plt.contour(X, Y, psi, levels=[level])
    plt.close()

    separatrix = [x.vertices for x in cs.collections[0].get_paths()]
    # Keep only 2 largest lists of points in separatrix
    lengths = [len(elt) for elt in separatrix]

    for i, elt in enumerate(separatrix):
        if len(separatrix) > 1:
            for point in elt:
                if (point[1] > Yxptup) or (True in [((point[0] - 0.51*Lx)**2 + (point[1] - 0.5*Ly)**2) > ((xcurrents[k+1] - 0.51*Lx)**2 + (ycurrents[k+1] - 0.5*Ly)**2) for k in range(len(xcurrents) - 1)]):
                    del separatrix[i]
                    break

        else:
            break

    separatrix = np.concatenate(separatrix)

    sepX = separatrix[:, 0]
    sepY = separatrix[:, 1]

    # Rs = [separatrix[i][0] for i in range(separatrix.shape[0]) if (
    #   separatrix[i][1] > Yxptlow and separatrix[i][1] < Yxptup)]

    Rs = sepX[[i for i in range(len(sepX)) if (
        sepY[i] > Yxptlow) and (sepY[i] <= Yxptup)]]
    Zs = sepY[[i for i in range(len(sepX)) if (
        sepY[i] > Yxptlow) and (sepY[i] <= Yxptup)]]

    Zmin_idx = np.argmin(Zs)
    Zmax_idx = np.argmax(Zs)

    Zmin = Zs[Zmin_idx]
    Zmax = Zs[Zmax_idx]

    RZmin = Rs[Zmin_idx]    # R(Z = Zmin)
    RZmax = Rs[Zmax_idx]    # R(Z = Zmax)

    Rmin = min(Rs)
    Rmax = max(Rs)

    R0 = (Rmin+Rmax)/2
    a = (Rmax - Rmin)/2

    kappa = float((Zmax-Zmin)/(Rmax-Rmin))
    # delta = (R0 - Xxpt)/a   # Assumed same X for both Xpoints

    dup = (R0 - RZmax)/a
    ddown = (R0 - RZmin)/a

    if full:
        return kappa, (dup, ddown), (RZmin, Zmin), (RZmax, Zmax)

    return kappa, (dup, ddown)


def optimizeX0(C, toOp):
    xcurrents, ycurrents, ccurrents = C
    global Xlb, Xub

    def cost(p):

        Cnew = list(C)
        Cnew[0][0] = p

        Cnew = centerX0(Cnew)
        Xxpt, Yxpt = XptCoords(Cnew)

        if (not isXptInBnds(Cnew, Xlb, Xub)):
            return np.inf

        try:
            kappa, delta = specs(Cnew)
        except ValueError:
            return np.inf

        dup, ddown = delta

        if toOp == 'dup':
            cost = abs(dupTarg - dup)  # +abs(ddownTarg-ddown)
        elif toOp == 'ddown':
            cost = abs(ddownTarg-ddown)  # +abs(dupTarg-dup)
        elif toOp == 'k':
            cost = abs(kappaTarg - kappa) + \
                2*(abs(dupTarg - dup) + abs(ddownTarg-ddown))  # Priority on d
        else:
            cost = abs(dupTarg - dup) + abs(ddownTarg-ddown) + \
                abs(kappaTarg - kappa)
        return cost

    p = xcurrents[0]
    x0Op = fmin(cost, p)

    xcurrents[0] = x0Op
    C = centerX0(C)
    return C


def optimizeY0(C, toOp):
    xcurrents, ycurrents, ccurrents = C
    global Xlb, Xub

    def cost(p):

        Cnew = list(C)
        Cnew[1][0] = p

        Cnew = centerX0(Cnew)
        Xxpt, Yxpt = XptCoords(Cnew)

        if (not isXptInBnds(Cnew, Xlb, Xub)):
            return np.inf

        try:
            kappa, delta = specs(Cnew)
        except ValueError:
            return np.inf
        dup, ddown = delta

        if toOp == 'dup':
            cost = abs(dupTarg - dup) + abs(ddownTarg-ddown)
        elif toOp == 'ddown':
            cost = abs(ddownTarg-ddown) + abs(dupTarg - dup)
        elif toOp == 'k':
            cost = abs(kappaTarg - kappa) + \
                2*(abs(dupTarg - dup) + abs(ddownTarg-ddown))  # Priority on d
        else:
            cost = abs(dupTarg - dup) + abs(ddownTarg-ddown) + \
                abs(kappaTarg - kappa)
        return cost

    p = ycurrents[0]
    y0Op = fmin(cost, p)

    ycurrents[0] = y0Op
    C = centerX0(C)
    return C


def optimizeC(C, i, toOp, keepDelta=False):
    """Optimizes parameter ccurrents[i]"""

    xcurrents, ycurrents, ccurrents = C
    global Xlb, Xub

    def cost(c):

        #     # Conserve DN symmetry
        # if (i > 1) and (i < (nbaux/2+1)):
        #     ccurrents[-i+1] = c

        Cnew = [list(xcurrents), list(ycurrents),
                list(ccurrents)]  # Copy config
        Cnew[-1][i] = c     # change appropriate coeff

        Xxpt, Yxpt = XptCoords(Cnew)

        if (not isXptInBnds(Cnew, Xlb, Xub)):
            return np.inf

        try:
            kappa, delta = specs(Cnew)
        except ValueError:
            return np.inf

        dup, ddown = delta

        if toOp == 'dup':
            cost = abs(dupTarg - dup) + abs(ddownTarg -
                                            ddown)  # + abs(kappaTarg - kappa)
        elif toOp == 'ddown':
            # + 0.5*abs(kappaTarg - kappa)
            cost = abs(ddownTarg - ddown) + abs(dupTarg - dup)
        elif toOp == 'k':
            cost = abs(kappaTarg - kappa)

            if keepDelta:
                cost += 0.5*(abs(ddownTarg-ddown) +
                             abs(kappaTarg - kappa))
        else:
            cost = abs(dupTarg - dup) + abs(ddownTarg-ddown) + \
                abs(kappaTarg - kappa)

        return cost

    c = ccurrents[i]
    cOp = fmin(cost, c)

    ccurrents[i] = cOp
    return C


def centerX0(C):

    global Lx
    xcurrents, ycurrents, ccurrents = C

    xshift = 0.51*Lx - xcurrents[0]
    yshift = 0.5*Ly - ycurrents[0]

    xcurrentsnew = [x + xshift for x in xcurrents]
    ycurrentsnew = [y + yshift for y in ycurrents]

    Cnew = [xcurrentsnew, ycurrentsnew, ccurrents]
    return Cnew


def updateToOp(dup, ddown, k):

    cost_dup = abs(dup-dupTarg)
    cost_ddown = abs(ddown-ddownTarg)
    cost_kappa = abs(kappa-kappaTarg)

    toOp = 'dup'
    if cost_ddown > cost_dup:
        toOp = 'ddown'
    if (cost_kappa > cost_dup) and (cost_kappa > cost_ddown):
        toOp = 'k'

    if (cost_dup <= tol) and (cost_ddown <= tol) and (cost_kappa < tol):
        toOp = None

    # if (abs(dup-dupTarg) <= tol) and (abs(ddown-ddownTarg) <= tol) and (abs(kappa-kappaTarg) > tol):
    #     toOp = 'k'
    # elif (abs(ddown-ddownTarg) > tol):
    #     toOp = 'ddown'
    # elif (abs(ddown-ddownTarg) <= tol) and (abs(dup-dupTarg) > tol):
    #     toOp = 'dup'
    # else:
    #     toOp = None

    return toOp


# %% Optimize kappa

# Load previous initial config

if os.path.isfile(f"resultsPy/C0_kappa_{kappaTarg}.pkl"):

    with open(f"resultsPy/C0_kappa_{kappaTarg}.pkl", 'rb') as file:
        C0 = load(file)

tol = tolerance
iteration = 1

print("OPTIMIZING ELONGATION...")
print(f"Target elongation : kappa = {kappaTarg}")
kappa, delta = specs(C0)
while abs(kappaTarg-kappa) > tol:

    if iteration % 20 == 0:
        tol += tolerance
        print(
            f"WARNING : tolerance on elongation increased by {tolerance} to {tol}")

    for i in range(nbaux+1):

        kappa, _ = specs(C0)

        print(f"Optimization of current {i}")
        COld = C0
        costOld = abs(kappaTarg-kappa)

        C0 = optimizeC(C0, i, 'k')
        kappa, delta = specs(C0)
        dup, ddown = delta

        cost = abs(kappaTarg-kappa)

        # If previous config was better when targets reached, do not make it worse plz
        if (costOld < cost):
            C0 = COld
    iteration += 1

# Save optimized initial config

with open(f"resultsPy/C0_kappa_{kappaTarg}.pkl", 'wb') as file:
    dump(C0, file)
# %% Plot init config
plt.figure()
plotMagField(C0)
plt.show()
plt.close()

# %% OPTIMIZE TRIANGULARITY

# print("Round 0 : fine-tuning all-in-1...")
# toOp = None to take into account both delta and kappa

# kappa, delta = specs(C)
# dup, ddown = delta
# iteration = 0
# tol = tolerance

# for dupTarg in dupTargs:
#     for ddownTarg in ddownTargs:
#         iteration += 1
#         while ((abs(kappa - kappaTarg) > tol) or (abs(dup - dupTarg) > tol) or (abs(ddown - ddownTarg) > tol) or (not isXptInBnds(C, Xlb, Xub))):
#             print(f"Round n° {iteration} : x0 optimization ...")
#             C = optimizeX0(C, None)
#             C = optimizeY0(C, None)
#             for i in range(int((nbaux)/2+2)):
#                 C = optimizeC(C, i, None)
#             plt.figure()
#             plotMagField(C)
#             plt.show()


for dupTarg in dupTargs:
    for ddownTarg in ddownTargs:
        print(f"Target up triangularity : dup = {dupTarg}")
        print(f"Target down triangularity : ddown = {ddownTarg}")

        iteration = 0
        tol = tolerance
        C = list(C0)

        kappa, delta = specs(C)
        dup, ddown = delta

        # d if delta to optimize (priority), k if kappa, None else
        toOp = 'dup'

        while ((abs(kappa - kappaTarg) > tol) or (abs(dup - dupTarg) > tol) or (abs(ddown - ddownTarg) > tol) or (not isXptInBnds(C, Xlb, Xub))):

            print("Optimizing parameters one by one...")
            iteration += 1

            # Stop condition
            if iteration % 100 == 0:
                print(
                    "WARNING : Algorithm did not converge. Try modifying dimensions of box and/or boundaries on X-point.")
                print("Aborting...")
                break

            # Change initial conditions to avoid local minimums
            if iteration % 20 == 0:
                tol += tolerance
                #xcurrents[0] += int(((iteration//10)*nx/100)*np.sign(delta))
                print(
                    f"WARNING : 20 rounds since last update. Tol incremented by 1. Current tol level : {tol}")
                print("Configuration reset.")
            print(f"Round n° {iteration} : x0 optimization for {toOp}...")

            # Optimization of x-coord of main divertor
            COld = list(C)
            costOld = abs(kappaTarg-kappa) + \
                abs(dupTarg-dup) + abs(ddownTarg-ddown)

            C = optimizeX0(C, toOp)
            kappa, delta = specs(C)
            dup, ddown = delta

            cost = abs(kappaTarg-kappa) + \
                abs(dupTarg-dup) + abs(ddownTarg-ddown)

            # If previous config was better when targets reached, do not make it worse plz
            if (toOp == None) and (costOld < cost):
                C = list(COld)

            # Check parameters to optimize
            toOp = updateToOp(dup, ddown, kappa)

            # Compute Xpoint coordinates
            Xxpt, Yxpt = XptCoords(C)

            # Plot current magnetic profile
            plt.figure()
            plotMagField(C)
            plt.show()

            print(f"Round n° {iteration} : y0 optimization for {toOp}...")

            # Optimization of y-coord of main divertor
            COld = list(C)
            costOld = abs(kappaTarg-kappa) + \
                abs(dupTarg-dup) + abs(ddownTarg-ddown)

            C = optimizeY0(C, toOp)
            kappa, delta = specs(C)
            dup, ddown = delta

            cost = abs(kappaTarg-kappa) + \
                abs(dupTarg-dup) + abs(ddownTarg-ddown)

            # If previous config was better when targets reached, do not make it worse plz
            if (toOp == None) and (costOld < cost):
                C = list(COld)

            # Check parameters to optimize
            toOp = updateToOp(dup, ddown, kappa)

            # Compute Xpoint coordinates
            Xxpt, Yxpt = XptCoords(C)

            # Plot current magnetic profile
            plt.figure()
            plotMagField(C)
            plt.show()

            print(f"Round n° {iteration} : c-by-c optimization...")

            # Optimize currents amplitude
            # Only upper and horizontal currents are varied : others are equalized
            # by symmetry to keep DN configuration
            for i in range(nbaux+1):

                print(f"Optimization of current {i} for {toOp}")
                COld = list(C)
                costOld = abs(kappaTarg-kappa) + \
                    abs(dupTarg-dup) + abs(ddownTarg-ddown)

                C = optimizeC(C, i, toOp, True)
                kappa, delta = specs(C)
                dup, ddown = delta

                cost = abs(kappaTarg-kappa) + \
                    abs(dupTarg-dup) + abs(ddownTarg-ddown)

                # If previous config was better when targets reached, do not make it worse plz
                if (toOp == None) and (costOld < cost):
                    C = list(COld)

                # Check parameters to optimize after each modification
                toOp = updateToOp(dup, ddown, kappa)

                plt.figure()
                plotMagField(C)
                plt.show()

            print("----------------------------")

        if ((abs(kappa - kappaTarg) < tol) and (abs(dup-dupTarg) < tol) and (abs(ddown-ddownTarg) < tol)):

            # Display execution time
            executionTime = (time.time() - startTime)
            print('Execution time in seconds: ' + str(executionTime))
            print('Execution time in minutes: ' + str(executionTime/60))
            print("Optimization success!")

        # Display final results

        print(f"Final delta: {delta}")
        print(f"Final kappa: {kappa}")

        Xxpt, Yxpt = XptCoords(C)
        print(f"Final Yxpt: {Yxpt}")

        # Save data and plot profile

        with open(respath+f"C_dup_{dup}_ddown_{ddown}_kappa_{kappaTarg}_Yl_{Xlb}_Yu_{Xub}.pkl", "wb") as file:
            dump(C, file)

        plt.figure()
        plotMagField(C)
        plt.savefig(
            respath+f"C_dup_{dup}_ddown_{ddown}_kappa_{kappaTarg}.png", format="png")
        plt.show()


# %% SAVE TO GBS FORMAT

# if True:

#     C = importC(-0.4, 1.8, 140, 170)

    # if GBS:
    #     Xxpt, Yxpt = XptCoords(C)

    #     psi = Psi(C)
    #     dpdx, dpdy = gradPsi(C)

    #     iX, iY = XptCoordsIdx(gradPsi(C))
    #     iYup = len(y) - iY
    #     Yxptup = y[iYup]

    #     d2pdx2, d2pdy2, d2pdxy = grad2Psicheat(C)
    #     triang = ("_NT_" if deltaTarg < 0 else "_PT_") + \
    #         "d" + (str(deltaTarg).replace('.', 'p'))

    #     with h5.File(hp5path+"Equil_DN"+triang+".h5", "w") as f:
    #         psi_eq = f.create_dataset("psi_eq", (324, 244), dtype='float64')
    #         psi_eq[...] = psi
    #         dpdx_v = f.create_dataset("dpsidx_v", (324, 244), dtype='float64')
    #         dpdx_v[...] = dpdx
    #         dpdy_v = f.create_dataset("dpsidy_v", (324, 244), dtype='float64')
    #         dpdy_v[...] = dpdy
    #         d2psidx2_v = f.create_dataset(
    #             "d2psidx2_v", (324, 244), dtype='float64')
    #         d2psidx2_v[...] = d2pdx2
    #         d2psidy2_v = f.create_dataset(
    #             "d2psidy2_v", (324, 244), dtype='float64')
    #         d2psidy2_v[...] = d2pdy2
    #         xmain = f.create_dataset("xmag1", (1, 1), dtype='float64')
    #         xmain[...] = C[0][0]    # x0
    #         ymain = f.create_dataset("y0_source", (1, 1), dtype='float64')
    #         ymain[...] = C[1][0]
    #         yxpt = f.create_dataset("Yxpt_low", (1, 1), dtype='float64')
    #         yxpt[...] = Yxpt
    #         yxptup = f.create_dataset("Yxpt_up", (1, 1), dtype='float64')
    #         yxptup[...] = Yxptup


# %% PLOT
# C_delta_-0.5_kappa_1.8_Yl_140_Yu_170
# Cp = importC(-0.5, 1.8, 140, 170)
# plotMagField(Cp)
# plt.show()
