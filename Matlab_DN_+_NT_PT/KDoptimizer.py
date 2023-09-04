# -*- coding: utf-8 -*-
"""
Script aimed at shaping a tokamak plasma magnetic profile for given 
elongation and triangularity

@author: LL
"""
# %% Import necessary modules

import os
import scipy.special as sc
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
deltaTargs = [-0.5, -0.4, -0.3, 0.3, 0.4, 0.5]
Xlb = 125
Xub = 155
tolerance = 0.01
respath = "resultsPy/"
hp5path = "hp5_GBS/"
GBS = True

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
c0 = prop
caux = [1.85*prop if (i+nbaux/4) % (nbaux/2) ==
        0 else 0 for i in range(nbaux)]


ccurrents = [c0] + caux

C0 = [xcurrents, ycurrents, ccurrents]
C = C0


# %% FUNCTIONS


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

    r0 = ((X-xcurrents[0]) ** 2) + ((Y-ycurrents[0]) ** 2)

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


def grad2Psi(C):

    xcurrents, ycurrents, ccurrents = C

    global X, Y, I0, s

    x0 = xcurrents[0]
    y0 = ycurrents[0]
    r0 = ((X-x0) ** 2) + ((Y-y0) ** 2)

    # Compute separations
    r = [r0]
    for i in range(len(xcurrents)-1):
        r.append(((X-xcurrents[i+1]) ** 2) + ((Y-ycurrents[i+1]) ** 2))

    drdx = 2*(X - xcurrents[i])
    drdy = 2*(Y - ycurrents[i])

    # d/dx (1/r)
    d1rdx = [-(1/r[i]**2)*drdx for i in range(len(r))]
    d1rdy = [-(1/r[i]**2)*drdy for i in range(len(r))]

    F = 1 - r[0]*np.exp(-r0/(s**2))/(s**2)
    dFdx = drdx*((F-1)/r[0] + (1-F)/(s**2))
    dFdy = drdy*((F-1)/r[0] + (1-F)/(s**2))

    c0 = ccurrents[0]
    d2psidx2 = c0*(dFdx * (X - x0)/r[0] + F *
                   ((1/r[0]) + (X-x0)*d1rdx[0]))
    d2psidy2 = c0*(dFdy * (Y - y0)/r[0] + F *
                   ((1/r[0]) + (Y-y0)*d1rdy[0]))
    d2psidxdy = c0 * (dFdy * (X - x0)/r[0] +
                      F*(1/r[0] + (X-x0)*d1rdy[0]))

    for i in range(len(xcurrents)-1):
        d2psidx2 += ccurrents[i+1]*(1/r[i+1] + (X - xcurrents[i+1])*d1rdx[i+1])
        d2psidy2 += ccurrents[i+1]*(1/r[i+1] + (Y - ycurrents[i+1])*d1rdy[i+1])
        d2psidxdy += ccurrents[i+1] * \
            (1/r[i+1] + (X - xcurrents[i+1])*d1rdy[i+1])

    d2psidx2 *= I0
    d2psidy2 *= I0
    d2psidxdy *= I0

    return d2psidx2, d2psidy2, d2psidxdy


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


def positivityConstraint(p):

    for i, elt in enumerate(p[1:]):
        if elt < 0:
            return elt

    return 1


def XptConstraint(p):

    global C
    Cnew = list(C)

    xc, yc, cc = Cnew
    # xc[0] = p[0] * Lx

    for i, elt in enumerate(p[0:]):

        cc[i] = np.abs(elt)  # Abs to keep ci > 0

        if (i > 1) and (i < ((nbaux+1)/2)):  # Keep DN symmetry
            cc[-i+1] = np.abs(elt)

    Xxpt, Yxpt = XptCoords(Cnew)

    if (Yxpt > Xub):
        return Xub - Yxpt
    elif Yxpt < Xlb:
        return Yxpt - Xlb
    return 0


def centralConstraint(p):

    c0 = p[0]

    for i, elt in enumerate(p[1:]):

        ratio = elt/c0
        diff = 1.7501-ratio

        if diff < 0:
            return diff

    return 0


def toMinimize(p):

    global C

    Cnew = list(C)

    xc, yc, ccurrents = Cnew

    xc[0] = p[0]*Lx    # First elt of p is x0

    # All following elts of p are currents
    for i, elt in enumerate(p[1:]):

        ccurrents[i] = np.abs(elt)  # Abs to keep ci > 0

        if (i > 1) and (i < ((nbaux+1)/2)):  # Keep DN symmetry
            ccurrents[-i+1] = np.abs(elt)

    if (not isXptInBnds(Cnew, Xlb, Xub)):
        return 9999999

    try:
        Cnew = centerX0(Cnew)
        kappa, delta = specs(Cnew)
    except ValueError:
        return 9999999

    cost = (deltaTarg - delta)**2 + (kappaTarg - kappa)**2
    # If ci > c0, punish
    # for i, current in enumerate(ccurrents):
    #     if current > ccurrents[0]:
    #         cost += abs(ccurrents[0] - current)**2

    return cost


def optimizeConfig(C, deltaTarg, kappaTarg):
    """ p: x0, y0, c*"""
    kappa, delta = specs(C)

    cost = (deltaTarg - delta)**2 + (kappaTarg - kappa)**2
    print(f'Initial cost is {cost}')

    xc, yc, cc = C

    p = [xc[0]/Lx] + list(cc[0:int(np.ceil(nbaux/2 + 2))])
    #p = list(cc[0:int(np.ceil(nbaux/2 + 2))])

    bounds = [(-1, 1)]
    bounds += [(0, 10) for c in p[1:]]

    # constraints = [
    #     {'type': 'ineq', 'fun': XptConstraint}]

    res = minimize(toMinimize, p,  method='Nelder-Mead',
                   bounds=bounds)  # , constraints=constraints)

    p = res.x

    print(p)

    xc[0] = p[0]*Lx
    for i, elt in enumerate(p[1:]):
        cc[i] = np.abs(elt)
        if (i > 1) and (i < ((nbaux+1)/2)):
            cc[-i+1] = np.abs(elt)

    C = centerX0(C)
    kappa, delta = specs(C)

    cost = (deltaTarg - delta)**2 + (kappaTarg - kappa)**2
    print(f"Final cost is {cost}")

    return C


def optimizeX0(C, toOp):
    xcurrents, ycurrents, ccurrents = C
    global Xlb, Xub

    def cost(p):

        Cnew = [list(xcurrents), list(ycurrents), list(ccurrents)]
        Cnew[0][0] = p

        Cnew = centerX0(Cnew)
        Xxpt, Yxpt = XptCoords(Cnew)

        if (not isXptInBnds(Cnew, Xlb, Xub)):
            return np.inf

        kappa, delta = specs(Cnew)

        if toOp == 'd':
            cost = abs(deltaTarg - delta)**2
        elif toOp == 'k':
            cost = abs(kappaTarg - kappa)**2 + \
                abs(deltaTarg - delta)**2  # Priority on d
        else:
            cost = abs(deltaTarg - delta)**2 + abs(kappaTarg - kappa)**2
        return cost

    p = xcurrents[0]
    x0Op = fmin(cost, p)

    xcurrents[0] = x0Op
    C = centerX0(C)
    return C


def optimizeC(C, i, toOp):
    """Optimizes parameter ccurrents[i]"""

    xcurrents, ycurrents, ccurrents = C
    global Xlb, Xub

    def cost(c):

        Cnew = [list(xcurrents), list(ycurrents), list(ccurrents)]
        Cnew[-1][i] = np.abs(c)

        # Conserve DN symmetry
        if (i > 1) and (i < ((nbaux+1)/2)):
            Cnew[-1][-i+1] = np.abs(c)

        Xxpt, Yxpt = XptCoords(Cnew)

        if (not isXptInBnds(Cnew, Xlb, Xub)):
            return np.inf

        kappa, delta = specs(Cnew)

        if toOp == 'd':
            cost = abs(deltaTarg - delta)**2 + abs(kappaTarg - kappa)**2
        elif toOp == 'k':
            cost = abs(kappaTarg - kappa)**2
        else:
            cost = abs(deltaTarg - delta)**2 + abs(kappaTarg - kappa)**2

        # if (i != 0) and (ccurrents[i] > ccurrents[0]):
        #     cost += abs(ccurrents[i] - ccurrents[0])**2

        return cost

    c = ccurrents[i]
    cOp = fmin(cost, c)

    ccurrents[i] = np.abs(cOp)  # Abs makes sure c > 0
    # Conserve DN symmetry
    if (i > 1) and (i < ((nbaux+1)/2)):
        ccurrents[-i+1] = np.abs(cOp)

    return C


def centerX0(C):

    global Lx
    xcurrents, ycurrents, ccurrents = C

    xshift = 0.51*Lx - xcurrents[0]
    xcurrentsnew = [x + xshift for x in xcurrents]

    Cnew = [xcurrentsnew, ycurrents, ccurrents]
    return Cnew


# %% Plot initial config
plt.figure()
plotMagField(C)
plt.show()
plt.close()

# %% OPTIMIZE TRIANGULARITY


kappa, delta = specs(C)

for deltaTarg in deltaTargs:

    iteration = 0
    tol = tolerance
    C = [list(elt) for elt in C0]
    print("--------------------------------")
    print("NEW OPTIMIZATION")
    print(f"Target triangularity : delta = {deltaTarg}")
    print(f"Target elongation : kappa = {kappaTarg}")

    print("Round 0 : fine-tuning all-in-1...")

    C = optimizeConfig(C, deltaTarg, kappaTarg)
    plt.figure()
    plotMagField(C)
    plt.show()

    kappa, delta = specs(C)

    # Check parameters to optimize after each modification
    if (abs(delta-deltaTarg) <= tol) and (abs(kappa-kappaTarg) > tol):
        toOp = 'k'
    elif abs(delta-deltaTarg) > tol:
        toOp = 'd'
    else:
        toOp = None

    while ((abs(kappa - kappaTarg) > tol) or (abs(delta - deltaTarg) > tol) or (not isXptInBnds(C, Xlb, Xub))):

        print("Previous round failed. Optimizing parameters one by one...")
        iteration += 1

        # Stop condition
        if iteration % 100 == 0:
            print("WARNING : Algorithm did not converge. Try modifying dimensions of box and/or boundaries on X-point.")
            print("Aborting...")
            break

        # Change initial conditions to avoid local minimums
        if iteration % 20 == 0:
            tol += tolerance
            C = list(C0)
            #xcurrents[0] += int(((iteration//10)*nx/100)*np.sign(delta))
            print(
                f"WARNING : 20 rounds since last update. Tol incremented by 1. Current tol level : {tol}")
            print("Configuration reset.")
        print(f"Round n° {iteration} : x0 optimization for {toOp}...")

        # Optimization of x-coord of main divertor
        COld = list(C)
        costOld = abs(kappaTarg-kappa) + abs(deltaTarg-delta)

        C = optimizeX0(C, toOp)
        kappa, delta = specs(C)
        cost = abs(kappaTarg-kappa) + abs(deltaTarg-delta)

        # If previous config was better when targets reached, do not make it worse plz
        if (toOp == None) and (costOld < cost):
            C = list(COld)

        # Check parameters to optimize
        if (abs(delta-deltaTarg) <= tol) and (abs(kappa-kappaTarg) > tol):
            toOp = 'k'
        elif abs(delta-deltaTarg) > tol:
            toOp = 'd'
        else:
            toOp = None

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
        for i in range(int((nbaux)/2+2)):

            print(f"Optimization of current {i} for {toOp}")
            COld = list(C)
            costOld = abs(kappaTarg-kappa) + abs(deltaTarg-delta)

            C = optimizeC(C, i, toOp)
            kappa, delta = specs(C)
            cost = abs(kappaTarg-kappa) + abs(deltaTarg-delta)

            # If previous config was better when targets reached, do not make it worse plz
            if (toOp == None) and (costOld < cost):
                C = list(COld)

            # Check parameters to optimize after each modification
            if (abs(delta-deltaTarg) <= tol) and (abs(kappa-kappaTarg) > tol):
                toOp = 'k'
            elif abs(delta-deltaTarg) > tol:
                toOp = 'd'
            else:
                toOp = None

            plt.figure()
            plotMagField(C)
            plt.show()

        print("----------------------------")

    if ((abs(kappa - kappaTarg) < tol) and (abs(delta - deltaTarg) < tol)):

        # Display execution time
        executionTime = (time.time() - startTime)
        print('Execution time in seconds: ' + str(executionTime))
        print('Execution time in minutes: ' + str(executionTime/60))
        print("Optimization success!")

    # Display final results
    print("Final configuration : ")
    print(C)
    print(f"Final delta: {delta}")
    print(f"Final kappa: {kappa}")

    Xxpt, Yxpt = XptCoords(C)
    print(f"Final Yxpt: {Yxpt}")

    # Save data and plot profile

    with open(respath+f"C_delta_{deltaTarg}_kappa_{kappaTarg}_Yl_{Xlb}_Yu_{Xub}.pkl", "wb") as file:
        dump(C, file)

    plt.figure()
    plotMagField(C)
    plt.savefig(
        respath+f"C_delta_{deltaTarg}_kappa_{kappaTarg}.png", format="png")
    plt.show()

    plt.figure()
    psi = Psi(C)
    plt.pcolormesh(psi)
    plt.colorbar()
    plt.show()


# %% SAVE TO GBS FORMAT

# if True:

#     C = importC(-0.4, 1.8, 140, 170)

    if GBS:
        Xxpt, Yxpt = XptCoords(C)

        psi = Psi(C)
        dpdx, dpdy = gradPsi(C)

        iX, iY = XptCoordsIdx(gradPsi(C))
        iYup = len(y) - iY
        Yxptup = y[iYup]

        d2pdx2, d2pdy2, d2pdxy = grad2Psicheat(C)
        triang = ("_NT_" if deltaTarg < 0 else "_PT_") + \
            "d" + (str(deltaTarg).replace('.', 'p'))

        with h5.File(hp5path+"Equil_DN"+triang+".h5", "w") as f:
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


# %% PLOT
# C_delta_-0.5_kappa_1.8_Yl_140_Yu_170
# Cp = importC(-0.5, 1.8, 140, 170)
# plotMagField(Cp)
# plt.show()
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
