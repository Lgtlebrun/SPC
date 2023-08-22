# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 10:07:02 2023

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
startTime = time.time()

plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({
    "font.family": "Helvetica"
})

# %% Initial parameters


kappaTarg = 1.6
deltaTargs = [-0.3, -0.5, 0.3, 0.5]
Xlb = 120
Xub = 200
tolerance = 0.01
respath = "resultsPy/"

Lx = 600
Ly = 800
R = 700
nx = 244        # Take even int
ny = 324        # Take even int

dx = Lx/(nx-4)
dy = Ly/(ny-4)

x = np.arange(-3/2*dx, Lx+3*dx/2, step=dx)
y = np.arange(-3/2*dy, Ly+3*dy/2, step=dy)

X, Y = np.meshgrid(x, y)

I0 = 2000*Ly/400*Lx/300         # Reference current amplitude
s = 60*Ly/400

# %% Initial symmetric profile

x0 = 0.6*Lx           # X-coord. of the main plasma current
x1 = Lx/2 - 0.1*Lx    # Lower divertor current
x2 = x1               # Upper divertor current
x3 = 2*x0 - x1       # Symmetric for elliptical initial guess
x4 = x3              # Left upper shaping current

Yxpt_init = 0.18*Ly   # Yxpt_low


y0 = 0.5*Ly           # Y-coord. of the main plasma current
y1 = y0 - np.sqrt(3)*(y0-Yxpt_init)
y2 = y0 + np.sqrt(3)*(y0-Yxpt_init)
y3 = y0 - np.sqrt(3)*(y0-Yxpt_init)  # Upper shaping current
y4 = y0 + np.sqrt(3)*(y0-Yxpt_init)  # Lower shaping current

c0 = 1
caux = 0.85
c1 = c2 = c3 = c4 = caux

C0 = [x0, x1, x2, x3, x4, y0, y1, y2, y3, y4, c0, c1, c2, c3, c4]
C = C0


# %% FUNCTIONS


def Psi(C):
    """Computes the magnetic flux out of a given configuration C"""

    x0, x1, x2, x3, x4, y0, y1, y2, y3, y4, c0, c1, c2, c3, c4 = C

    global X, Y

    Psi0 = I0/2*c0*(np.log((X-x0) ** 2+((Y-y0) ** 2)) +
                    np.exp(-((X-x0) ** 2+((Y-y0) ** 2))/(s ** 2)))  # + gaussian profile
    Psi1 = I0/2*c1*(np.log((X-x1) ** 2+(Y-y1) ** 2))
    Psi2 = I0/2*c2*(np.log((X-x2) ** 2+(Y-y2) ** 2))
    Psi3 = I0/2*c3*(np.log((X-x3) ** 2+(Y-y3) ** 2))
    Psi4 = I0/2*c4*(np.log((X-x4) ** 2+(Y-y4) ** 2))
    Psi = Psi0 + Psi1 + Psi2 + Psi3 + Psi4

    return Psi


def gradPsi(C):
    """Computes the gradient of magnetic flux out of a given configuration C"""

    x0, x1, x2, x3, x4, y0, y1, y2, y3, y4, c0, c1, c2, c3, c4 = C

    global X, Y, I0, s

    # F = 1-np.exp(-((X-x0) ** 2+(Y-y0) ** 2)/(s ** 2)) * \
    # ((X-x0)**2 + (Y-y0) ** 2)/(s**2)

    r0 = ((X-x0) ** 2) + ((Y-y0) ** 2)
    r1 = ((X-x1) ** 2) + ((Y-y1) ** 2)
    r2 = ((X-x2) ** 2) + ((Y-y2) ** 2)
    r3 = ((X-x3) ** 2) + ((Y-y3) ** 2)
    r4 = ((X-x4) ** 2) + ((Y-y4) ** 2)

    bx0 = c0*I0*(Y-y0)*(1/r0 -
                        np.exp(-r0/(s ** 2))/(s**2))
    by0 = -c0*I0*(X-x0)*(1/r0 -
                         np.exp(-r0/(s ** 2))/(s**2))
    bx1 = c1*I0*(Y-y1)/r1
    by1 = -c1*I0*(X-x1)/r1

    bx2 = c2*I0*(Y-y2)/r2
    by2 = -c2*I0*(X-x2)/r2

    bx3 = c3*I0*(Y-y3)/r3
    by3 = -c3*I0*(X-x3)/r3

    bx4 = c4*I0*(Y-y4)/r4
    by4 = -c4*I0*(X-x4)/r4

    bx = bx0 + bx1 + bx2 + bx3 + bx4  # Total mag fields
    by = by0 + by1 + by2 + by3 + by4

    dpsidx = -by
    dpsidy = bx

    return dpsidx, dpsidy


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def XptCoordsIdx(gradPsi):
    """Computes the coordinates of lower Xpoint /!\ assumes symmetry"""

    global x, y, x0

    dpdx, dpdy = gradPsi

    Bp2 = dpdx**2 + dpdy ** 2

    iY, iX = divmod(Bp2.argmin(), Bp2.shape[1])

    if iY > len(y)/2:
        iY = len(y) - iY

    # iX = find_nearest(x, x0)
    # iY = np.argmin(Bp2[:][iX])

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
    x0, x1, x2, x3, x4, y0, y1, y2, y3, y4, c0, c1, c2, c3, c4 = C

    currents = np.array([[x0, x1, x2, x3, x4], [y0, y1, y2, y3, y4]])

    psi = Psi(C)
    k, d = specs(C)
    Xxpt, Yxpt = XptCoords(C)

    iX, iY = XptCoordsIdx(gradPsi(C))
    levels = np.arange(psi[int(ny/2)][int(nx/2)], psi[int(ny/2)]
                       [nx - 5], step=psi[int(ny/2)][int(nx/2)]/200)

    # Plot level lines
    plt.figure(0)
    plt.axis("equal")
    plt.contour(X, Y, psi, levels=levels, colors='k')

    for i in range(5):
        plt.plot(currents[0][i], currents[1][i], marker='o')

    plt.contour(X, Y, psi, levels=[psi[iY, iX]], colors='r')
    plt.vline(0, 0, Ly, colors='grey', linestyles='dashed')
    plt.vline(Lx, 0, Ly, colors='grey', linestyles='dashed')
    plt.xlabel(r'$R/\rho_{s0}$')
    plt.ylabel(r'$Z/\rho_{s0}$')
    plt.title(r'$\Psi$')
    textstr = fr"$\delta = {d:.4}$""\n"fr"$\kappa = {k:.4}$""\n"r"$Z_{Xpt} = $"f"{Yxpt:.1f}"
    props = dict(boxstyle='round', facecolor='lightsteelblue', alpha=0.5)
    plt.text(-280, 820, textstr, fontsize=14,
             verticalalignment='top', bbox=props)


def specs(C):
    """Computes and returns : elongation kappa, triangularity delta
    """

    global X, Y, x, y
    x0, x1, x2, x3, x4, y0, y1, y2, y3, y4, c0, c1, c2, c3, c4 = C

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


def optimizeDelta0(C, deltaTarg):
    x0, x1, x2, x3, x4, y0, y1, y2, y3, y4, c0, c1, c2, c3, c4 = C
    global I0, s

    def cost(x0):

        Cnew = [x0, x1, x2, x3, x4, y0, y1, y2, y3, y4, c0, c1, c2, c3, c4]
        kappa, delta = specs(Cnew)

        cost = abs(deltaTarg - delta)
        # Xxpt, Yxpt = XptCoords(Cnew)

        # if ((Yxpt > Xub) or (Yxpt < Xlb)):
        #     cost += 1000

        return cost

    x0op = fmin(cost, x0)
    COp = [x0op, x1, x2, x3, x4, y0, y1, y2, y3, y4, c0, c1, c2, c3, c4]

    return COp


def optimizeKappa0(C, kappaTarg, lb, ub):

    x0, x1, x2, x3, x4, y0, y1, y2, y3, y4, c0, c1, c2, c3, c4 = C
    global I0, s

    def cost(P):

        cv, cx, c0 = P

        c1 = cv
        c2 = cv
        c3 = cx
        c4 = cx

        Cnew = [x0, x1, x2, x3, x4, y0, y1, y2, y3, y4, c0, c1, c2, c3, c4]
        kappa, delta = specs(Cnew)
        Xxpt, Yxpt = XptCoords(Cnew)

        cost = abs(kappaTarg - kappa) + ((Yxpt - ub) if (Yxpt > ub)
                                         else (lb-Yxpt) if (lb > Yxpt) else 0)

        return cost

    cv = c1
    cx = c3
    P = [cv, cx, c0]

    POp = fmin(cost, P)

    cv, cx, c0 = POp

    c1 = cv
    c2 = cv
    c3 = cx
    c4 = cx

    COp = [x0, x1, x2, x3, x4, y0, y1, y2, y3, y4, c0, c1, c2, c3, c4]

    return COp


def optimizeDelta(C, deltaTarg):

    x0, x1, x2, x3, x4, y0, y1, y2, y3, y4, c0, c1, c2, c3, c4 = C
    global I0, s

    def costNT(P):

        # cv, xv : amplitude and pos of delta-side ; cx : non-moving currents amplitude
        c0, cv, xv = P
        # print(P)

        c1 = cv
        c2 = cv
        x1 = xv
        x2 = xv

        Cnew = [x0, x1, x2, x3, x4, y0, y1, y2, y3, y4, c0, c1, c2, c3, c4]
        kappa, delta = specs(Cnew)

        cost = abs(deltaTarg - delta)
        # Xxpt, Yxpt = XptCoords(Cnew)

        # if ((Yxpt > Xub) or (Yxpt < Xlb)):
        #     cost += 1000

        return cost

    def costPT(P):

        # cv, xv : amplitude and pos of delta-side ; cx : non-moving currents amplitude
        c0, cv, xv = P

        x3 = xv
        x4 = xv
        c3 = cv
        c4 = cv

        Cnew = [x0, x1, x2, x3, x4, y0, y1, y2, y3, y4, c0, c1, c2, c3, c4]
        kappa, delta = specs(Cnew)

        cost = abs(deltaTarg - delta)
        # Xxpt, Yxpt = XptCoords(Cnew)

        # if ((Yxpt > Xub) or (Yxpt < Xlb)):
        #     cost += 1000

        return cost

    if deltaTarg < 0:
        cv = c1
        xv = x1
        P = [c0, cv, xv]
        POp = fmin(costNT, P, maxiter=1000)

    else:
        xv = x3
        cv = c3
        P = [c0, cv, xv]
        POp = fmin(costPT, P, maxiter=1000)

    c0, cv, xv = POp

    if deltaTarg < 0:
        c1 = cv
        c2 = cv
        x1 = xv
        x2 = xv

    else:
        x3 = xv
        x4 = xv
        c3 = cv
        c4 = cv

    COp = [x0, x1, x2, x3, x4, y0, y1, y2, y3, y4, c0, c1, c2, c3, c4]

    return COp


def optimizeKappa(C, kappaTarg):

    x0, x1, x2, x3, x4, y0, y1, y2, y3, y4, c0, c1, c2, c3, c4 = C
    global I0, s

    def costNT(P):

        # cv, xv : amplitude and pos of delta-side ; cx : non-moving currents amplitude
        c0, cv, cx = P
        # print(P)

        c1 = cv
        c2 = cv
        c3 = cx
        c4 = cx

        Cnew = [x0, x1, x2, x3, x4, y0, y1, y2, y3, y4, c0, c1, c2, c3, c4]
        kappa, delta = specs(Cnew)

        cost = abs(kappaTarg - kappa)
        # Xxpt, Yxpt = XptCoords(Cnew)

        # if (Yxpt > Xub):
        #     cost += 1000*(Yxpt - Xub)
        # elif(Yxpt < Xlb):
        #     cost += 1000*(Xlb-Yxpt)

        return cost

    def costPT(P):

        # cv, xv : amplitude and pos of delta-side ; cx : non-moving currents amplitude
        c0, cv, cx = P

        c1 = cx
        c2 = cx
        c3 = cv
        c4 = cv

        Cnew = [x0, x1, x2, x3, x4, y0, y1, y2, y3, y4, c0, c1, c2, c3, c4]
        kappa, delta = specs(Cnew)

        cost = abs(kappaTarg - kappa)
        # Xxpt, Yxpt = XptCoords(Cnew)

        # if (Yxpt > Xub):
        #     cost += 1000*(Yxpt - Xub)
        # elif(Yxpt < Xlb):
        #     cost += 1000*(Xlb-Yxpt)

        return cost

    if deltaTarg < 0:
        cv = c1
        cx = c3
        P = [c0, cv, cx]
        POp = fmin(costNT, P)

    else:
        cx = c1
        cv = c3
        P = [c0, cv, cx]
        POp = fmin(costPT, P)

    c0, cv, cx = POp

    if deltaTarg < 0:
        c1 = cv
        c2 = cv
        c3 = cx
        c4 = cx

    else:
        c1 = cx
        c2 = cx
        c3 = cv
        c4 = cv

    COp = [x0, x1, x2, x3, x4, y0, y1, y2, y3, y4, c0, c1, c2, c3, c4]

    return COp


def optimizeXpt(C, Xlb, Xub):

    x0, x1, x2, x3, x4, y0, y1, y2, y3, y4, c0, c1, c2, c3, c4 = C
    global I0, s

    def cost(P):

        coeff, coeffc0 = P

        c0new = c0*coeffc0
        c1new = c1*coeff
        c2new = c2*coeff
        c3new = c3*coeff
        c4new = c4*coeff

        Cnew = [x0, x1, x2, x3, x4, y0, y1, y2,
                y3, y4, c0new, c1new, c2new, c3new, c4new]
        Xxpt, Yxpt = XptCoords(Cnew)

        if (Yxpt > Xub):
            return Yxpt - Xub
        elif(Yxpt < Xlb):
            return Xlb-Yxpt
        else:
            return 0

    P = [1, c0]
    POp = fmin(cost, P)

    coeff, c0 = POp

    c0 = c0*coeff
    c1 = c1*coeff
    c2 = c2*coeff
    c3 = c3*coeff
    c4 = c4*coeff

    COp = [x0, x1, x2, x3, x4, y0, y1, y2, y3, y4, c0, c1, c2, c3, c4]

    return COp


def optimizeAll(C, deltaTarg, kappaTarg, lb, ub):

    x0, x1, x2, x3, x4, y0, y1, y2, y3, y4, c0, c1, c2, c3, c4 = C
    global I0, s

    def costNT(P):

        x0, xv, c0, cv, cx = P

        c1 = cv
        c2 = cv
        c3 = cx
        c4 = cx
        x1 = xv
        x2 = xv

        Cnew = [x0, x1, x2, x3, x4, y0, y1, y2, y3, y4, c0, c1, c2, c3, c4]
        Xxpt, Yxpt = XptCoords(Cnew)

        if (not isXptInBnds(Cnew, lb, ub)):
            return np.inf

        kappa, delta = specs(Cnew)

        cost = abs(kappaTarg - kappa) + abs(deltaTarg-delta)
        return cost

    def costPT(P):

        x0, xv, c0, cv, cx = P

        x3 = xv
        x4 = xv
        c1 = cx
        c2 = cx
        c3 = cv
        c4 = cv

        Cnew = [x0, x1, x2, x3, x4, y0, y1, y2, y3, y4, c0, c1, c2, c3, c4]
        Xxpt, Yxpt = XptCoords(Cnew)

        if (not isXptInBnds(Cnew, lb, ub)):
            return np.inf

        kappa, delta = specs(Cnew)

        cost = abs(kappaTarg - kappa) + abs(deltaTarg-delta)
        return cost

    if deltaTarg < 0:
        cv = c1
        cx = c3
        xv = x1
        P = [x0, xv, c0, cv, cx]
        POp = fmin(costNT, P)

    else:
        xv = x3
        cv = c3
        cx = c1
        P = [x0, xv, c0, cv, cx]
        POp = fmin(costPT, P)

    x0, xv, c0, cv, cx = POp

    if deltaTarg < 0:
        c1 = cv
        c2 = cv
        c3 = cx
        c4 = cx
        x1 = xv
        x2 = xv

    else:
        x3 = xv
        x4 = xv
        c1 = cx
        c2 = cx
        c3 = cv
        c4 = cv

    COp = [x0, x1, x2, x3, x4, y0, y1, y2, y3, y4, c0, c1, c2, c3, c4]
    return COp


# %% Plot initial config
plt.figure()
plotMagField(C)
plt.show()
plt.close()

# %% OPTIMIZE TRIANGULARITY


kappa, delta = specs(C)
last = ""

for deltaTarg in deltaTargs:

    iteration = 0
    tol = tolerance
    C = C0
    print(f"Target triangularity : delta = {deltaTarg}")
    print(f"Target elongation : kappa = {kappaTarg}")
    while ((abs(kappa - kappaTarg) > tol) or (abs(delta - deltaTarg) > tol) or (not isXptInBnds(C, Xlb, Xub))):

        iteration += 1

        if iteration % 20 == 0:
            tol += tolerance

        print(f"Round n° {iteration} : All-in-1-optimization...")

        C = optimizeAll(C, deltaTarg, kappaTarg, Xlb, Xub)
        kappa, delta = specs(C)
        Xxpt, Yxpt = XptCoords(C)
        plt.figure()
        plotMagField(C)
        plt.show()

        print(f"Kappa after D-op : {kappa}")
        print(f"Delta after D-op : {delta}")
        print(f"Yxpt after Xpt-op : {Yxpt}")
        print("Current configuration : ")
        print(C)

        # print(f"Round n° {iteration} : delta-optimization...")

        # #C = optimizeDelta(C, deltaTarg)
        # C = optimizeDelta0(C, deltaTarg)

        # kappa, delta = specs(C)

        # Xxpt, Yxpt = XptCoords(C)
        # print(f"Kappa after D-op : {kappa}")
        # print(f"Delta after D-op : {delta}")
        # print(f"Yxpt after Xpt-op : {Yxpt}")

        # print(f"Round n° {iteration} : kappa-optimization...")

        # C = optimizeKappa0(C, kappaTarg, Xlb, Xub)
        # kappa, delta = specs(C)
        # Xxpt, Yxpt = XptCoords(C)
        # print(f"Kappa after K-op : {kappa}")
        # print(f"Delta after K-op : {delta}")
        # print(f"Yxpt after Xpt-op : {Yxpt}")

        # C = optimizeXpt(C, Xlb, Xub)
        # kappa, delta = specs(C)
        # Xxpt, Yxpt = XptCoords(C)

        # print(f"Kappa after Xpt-op : {kappa}")
        # print(f"Delta after Xpt-op : {delta}")
        # print(f"Yxpt after Xpt-op : {Yxpt}")

        print("----------------------------")

    if ((abs(kappa - kappaTarg) < tol) and (abs(delta - deltaTarg) < tol)):
        executionTime = (time.time() - startTime)
        print('Execution time in seconds: ' + str(executionTime))
        print('Execution time in minutes: ' + str(executionTime/60))
        print("Optimization success!")

    print("Final configuration : ")
    print(C)

    with open(respath+f"C_delta_{delta}_kappa_{kappa}.pkl", "wb") as file:
        dump(C, file)

    plt.figure()
    plotMagField(C)
    plt.show()

    plt.savefig(respath+f"C_delta_{delta}_kappa_{kappa}.png", format="png")


# %% save figures from previous sims


# path = "C:/Users/lgtle/Desktop/SPC/Matlab_DN_+_NT_PT/resultsPy/"
# files = [i for i in os.listdir(path) if os.path.isfile(os.path.join(path, i)) and
#          'C_delta' in i]
# for file in files:
#     print("yo")
#     with open(path+file, "rb") as f:
#         C = load(f)
#         plt.figure()
#         plotMagField(C)
#         plt.savefig(respath+f"{file}.png", format="png")
#         plt.close()
