# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 13:18:33 2023

@author: lgtle
"""
# %% Import necessary modules

from scipy.spatial import Delaunay
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import re
import pandas as pd

plt.rcParams['figure.dpi'] = 500

# %% Functions


def cleanParams(f):

    # ASSUMES ONLY ONE ELEMENT IN f["files"] ! Always the case?

    for key in f["files"].keys():
        paramsName = key

    params = str(f["files"][paramsName][0].decode('ASCII'))
    params = params.removeprefix(
        "!\n!\n!\n!\n&BASIC\n").removesuffix("\n/\n").replace(',', '\n').strip().split("\n")
    params = [el.strip() for el in params if ('&' not in el)
              and ('/' not in el) and (el.strip() != "")]
    keys = [el.split('=')[0].strip() for el in params]
    vals = [el.split('=')[1].strip(" \n'\t") for el in params]
    vals = [el.split('!')[0].strip() if '!' in el else el for el in vals]

    # Processing of number-like values
    for i, elt in enumerate(vals):
        # if some digits in elt
        if re.search('\d', elt):
            # if character D in elt and everything before and after is digit
            if ('D' in elt) and (elt[:elt.find("D")].replace('.', '', 1).replace('-', '', 1).isdigit()) and elt[elt.find("D")+1:].replace('-', '', 1).replace('+', '', 1).isdigit():
                power = elt[elt.find("D")+1:]
                mantisse = elt[:elt.find("D")]
                elt = float(mantisse)*(10**int(power))
                # if int, transform
                if elt - int(elt) == 0:
                    elt = int(elt)
            # if float-like string
            elif elt.replace('.', '', 1).isdigit():
                elt = float(elt)
                if elt - int(elt) == 0:
                    elt = int(elt)
            elif 'e' in elt and (elt[:elt.find("e")].replace('.', '', 1).replace('-', '', 1).isdigit()) and elt[elt.find("e")+1:].replace('-', '', 1).isdigit():
                elt = float(elt)
                if elt - int(elt) == 0:
                    elt = int(elt)

            vals[i] = elt  # save modif

    vals = [True if (el in ('.true.', ' .true.')) else el for el in vals]
    vals = [False if el in (".false.", " .false.") else el for el in vals]
    vals = [None if el == 'none' else el for el in vals]

    params = dict(zip(keys, vals))
    return params


def XptCoordsIdx(gradPsi):
    """Computes the coordinates of lower Xpoint /!\ assumes symmetry"""

    global x, y

    dpdx, dpdy = gradPsi

    dpdx = dpdx
    dpdy = dpdy

    Bp2 = dpdx**2 + dpdy ** 2

    iY, iX = divmod(Bp2.argmin(), Bp2.shape[1])

    if iY > len(y)/2:
        iY = len(y) - iY

    return iX, iY


def XptCoords(gradPsi):

    iX, iY = XptCoordsIdx(gradPsi)

    return x[iX], y[iY]


def getSep(psi, gradPsi):
    """Returns points of the contour of separatrix"""

    global X, Y, x, y

    iX, iY = XptCoordsIdx(gradPsi)
    Xxpt = x[iX]
    Yxptlow = y[iY]
    Yxptup = y[-1] - Yxptlow

    level = psi[iY, iX]

    plt.figure(1)
    cs = plt.contour(X, Y, psi, levels=[level])
    plt.close()

    separatrix = [x.vertices for x in cs.collections[0].get_paths()]
    separatrix = np.concatenate(separatrix)
    return separatrix


def getInSep(psi, gradPsi):
    """Returns indices of raveled points of mesh located inside separatrix"""

    global X, Y, x, y, meshPoints

    Xxpt, Yxpt = XptCoords(gradPsi)
    Yup = Ly - Yxpt

    sep = getSep(psi, gradPsi)

    sep = np.array([p for p in sep if (p[1] < Yup) and (p[1] > Yxpt)])

    contour_points = Path(sep)

    inside = contour_points.contains_points(meshPoints)
    idxInside = np.where(inside)

    return idxInside


def specs(psi, gradPsi):
    """Computes and returns : elongation kappa, triangularity delta
    """

    global X, Y, x, y

    iX, iY = XptCoordsIdx(gradPsi)
    Xxpt = x[iX]
    Yxptlow = y[iY]
    Yxptup = y[-1] - Yxptlow

    # level = psi[iY, iX]

    # plt.figure(1)
    # cs = plt.contour(X, Y, psi, levels=[level])
    # plt.close()

    # separatrix = [x.vertices for x in cs.collections[0].get_paths()]
    # separatrix = np.concatenate(separatrix)
    # print(separatrix)

    separatrix = getSep(psi, gradPsi)

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

    return kappa, delta, Rmin, Rmax


def plotMagField(dpdx, dpdy, Psi, delta):
    """Plots magnetic field"""

    global x, y

    iX, iY = XptCoordsIdx((dpdx, dpdy))
    Yxpt = y[iY]

    sgn = -1 if delta < 0 else 1
    iY0, iX0 = divmod(Psi.argmin(), Psi.shape[1])

    levels = np.arange(Psi[iY0][iX0], Psi[int(ny/2)]
                       [nx - 5 if sgn == -1 else 6], step=Psi[int(ny/2)][int(nx/2)]/200)

    k, d, Rmin, Rmax = specs(Psi, (dpdx, dpdy))

    # Plot level lines
    plt.figure()
    plt.axis("equal")
    plt.contour(X, Y, Psi, levels=levels, colors='k')

    plt.contour(X, Y, Psi, levels=[Psi[iY, iX]], colors='r')
    plt.vlines(0, 0, Ly, colors='grey',
               linestyles='dashed', label='_nolegend_')
    plt.vlines(Lx, 0, Ly, colors='grey',
               linestyles='dashed', label='_nolegend_')
    plt.xlabel(r'$R/\rho_{s0}$')
    plt.ylabel(r'$Z/\rho_{s0}$')
    plt.title(r'$\Psi$')
    textstr = fr"$\delta = {d:.4}$""\n"fr"$\kappa = {k:.4}$""\n"r"$Z_{Xpt} = $"f"{Yxpt:.1f}"
    props = dict(boxstyle='round',
                 facecolor='lightsteelblue', alpha=0.5)
    plt.text(0.02, 0.98, textstr, fontsize=14,
             verticalalignment='top', bbox=props, transform=plt.gca().transAxes)


def pltColorMap(data, vmin=None, vmax=None, title=None):

    global x, y, PsiEquil

    iX, iY = XptCoordsIdx((dpdxEquil, dpdyEquil))

    data = data.transpose()

    plt.figure()
    plt.imshow(data, cmap='seismic', origin='lower',
               extent=[x[0], x[-1], y[0], y[-1]], vmin=vmin, vmax=vmax)

    plt.colorbar()
    plt.contour(X, Y, PsiEquil, levels=[
                PsiEquil[iY, iX]], colors='magenta', linestyles='dashed')
    plt.xlabel(r'$R/\rho_{s0}$')
    plt.ylabel(r'$Z/\rho_{s0}$')
    if title != None:
        plt.title(title)


# %% Import data

figPath = "figures/"
resID = [132, 133]
resID.sort()

files = [h5.File(f'results_{i}.h5', 'r') for i in resID]

flast = files[-1]

dataEquil = flast['equil']

PsiEquil = np.array(dataEquil['Psi']).transpose()
dpdxEquil = np.array(dataEquil['dpsidx_v']).transpose()
dpdyEquil = np.array(dataEquil['dpsidy_v']).transpose()


q = 4  # TEMPORARY

# Lx = 600
# Ly = 800
params = cleanParams(flast)
nu = params['nu']
x0n = params['x0_EC_theta']
x0t = params['x0_EC_tempe']


wn = params['wEC_theta']
wt = params['wEC_tempe']
An = params['SEC_theta']
At = params['SEC_tempe']
rorho = params['rorho_s']

Lx = params["xmax"] - params["xmin"]
Ly = params["ymax"] - params["ymin"]

y0 = 0.5*Ly

# WEIRD CONVENTION

nx = params['nx'] + 4
ny = params['ny'] + 4
nz = params['nz']

# dx = (params["xmax"] - params["xmin"])/params['nx']
# dy = (params["ymax"] - params["ymin"])/params['ny']
# nx = 244
# ny = 324

dx = Lx/(nx-4)
dy = Ly/(ny-4)

x = np.arange(-3/2*dx, Lx+5*dx/2, step=dx)
y = np.arange(-3/2*dy, Ly+5*dy/2, step=dy)


# x = np.arange(params['xmin'], params['xmax'], step=dx)
# y = np.arange(params['ymin'], params['ymax'], step=dy)


X, Y = np.meshgrid(x, y)
meshPoints = np.column_stack((X.ravel(), Y.ravel()))

k, d, rmin, rmax = specs(PsiEquil, (dpdxEquil, dpdyEquil))

# %% Plot data

df = pd.DataFrame(columns=['theta', 'n', 'T_e', 'T_i'])

iX, iY = XptCoordsIdx((dpdxEquil, dpdyEquil))
delta = 0
plt.figure()
plotMagField(dpdxEquil, dpdyEquil, PsiEquil, delta)
plt.show()


nbIter = 0

# Sum all iterations
for f in files:
    thetaCatalogue = f['data']['var3d']['theta']
    TECat = f['data']['var3d']['temperature']
    TICat = f['data']['var3d']['temperaturi']
    # Add a row to df for each iteration
    for key, iteration in thetaCatalogue.items():
        if 'coord' in key:
            continue

        nbIter += 1
        df.loc[key] = [np.array(iteration), np.exp(
            np.array(iteration)), np.array(TECat[key]), np.array(TICat[key])]

# Compute pressure
df['p_e'] = df['n'] * df['T_e']

# Mean along z axis


def mean0(a):
    return np.mean(a, axis=0)


# Toroidally averaged data
dfAvTor = pd.DataFrame(
    dict([(col, df[col].apply(mean0)) for col in df.columns]))

dfAvIter = pd.Series(dict([(col, df[col].sum()/nbIter)
                           for col in df.columns]))

# Toroidally and iterations-averaged data
dfAv = pd.Series(dict([(col, dfAvTor[col].sum()/nbIter)
                       for col in dfAvTor.columns]))

nTilde = dfAvTor['n'][-1] - dfAv['n']


# %%
pltColorMap(dfAv['theta'])
plt.title(r"$\overline{\theta}$")
plt.savefig(fname=figPath + f"thetAverage.eps", format='eps')
plt.show()

pltColorMap(dfAv['T_e'])
plt.title(r"$\overline{T_e}/T_{e0}$")
plt.savefig(fname=figPath + f"TEAverage.eps", format='eps')
plt.show()

pltColorMap(dfAv['T_i'])
plt.title(r"$\overline{T_i}/T_{i0}$")
plt.savefig(fname=figPath + f"TIAverage.eps", format='eps')
plt.show()

pltColorMap(nTilde)
plt.title(r"$\tilde{n}_{tfin}$")
plt.savefig(fname=figPath + f"ntilde.eps", format='eps')
plt.show()
# %%
pltColorMap(dfAv['p_e'])
plt.title(r"$p_e$")
plt.savefig(fname=figPath + f"pe.eps", format='eps')
plt.show()
# %%
# pltColorMap(LpSim, vmin=-100, vmax=100)
# plt.title(r"$L_p$")
# plt.savefig(fname=figPath + f"Lp.eps", format='eps')
# plt.show()

# plt.figure()
# plt.plot(x, LpRad)
# plt.show()

# %% sim Lp

# xzoom = [c for c in x if (c < 500) and (c > 400)]
# LpRadzoom = [LpRad[i]
#              for i in range(len(LpRad)) if (x[i] < 500) and (x[i] > 400)]

# plt.figure()
# plt.plot(xzoom, LpRadzoom)
# plt.show()

# Compute pressure gradient at last snapshot
# LpSim = np.array(
#     -dfAvTor['p_e'][-1] / np.gradient(dfAvTor['p_e'][-1], axis=0))


# Compute toroidally averaged Lp at each iteration
LpSim = pd.Series([np.array(-dfAvTor['p_e'][it] / np.gradient(dfAvTor['p_e'][it], axis=0))
                  for it in dfAvTor.index], name='Lp', index=dfAvTor.index)

# Selects region y = y_outer-middle-plane for plotting radial profile
LpRad = pd.Series([Lpit[:, round(len(y)/2)]
                  for Lpit in LpSim], name='Lp_y=omp', index=LpSim.index)

# Selects closest mesh point to outer middle plane
idxOMP = np.argmin(abs(x-rmax))  # Index in mesh of outer middle plane

LpOMP = pd.Series([LpRadit[idxOMP] for LpRadit in LpRad],
                  name='Lp@OMP', index=LpRad.index)


# %% Compute and integrate Sp


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


ix0n = find_nearest(x, 300 + x0n)
ix0t = find_nearest(x, 300+x0t)
iy0 = find_nearest(y, y0)

psi0n = PsiEquil[iy0, ix0n]
psi0t = PsiEquil[iy0, ix0t]

iYup = ny - iY
Yxpt = y[iY]
YxptUp = y[iYup]

dA = dx*dy

Sn = An*np.exp(-((PsiEquil - psi0n)**2)/(wn**2))/rorho
# Temperature source (Tanh)
St = At/2 * (np.tanh(-(PsiEquil - psi0t)/wt) + 1)/rorho
# Power source iteration by iteration
Sp = pd.Series([dfAvTor['n'][it] * St.transpose() + dfAvTor['T_e']
               [it] * Sn.transpose() for it in dfAvTor.index], index=dfAvTor.index, name='Sp')


pltColorMap(St.transpose(), title=r'$St$')
plt.show()

pltColorMap(Sn.transpose(), title=r'$Sn$')
plt.show()

# Plot last iteration of Sp
pltColorMap(Sp[-1], title=r'$Sp_{last}$')
plt.show()


# Integration inside separatrix, it by it

idxInside = getInSep(PsiEquil, (dpdxEquil, dpdyEquil))[0]
# idxInside = np.array([i for i in idxInside if (
#     (meshPoints[i][1] > Yxpt) and (meshPoints[i][1] < YxptUp))])

triangulation = Delaunay(meshPoints)
surface_triangles = triangulation.simplices[idxInside]
triangle_vertices = meshPoints[surface_triangles]
triangle_areas = 0.5 * np.abs(
    triangle_vertices[:, 0, 0] * (triangle_vertices[:, 1, 1] - triangle_vertices[:, 2, 1]) +
    triangle_vertices[:, 1, 0] * (triangle_vertices[:, 2, 1] - triangle_vertices[:, 0, 1]) +
    triangle_vertices[:, 2, 0] *
    (triangle_vertices[:, 0, 1] - triangle_vertices[:, 1, 1])
)

#  Integrate the function over the surface using the calculated triangle areas:


#integral = np.sum(function_values[surface_indices] * triangle_areas)


Sp_tot = pd.Series([np.sum(SpIt.ravel()[idxInside] * dA)
                   for SpIt in Sp], index=Sp.index, name='Sp_tot')
Sp_tot2 = pd.Series([np.sum(SpIt.ravel()[idxInside] * triangle_areas)
                     for SpIt in Sp], index=Sp.index, name='Sp_tot')


pointsIn = meshPoints[idxInside]
# pointsIn = np.array([p for p in pointsIn if (p[1] > Yxpt) and (p[1] < YxptUp)])


# %% Plot zone of integration
plt.figure()
plotMagField(dpdxEquil, dpdyEquil, PsiEquil, delta)
plt.plot(pointsIn[:, 0], pointsIn[:, 1])
plt.plot(x[idxOMP], y[round(len(y)/2)], 'r+', markersize=12)
plt.legend(['int zone', 'OMP'])
plt.show()


# %% compute analytical Lp


nOMP = pd.Series([nIt[idxOMP][round(len(y)/2)]
                  for nIt in dfAvTor['n']], name='nOMP', index=dfAvTor.index)
pOMP = pd.Series([pIt[idxOMP][round(len(y)/2)]
                  for pIt in dfAvTor['p_e']], name='nOMP', index=dfAvTor.index)

# nOMP = dfAv['n'][idxOMP][round(len(y)/2)]

# pOMP = dfAv['p_e'][idxOMP][round(len(y)/2)]


def Lpth(psi, dpdx, dpdy, Sp_tot):
    """Computes theoretical value of L_p"""

    global q, nu, dfAvTor
    rho = 1/700

    k, d, Rmin, Rmax = specs(psi, (dpdx, dpdy))
    a = (Rmax - Rmin)/2

    C = 1-((k-1)/(k+1)) * (3*q/(q+2)) + delta*q/(1+q) + ((k-1)**2) * \
        (5*q-2)/(2*((k+1)**2)*(q+2)) + (d**2)*(7*q-1)/(16*(1+q))
    Lchi = np.pi*a*(0.45 + 0.55*k) + 1.33*a*d

    Lp = C * ((rho * ((nu*nOMP*(q**2))**2)*((Lchi*pOMP/Sp_tot)**4))**(1/3))
    Lp.name = 'Lp_th'
    return Lp


Lp_th = Lpth(PsiEquil, dpdxEquil, dpdyEquil, Sp_tot)
Lp_th2 = Lpth(PsiEquil, dpdxEquil, dpdyEquil, Sp_tot2)
