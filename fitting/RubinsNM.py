import copy

import numpy as np

try:
    import commands
except:
    import subprocess

import sys
import time

import astropy.io.fits as pyfits
from scipy.sparse import lil_matrix

# Version 1: Starting to keep track of versions
# 1.01: added wmat_to_rho
# 1.02: Prints out diagnostics for incorrectly-sized Wmat in LM
# 1.03: Added 'verbose' option to secderiv
# 1.04: Added wmat_to_errormat
# 1.05: Added option to save jacobian from LM as something
# 1.06: Added wrapper around miniLM, including cov mat return
# 1.07: Added wrapper around miniNM
# 1.08: Added option to turn Cmat computation off for miniNM_new
# 1.1: Jacobian now stored as sparse matrix.
# 1.11: 'verbose' option for secderiv passed through miniNM_new
# 1.12: Added 'save_patches'
# 1.13: Added nan check to miniLM
# 1.14: Added option to miniLM to use dense jacobian matrices (faster, if they are dense)
# 1.15: Added option to miniLM_new to not return Cmat, won't recompute Jacobian after search!
# 1.16: Added option to miniLM to use multiprocessing pool
# 1.17: Added eigenvector decomposition (better linalg.eig)
# 1.18: Added save_jacobian to miniLM
# 1.19: pool works now in miniLM
# 1.20: Warns about determinant
# 1.21: Secder automatic sign flip if it sees the fit is up against a bound
# 1.22: Improved version of save_patches
# 1.23: Linting
# 1.24: Verbosity


def print_verb(v, *args):
    if v:
        print(*args)


verbose = False
version = 1.24
print_verb(verbose, "DavidsNM Version ", version)


def eig(the_matrix, verbose=False):
    if any(np.isnan(the_matrix)):
        print_verb(verbose, "Couldn't decompose matrix!")
        return 0, 0, 0

    evals, evecs = np.linalg.eig(the_matrix)

    inds = np.argsort(evals)[::-1]

    evecs = np.transpose(evecs)
    evecs = evecs[inds]
    evals = evals[inds]
    evecs_norm = np.array([evecs[i] * np.sqrt(evals[i]) for i in range(len(evecs))])
    return evals, evecs, evecs_norm


def save_img(dat, imname):

    try:
        subprocess.getoutput("rm -f " + imname)
    except:
        commands.getoutput("rm -f " + imname)

    fitsobj = pyfits.HDUList()
    hdu = pyfits.PrimaryHDU()
    hdu.data = dat
    fitsobj.append(hdu)
    fitsobj.writeto(imname)
    fitsobj.close()


def save_patches(patch_list, imname):
    try:
        patch1 = len(patch_list[0])
        patch2 = len(patch_list[0][0])
    except:
        save_img(np.zeros([0, 0]), imname)
        return 0

    ideal_pix_on_a_side = np.sqrt(len(patch_list) * patch1 * patch2 * 1.0)
    grid1 = int(np.ceil(ideal_pix_on_a_side / patch1))
    grid2 = int(np.ceil(ideal_pix_on_a_side / patch2))

    all_data = np.zeros([grid1 * patch1, grid2 * patch2], dtype=np.float64) + np.sqrt(
        -1.0
    )

    max_i = 0
    max_j = 0
    for i in range(len(patch_list)):
        ipos = i % grid1
        jpos = int(np.floor(i / grid1))

        all_data[
            ipos * patch1 : (ipos + 1) * patch1, jpos * patch2 : (jpos + 1) * patch2
        ] = patch_list[i]

        max_i = max(max_i, (ipos + 1) * patch1)
        max_j = max(max_j, (jpos + 1) * patch2)

    save_img(all_data[:max_i, :max_j], imname)


def get_simpons_weights(n, verbose=False):
    if n % 2 != 1:
        print_verb(verbose, "n should be odd!", n)
        return None
    weights = np.zeros(n, dtype=np.float64)
    weights[1::2] = 4.0
    weights[0::2] = 2.0
    weights[0] = 1.0
    weights[n - 1] = 1.0

    return weights / sum(weights)


# Start of minimization routines


def wmat_to_rhomat(wmat):
    covmat = np.linalg.inv(wmat)
    rhomat = np.zeros(wmat.shape, dtype=np.float64)

    for i in range(len(wmat)):
        for j in range(len(wmat)):
            rhomat[i, j] = covmat[i, j] / np.sqrt(covmat[i, i] * covmat[j, j])
    return rhomat


def wmat_to_errormat(wmat):
    covmat = np.linalg.inv(wmat)
    errormat = np.zeros(wmat.shape, dtype=np.float64)

    for i in range(len(wmat)):
        for j in range(len(wmat)):
            errormat[i, j] = np.sign(covmat[i, j]) * np.sqrt(abs(covmat[i, j]))
    return errormat


def smoothprior(x):  # quadratic for small x, linear for large
    return 1000.0 * (np.sqrt(1.0 + x**2.0) - 1.0)


def f(P, merged_list):
    chi2fn = merged_list[0]
    return chi2fn(P, merged_list[2:])


def listsort(pf):
    [P, F] = pf

    tmplist = []
    for i in range(len(P)):
        tmplist.append([F[i], P[i].tolist()])
    tmplist.sort()

    P = []
    F = []
    for i in range(len(tmplist)):
        P.append(tmplist[i][1])
        F.append(tmplist[i][0])
    P = np.array(P, dtype=np.float64)
    return [P, F]


def strlist(list):
    newlist = []
    for i in range(len(list)):
        newlist.append(str(list[i]))
    return newlist


def find_best_dx(
    displ_params,
    free_params,
    goal,
    i,
    P,
    minchi,
    merged_list,
    sign_to_use=1.0,
    verbose=False,
):
    dx_scale_max = -1e6
    dx_scale_min = 1e6

    dx_scale = 1.0
    dP = np.zeros(len(P[0]), dtype=np.float64)
    triesfordx = 0

    dP[i] = displ_params[free_params.index(i)] * dx_scale * sign_to_use
    fPdP = f(P[0] + dP, merged_list)

    while abs(fPdP - minchi) > 2.0 * goal or abs(fPdP - minchi) < goal / 2.0:
        print_verb(verbose, i, dx_scale, dx_scale_min, dx_scale_max, fPdP)

        if abs(fPdP - minchi) < goal / 2.0:  # point is too close to chi2 minimum
            dx_scale_min = min([dx_scale_min, dx_scale])
            dx_scale *= 2.0

        if abs(fPdP - minchi) > 2.0 * goal:  # point is too far away from chi2 minmum
            dx_scale_max = max([dx_scale_max, dx_scale])
            dx_scale /= 2.0

        if dx_scale_min < dx_scale_max:
            dx_scale = (dx_scale_max + dx_scale_min) / 2.0

        dP[i] = displ_params[free_params.index(i)] * dx_scale * sign_to_use
        fPdP = f(P[0] + dP, merged_list)

        triesfordx += 1

        if triesfordx > 100:
            print_verb(verbose, "Couldn't get dx, i = ", i)
            return None
    return displ_params[free_params.index(i)] * dx_scale * sign_to_use


def secderiv(
    P, merged_list, displ_list, goal, verbose=False
):  # goal is delta chi2 from minimum, sqrt(goal) is about the sigma
    print_verb(verbose, "goal ", goal)

    free_params = []
    displ_params = []
    for i in range(len(displ_list)):
        if displ_list[i] != 0:
            free_params.append(i)
            displ_params.append(displ_list[i] / 10.0)
    print_verb(verbose, "free_params ", free_params)
    print_verb(verbose, "displ_params ", displ_params)

    W = np.zeros([len(free_params)] * 2, dtype=np.float64)

    minchi = f(P[0], merged_list)
    dx = np.zeros(len(free_params), dtype=np.float64)

    for i in free_params:
        tmp_dx = find_best_dx(
            displ_params=displ_params,
            free_params=free_params,
            goal=goal,
            i=i,
            P=P,
            minchi=minchi,
            merged_list=merged_list,
            sign_to_use=1.0,
            verbose=verbose,
        )
        if tmp_dx is None:
            print_verb(verbose, "Flipping sign!")
            tmp_dx = find_best_dx(
                displ_params=displ_params,
                free_params=free_params,
                goal=goal,
                i=i,
                P=P,
                minchi=minchi,
                merged_list=merged_list,
                sign_to_use=-1.0,
                verbose=verbose,
            )
        if tmp_dx is None:
            print_verb(verbose, "Tried sign flip, gave up!")
            sys.exit(1)

        dx[free_params.index(i)] = tmp_dx
        print_verb(verbose, "dx[free_params.index(i)], ", dx[free_params.index(i)])
    print_verb(verbose, "dx ", dx)

    F0 = f(P[0], merged_list)

    Pcollection = [P[0]]
    Fcollection = [f(P[0], merged_list)]

    for i in free_params:
        print_verb(verbose, i)
        for j in free_params:

            if j >= i:
                dP1 = np.zeros(len(P[0]), dtype=np.float64)
                dP1[i] = dx[free_params.index(i)]
                dP2 = np.zeros(len(P[0]), dtype=np.float64)
                dP2[j] = dx[free_params.index(j)]

                F0 = f(P[0] - (dP1 + dP2) * 0.5, merged_list)
                F1 = f(P[0] + dP1 - (dP1 + dP2) * 0.5, merged_list)
                F12 = f(P[0] + dP1 + dP2 - (dP1 + dP2) * 0.5, merged_list)
                F2 = f(P[0] + dP2 - (dP1 + dP2) * 0.5, merged_list)

                Pcollection.append(P[0] + dP1)
                Fcollection.append(F1)
                Pcollection.append(P[0] + dP2)
                Fcollection.append(F2)
                Pcollection.append(P[0] + dP1 + dP2)
                Fcollection.append(F12)

                W[free_params.index(i), free_params.index(j)] = (F12 - F1 - F2 + F0) / (
                    dx[free_params.index(i)] * dx[free_params.index(j)]
                )
                W[free_params.index(j), free_params.index(i)] = W[
                    free_params.index(i), free_params.index(j)
                ]

    W /= 2.0

    print_verb(verbose, "Weight Matrix ", W)
    return [W, Pcollection, Fcollection]


def fillindx(dx, free_params, displ_list):
    tmp_dx = np.zeros(len(displ_list), dtype=np.float64)

    for i in range(len(free_params)):
        tmp_dx[free_params[i]] = dx[i]
    return tmp_dx


def better_secderiv(
    P, merged_list, displ_list, goal, verbose=False
):  # goal is delta chi2 from minimum, sqrt(goal) is about the sigma
    print_verb(verbose, "goal ", goal)

    free_params = []
    displ_params = []
    for i in range(len(displ_list)):
        if displ_list[i] != 0:
            free_params.append(i)
            displ_params.append(abs(displ_list[i]))
    print_verb(verbose, "free_params ", free_params)
    print_verb(verbose, "displ_params ", displ_params)

    W = np.zeros([len(free_params)] * 2, dtype=np.float64)

    minchi = f(P[0], merged_list)

    dx = []
    for i in range(len(free_params)):
        dx.append(np.zeros(len(free_params), dtype=np.float64))
        dx[i][i] = displ_params[i]

    for k in range(2):
        print_verb(verbose, "dx ")
        print_verb(verbose, dx)

        for i in range(len(free_params)):  # Normalize the dxs
            scale = 1.0  # scale factor for dx[i]

            scale_max = 1.0e10
            scale_min = 0.0

            triesforscale = 0

            tmp_dx = fillindx(dx[i], free_params, displ_list)

            fPdP = f(P[0] + scale * tmp_dx, merged_list)

            while abs(fPdP - minchi) > 2.0 * goal or abs(fPdP - minchi) < goal / 2.0:
                # print i, tmp_dx, tmp_max, tmp_min, f(P[0] + dP, merged_list)
                if abs(fPdP - minchi) < goal / 2.0:  # too close to minimum
                    scale_min = max(scale_min, scale)  # must be at least this far away
                    scale *= 2.0

                if abs(fPdP - minchi) > 2.0 * goal:
                    scale_max = min(scale_max, scale)
                    scale /= 2.0

                if scale_max != 1.0e10 and scale_min != 0.0:
                    scale = (scale_max + scale_min) / 2.0

                fPdP = f(P[0] + tmp_dx * scale, merged_list)

                if (
                    abs(fPdP - minchi) <= 2.0 * goal
                    and abs(fPdP - minchi) >= goal / 2.0
                ):
                    dx[i] *= scale

                triesforscale += 1

                if triesforscale > 100:
                    print_verb(verbose, "Couldn't get dx, i = ", i)
                    sys.exit(1)

            print_verb(verbose, "dx[i], ", dx[i])
        print_verb(verbose, "dx ", dx)

        for i in range(len(free_params)):
            for j in range(len(free_params)):

                tmp_dx1 = fillindx(dx[i], free_params, displ_list)
                tmp_dx2 = fillindx(dx[j], free_params, displ_list)

                pt0 = P[0]
                pt1 = P[0] + tmp_dx1
                pt2 = P[0] + tmp_dx2
                pt3 = P[0] + tmp_dx1 + tmp_dx2

                W[i, j] = (
                    f(pt0, merged_list)
                    - f(pt1, merged_list)
                    - f(pt2, merged_list)
                    + f(pt3, merged_list)
                ) / (
                    np.sqrt(np.dot(tmp_dx1, tmp_dx1))
                    * np.sqrt(np.dot(tmp_dx2, tmp_dx2))
                )

                W[j, i] = W[i, j]
                # I guess this line explicitly enforces symmetry
        print_verb(verbose, "Weight Matrix iter ", k)
        print_verb(verbose, W)

        eig_vec = np.linalg.eig(W)[1]
        print_verb(verbose, "eig_vec")
        print_verb(verbose, eig_vec)

        dx = eig_vec

    W = np.dot(np.transpose(dx), np.dot(W, np.linalg.inv(np.transpose(dx))))
    W /= 2.0
    print_verb(verbose, "Weight Matrix ")
    print_verb(verbose, W)

    return W


def linfit(x1, y1, x2, y2, targ, limit1, limit2, verbose=False):
    print_verb(verbose, "linfit ", [x1, y1], [x2, y2])
    slope = (y1 - y2) / (x1 - x2)
    inter = (x1 * y2 - x2 * y1) / (x1 - x2)

    bestguess = (targ - inter) / slope

    if bestguess > limit1 and bestguess > limit2:  # too high
        bestguess = max(limit1, limit2)

    if bestguess < limit1 and bestguess < limit2:  # too low
        bestguess = min(limit1, limit2)

    return bestguess


def minos_f(ministarts, minioffsets, merged_list, dx, pos, verbose=False):

    bestF = -1

    tmpstarts = np.array(ministarts, dtype=np.float64)

    try:
        len(pos)
        tmpstarts += pos * dx

        if any(tmpstarts * pos != tmpstarts[0] * pos):
            print_verb(verbose, "tmpstarts are conflicting!")
            print_verb(verbose, "tmpstarts ", tmpstarts)
            print_verb(verbose, "ministarts ", ministarts)
            sys.exit(1)
    except:
        tmpstarts[:, pos] += dx

        if any(tmpstarts[:, pos] != tmpstarts[0, pos]):
            print_verb(verbose, "tmpstarts are conflicting!")
            print_verb(verbose, "tmpstarts ", tmpstarts)
            print_verb(verbose, "ministarts ", ministarts)
            sys.exit(1)

    tmpstarts = tmpstarts.tolist()

    for i in range(len(ministarts)):
        [P, F] = miniNM(tmpstarts[i], [1.0e-6, 1.0e-8], merged_list, minioffsets[i], 0)
        # [P, F] = miniNM(tmpstarts[i], [1.e-8, 1.e-10], merged_list, minioffsets[i], 0)

        if F[0] < bestF or bestF == -1:
            bestF = F[0]
            bestP = P[0]
        if np.isnan(bestF):
            print_verb(verbose, "Nan!")
            return [1.0e20, P[0]]

    return [bestF, bestP]


def minos(
    Pmins,
    minioffsets,
    chi2fn,
    dx,
    targetchi2,
    pos,
    minichi2,
    inlimit=lambda x: 1,
    passdata=None,
    verbose=False,
):  # Pmins, minioffsets are lists of starting conditions
    """This is the version to use, not the newer one."""
    merged_list = [chi2fn, inlimit]
    merged_list.extend([passdata])

    toohigh = 0.0
    toolow = 0.0
    toohighchi2 = 0.0
    toolowchi2 = 0.0
    print_verb(verbose, "targetchi2 ", targetchi2)

    [cur_chi, bestP] = minos_f(
        Pmins, minioffsets, merged_list, dx, pos, verbose=verbose
    )
    if cur_chi < minichi2:
        print_verb(verbose, "You didn't converge the starting fit!")
        print_verb(verbose, bestP)
        return [0.0, bestP]

    if cur_chi == 1000000:
        print_verb(verbose, "Minos Start Error!")
        sys.exit(1)

    dxscale = 1.0

    chi2list = []

    minostries = 0

    while (
        abs(cur_chi - targetchi2) > 0.000001
        and abs(toohigh - toolow) > 0.00001 * abs(toohigh)
        or toohigh == 0.0
        or toolow == 0.0
    ):

        dxscale *= 1.5
        print_verb(
            verbose,
            "[abs(cur_chi - targetchi2), max(abs(toohigh - toolow))] ",
            [abs(cur_chi - targetchi2), abs(toohigh - toolow)],
        )

        chi2list.append([abs(cur_chi - targetchi2), dx, cur_chi])
        chi2list.sort()  # low to high

        if cur_chi > targetchi2:
            print_verb(verbose, cur_chi, " > ", targetchi2)
            toohigh = dx

            if toohigh == 0.0 or toolow == 0.0:
                dx /= dxscale
            toohighchi2 = cur_chi
        else:
            print_verb(verbose, cur_chi, " <= ", targetchi2)
            toolow = dx

            if toohigh == 0.0 or toolow == 0.0:
                dx *= dxscale
            toolowchi2 = cur_chi
        if toohigh != 0.0 and toolow != 0.0:

            print_verb(
                verbose, "chi2list [abs(cur_chi - targetchi2), dx, cur_chi] ", chi2list
            )

            if minostries < 15:
                dx = linfit(
                    chi2list[0][1],
                    chi2list[0][2],
                    chi2list[1][1],
                    chi2list[1][2],
                    targetchi2,
                    0.9 * toolow + 0.1 * toohigh,
                    0.9 * toohigh + 0.1 * toolow,
                    verbose=verbose,
                )
                print_verb(verbose, "dxlinfit ", dx)
            else:
                dx = (toohigh + toolow) / 2.0
                print_verb(verbose, "dx the slow way ", dx)

        [cur_chi, bestP] = minos_f(
            Pmins, minioffsets, merged_list, dx, pos, verbose=verbose
        )
        if cur_chi < minichi2:
            print_verb(verbose, "You didn't converge the starting fit!")
            print_verb(verbose, bestP)
            return [0.0, bestP]
        print_verb(
            verbose, "toohigh, toolow, dx, cur_chi ", toohigh, toolow, dx, cur_chi
        )
        if np.isinf(dx):
            return [1.0e100, None]
        minostries += 1

    print_verb(verbose, "Finished getting dx")
    return [dx, bestP]


def better_minos(P0, Pstart, minioffset, merged_list, dx, target_chi2, verbose=False):
    print_verb(verbose, "target_chi2 ", target_chi2)

    newmerged_list = copy.deepcopy(merged_list)
    minos_params = [target_chi2, dx, P0, merged_list[0]]
    newmerged_list.append(minos_params)

    newmerged_list[0] = better_minos_chi2fn

    [P, F] = miniNM(Pstart, [1.0e-6, 0.0], newmerged_list, minioffset, verbose)

    return [
        np.dot(P[0] - P0, dx) / np.sqrt(dot(dx, dx)),
        P[0] - P0,
        merged_list[0](P[0], merged_list[2:]),
    ]


def better_minos_chi2fn(P, merged_list):
    minos_params = merged_list[-1]

    [target_chi2, dx, P0, chi2fn] = minos_params

    chi2 = chi2fn(P, merged_list[:-1])

    gradient = -np.dot(P - P0, dx) / np.dot(dx, dx)
    if abs(gradient) > 1.0e2:
        return 0.0

    # 1.e5 assures chi2 is positive
    return chi2 + smoothprior(chi2 - target_chi2) + gradient + 1.0e5


def improve(pf, merged_list):  # P is sorted lowest chi2 to highest
    [P, F] = pf

    # Got an error saying
    # TypeError: sum() takes no keyword arguments
    # M = sum(P[:-1], axis=0) / len(P[:-1])
    M = sum(P[:-1]) / len(P[:-1])

    W = P[-1]  # worst
    fW = F[-1]

    R = M + M - W  # ave + move away from worst
    E = R + (R - M)  # ave + 2 moves away from worst

    if merged_list[1](E):
        fE = f(E, merged_list)
        if fE < fW:
            P[-1] = E
            F[-1] = fE
            return [P, F]

    if merged_list[1](R):
        fR = f(R, merged_list)
        if fR < fW:
            P[-1] = R
            F[-1] = fR
            return [P, F]

    C1 = 0.5 * (M + W)  # ave + half move towards worst
    C2 = 0.5 * (M + R)  # ave + half move away from worst

    if merged_list[1](C1):
        fC1 = f(C1, merged_list)
        if fC1 < fW:
            P[-1] = C1
            F[-1] = fC1
            return [P, F]

    if merged_list[1](C2):
        fC2 = f(C2, merged_list)
        if fC2 < fW:
            P[-1] = C2
            F[-1] = fC2
            return [P, F]

    # Squish things towards best chi2.
    for i in range(1, len(P)):
        P[i] = 0.5 * (P[0] + P[i])
        F[i] = f(P[i], merged_list)

    return [P, F]


def get_start(P0, displ_list, merged_list, verbose=False):
    P = np.array([P0] * (len(displ_list) - displ_list.count(0.0) + 1), dtype=np.float64)

    F = [f(P[0], merged_list)]

    j = 1
    for i in range(len(displ_list)):
        if displ_list[i] != 0.0:

            P[j, i] += displ_list[i]

            if merged_list[1](P[j]) == 1:  # merged_list[1] is inlimit
                F.append(f(P[j], merged_list))
            else:
                print_verb(verbose, "Changing sign! ", displ_list[i])
                P[j, i] -= 2.0 * displ_list[i]

                if merged_list[1](P[j]) == 1:  # merged_list[1] is inlimit
                    print_verb(verbose, "Change worked!")
                    F.append(f(P[j], merged_list))
                else:
                    print_verb(verbose, "start out of range!")
                    print_verb(verbose, P[j])
                    return [P, F]
            j += 1

    [P, F] = listsort([P, F])

    return [P, F]


def miniNM(
    P0,
    e,
    merged_list,
    displ_list,
    maxruncount=15,
    negativewarning=True,
    maxiter=100000,
    verbose=False,
):
    runcount = 0

    print_verb(verbose, "maxruncount ", maxruncount)

    old_F = -1.0
    F = [-2.0]

    while runcount < maxruncount and old_F != F[0]:
        [P, F] = get_start(P0, displ_list, merged_list, verbose=verbose)
        if len(F) != len(P):  # Started against a limit
            print_verb(verbose, "Returning starting value!")
            return [
                np.array([P0], dtype=np.float64),
                np.array(
                    [f(np.array(P0, dtype=np.float64), merged_list)], dtype=np.float64
                ),
            ]
        old_F = F[0]

        k = 1

        noimprovement = 0
        noimprove_F = -1.0

        if runcount == 0:
            tmpe = e[0] / 10.0
            tmpe2 = e[1] / 10.0
        else:
            tmpe = e[0]  # Just checking previous result
            tmpe2 = e[1]

        while (
            F[-1] > F[0] + tmpe
            and k < maxiter
            and noimprovement < 200
            and max(abs((P[0] - P[-1]) / max(max(P[0]), 1.0e-10))) > tmpe2
        ) or k < 2:

            last_F = F[0]
            [P, F] = improve([P, F], merged_list)  # Run an iteration
            [P, F] = listsort([P, F])

            if last_F == F[0] or F[0] == old_F:
                noimprovement += 1
            else:
                noimprovement = 0
                tmpe = e[0] / 10.0  # If improvement, run extra
                tmpe2 = e[1] / 10.0

            print_verb(verbose, P[0], F[0], F[-1] - F[0], k, noimprovement)

            if F[0] < -1.0e-8 and negativewarning:
                print_verb(verbose, "F ", F)
                print_verb(verbose, "P ", P)
                print_verb(verbose, "Negative Chi2")
                sys.exit(1.0)

            k += 1

        print_verb(verbose, f"iter {k} F[0]={F[0]}")

        P0 = P[0].tolist()

        runcount += 1

    return [P, F]


def miniNM_new(
    ministart,
    miniscale,
    passdata,
    chi2fn=None,
    residfn=None,
    inlimit=lambda x: True,
    maxruncount=15,
    negativewarning=False,
    maxiter=10000,
    tolerance=[1.0e-8, 0],
    compute_Cmat=True,
    verbose=False,
):
    if chi2fn is None:

        def chi2fn(x, y):
            return (residfn(x, y) ** 2.0).sum()

    try:
        miniscale = miniscale.tolist()
    except:
        pass

    try:
        ministart = ministart.tolist()
    except:
        pass

    [P, F] = miniNM(
        ministart,
        tolerance,
        merged_list=[chi2fn, inlimit, passdata],
        displ_list=miniscale,
        maxruncount=maxruncount,
        negativewarning=negativewarning,
        maxiter=maxiter,
        verbose=verbose,
    )

    if compute_Cmat:
        [Wmat, NA, NA] = secderiv(
            P,
            [chi2fn, inlimit, passdata],
            miniscale,
            1.0e-1,
            verbose=verbose,
        )
    else:
        Wmat = []

    Cmat = None
    if len(Wmat) > 0:
        if np.linalg.det(Wmat) != 0:
            Cmat = np.linalg.inv(Wmat)

    return P[0], F[0], Cmat


def err_from_cov(matrix):
    errs = []
    for i in range(len(matrix)):
        errs.append(np.sqrt(matrix[i, i]))
    return errs


# Start L-M


def Jacobian(
    modelfn,
    unpad_offsetparams,
    merged_list,
    unpad_params,
    displ_list,
    params,
    datalen,
    use_dense_J,
    pool=None,
):

    if use_dense_J:
        J = np.zeros([datalen, len(unpad_params)], dtype=np.float64, order="F")
    else:
        J = np.zeros([datalen, len(unpad_params)], dtype=np.float64)  # , order = 'F')

    # J = lil_matrix((datalen, len(unpad_params)))

    if pool is None:
        base_mod_list = modelfn(
            get_pad_params(unpad_params, displ_list, params), merged_list
        )
    else:
        base_mod_list = modelfn(
            (get_pad_params(unpad_params, displ_list, params), merged_list)
        )

    if pool is None:
        for j in range(len(unpad_params)):
            dparams = copy.deepcopy(unpad_params)
            dparams[j] += unpad_offsetparams[j]
            J[:, j] = (
                modelfn(get_pad_params(dparams, displ_list, params), merged_list)
                - base_mod_list
            ) / unpad_offsetparams[j]
    else:
        arg_list = []
        for j in range(len(unpad_params)):
            dparams = copy.deepcopy(unpad_params)
            dparams[j] += unpad_offsetparams[j]
            arg_list.append((get_pad_params(dparams, displ_list, params), merged_list))

        Jtmp = pool.map(modelfn, arg_list)

        Jtmp = [
            (Jtmp[j] - base_mod_list) / unpad_offsetparams[j]
            for j in range(len(unpad_params))
        ]
        J = np.transpose(np.array(Jtmp))

    if not use_dense_J:
        J = lil_matrix(J)
        J = J.tocsr()

    return J


def get_pad_params(unpad_params, displ_list, params):
    pad_params = copy.deepcopy(params)

    # for i in range(len(displ_list)):
    #    if displ_list[i] == 0:
    #        pad_params = insert(pad_params, i, params[i])
    inds = np.where(displ_list != 0)
    pad_params[inds] = unpad_params
    return pad_params


def get_unpad_params(pad_params, displ_list):
    unpad_params = copy.deepcopy(pad_params)
    unpad_params = unpad_params.compress(displ_list != 0)
    return unpad_params


def chi2fromresid(resid, Wmat, verbose=False):
    if Wmat is None:
        chi2 = sum(resid**2.0)
    else:
        chi2 = np.dot(np.dot(resid, Wmat), resid)

    if np.isnan(chi2):
        print_verb(verbose, "nan found! ", resid, Wmat)
        return 1.0e100
    else:
        return chi2


def miniLM(
    params,
    orig_merged_list,
    displ_list,
    maxiter=150,
    maxlam=100000,
    Wmat=None,
    jacobian_name="Jacob.fits",
    return_wmat=False,
    use_dense_J=False,
    pool=None,
    save_jacobian=True,
    verbose=False,
):
    params = np.array(params, dtype=np.float64)
    displ_list = np.array(displ_list, dtype=np.float64)

    # fix_list -- 1 = fix

    merged_list = copy.deepcopy(orig_merged_list)
    modelfn = orig_merged_list[0]
    del merged_list[0]
    del merged_list[0]  # placeholder for inlimit

    lam = 1.0e-6
    lamscale = 2.0

    converged = 0

    if pool is None:
        curchi2 = modelfn(params, merged_list)
    else:
        curchi2 = modelfn((params, merged_list))

    unpad_offsetparams = (
        get_unpad_params(np.array(displ_list, dtype=np.float64), displ_list) * 1.0e-6
    )

    unpad_params = get_unpad_params(params, displ_list)
    print_verb(verbose, "unpad_params ", unpad_params)

    itercount = 0
    was_just_searching = 0
    while lam < maxlam and itercount < maxiter:
        itercount += 1

        print_verb(
            verbose,
            len(unpad_offsetparams),
            len(unpad_params),
            len(displ_list),
            len(params),
            len(curchi2),
        )
        if not was_just_searching:
            Jacob = Jacobian(
                modelfn,
                unpad_offsetparams,
                merged_list,
                unpad_params,
                displ_list,
                params,
                len(curchi2),
                use_dense_J,
                pool=pool,
            )
        # save_img(Jacob.todense(), "tmpjacob.fits")
        was_just_searching = 0

        print_verb(verbose, "Jacob.shape ", Jacob.shape)
        Jacobt = np.transpose(Jacob)

        print_verb(verbose, "Dot start ", time.asctime())
        if Wmat is None:
            JtJ = Jacobt.dot(Jacob)
            if not use_dense_J:
                JtJ = JtJ.todense()
        else:
            try:
                JtJ = Jacobt.dot(np.transpose(Jacobt.dot(Wmat)))
            except:
                print_verb(verbose, "Couldn't do dot product!")
                print_verb(verbose, "Sizes ", Jacobt.shape, Wmat.shape)
                sys.exit(1)
        print_verb(verbose, "Dot end ", time.asctime())

        JtJ_lam = copy.deepcopy(JtJ)

        for i in range(len(JtJ)):
            JtJ_lam[i, i] *= 1.0 + lam
        try:
            if Wmat is None:
                delta1 = -np.linalg.solve(JtJ_lam, Jacobt.dot(curchi2))
            else:
                delta1 = -np.linalg.solve(JtJ_lam, Jacobt.dot(np.dot(Wmat, curchi2)))
        except:

            print_verb(verbose, "Uninvertible Matrix!")
            if save_jacobian:
                # Jdense = Jacob.todense()
                # print Jdense.shape
                # for i in range(len(Jdense)):
                #    print i
                #    if all(Jdense[i] == 0):
                #        print "fdsakfdskjahfdkjs"
                if not use_dense_J:
                    save_img(Jacob.todense(), jacobian_name)
                else:
                    save_img(Jacob, jacobian_name)

            return [
                np.array(
                    [get_pad_params(unpad_params, displ_list, params)], dtype=np.float64
                ),
                np.array(
                    [chi2fromresid(curchi2, Wmat), -1],
                    dtype=np.float64,
                    verbose=verbose,
                ),
            ] + [Jacob.todense()] * return_wmat

        JtJ_lam2 = copy.deepcopy(JtJ)

        for i in range(len(JtJ)):
            JtJ_lam2[i, i] *= 1.0 + lam / lamscale
        if Wmat is None:
            delta2 = -np.linalg.solve(JtJ_lam2, Jacobt.dot(curchi2))
        else:
            delta2 = -np.linalg.solve(JtJ_lam2, Jacobt.dot(np.dot(Wmat, curchi2)))

        unpad_params1 = get_pad_params(unpad_params + delta1, displ_list, params)
        unpad_params2 = get_pad_params(unpad_params + delta2, displ_list, params)

        if pool is None:
            chi2_1 = modelfn(unpad_params1, merged_list)
            chi2_2 = modelfn(unpad_params2, merged_list)
        else:
            chi2_1 = modelfn((unpad_params1, merged_list))
            chi2_2 = modelfn((unpad_params2, merged_list))

        if chi2fromresid(chi2_2, Wmat, verbose=verbose) < chi2fromresid(
            curchi2, Wmat, verbose=verbose
        ):
            curchi2 = chi2_2
            unpad_params = unpad_params + delta2
            lam /= lamscale

        elif chi2fromresid(chi2_1, Wmat, verbose=verbose) < chi2fromresid(
            curchi2, Wmat, verbose=verbose
        ):
            curchi2 = chi2_1
            unpad_params = unpad_params + delta1
        else:

            itercount -= 1
            was_just_searching = 1

            while (
                chi2fromresid(chi2_1, Wmat, verbose=verbose)
                >= chi2fromresid(curchi2, Wmat, verbose=verbose)
                and lam < maxlam
            ) or np.isnan(chi2fromresid(chi2_1, Wmat, verbose=verbose)):

                print_verb(verbose, "Searching... ", lam)
                lam *= lamscale

                JtJ_lam = copy.deepcopy(JtJ)

                for i in range(len(JtJ)):
                    JtJ_lam[i, i] *= 1.0 + lam

                if Wmat is None:
                    delta1 = -np.linalg.solve(JtJ_lam, Jacobt.dot(curchi2))
                else:
                    delta1 = -np.linalg.solve(
                        JtJ_lam, Jacobt.dot(np.dot(Wmat, curchi2))
                    )

                unpad_params1 = get_pad_params(
                    unpad_params + delta1, displ_list, params
                )
                if pool is None:
                    chi2_1 = modelfn(unpad_params1, merged_list)
                else:
                    chi2_1 = modelfn((unpad_params1, merged_list))

        print_verb(
            verbose,
            "itercount, unpad_params, lam, curchi2 ",
            itercount,
            unpad_params,
            lam,
            chi2fromresid(curchi2, Wmat, verbose=verbose),
        )

    if save_jacobian:
        if not use_dense_J:
            save_img(Jacob.todense(), jacobian_name)
        else:
            save_img(Jacob, jacobian_name)

    return [
        np.array([get_pad_params(unpad_params, displ_list, params)], dtype=np.float64),
        np.array([chi2fromresid(curchi2, Wmat, verbose=verbose)], dtype=np.float64),
    ] + [JtJ] * return_wmat


def miniLM_new(
    ministart,
    miniscale,
    residfn,
    passdata,
    maxiter=150,
    maxlam=100000,
    Wmat=None,
    jacobian_name="Jacob.fits",
    use_dense_J=False,
    return_Cmat=True,
    pad_Cmat=False,
    pool=None,
    save_jacobian=False,
    verbose=False,
):
    [P, F, param_wmat] = miniLM(
        ministart,
        [residfn, None, passdata],
        miniscale,
        maxiter=maxiter,
        maxlam=maxlam,
        Wmat=Wmat,
        jacobian_name=jacobian_name,
        return_wmat=True,
        use_dense_J=use_dense_J,
        pool=pool,
        save_jacobian=save_jacobian,
        verbose=verbose,
    )

    if len(param_wmat) > 0 and return_Cmat:
        if np.linalg.det(param_wmat) != 0.0:
            Cmat = np.linalg.inv(param_wmat)
        else:
            print_verb(verbose, "Determinant is zero!")
            Cmat = None
    else:
        Cmat = None
    return P[0], F[0], Cmat


# End L-M


def miniGN(
    params,
    orig_merged_list,
    displ_list,
    Wmat,
    use_dense_J=False,
    pool=None,
    verbose=False,
):
    params = np.array(params, dtype=np.float64)
    displ_list = np.array(displ_list, dtype=np.float64)

    # fix_list -- 1 = fix

    merged_list = copy.deepcopy(orig_merged_list)
    modelfn = orig_merged_list[0]
    del merged_list[0]
    del merged_list[0]  # placeholder for inlimit

    lam = 1.0
    lamscale = 2.0

    maxiter = 150

    converged = 0

    curchi2 = modelfn(params, merged_list)

    unpad_offsetparams = (
        get_unpad_params(np.array(displ_list, dtype=np.float64), displ_list) * 1.0e-6
    )

    unpad_params = get_unpad_params(params, displ_list)
    print_verb(verbose, "unpad_params ", unpad_params)

    itercount = 0
    while itercount < maxiter:
        itercount += 1

        print_verb(
            verbose,
            len(unpad_offsetparams),
            len(unpad_params),
            len(displ_list),
            len(params),
            len(curchi2),
        )

        Jacob = Jacobian(
            modelfn,
            unpad_offsetparams,
            merged_list,
            unpad_params,
            displ_list,
            params,
            len(curchi2),
            use_dense_J,
            pool=pool,
        )

        print_verb(verbose, "Jacob.shape ", Jacob.shape)
        Jacobt = np.transpose(Jacob)
        print_verb(verbose, "Dot Start")
        JtJ = np.dot(Jacobt, Jacob)
        print_verb(verbose, "Dot Finish")

        chi2_1 = 1.0e10
        lam = 1.0e-10

        JtJ_lam = copy.deepcopy(JtJ)

        try:
            print_verb(verbose, "delta1")
            delta1 = -np.dot(np.linalg.inv(JtJ_lam), np.dot(Jacobt, curchi2))
        except:

            print_verb(verbose, "Uninvertible Matrix!")
            return [
                np.array(
                    [get_pad_params(unpad_params, displ_list, params)], dtype=np.float64
                ),
                np.array(
                    [chi2fromresid(curchi2, Wmat, verbose=verbose), -1],
                    dtype=np.float64,
                ),
            ]

        while sum(chi2_1**2.0) >= sum(curchi2**2.0):

            unpad_params1 = get_pad_params(
                unpad_params + delta1 * lam, displ_list, params
            )

            chi2_1 = modelfn(unpad_params1, merged_list)

            if sum(chi2_1**2.0) < sum(curchi2**2.0):
                curchi2 = chi2_1
                unpad_params = unpad_params + delta1 * lam
            else:
                lam /= lamscale
                print_verb(verbose, "lam ", lam)

        print_verb(
            verbose,
            "itercount, unpad_params, lam, curchi2 ",
            itercount,
            unpad_params,
            lam,
            sum(curchi2**2.0),
        )

    return [
        np.array([get_pad_params(unpad_params, displ_list, params)], dtype=np.float64),
        np.array([sum(curchi2**2.0)], dtype=np.float64),
    ]


# Fitting Tangent Parab doesn't always work:


"""

def parabchi2(P, merged_list):
    chi2list = parabmodel(P, merged_list)
    chi2 = merged_list[2]

    return dot(chi2list - chi2, chi2list - chi2)


def inlimit_parabchi2(x):

    
    if abs(x[-1] - bestSNF) > 0.1:
        return 0

    #nparams =  int(round(   (-3 + sqrt(1 + 8*len(x)))/2   )) # not so pretty



    fitted_cov = params_to_matrix(x)
    fitted_cov_unpad = unpadm(fitted_cov, fixlist)
    

    for i in range(len(fitted_cov_unpad)):
        if fitted_cov_unpad[i, i] <= 0.:
            #print "inlimit1"
            return 0


    for i in range(len(fitted_cov_unpad)):
        for j in range(i + 1, len(fitted_cov_unpad)):
            if abs(fitted_cov_unpad[i, j]) >= 0.8*sqrt(fitted_cov_unpad[i, i]*fitted_cov_unpad[j, j]): #no correlations larger than 0.8
                #print "inlimit2 ", i, j

                return 0


    try:
        fitted_W = linalg.inv(fitted_cov_unpad) #Must be invertible
    except:
        #print "inlimit3"

        return 0
    return 1


def unpadm(padmatrix, fixlist):
    # unpadds matrix to get rid of fixed parameters

    #1 = fixed

    returnmatrix = copy.deepcopy(padmatrix)

    tmprange = range(len(fixlist))
    tmprange.reverse()
    
    for i in tmprange:
        if fixlist[i] == 1:
            returnmatrix = delete(returnmatrix, i, 0)
            returnmatrix = delete(returnmatrix, i, 1)
                        
    return returnmatrix


def padm(unpadmatrix, fixlist):
    returnmatrix = copy.deepcopy(unpadmatrix)

    tmprange = range(len(fixlist))
    tmprange.reverse()
    
    for i in tmprange:
        if fixlist[i] == 1:
            returnmatrix = insert(returnmatrix, i, zeros(returnmatrix.shape[0]), 0)
            returnmatrix = insert(returnmatrix, i, zeros(returnmatrix.shape[0]), 1)
                        
    return returnmatrix
    


def fit_covmatrix(newmerged_list, bestSNP, bestSNF, Wderiv, offdiag_scale, verbose=False):

    nparams = newmerged_list[2]

    
    covparams = zeros(nparams*(nparams + 1)/2, dtype = float64)

    

    Covderiv = linalg.inv(Wderiv)

    for i in range(len(Covderiv)):
        for j in range(len(Covderiv)):
            if i != j:
                maxcov = 0.75*sqrt(abs(Covderiv[i, i]*Covderiv[j, j]))
                if abs(Covderiv[i, j]) > maxcov:
                    Covderiv[i, j] = sign(Covderiv[i, j])*maxcov
    
    Covderivpad = padm(Covderiv, fixlist)
    count = 0

    covoffset = []
    for i in range(nparams):
        for j in range(i, nparams):
            
            #diagonal

            if Covderivpad[i, j] == 0:
                covparams[count] = 0.
                covoffset.append(0.)

            else:
                if i == j:
                    covparams[count] = abs(Covderivpad[i, i])
                    covoffset.append(covparams[count]/10.) # guess in the direction of larger
                else:
                    covparams[count] = Covderivpad[i, j]*offdiag_scale
                    covoffset.append(-covparams[count]/10.) # guess in the direction of smaller


            
            count += 1


    print_verb(verbose, covparams)


    print_verb(verbose, len(covoffset))
    
    ministart = bestSNP.tolist() + covparams.tolist() + [bestSNF]
    minioffset = [0.0]*(nparams) + covoffset + [0.0]


    print_verb(verbose, "ministart ", ministart)
    print_verb(verbose, "minioffset ", minioffset)

    [P, F] = miniNM(ministart, [1.e-20, 1.e-10], newmerged_list, minioffset, 0, verbose=verbose)

    oldF = F[0]*2.

    runs = 0
    while F[0] < 0.99*oldF and runs < 10 - 7*(F[0] < 1.): # stop early if F < 1
        print_verb(verbose, "runs ", runs)
        oldF = F[0]


        P[0][nparams:-1] = clip_params(P[0], verbose=verbose)
        minioffsetmatrix = params_to_matrix(P[0])
        
        
        for i in range(nparams):
            for j in range(nparams):
                if minioffsetmatrix[i, j] != 0.:
                    if i == j:
                        minioffsetmatrix[i, j] = minioffsetmatrix[i, j]*0.1
                    else:
                        minioffsetmatrix[i, j] = -minioffsetmatrix[i, j]*0.1
        minioffset = [0.]*nparams + matrix_to_params(minioffsetmatrix).tolist() + [0.]

        print_verb(verbose, "minioffset ", minioffset)
        print_verb(verbose, "ministart ", P[0].tolist())
        [P, F] = miniNM(P[0].tolist(), [1.e-20, 1.e-10], newmerged_list, minioffset, 0, verbose=verbose)
        runs += 1
    # this is tens of thousands of iterations-- don't use verbose

    print_verb(verbose, "P[0] and F[0]")
    print_verb(verbose, P[0])
    print_verb(verbose, F[0])


    fitted_cov = params_to_matrix(P[0])

    print_verb(verbose, "fitted_cov")
    print_verb(verbose, fitted_cov)
    print_verb(verbose, "err_from_cov")
    print_verb(verbose, err_from_cov(fitted_cov))

    print_verb(verbose, "fitted_W")

    fitted_cov_unpad = unpadm(fitted_cov, fixlist)
    fitted_W = linalg.inv(fitted_cov_unpad)
    print_verb(verbose, fitted_W)



    fitted_cor = zeros([len(fitted_cov_unpad), len(fitted_cov_unpad)], dtype=float64)
    for i in range(len(fitted_cor)):
        for j in range(len(fitted_cor)):
            fitted_cor[i, j] = fitted_cov_unpad[i, j]/sqrt(fitted_cov_unpad[i, i]*fitted_cov_unpad[j, j])

    print_verb(verbose, "fitted_cor")
    print_verb(verbose, fitted_cor)




    return [fitted_cov, fitted_W, fitted_cor, P[0], F[0]]

def clip_params(P, verbose=False):
    thematrix = params_to_matrix(P)
    for i in range(len(thematrix)):
        for j in range(len(thematrix)):
            if i != j:
                if thematrix[i, j] != 0.:
                    corr = thematrix[i, j]/sqrt(thematrix[i,i]*thematrix[j,j])
                    if abs(corr) >= 0.8: # 0.79 to allow for rounding errors
                        print_verb(verbose, "Clipping ", i, j, corr)
                        thematrix[i, j] = sign(thematrix[i, j])*0.79*sqrt(thematrix[i, i]*thematrix[j, j])
    return matrix_to_params(thematrix)

def params_to_matrix(P):
    count = nparams
    
    thematrix = zeros([nparams, nparams], dtype=float64)

    for i in range(nparams):
        for j in range(i, nparams):
            thematrix[i, j] = P[count]
            thematrix[j, i] = P[count]
            count += 1
    return thematrix

def matrix_to_params(thematrix):
    P = []
    for i in range(nparams):
        for j in range(i, nparams):
            P.append(thematrix[i, j])
    return array(P, dtype=float64)


def parabmodel(P, merged_list, verbose=False):
    #newmerged_list = [parabchi2, inlimit_parabchi2, nparams, Pcollection, Fcollection]

    
    nparams = merged_list[0]
    
    centroid = P[:nparams]
    
    fitted_cov = params_to_matrix(P)
    fitted_cov_unpad = unpadm(fitted_cov, fixlist)

    try:
        fitted_W_unpad = linalg.inv(fitted_cov_unpad)
    except:
        print_verb(verbose, "Singular; error in inlimit?", fitted_cov_unpad)
        return 1.e20
    fitted_W = padm(fitted_W_unpad, fixlist)

    offset = P[nparams + nparams*(nparams + 1)/2]

    chi2list = []

    for i in range(len(merged_list[1])):
        P = merged_list[1][i] # the evaluation point

        if len(P) == len(centroid) + 1:
            P = P[:-1] # remove the last element


        res = P - centroid


        chi2list.append(dot(res, dot(fitted_W, res)) + offset)

    return array(chi2list, dtype = float64)

"""


# End


"""
def inlimit(x):
    return 1

def chi2fn(P, merged_list):
    chi2 = P[0]**2. + (0.9*P[0] - P[1])**2. + 3.*P[2]**2. + 0.2*P[3]**2. - 0.1*(P[0] - P[2])**2

    return chi2

merged_list=[chi2fn,inlimit]
ministart = [1., 2.,3.,1.]
minioffset = [1., 1., 1., 1.]

[P, F] = miniNM(ministart, [1.e-9, 0.], merged_list, minioffset, 0, verbose=verbose)

print_verb(verbose, P)
print_verb(verbose, F)

[Wmat, NA, NA] = secderiv(P, merged_list, minioffset, 1.e-2, verbose=verbose)
print_verb(verbose, linalg.inv(Wmat))
print_verb(verbose, err_from_cov(linalg.inv(Wmat)))

#print better_minos(P[0], minioffset, merged_list, array([1., 0., 0., 0.]), F[0] + 1. , 0)
#print better_minos(P[0], minioffset, merged_list, array([-1., 0., 0., 0.]), F[0] + 1. , 0)

#print better_minos(P[0], minioffset, merged_list, array([0., 1., 0., 0.]), F[0] + 1. , 0)

minos_minioffset = copy.deepcopy(minioffset)

minos_minioffset[2] = 0.0
initialdx = 1

dx = minos([P[0].tolist()], [minos_minioffset], merged_list, initialdx, F[0] + 1., 2, -1) # 2nd parameter
dx = minos([P[0].tolist()], [minos_minioffset], merged_list, initialdx, F[0] + 1., array([0., 0., 1., 0.]), -1) # 2nd parameter

print_verb(verbose, 'dx')
print_verb(verbose, dx)
print_verb(verbose, P,F)
"""

"""

#test:
def chi2fn(x, merged_list):
    return x[0]**2. + 0.2*x[1]**2. + x[1]*x[0]*0.1 + x[1]**3. - x[1]*x[0]**3.

Hessian([1, 2], [0.00001, 0.00001], [chi2fn])
"""
