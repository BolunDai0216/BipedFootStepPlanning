import json
from pdb import set_trace

import munch
import numpy as np
from scipy.linalg import block_diag

from biped_params import (
    rectangular_foot_CoP,
    support_foot_limit,
    walking_straight,
    init_double_support_CoP,
)


def angle2RotMat(angle):
    """
    INPUTS
    angle (float): rotation angle in radians;

    OUTPUTS
    RotMat: size [2, 2] rotation matrix;
    """
    RotMat = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

    return RotMat


def get_Up(N=16, dt=0.1):
    """
    INPUTS
    N (int): is the length of the preview horizon;
    dt (float): time step size;

    OUTPUTS
    Up: size [N, N] matrix;
    """
    Up = np.tril(np.ones((N, N)), 0) * (1 / 6)

    for i in range(N):
        Up += np.diag(np.ones(N - i) * i, k=-i) / 2
        Up += np.diag(np.ones(N - i) * (i ** 2), k=-i) / 2

    Up *= np.power(dt, 3)

    return Up


def get_Uv(N=16, dt=0.1):
    """
    INPUTS
    N (int): is the length of the preview horizon;
    dt (float): time step size;

    OUTPUTS
    Uv: size [N, N] matrix;
    """
    Uv = np.tril(np.ones((N, N)), 0) * 0.5

    for i in range(N):
        Uv += np.diag(np.ones(N - i) * i, k=-i)

    Uv *= dt ** 2

    return Uv


def get_Ua(N=16, dt=0.1):
    """
    INPUTS
    N (int): is the length of the preview horizon;
    dt (float): time step size;

    OUTPUTS
    Uz: size [N, N] matrix;
    """
    Ua = np.tril(np.ones((N, N)), 0) * dt

    return Ua


def get_Uz(N=16, dt=0.1, h=1.0, g=9.81):
    """
    INPUTS
    N (int): is the length of the preview horizon;
    dt (float): time step size;
    h (float): CoM height;
    g (float): gravitational acceleration;

    OUTPUTS
    Uz: size [N, N] matrix;

    The calculations can be proven by:
    "Trajectory Free Linear Model Predictive Control
    for Stable Walking in the Presence of Strong Perturbations"
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4115592
    Eq.(8)
    """
    Up = get_Up(N=N, dt=dt)
    Ua = get_Ua(N=N, dt=dt)
    Uz = Up - (h / g) * Ua

    return Uz


def get_Sp(N=16, dt=0.1):
    """
    INPUTS
    N (int): is the length of the preview horizon;
    dt (float): time step size;

    OUTPUTS
    Sp: size [N, 3] matrix;
    """
    Sp1 = np.ones((N, 1))
    Sp2 = np.arange(1, N + 1).reshape(N, 1) * dt
    Sp3 = (np.arange(1, N + 1) ** 2).reshape(N, 1) * (dt ** 2) / 2
    Sp = np.concatenate((Sp1, Sp2, Sp3), axis=1)

    return Sp


def get_Sv(N=16, dt=0.1):
    """
    INPUTS
    N (int): is the length of the preview horizon;
    dt (float): time step size;

    OUTPUTS
    Sv: size [N, 3] matrix;
    """
    Sp = get_Sp(N=N, dt=dt)
    Sv1 = np.zeros((N, 1))
    Sv = np.concatenate((Sv1, Sp[:, :2]), axis=1)

    return Sv


def get_Sa(N=16, dt=0.1):
    """
    INPUTS
    N (int): is the length of the preview horizon;
    dt (float): time step size;

    OUTPUTS
    Sa: size [N, 3] matrix;
    """
    Sp = get_Sp(N=N, dt=dt)
    Sa12 = np.zeros((N, 2))
    Sa = np.concatenate((Sa12, Sp[:, :1]), axis=1)

    return Sa


def get_Sz(N=16, dt=0.1, h=1.0, g=9.81):
    """
    INPUTS
    N (int): is the length of the preview horizon;
    dt (float): time step size;
    h (float): CoM height;
    g (float): gravitational acceleration;

    OUTPUTS
    Sz: size [N, 3] matrix;

    The calculations can be proven by:;
    "Trajectory Free Linear Model Predictive Control
    for Stable Walking in the Presence of Strong Perturbations"
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4115592
    Eq.(8)
    """
    Sp = get_Sp(N=N, dt=dt)
    Sa = get_Sa(N=N, dt=dt)
    Sz = Sp - (h / g) * Sa

    return Sz


def stepsInCurrentStepVec(m, N=16, tPf=8):
    """
    INPUTS
    m (int): remaining time steps in current foot step;
    N (int): is the length of the preview horizon;
    tPf (int): time steps per foot step;

    OUTPUTS
    CurrentStepVec: size [N, 1] vector;;
    """
    CurrentStepVec = np.vstack((np.ones((m, 1)), np.zeros((N - m, 1))))

    return CurrentStepVec


def stepsInFutureStepsMat(m, N=16, tPf=8):
    """
    INPUTS
    m (int): remaining time steps in current foot step;
    N (int): is the length of the preview horizon;
    tPf (int): time steps per foot step;

    OUTPUTS
    CurrentStepVec: size [N, l] Matrix, where l is the number
    of remaining foots steps contained in the preview horizon;
    """

    l = int(np.ceil((N - m) / tPf))
    ones = np.ones((tPf, 1))
    zeros = np.zeros((m, l))

    for i in range(l):
        if i == 0:
            FutureStepsMat = block_diag(ones)
        else:
            FutureStepsMat = block_diag(FutureStepsMat, ones)

    FutureStepsMat = np.vstack((zeros, FutureStepsMat))
    FutureStepsMat = FutureStepsMat[:N, :]

    return FutureStepsMat


def CoP_constraints(
    m,
    foot_angles,
    support_foot_pos,
    stateX,
    stateY,
    N=16,
    dt=0.1,
    h=1.0,
    g=9.81,
    tPf=8,
):
    """
    INPUTS
    m (int): remaining time steps in current foot step;
    foot_angles ([N, 1] vector): containing the orientations in radians
    of the foot steps at each time step;
    support_foot_pos ([2, 1] vec): current support foot position;
    stateX ([3, 1] matrix): position, velocity, acceleration of CoM along x-axis;
    stateY ([3, 1] matrix): position, velocity, acceleration of CoM along y-axis;
    N (int): is the length of the preview horizon;
    dt (float): time step size;
    h (float): CoM height;
    g (float): gravitational acceleration;
    tPf (int): time steps per foot step;

    Also calls a function that load the data for the foot edge
    normal vectors and edge to center distances;

    OUTPUTS
    leftHandSide: size [ef*N, 2N+2l] Matrix, where l is the number
    of remaining foots steps contained in the preview horizon and
    ef is the number of edges in the robot foot, e being the
    number of the edges of the foot, using a rectangular foot, ef=4;
    rightHandSide: size [ef*N, 1] Matrix;
    """
    Uz = get_Uz(N=N)
    FutureStepsMat = stepsInFutureStepsMat(m, N=N)
    middleMat_diag = np.hstack((Uz, -FutureStepsMat))
    middleMat = block_diag(middleMat_diag, middleMat_diag)

    CurrentStepVec = stepsInCurrentStepVec(m, N=N)
    Sz = get_Sz(N=N)
    rightVecX = CurrentStepVec * support_foot_pos[0] - Sz @ stateX
    rightVecY = CurrentStepVec * support_foot_pos[1] - Sz @ stateY
    rightVex = np.vstack((rightVecX, rightVecY))

    d, b = rectangular_foot_CoP()
    for i in range(N):
        RotMat = angle2RotMat(foot_angles[i])
        # (Rd^T)^T = dR^T
        dRot = d @ RotMat.T
        if i == 0:
            DMatX = block_diag(dRot[:, :1])
            DMatY = block_diag(dRot[:, 1:])
            bVec = b
        else:
            DMatX = block_diag(DMatX, dRot[:, :1])
            DMatY = block_diag(DMatY, dRot[:, 1:])
            bVec = np.vstack((bVec, b))

    DMat = np.hstack((DMatX, DMatY))

    leftHandSide = DMat @ middleMat
    rightHandSide = bVec + DMat @ rightVex

    return leftHandSide, rightHandSide


def CoP_constraints_ds(
    m,
    foot_angles,
    next_support_foot_pos,
    stateX,
    stateY,
    N=16,
    dt=0.1,
    h=1.0,
    g=9.81,
    tPf=8,
):
    """
    INPUTS
    m (int): remaining time steps in current foot step;
    foot_angles ([N, 1] vector): containing the orientations in radians
    of the foot steps at each time step;
    next_support_foot_pos ([2, 1] vec): next support foot position;
    stateX ([3, 1] matrix): position, velocity, acceleration of CoM along x-axis;
    stateY ([3, 1] matrix): position, velocity, acceleration of CoM along y-axis;
    N (int): is the length of the preview horizon;
    dt (float): time step size;
    h (float): CoM height;
    g (float): gravitational acceleration;
    tPf (int): time steps per foot step;

    Also calls a function that load the data for the foot edge
    normal vectors and edge to center distances;

    OUTPUTS
    leftHandSide: size [ef*N, 2N+2l] Matrix, where l is the number
    of remaining foots steps contained in the preview horizon and
    ef is the number of edges in the robot foot, e being the
    number of the edges of the foot, using a rectangular foot, ef=4;
    rightHandSide: size [ef*N, 1] Matrix;
    """
    Uz = get_Uz(N=N)
    FutureStepsMat = stepsInFutureStepsMat(m, N=N)
    middleMat_diag = np.hstack((Uz, -FutureStepsMat[:, 1:]))
    middleMat = block_diag(middleMat_diag, middleMat_diag)

    Sz = get_Sz(N=N)
    rightVecX = FutureStepsMat[:, :1] * next_support_foot_pos[0] - Sz @ stateX
    rightVecY = FutureStepsMat[:, :1] * next_support_foot_pos[1] - Sz @ stateY
    rightVex = np.vstack((rightVecX, rightVecY))

    # set_trace()

    for i in range(N):
        RotMat = angle2RotMat(foot_angles[i])
        if i < m:
            d, b = init_double_support_CoP()
        else:
            d, b = rectangular_foot_CoP()
        # (Rd^T)^T = dR^T
        dRot = d @ RotMat.T

        if i == 0:
            DMatX = block_diag(dRot[:, :1])
            DMatY = block_diag(dRot[:, 1:])
            bVec = b
        else:
            DMatX = block_diag(DMatX, dRot[:, :1])
            DMatY = block_diag(DMatY, dRot[:, 1:])
            bVec = np.vstack((bVec, b))

    DMat = np.hstack((DMatX, DMatY))

    leftHandSide = DMat @ middleMat
    rightHandSide = bVec + DMat @ rightVex

    return leftHandSide, rightHandSide


def support_foot_constraints(
    m, which_current_support, support_foot_pos, next_foot_angles, N=16, tPf=8
):
    """
    INPUTS
    m (int): remaining time steps in current foot step;
    which_current_support (int): current support foot being 'left' 0 or 'right' 1;
    support_foot_pos ([2, 1] vector): positions of the foot for the current support foot;
    next_foot_angles ([l, 1] vector): containing the orientations in radians of next foot steps;
    N (int): is the length of the preview horizon;
    tPf (int): time steps per foot step;

    OUTPUTS
    leftHandSide: size [es*l, 2N+2l] Matrix, where l is the number
    of remaining foots steps contained in the preview horizon and
    es being the number of the edges in the constraint region, using
    the region outlined in the 'Walking without thinking about it'
    paper, es=5;
    rightHandSide: size [es*l, 1] Matrix;
    """
    l = int(np.ceil((N - m) / tPf))
    foot_diff_mat = np.diag(np.ones(l)) + np.diag(np.ones(l - 1), k=-1) * (-1)
    M_quarter = np.hstack((np.zeros((l, N)), foot_diff_mat))
    M = block_diag(M_quarter, M_quarter)

    current_foot_vecX = np.zeros((l, 1))
    current_foot_vecX[0] = support_foot_pos[0]
    current_foot_vecY = np.zeros((l, 1))
    current_foot_vecY[0] = support_foot_pos[1]
    current_foot_vec = np.vstack((current_foot_vecX, current_foot_vecY))

    foots = ["left", "right"]
    next_foot = which_current_support

    for i in range(l):
        # which foot will be landed next
        next_foot = 1 - next_foot
        support_norm_vecs, support_dis = support_foot_limit(foot=foots[next_foot])
        RotMat = angle2RotMat(next_foot_angles[i])
        # (Rd^T)^T = dR^T
        normVecsRot = support_norm_vecs @ RotMat.T
        if i == 0:
            normVecX = normVecsRot[:, :1]
            normVecY = normVecsRot[:, 1:]
            disVec = support_dis
        else:
            normVecX = block_diag(normVecX, normVecsRot[:, :1])
            normVecY = block_diag(normVecY, normVecsRot[:, 1:])
            disVec = np.vstack((disVec, support_dis))

    normVec = np.hstack((normVecX, normVecY))

    leftHandSide = normVec @ M
    rightHandSide = disVec + normVec @ current_foot_vec

    return leftHandSide, rightHandSide


def support_foot_constraints_ds(
    m, which_next_support, next_support_foot_pos, next_foot_angles, N=16, tPf=8
):
    """
    INPUTS
    m (int): remaining time steps in current foot step;
    which_next_support (int): next support foot being 'left' 0 or 'right' 1;
    next_support_foot_pos ([2, 1] vector): positions of the foot for the next support foot;
    next_foot_angles ([l, 1] vector): containing the orientations in radians of next foot steps;
    N (int): is the length of the preview horizon;
    tPf (int): time steps per foot step;

    OUTPUTS
    leftHandSide: size [es*(l-1), 2N+2l-2] Matrix, where l is the number
    of remaining foots steps contained in the preview horizon and
    es being the number of the edges in the constraint region, using
    the region outlined in the 'Walking without thinking about it'
    paper, es=5;
    rightHandSide: size [es*(l-1), 1] Matrix;
    """
    l = int(np.ceil((N - m) / tPf))
    foot_diff_mat = np.diag(np.ones(l - 1)) + np.diag(np.ones(l - 2), k=-1) * (-1)
    M_quarter = np.hstack((np.zeros((l - 1, N)), foot_diff_mat))
    M = block_diag(M_quarter, M_quarter)

    current_foot_vecX = np.zeros((l - 1, 1))
    current_foot_vecX[0] = next_support_foot_pos[0]
    current_foot_vecY = np.zeros((l - 1, 1))
    current_foot_vecY[0] = next_support_foot_pos[1]
    current_foot_vec = np.vstack((current_foot_vecX, current_foot_vecY))

    foots = ["left", "right"]
    next_foot = which_next_support

    for i in range(l - 1):
        # which foot will be landed next
        next_foot = 1 - next_foot
        support_norm_vecs, support_dis = support_foot_limit(foot=foots[next_foot])
        RotMat = angle2RotMat(next_foot_angles[i])
        # (Rd^T)^T = dR^T
        normVecsRot = support_norm_vecs @ RotMat.T
        if i == 0:
            normVecX = normVecsRot[:, :1]
            normVecY = normVecsRot[:, 1:]
            disVec = support_dis
        else:
            normVecX = block_diag(normVecX, normVecsRot[:, :1])
            normVecY = block_diag(normVecY, normVecsRot[:, 1:])
            disVec = np.vstack((disVec, support_dis))

    normVec = np.hstack((normVecX, normVecY))

    leftHandSide = normVec @ M
    rightHandSide = disVec + normVec @ current_foot_vec

    return leftHandSide, rightHandSide


def swing_foot_constraints(m, swing_foot_pos, N=16, dt=0.1, tPf=8):
    """
    INPUTS
    m (int): remaining time steps in current foot step;
    swing_foot_pos ([2, 1] vec): position of the swing foot;
    N (int): is the length of the preview horizon;
    dt (float): time step size;
    tPf (int): time steps per foot step;

    OUTPUTS
    leftHandSide: size [2, 2N+2l] Matrix, where l is the number
    of remaining foots steps contained in the preview horizon;
    rightHandSide: size [2, 1] Matrix;
    """
    # remaining time in current foot step
    tr = (tPf - m) * dt
    remainSteps = int(np.ceil((N - m) / tPf))

    # vectors representing sagittal and frontal plane direction
    Nmat, vmax = walking_straight()

    zeroN = np.zeros((2, N))
    zerolm1 = np.zeros((2, remainSteps - 1))

    leftHandSide = np.concatenate(
        (zeroN, Nmat[:, :1], zerolm1, zeroN, Nmat[:, 1:], zerolm1), axis=1
    )
    rightHandSide = tr * vmax + Nmat @ swing_foot_pos.reshape(-1, 1)

    return leftHandSide, rightHandSide


def get_cost(
    m,
    stateX,
    stateY,
    vx_ref,
    vy_ref,
    support_foot_pos,
    alpha=1e-6,
    beta=10,
    gamma=1e-5,
    N=16,
    dt=0.1,
    h=1.0,
    g=9.81,
    tPf=8,
):
    """
    INPUTS
    m (int): remaining time steps in current foot step;
    stateX ([3, 1] matrix): position, velocity, acceleration of CoM along x-axis;
    stateY ([3, 1] matrix): position, velocity, acceleration of CoM along y-axis;
    vx_ref (float): reference velocity along X axis;
    vy_ref (float):reference velocity along Y axis;
    support_foot_pos ([2, 1] vector): (x, y) position of the current support foot;
    alpha (float): weight on jerk cost;
    beta (float): weight of velocity cost;
    gamma (float): weight of CoP cost;
    N (int): is the length of the preview horizon;
    dt (float): time step size;
    h (float): CoM height;
    g (float): gravitational acceleration;
    tPf (int): time steps per foot step;

    OUTPUTS
    Qk ([2N+2l, 2N+2l] matrix): Q matrix in standard quadratic programming;
    pk ([2N+2l, 1] matrix): p matrix in standard quadratic programming;
    """
    I = np.identity(N)
    Uz = get_Uz(N=N, dt=dt, h=h, g=g)
    Uv = get_Uv(N=N, dt=dt)
    FutureStepsMat = stepsInFutureStepsMat(m, N=N, tPf=tPf)

    _Qk_upper_left = alpha * I + beta * Uv.T @ Uv + gamma * Uz.T @ Uz
    _Qk_upper_right = -gamma * Uz.T @ FutureStepsMat
    _Qk_lower_left = -gamma * FutureStepsMat.T @ Uz
    _Qk_lower_right = gamma * FutureStepsMat.T @ FutureStepsMat

    _Qk = np.bmat(
        [[_Qk_upper_left, _Qk_upper_right], [_Qk_lower_left, _Qk_lower_right]]
    )
    Qk = block_diag(_Qk, _Qk)

    Sv = get_Sv(N=N, dt=dt)
    Sz = get_Sz(N=N, dt=dt, h=h, g=g)
    dx_ref = np.ones((N, 1)) * vx_ref
    dy_ref = np.ones((N, 1)) * vy_ref
    CurrentStepVec = stepsInCurrentStepVec(m, N=N, tPf=tPf)

    _pk1 = beta * Uv.T @ (Sv @ stateX - dx_ref) + gamma * Uz.T @ (
        Sz @ stateX - CurrentStepVec * support_foot_pos[0]
    )
    _pk2 = (
        -gamma * FutureStepsMat.T @ (Sz @ stateX - CurrentStepVec * support_foot_pos[0])
    )
    _pk3 = beta * Uv.T @ (Sv @ stateY - dy_ref) + gamma * Uz.T @ (
        Sz @ stateY - CurrentStepVec * support_foot_pos[1]
    )
    _pk4 = (
        -gamma * FutureStepsMat.T @ (Sz @ stateY - CurrentStepVec * support_foot_pos[1])
    )

    pk = np.bmat([[_pk1], [_pk2], [_pk3], [_pk4]])

    return Qk, pk


def get_cost_ds(
    m,
    stateX,
    stateY,
    vx_ref,
    vy_ref,
    next_support_foot_pos,
    alpha=1e-6,
    beta=1,
    gamma=1e-6,
    N=16,
    dt=0.1,
    h=1.0,
    g=9.81,
    tPf=8,
):
    """
    INPUTS
    m (int): remaining time steps in current foot step;
    stateX ([3, 1] matrix): position, velocity, acceleration of CoM along x-axis;
    stateY ([3, 1] matrix): position, velocity, acceleration of CoM along y-axis;
    vx_ref (float): reference velocity along X axis;
    vy_ref (float):reference velocity along Y axis;
    next_support_foot_pos ([2, 1] vector): (x, y) position of the next support foot;
    alpha (float): weight on jerk cost;
    beta (float): weight of velocity cost;
    gamma (float): weight of CoP cost;
    N (int): is the length of the preview horizon;
    dt (float): time step size;
    h (float): CoM height;
    g (float): gravitational acceleration;
    tPf (int): time steps per foot step;

    OUTPUTS
    Qk ([2N+2l, 2N+2l] matrix): Q matrix in standard quadratic programming;
    pk ([2N+2l, 1] matrix): p matrix in standard quadratic programming;
    """
    I = np.identity(N)
    Uz = get_Uz(N=N, dt=dt, h=h, g=g)
    Uv = get_Uv(N=N, dt=dt)
    FutureStepsMat = stepsInFutureStepsMat(m, N=N, tPf=tPf)

    _Qk_upper_left = alpha * I + beta * Uv.T @ Uv + gamma * Uz.T @ Uz
    _Qk_upper_right = -gamma * Uz.T @ FutureStepsMat[:, 1:]
    _Qk_lower_left = -gamma * FutureStepsMat[:, 1:].T @ Uz
    _Qk_lower_right = gamma * FutureStepsMat[:, 1:].T @ FutureStepsMat[:, 1:]

    _Qk = np.bmat(
        [[_Qk_upper_left, _Qk_upper_right], [_Qk_lower_left, _Qk_lower_right]]
    )
    Qk = block_diag(_Qk, _Qk)

    Sv = get_Sv(N=N, dt=dt)
    Sz = get_Sz(N=N, dt=dt, h=h, g=g)

    dx_ref = np.vstack((np.zeros((m, 1)), np.ones((N - m, 1)) * vx_ref))
    dy_ref = np.vstack((np.zeros((m, 1)), np.ones((N - m, 1)) * vy_ref))

    _pk1 = beta * Uv.T @ (Sv @ stateX - dx_ref) + gamma * Uz.T @ (
        Sz @ stateX - FutureStepsMat[:, :1] * next_support_foot_pos[0]
    )
    _pk2 = (
        -gamma
        * FutureStepsMat[:, 1:].T
        @ (Sz @ stateX - FutureStepsMat[:, :1] * next_support_foot_pos[0])
    )
    _pk3 = beta * Uv.T @ (Sv @ stateY - dy_ref) + gamma * Uz.T @ (
        Sz @ stateY - FutureStepsMat[:, :1] * next_support_foot_pos[1]
    )
    _pk4 = (
        -gamma
        * FutureStepsMat[:, 1:].T
        @ (Sz @ stateY - FutureStepsMat[:, :1] * next_support_foot_pos[1])
    )
    pk = np.bmat([[_pk1], [_pk2], [_pk3], [_pk4]])

    return Qk, pk


def get_constraints(
    m,
    foot_angles,
    which_current_support,
    next_foot_angles,
    support_foot_pos,
    swing_foot_pos,
    stateX,
    stateY,
    N=16,
    dt=0.1,
    h=1.0,
    g=9.81,
    tPf=8,
):
    """
    INPUTS
    m (int): remaining time steps in current foot step;
    foot_angles ([N, 1] vector): containing the orientations in radians
    of the foot steps at each time step;
    which_current_support (int): current support foot being 'left' 0 or 'right' 1;
    next_foot_angles ([l, 1] vector): containing the orientations in radians of next foot steps;
    support_foot_pos ([2, 1] vec): current support foot step position;
    swing_foot_pos ([2, 1] vec): position of the swing foot;
    stateX ([3, 1] matrix): position, velocity, acceleration of CoM along x-axis;
    stateY ([3, 1] matrix): position, velocity, acceleration of CoM along y-axis;
    N (int): is the length of the preview horizon;
    dt (float): time step size;
    h (float): CoM height;
    g (float): gravitational acceleration;
    tPf (int): time steps per foot step;

    OUTPUTS
    leftHandside ([ef*N+es*l+2, 2N+2l] matrix): left hand side of the qp constraint,
    G in standard qp (G leq h);
    rightHandside ([ef*N+es*l+2, 1] matrix): right hand side of the qp constraint,
    h in standard qp (G leq h);
    """
    l_cop, r_cop = CoP_constraints(
        m,
        foot_angles,
        support_foot_pos,
        stateX,
        stateY,
        N=N,
        dt=dt,
        h=h,
        g=g,
        tPf=tPf,
    )
    l_support, r_support = support_foot_constraints(
        m, which_current_support, support_foot_pos, next_foot_angles, N=N, tPf=tPf
    )

    # l_swing, r_swing = swing_foot_constraints(m, swing_foot_pos, N=N, dt=dt, tPf=tPf)
    # leftHandside = np.concatenate((l_cop, l_support, l_swing), axis=0)
    # rightHandside = np.concatenate((r_cop, r_support, r_swing), axis=0)

    leftHandside = np.concatenate((l_cop, l_support), axis=0)
    rightHandside = np.concatenate((r_cop, r_support), axis=0)

    return leftHandside, rightHandside


def get_constraints_ds(
    m,
    foot_angles,
    which_next_support,
    next_foot_angles,
    next_support_foot_pos,
    stateX,
    stateY,
    N=16,
    dt=0.1,
    h=1.0,
    g=9.81,
    tPf=8,
):
    """
    INPUTS
    m (int): remaining time steps in current foot step;
    foot_angles ([N, 1] vector): containing the orientations in radians
    of the foot steps at each time step;
    which_current_support (int): next support foot being 'left' 0 or 'right' 1;
    next_foot_angles ([l, 1] vector): containing the orientations in radians of next foot steps;
    next_support_foot_pos ([2, 1] vec): next support foot step position;
    swing_foot_pos ([2, 1] vec): position of the swing foot;
    stateX ([3, 1] matrix): position, velocity, acceleration of CoM along x-axis;
    stateY ([3, 1] matrix): position, velocity, acceleration of CoM along y-axis;
    N (int): is the length of the preview horizon;
    dt (float): time step size;
    h (float): CoM height;
    g (float): gravitational acceleration;
    tPf (int): time steps per foot step;

    OUTPUTS
    leftHandside ([ef*N+es*l+2, 2N+2l] matrix): left hand side of the qp constraint,
    G in standard qp (G leq h);
    rightHandside ([ef*N+es*l+2, 1] matrix): right hand side of the qp constraint,
    h in standard qp (G leq h);
    """

    l_cop, r_cop = CoP_constraints_ds(
        m, foot_angles, next_support_foot_pos, stateX, stateY, N=N
    )
    l_support, r_support = support_foot_constraints_ds(
        m, which_next_support, next_support_foot_pos, next_foot_angles, N=N
    )
    leftHandside = np.concatenate((l_cop, l_support), axis=0)
    rightHandside = np.concatenate((r_cop, r_support), axis=0)

    return leftHandside, rightHandside


def main():
    config_path = "biped_config.json"
    with open(config_path) as json_file:
        config = json.load(json_file)
    config = munch.munchify(config)

    Up = get_Up(2, 1)
    Uv = get_Uv(config.preview_horizon, config.dt)
    Ua = get_Ua(config.preview_horizon, config.dt)
    FutureStepsMat = stepsInFutureStepsMat(3, N=16, tPf=8)

    m = 3
    foot_angles = np.zeros(16)
    foot_position = np.ones((2, 1))
    which_current_support = 0
    stateX = np.ones((3, 1))
    stateY = np.ones((3, 1))
    support_foot_pos = np.array([0, 0])
    next_foot_angles = np.array([0, 0])
    # foot_angles = np.zeros((16, 1))
    swing_foot_pos = np.array([[0], [0]])

    # leftHandSide, rightHandSide = CoP_constraints(
    #     m, foot_angles, foot_position, stateX, stateY
    # )
    # leftHandSide, rightHandSide = support_foot_constraints(
    #     m, current_foot, next_foot_positions, next_foot_angles
    # )
    # leftHandSide, rightHandSide = swing_foot_constraints(m, swing_foot_pos)

    vx_ref = 1.0
    vy_ref = 0.0
    Q, p = get_cost(m, stateX, stateY, vx_ref, vy_ref, support_foot_pos)

    G, h = get_constraints(
        m,
        foot_angles,
        which_current_support,
        next_foot_angles,
        support_foot_pos,
        swing_foot_pos,
        stateX,
        stateY,
    )


if __name__ == "__main__":
    main()
