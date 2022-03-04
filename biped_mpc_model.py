import json
from pdb import set_trace

import cvxopt
import cvxpy as cp
import matplotlib.pyplot as plt
import munch
import numpy as np

from biped_mpc_opt_utils import (
    get_constraints,
    get_cost,
    CoP_constraints,
    get_constraints_ds,
    get_cost_ds,
)
from biped_params import rectangular_foot_CoP


class BipedModel:
    def __init__(self, config):
        self.config = config
        # Time steps per Foot steps
        self.tPf = int(config.step_time / config.dt)
        self.vx_ref = config.vx
        self.vy_ref = config.vy
        self.foot_angles = np.zeros(config.preview_horizon)
        self.preview_horizon = config.preview_horizon

    def reset(self):
        self.stateX = np.zeros((3, 1))
        self.stateY = np.zeros((3, 1))
        self.A, self.B = get_state_matrices(self.config.dt)
        self.step_count = 0
        self.counter = 0
        # 0 for left, 1 for right, -1 for double support
        self.support_foot = -1
        self.left_foot_pos = np.array([0, 0.15])
        self.right_foot_pos = np.array([0, -0.15])

    def step(self, controlX, controlY, next_step_planned, noise=False, ratio=1):
        noiseX = np.array([[0], [1], [0]]) * np.random.normal(0.0, 0.02) * ratio
        noiseY = np.array([[0], [1], [0]]) * np.random.normal(0.0, 0.02) * ratio
        if not noise:
            noiseX = 0 * noiseX
            noiseY = 0 * noiseY

        next_stateX = self.A @ self.stateX + self.B @ controlX + noiseX
        next_stateY = self.A @ self.stateY + self.B @ controlY + noiseY
        self.stateX = next_stateX
        self.stateY = next_stateY
        # update swing foot
        if self.support_foot != -1:
            self.update_swing_foot(next_step_planned)
        # finite state machine
        self.fsm_update()

        return next_stateX, next_stateY

    def get_CoP(self):
        copX = np.array([[1, 0, -1 / self.config.g]]) @ self.stateX
        copY = np.array([[1, 0, -1 / self.config.g]]) @ self.stateY

        return np.array([copX[0][0], copY[0][0]])

    def get_support_foot_pos(self):
        # Initial double support phase
        if self.counter < self.tPf:
            return np.array([0.0, 0.0])
        else:
            if self.support_foot == 0:
                return self.left_foot_pos
            elif self.support_foot == 1:
                return self.right_foot_pos

    def get_swing_foot_pos(self):
        # Initial double support phase
        if self.counter < self.tPf:
            return np.array([0.0, 0.0])
        else:
            if self.support_foot == 0:
                return self.right_foot_pos
            elif self.support_foot == 1:
                return self.left_foot_pos

    def update_swing_foot(self, next_step_planned, max_speed=1.0):
        swing_foot_pos = self.get_swing_foot_pos()
        tr = (self.tPf - self.step_count % self.tPf) * self.config.dt

        v_swing = (next_step_planned - swing_foot_pos) / tr
        v_swing = np.clip(v_swing, -max_speed, max_speed)

        if self.support_foot == 0:
            self.right_foot_pos += v_swing * self.config.dt
        elif self.support_foot == 1:
            self.left_foot_pos += v_swing * self.config.dt

    def fsm_update(self):
        self.step_count += 1
        self.counter += 1

        if self.step_count == self.tPf:
            self.step_count = 0

            if self.support_foot == -1:
                # If in double support move right leg, making the left leg the support leg
                self.support_foot = 0
            else:
                # 0 -> 1, 1 -> 0
                self.support_foot = 1 - self.support_foot

    def get_control(self, solver="cvxopt", init=False, next_support_foot="left"):
        remainTimeSteps = self.tPf - self.step_count - 1
        support_foot_pos = self.get_support_foot_pos()
        remainSteps = int(np.ceil((self.preview_horizon - remainTimeSteps) / self.tPf))

        if init:
            if next_support_foot == "left":
                next_support_pos = self.left_foot_pos
                which_next_support = 0
            elif next_support_foot == "right":
                next_support_pos = self.right_foot_pos
                which_next_support = 1

            Qk, pk = get_cost_ds(
                remainTimeSteps,
                self.stateX,
                self.stateY,
                self.vx_ref,
                self.vy_ref,
                next_support_pos,
                N=self.preview_horizon,
            )
            leftHandside, rightHandside = get_constraints_ds(
                remainTimeSteps,
                self.foot_angles,
                which_next_support,
                np.zeros(10),
                next_support_pos,
                self.stateX,
                self.stateY,
                N=self.preview_horizon,
            )
            remainSteps -= 1
        else:
            Qk, pk = get_cost(
                remainTimeSteps,
                self.stateX,
                self.stateY,
                self.vx_ref,
                self.vy_ref,
                support_foot_pos,
                N=self.preview_horizon,
            )

            swing_foot_pos = self.get_swing_foot_pos()
            leftHandside, rightHandside = get_constraints(
                remainTimeSteps,
                self.foot_angles,
                self.support_foot,
                np.zeros(10),
                support_foot_pos,
                swing_foot_pos,
                self.stateX,
                self.stateY,
                N=self.preview_horizon,
            )

        if solver == "cvxpy":
            # Solve QP using cvxpy
            x = cp.Variable(2 * self.config.preview_horizon + 2 * remainSteps)
            prob = cp.Problem(
                cp.Minimize(cp.quad_form(x, Qk) + pk.T @ x),
                [leftHandside @ x <= rightHandside[:, 0]],
            )
            prob.solve(solver=cp.MOSEK)
            x = np.expand_dims(x.value, axis=1)
        elif solver == "cvxopt":
            # Solve QP using cvxopt
            # we transform our data into cvxopt complicant data
            Q = cvxopt.matrix(Qk)
            p = cvxopt.matrix(pk)

            G = cvxopt.matrix(leftHandside)
            h = cvxopt.matrix(rightHandside)

            # now we call cvxopt to solve the quadratic program constructed above
            cvxopt.solvers.options["show_progress"] = False
            sol = cvxopt.solvers.qp(Q, p, G, h)
            x = np.array(sol["x"])
            print("primal objective: {}".format(sol["primal objective"]))

        X_jerks = x[: self.preview_horizon]
        X_next_foot_pos = x[self.preview_horizon : (self.preview_horizon + remainSteps)]
        Y_jerks = x[
            (self.preview_horizon + remainSteps) : (
                self.preview_horizon + remainSteps + self.preview_horizon
            )
        ]
        Y_next_foot_pos = x[
            (self.preview_horizon + remainSteps + self.preview_horizon) :
        ]

        return X_jerks, X_next_foot_pos, Y_jerks, Y_next_foot_pos


def get_state_matrices(t):
    A = np.diag([1, 1, 1]) + t * np.diag([1, 1], k=1) + (t ** 2 / 2) * np.diag([1], k=2)
    B = np.array([[np.power(t, 3) / 6], [t ** 2 / 2], [t]])
    return A, B


def main():
    config_path = "biped_config.json"
    with open(config_path) as json_file:
        config = json.load(json_file)
    config = munch.munchify(config)

    model = BipedModel(config)
    model.reset()

    xs = []
    ys = []
    x_support = []
    y_support = []
    x_cop = []
    y_cop = []

    for i in range(100):
        if i < 8:
            init = True
            print("double support control")
        else:
            init = False
            print("single support control")
        print("step {}, remain steps {}".format(i, model.tPf - model.step_count - 1))
        X_jerks, X_next_foot_pos, Y_jerks, Y_next_foot_pos = model.get_control(
            solver="cvxopt", init=init
        )

        print(i, X_next_foot_pos[0][0], Y_next_foot_pos[0][0])

        controlX = X_jerks[:1, :]
        controlY = Y_jerks[:1, :]
        next_step_planned = np.array([X_next_foot_pos[0][0], Y_next_foot_pos[0][0]])
        next_stateX, next_stateY = model.step(controlX, controlY, next_step_planned)

        sup = model.get_support_foot_pos()
        cop = model.get_CoP()

        xs.append(next_stateX[0][0])
        ys.append(next_stateY[0][0])
        x_support.append(sup[0])
        y_support.append(sup[1])
        x_cop.append(cop[0])
        y_cop.append(cop[1])
        # print(y_cop)

        # d, b = rectangular_foot_CoP()
        # cop = model.get_CoP().reshape(-1, 1)
        # sup = model.get_support_foot_pos().reshape(-1, 1)
        # cop_constraint = d @ (cop - sup) <= b
        # cop_satisfy = np.all(cop_constraint)
        # print("Step {} CoP Satisfy: {}".format(i, cop_satisfy))

        # print(
        #     "{}: support foot: {}, CoP: {}".format(
        #         i, model.get_support_foot_pos(), model.get_CoP()
        #     )
        # )

        # print(
        #     "left foot: {}, right foot: {}".format(
        #         model.left_foot_pos, model.right_foot_pos
        #     )
        # )

    plt.figure()
    plt.plot(xs, ys, label="CoM")
    # plt.plot(xs, label="CoM")
    # plt.plot(ys, label="CoM")
    plt.plot(x_cop, y_cop, label="CoP")
    # plt.plot(y_cop, label="CoP")
    plt.legend()

    plt.savefig("demo.png", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
