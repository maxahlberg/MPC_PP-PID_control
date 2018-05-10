#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import copy
import math
import cubic_spline_planner
import cvxpy
from cvxpy import *

# Parameters
iteration = 0
MAX_ACCEL = 10.0  # maximum acceleration [m/ss]
MAX_ROAD_WIDTH = 6.0  # maximum road width [m]
N = 10  #Horizon
Spaceing = 10 #how many points between yeach MPC point for the PP+PID
target_speed = 10.0
k = 0.4  # look forward gain
k_c = 400.0 #look forward curvature gain
Lfc = 8.0  # Min look-ahead distance
Kp = 1.0  # speed proportional gain
dt = 0.2  # [s]
L = 3.0  # [m] wheel base of vehicle
Width = 2.0  # [m] Width of the vehicle

#Vehicle parameters
lr = L*0.5 #[m]
lf = L*0.5 #[m]


show_animation = True


class quinic_polynomial: #Skapar ett 5e grads polynom som beräknar position, velocity och acceleration

    def __init__(self, xs, vxs, axs, xe, vxe, axe, T): # (position_xs, Velocity_xs, Acceleration_xs, P_xe, V_xe, A_xe, Time )

        # calc coefficient of quinic polynomial
        self.xs = xs 
        self.vxs = vxs
        self.axs = axs
        self.xe = xe
        self.vxe = vxe
        self.axe = axe

        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0 #Varför accelerationen delat med 2? -För att de skall bli rätt dimensioner i slutandan

        A = np.array([[T**3, T**4, T**5],
                      [3 * T ** 2, 4 * T ** 3, 5 * T ** 4],
                      [6 * T, 12 * T ** 2, 20 * T ** 3]])
        b = np.array([xe - self.a0 - self.a1 * T - self.a2 * T**2,
                      vxe - self.a1 - 2 * self.a2 * T,
                      axe - 2 * self.a2])
        x = np.linalg.solve(A, b) #Antagligen matris invers som löser a3,a4,a5. En form av jerk, jerk_derivata, jerk dubbelderivata

        self.a3 = x[0]
        self.a4 = x[1]
        self.a5 = x[2]
    def calc_point(self, t): # point on xs at t
        xt = self.a0 + self.a1 * t + self.a2 * t**2 + \
            self.a3 * t**3 + self.a4 * t**4 + self.a5 * t**5

        return xt

    def calc_first_derivative(self, t): #speed in point at t
        xt = self.a1 + 2 * self.a2 * t + \
            3 * self.a3 * t**2 + 4 * self.a4 * t**3 + 5 * self.a5 * t**4

        return xt

    def calc_second_derivative(self, t): # acceleration in point at t
        xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t**2 + 20 * self.a5 * t**3

        return xt

class quartic_polynomial:

    def __init__(self, xs, vxs, axs, vxe, axe, T):

        # calc coefficient of quinic polynomial
        self.xs = xs
        self.vxs = vxs
        self.axs = axs
        self.vxe = vxe
        self.axe = axe

        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array([[3 * T ** 2, 4 * T ** 3],
                      [6 * T, 12 * T ** 2]])
        b = np.array([vxe - self.a1 - 2 * self.a2 * T,
                      axe - 2 * self.a2])
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]

    def calc_point(self, t):
        xt = self.a0 + self.a1 * t + self.a2 * t**2 + \
            self.a3 * t**3 + self.a4 * t**4

        return xt

    def calc_first_derivative(self, t):
        xt = self.a1 + 2 * self.a2 * t + \
            3 * self.a3 * t**2 + 4 * self.a4 * t**3

        return xt

    def calc_second_derivative(self, t):
        xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t**2

        return xt

class Frenet_path:

    def __init__(self):
        self.t = []
        self.d = []
        self.d_d = []
        self.d_dd = []
        self.s = []
        self.s_d = []
        self.s_dd = []
        self.cd = 0.0
        self.cv = 0.0
        self.cf = 0.0

        self.x = []
        self.y = []
        self.yaw = []
        self.ds = []
        self.c = []

        #self.xx = []
        #self.yy = []

class Low_control:

    def __init__(self):
        self.t = []
        self.d = []
        self.d_d = []
        self.d_dd = []
        self.s = []
        self.s_d = []
        self.s_dd = []
        self.cd = 0.0
        self.cv = 0.0
        self.cf = 0.0

        self.x = []
        self.y = []
        self.yaw = []
        self.ds = []
        self.c = []

class State:

    def __init__(self):
        self.t = []
        self.d = []
        self.d_d = []
        self.d_dd = []
        self.s = []
        self.s_d = []
        self.s_dd = []
        self.cd = 0.0
        self.cv = 0.0
        self.cf = 0.0

        self.x = []
        self.y = []
        self.yaw = []
        self.ds = []
        self.c = []
        self.v = []
        self.psi = []

class Vehicle:

    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v

class LookUpp_sd_to_xy:

    def _init_(self):
        self.x = []
        self.y = []
        self.s = []
        self.d = []


def get_nparray_from_matrix(x): #to getarrays from matrix
    return np.array(x).flatten()

def calc_MPC_frenet_paths(c_speed, c_d, c_d_d, c_d_dd, s0, c_acc):
    tfp = Frenet_path()

    frenet_paths = []
    n = 4     #States x
    m = 2     #Control signals u
    dtt = dt  #[s]
    A = np.matrix([[1, dtt, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, dtt],
                   [0, 0, 0, 1]]) #in DT
    B = np.matrix([[0, 0],
                   [dtt, 0],
                   [0, 0],
                   [0, dtt]])

    X_0 = [s0, c_speed, c_d, c_d_d] #Start vector each iteration
    print('Print of X_0 och U_0: ')
    print(X_0)
    U_0 = [c_acc, c_d_dd]
    print(U_0)
    Q = np.eye(n, dtype=int)
    Q[0,0] = 1      #Position  S
    Q[1,1] = 100    #Hastighet S
    Q[2,2] = 100    #Position  Y
    Q[3,3] = 1      #Hastighet Y
    R = np.eye(m, dtype=int)
    R[0,0] = 10
    R[1,1] = 1
    ref = [s0, target_speed, 0, 0]
    uref = [0, 0]

    x = cvxpy.Variable(n, N+1)
    u = cvxpy.Variable(m, N)
    cost = 0.0
    constr = []
    
    for t in range(N):
        cost += cvxpy.quad_form(x[:,t]-ref, Q)
        cost += cvxpy.quad_form(u[:,t]-uref, R)
        constr += [x[:,t+1] == A*x[:,t] + B*u[:,t]]
        constr += [u[:,t] <= MAX_ACCEL]                 #Upper accel bound
        constr += [u[:,t] >= -MAX_ACCEL]                #Lower accel bound
        constr += [x[2, t] <= MAX_ROAD_WIDTH]           #Upper and lower road with bound
        constr += [x[2, t] >= -MAX_ROAD_WIDTH]
    constr += [x[:, 0] == X_0]
    constr += [u[:, 0] == U_0]
    prob = cvxpy.Problem(cvxpy.Minimize(cost), constr)
    prob.solve(verbose=False)
        
    if prob.status == cvxpy.OPTIMAL or prob.status == cvxpy.OPTIMAL_INACCURATE:
        tfp.s = get_nparray_from_matrix(x.value[0, :])
        tfp.s_d = get_nparray_from_matrix(x.value[1, :])
        tfp.d = get_nparray_from_matrix(x.value[2, :])
        tfp.d_d = get_nparray_from_matrix(x.value[3, :])
        tfp.s_dd = get_nparray_from_matrix(u.value[0, :])
        tfp.d_dd = get_nparray_from_matrix(u.value[1, :])


        frenet_paths.append(tfp)

    return frenet_paths

def calc_detailed_Traj(fplist):
    '''
    interpolating anf increasing the number os "s" and "d" coordinates
    '''
    for fpL in fplist:
        S_points = np.linspace(fpL.s[0], fpL.s[-1], N*Spaceing)
        Sd_points = np.linspace(fpL.s_d[0], fpL.s_d[-1], N*Spaceing)
        Sdd_points = np.linspace(fpL.s_dd[0], fpL.s_dd[-1], N * Spaceing)

    #linearize
    d_interp = np.interp(S_points, fpL.s, fpL.d)
    d_d_interp = np.interp(Sd_points, fpL.s_d, fpL.d_d)
    d_dd_interp = np.interp(Sdd_points, fpL.s_dd, fpL.d_dd)

    LC = Low_control()
    #build up similar vector
    LC.s = S_points
    LC.s_d = Sd_points
    LC.s_dd = Sdd_points
    LC.d = d_interp
    LC.d_d = d_d_interp
    LC.d_dd = d_dd_interp

    low_control = []

    low_control.append(LC)

    return low_control

def calc_global_paths(fplist, csp):  #From S to Global

    for fp in fplist:
        for i in range(N):
            ix, iy = csp.calc_position(fp.s[i])
            if ix is None:
                break
            iyaw = csp.calc_yaw(fp.s[i])                    #Yaw angle derivative wrt curvature
            di = fp.d[i]                                    #Lateral position
            fx = ix + di * math.cos(iyaw + math.pi / 2.0)   #x + closest catheter
            fy = iy + di * math.sin(iyaw + math.pi / 2.0)   #y + furthest catheter
            fp.x.append(fx)                                 #All global x coord in a list
            fp.y.append(fy)                                 #All global y coord in a list



        for i in range(len(fp.x) - 1):#Calc yaw and ds
            dx = fp.x[i + 1] - fp.x[i]
            dy = fp.y[i + 1] - fp.y[i]
            fp.yaw.append(math.atan2(dy, dx))       #Yaw angle as derivative of the curve
            fp.ds.append(math.sqrt(dx**2 + dy**2))  #Length of the tangent vector

        fp.yaw.append(fp.yaw[-1])                   #Last yaw one more time, to get equal length
        fp.ds.append(fp.ds[-1])

        for i in range(len(fp.yaw) - 1):            #Calc curvature
            fp.c.append((fp.yaw[i + 1] - fp.yaw[i]) / (fp.ds[i]+1)) #Derivative of yaw wrt step lenght ds.

        if iteration == 0: #Make a class out of this to save space.
            initial_g_x = fp.x[0]
            initial_g_y = fp.y[0]
            initial_g_yaw = fp.yaw[0]
            initial_g_c = fp.c[0]
            initial_psi = math.atan2(fp.y[1]-fp.y[0], fp.x[1]-fp.x[0])
            initial_x_d = fp.s_d[0] * math.cos(initial_psi) - fp.d_d[0]*math.sin(initial_psi)
            initial_y_d = fp.s_d[0] * math.sin(initial_psi) + fp.d_d[0]*math.cos(initial_psi)
            initial_v = math.sqrt(initial_x_d**2 + initial_y_d**2)

    return fplist, initial_g_x, initial_g_y, initial_g_yaw, initial_g_c, initial_v, initial_psi

def calc_global_paths_long(fplist, csp): #From S to Global

    for fp in fplist:#For each vector in the list of all vectors of fplist (mystical..)
        for i in range(N*Spaceing):
            ix, iy = csp.calc_position(fp.s[i])
            if ix is None:
                break
            iyaw = csp.calc_yaw(fp.s[i])                    #Yaw angle derivative wrt curvature
            di = fp.d[i]                                    #Lateral position
            fx = ix + di * math.cos(iyaw + math.pi / 2.0)   #x + closest catheter
            fy = iy + di * math.sin(iyaw + math.pi / 2.0)   #y + furthest catheter
            fp.x.append(fx)                                 #All global x coord in a list
            fp.y.append(fy)                                 #All global x coord in a list



        for i in range(len(fp.x) - 1):#Calc yaw and ds
            dx = fp.x[i + 1] - fp.x[i]
            dy = fp.y[i + 1] - fp.y[i]
            fp.yaw.append(math.atan2(dy, dx))       #Yaw angle as derivative of the curve
            fp.ds.append(math.sqrt(dx**2 + dy**2))  #Length of the tangent vector

        fp.yaw.append(fp.yaw[-1])                   #Last yaw one more time, to get equal length
        fp.ds.append(fp.ds[-1])

        # calc curvature
        for i in range(len(fp.yaw) - 1):
            fp.c.append((fp.yaw[i + 1] - fp.yaw[i]) / (fp.ds[i]+1)) #Derivative of yaw wrt step lenght ds.

    return fplist

def calc_target_index(low_control):
    # search nearest point index
    for LC in low_control:
        dx = [LC.x[0] - icx for icx in LC.x] #Distance from the red cx points to the state.x
        dy = [LC.y[0] - icy for icy in LC.y] #Same in y. icy takes the values in cy iteratively

    d = [(math.sqrt(idx ** 2 + idy ** 2)) for (idx, idy) in zip(dx, dy)]
    ind = d.index(min(d))

    L = 0.0
    curvature = 1/(math.sqrt((sum(LC.c[:]) * k_c)**2))
    print('curvature')
    print(curvature)
    Lf = Lfc

    #Search look ahead target point index
    #Jumps point by point forward untill we find a poiint cloase to the look ahead distance
    while Lf > L and (ind + 1) < len(LC.x):
        dx = LC.x[ind + 1] - LC.x[ind] #New defenition of dx and dy
        dy = LC.y[ind + 1] - LC.y[ind] #Now the distance between the next point in the x-y coordinates
        L += math.sqrt(dx ** 2 + dy ** 2)
        ind += 1
    print('Intex point')
    print(ind)
    return ind #Chooses the indexpoint that is close to the LookAhead distance

def PP_PID(low_control, target_ind, target_speed, initial_g_x, initial_g_y, initial_g_yaw, initial_g_c, initial_v, initial_psi, global_test):
    acceleration = []
    if iteration == 0:
        old_state_x = initial_g_x
        old_state_y = initial_g_y
        old_state_yaw = initial_g_yaw
        old_state_c = initial_g_c
        old_state_v = initial_v
        old_state_psi = initial_psi
    else:
        # Old state
        old_state_x = global_test.x[1]
        old_state_y = global_test.y[1]
        old_state_yaw = global_test.yaw[1]
        old_state_c = global_test.c[1]
        old_state_v = global_test.v[1]
        old_state_psi = global_test.psi[1]


    for LC in low_control:
        acceleration = Kp * (target_speed - old_state_v)

        tx = LC.x[target_ind]
        ty = LC.y[target_ind]

        #The difference in yaw between where you want to go and where you are looking(your yaw)
        alpha = math.atan2(ty - old_state_y, tx - old_state_x) - old_state_yaw
        alpha = alpha
        if LC.s_d[0] < 0:  #If going backwards correct with pi (rarely in racing)
            alpha = math.pi - alpha

        Lf = k * old_state_v + Lfc #LookForwardGain linearly dep. on speed. Lfc = constant
        #Yaw error at the nearest point (L is wheel base)
        SW_angle = 1.0 * math.atan2(2.0 * L * math.sin(alpha) / Lf, 1.0)

    print('Steering Wheel angle!')
    print(SW_angle)
    print('old_state_c')
    print(old_state_c)
    print('Lf')
    print(Lf)

    return SW_angle, acceleration, alpha

def global_vehicle_simulator(low_control, SW, a, initial_g_x, initial_g_y,
                             initial_g_yaw, initial_g_c, initial_v, initial_psi,
                             LUTs, LUTd, LUTx, LUTy, global_test):

    if iteration == 0:
        old_state_x = initial_g_x
        old_state_y = initial_g_y
        old_state_yaw = initial_g_yaw
        old_state_c = initial_g_c
        old_state_v = initial_v
        old_state_psi = initial_psi
    else:
        # Old state
        old_state_x = global_test.x[1]
        old_state_y = global_test.y[1]
        old_state_yaw = global_test.yaw[1]
        old_state_c = global_test.c[1]
        old_state_v = global_test.v[1]
        old_state_psi = global_test.psi[1]


    for LControl in low_control:


        #Global bicycle model here
        VP = lr/(lf+lr)
        beta = math.atan(VP*math.tan(SW)) #Body Slip angle
        new_state_c = math.sin(beta)/lr
        R = lr / math.sin(beta)
        v_p = a + old_state_v/R**2
        new_state_v = old_state_v + v_p * dt


        psi_d = new_state_v/lr * math.sin(beta)
        new_state_psi = old_state_psi + psi_d*dt

        x_dd = a * math.cos(new_state_psi)
        y_dd = a * math.sin(new_state_psi)
        x_d = new_state_v*math.cos(new_state_psi + beta)
        y_d = new_state_v*math.sin(new_state_psi + beta)
        new_state_x = old_state_x + x_d * dt #This is sent to the plot
        new_state_y = old_state_y + y_d * dt #This is sent to the plot
        new_state_yaw = math.atan2(new_state_y - old_state_y, new_state_x - old_state_x)

        #Transform the new coordinates to Frenet
        value = [new_state_x, new_state_y]
        X = np.sqrt(np.square(LUTx - value[0]) + np.square(LUTy - value[1]))
        idx = np.where(X == X.min())
        r_idx = np.asscalar(idx[0])
        c_idx = np.asscalar(idx[1])

        new_state_s = LUTs[r_idx, c_idx] #Position i Frenet
        new_state_d = LUTd[r_idx, c_idx]

        print('New s')
        print(new_state_s)
        print('New d')
        print(new_state_d)

        new_state_s_d = new_state_v * \
                        math.sin(((math.pi/2 - new_state_yaw)) + (new_state_psi-beta))
        new_state_d_d = new_state_v * \
                        math.cos(((math.pi/2 - new_state_yaw)) + (new_state_psi-beta))

        new_state_s_dd = v_p * math.sin(((math.pi/2 - new_state_yaw)) + (new_state_psi-beta))
        new_state_d_dd = v_p * math.cos(((math.pi/2 - new_state_yaw)) + (new_state_psi-beta))

        print('New s_dd')
        print(new_state_s_dd)
        print('New d_dd')
        print(new_state_d_dd)
        old_state_s, old_state_s_d, old_state_s_dd = 0, 0, 0
        old_state_d, old_state_d_d, old_state_d_dd = 0, 0, 0
        state = State()
        #Update
        state.d = [-old_state_d, -new_state_d]
        state.d_d = [old_state_d_d, new_state_d_d]
        state.d_dd = [old_state_d_dd, new_state_d_dd]
        state.s = [old_state_s, new_state_s]
        state.s_d = [old_state_s_d, new_state_s_d]
        state.s_dd = [old_state_s_dd, new_state_s_dd]

        state.x = [old_state_x, new_state_x]
        state.y = [old_state_y, new_state_y]
        state.yaw = [old_state_yaw, new_state_yaw]
        state.c = [old_state_c, new_state_c]
        state.v = [old_state_v, new_state_v]
        state.psi = [old_state_psi, new_state_psi]



        print('From the Vehicle model STATES')
        print('state.d')
        print(state.d)
        print('state.d_d')
        print(state.d_d)
        print('state.d_dd')
        print(state.d_dd)
        print('state.s')
        print(state.s)
        print(state.s_d)
        print(state.s_dd)

        test = []
        test.append(state)

    return test


def frenet_optimal_planning(csp, s0, c_speed, c_d, c_d_d, c_d_dd, c_acc, LUTs, LUTd, LUTx, LUTy, initial):
    #Calculating a traj. for position and speed
    fplist = calc_MPC_frenet_paths(c_speed, c_d, c_d_d, c_d_dd, s0, c_acc)

    #Interpolate between the points in the frenet frame
    low_control = calc_detailed_Traj(fplist)

    #The calculated traj. from Frenet to Globala
    fplist, initial_g_x, initial_g_y, initial_g_yaw, initial_g_c, initial_v, initial_psi = calc_global_paths(fplist, csp)

    #Generate target path to follow
    low_control = calc_global_paths_long(low_control, csp)

    target_ind = calc_target_index(low_control)

    SW_angle, acceleration, alpha = PP_PID(low_control, target_ind, target_speed, initial_g_x, initial_g_y, initial_g_yaw, initial_g_c, initial_v, initial_psi, initial)


    global_test = global_vehicle_simulator(low_control, SW_angle, acceleration,
                                           initial_g_x, initial_g_y, initial_g_yaw,
                                           initial_g_c, initial_v, initial_psi,
                                           LUTs, LUTd, LUTx, LUTy,
                                           initial)

    return fplist[0], low_control[0], SW_angle, acceleration, target_ind, global_test[0]

def generate_target_course(x, y):
    csp = cubic_spline_planner.Spline2D(x, y)
    s = np.arange(0, csp.s[-1], 0.1)
    d = np.arange(-MAX_ROAD_WIDTH, MAX_ROAD_WIDTH, 0.1) #Creates a 0.1 perp. line
    s_len = s.size
    d_len = d.size
    LUTs = np.zeros((s_len, d_len))
    LUTd = np.zeros((s_len, d_len))
    LUTx = np.zeros((s_len, d_len))
    LUTy = np.zeros((s_len, d_len))

    rx, ry, ryaw, rk = [], [], [], []
    for i_s in s:
        ix, iy = csp.calc_position(i_s) #i_s  = incremental s ->Visual constraints
        rx.append(ix)#Ref x
        ry.append(iy)#Ref y
        ryaw.append(csp.calc_yaw(i_s))#Ref yaw
        rk.append(csp.calc_curvature(i_s))#Ref curvature

    LTs, LTd, LTx, LTy, refx, refy, refyaw = [], [], [], [], [], [], []
    s_count, d_count = -1, -1
    for i_ss in s:
        s_count = s_count + 1
        LTs = i_ss
        refx, refy = csp.calc_position(i_ss)
        refyaw = csp.calc_yaw(i_ss)
        for i_dd in d:
            if i_dd == -MAX_ROAD_WIDTH:
                d_count = -1
            d_count = d_count + 1
            LTd = i_dd
            LTx = refx + i_dd*math.sin(refyaw)
            LTy = refy - i_dd*math.cos(refyaw)
            LUTs[s_count, d_count] = LTs
            LUTd[s_count, d_count] = LTd
            LUTx[s_count, d_count] = LTx
            LUTy[s_count, d_count] = LTy


    print('Matrix Analysis ')
    print(len(LUTs))
    print(len(LUTx))
    print(LUTs)
    print(LUTd)
    print(LUTx)
    print(LUTy)
    plt.plot(LUTx[:,:], LUTy[:,:])
    #plt.plot(LUTx[:,-1], LUTy[:,-1])
    plt.grid(True)
    plt.show()



    return rx, ry, ryaw, rk, csp, LUTs, LUTd, LUTx, LUTy


def main():
    print(__file__ + " start!!")
    wx, wy = [], []
    # way points fo the track
    #wxi = [0.0, 0.0, 5.0, 15.0, 17.0, 15.0,  17.0,  12.0,   0.0,   0.0, 0.0]
    #wyi = [0.0, 2.0, 5.0,  5.0,  0.0, -5.0, -10.0, -15.0, -15.0, -10.0, 0.0]

    wxi = [-20.0, -40.0, -70.0, -100.0, -120.0,  -140.0,  -150.0,   -160.0, -180.0,
           -200.0, -180.0, -160.0, -150.0, -140.0, -130.0, -120.0, -90.0, -60.0, -40.0, 0.0, 5.0, 0.0, -15.0, -20.0]

    wyi = [0.0, 0.0,  5.0,  0.0, 0.0, 0.0, 5.0, 0.0, 0.0,
           20.0, 40.0, 40.0, 40.0, 45.0, 40.0, 35.0, 40.0, 40.0, 40.0, 40.0, 20.0, 0.0, 0.0, 0.0]

    wx += wxi
    wy += wyi


    tx, ty, tyaw, tc, csp, LUTs, LUTd, LUTx, LUTy = generate_target_course(wx, wy) #Get ut target-x (tx) Target-y (ty) target yaw, target Course! csp(the whole function handle)osv.

    x_plus, y_plus, x_minus, y_minus = [], [], [], []#Creates visual constraints
    for i in range(len(tyaw)):
        x_plus.append(tx[i] + MAX_ROAD_WIDTH * math.sin(tyaw[i]))
        y_plus.append(ty[i] - MAX_ROAD_WIDTH * math.cos(tyaw[i]))
        x_minus.append(tx[i] - MAX_ROAD_WIDTH * math.sin(tyaw[i]))
        y_minus.append(ty[i] + MAX_ROAD_WIDTH * math.cos(tyaw[i]))

    # initial state
    s0 = 2.0      #current course position
    c_speed = 2   #current speed [m/s]
    c_acc = 0     #CURRENT LATTERAL ACCELERATION [m/s²]
    c_d = 3.0     #current lateral position [m]
    c_d_d = 0     #current lateral speed [m/s]
    c_d_dd = 0.0  #current latral acceleration [m/s²]

    state = State()
    state.s = s0
    state.s_d = c_speed
    state.s_dd = c_acc
    state.d = c_d
    state.d_d = c_d_d
    state.d_dd = c_d_dd
    state.v = s0
    state.x = 0
    state.y = 0
    state.yaw = 0
    initial = []
    initial.append(state)
    #initial measured state


    xx, yy = [], []

    area = 20.0  #Animation area length [m]
    vehicle = Vehicle()
    for i in range(600):  #Max number of iterations
        print('OMGÅNG:')
        print(i)


        path, L_follow, sw_angle, acceleration, target_ind, global_test = frenet_optimal_planning(
            csp, s0, c_speed, c_d, c_d_d, c_d_dd, c_acc, LUTs, LUTd, LUTx, LUTy, initial)  #Main function calling others
        initial = global_test
        print('DIFF .s[1]')
        print(path.s[1] - global_test.s[1])
        print('DIFF .d[1]')
        print(path.d[1] - - global_test.d[1])
        print('DIFF .s_d[1]')
        print(path.s_d[1] - global_test.s_d[1])
        print('DIFF .d_d[1]')
        print(path.d_d[1] - global_test.d_d[1])
        print('DIFF .s_dd[1]')
        print(path.s_dd[1] - global_test.s_dd[1])
        print('DIFF .d_dd[1]')
        print(path.d_dd[1] - global_test.d_dd[1])


        s0 = global_test.s[1]
        c_d = global_test.d[1]
        c_d_d = global_test.d_d[1]
        c_d_dd = global_test.d_dd[1]
        c_speed = global_test.s_d[1]
        c_acc = global_test.s_dd[1]

        xx.append(path.x)
        yy.append(path.y)

        if np.hypot(path.x[1] - tx[-1], path.y[1] - ty[-1]) <= 3.0: #hypot = sqrt(x²+y²) eucleadian norm
            print("The Goal has been Reached")
            break

        if show_animation:
            plt.cla()
            plt.plot(tx, ty, "k")
            plt.plot(x_plus, y_plus,"y")
            plt.plot(x_minus, y_minus,"b")
            #print('detta är path.x[1:]')
            #print(path.x[1:])
            #print('detta är path.y[1:]')
            #print(path.y[1:])
            #plt.plot(ob[:, 0], ob[:, 1], "xk")#x marker - black
            plt.plot(path.x[1:], path.y[1:], "-or")#circle marker - red MPC
            #plt.plot(path.x[0], path.y[0], "vc")#triangle down-cayan First MPC point
            #plt.plot(L_follow.x[0:], L_follow.y[0:], ".b")
            plt.plot(L_follow.x[target_ind], L_follow.y[target_ind], '+b')
            plt.xlim(path.x[1] - area, path.x[1] + area)
            plt.ylim(path.y[1] - area, path.y[1] + area)
            #plt.title("v[m/s]:" + str(c_speed)[0:4])
            plt.title("SW angle: " + '{:.2f}'.format(sw_angle) +
                      " yaw: " + '{:.2f}'.format(global_test.yaw[1])
                      + " s_d: " + '{:.2f}'.format(c_speed))
            plt.plot(global_test.x[1], global_test.y[1], 'gd')
            #Full vehicle coordinates
            FRx = lf*math.cos(global_test.yaw[1]) + (Width/2)*math.sin(global_test.yaw[1])
            FLx = lf*math.cos(global_test.yaw[1]) - (Width/2)*math.sin(global_test.yaw[1])
            RRx = lr*math.cos(global_test.yaw[1]) - (Width/2)*math.sin(global_test.yaw[1])
            RLx = lr*math.cos(global_test.yaw[1]) + (Width/2)*math.sin(global_test.yaw[1])
            FRy = lf*math.sin(global_test.yaw[1]) - (Width/2)*math.cos(global_test.yaw[1])
            FLy = lf*math.sin(global_test.yaw[1]) + (Width/2)*math.cos(global_test.yaw[1])
            RRy = lr*math.sin(global_test.yaw[1]) + (Width/2)*math.cos(global_test.yaw[1])
            RLy = lr*math.sin(global_test.yaw[1]) - (Width/2)*math.cos(global_test.yaw[1])
            carx = [global_test.x[1] - FRx, global_test.x[1] - FLx,
                    global_test.x[1] + RLx, global_test.x[1] + RRx, global_test.x[1] - FRx]
            cary = [global_test.y[1] - FRy, global_test.y[1] - FLy,
                    global_test.y[1] + RLy, global_test.y[1] + RRy, global_test.y[1] - FRy]
            plt.plot(carx, cary, 'r')

            plt.pause(0.01)





    print("Finish")
    print(i)

    if show_animation:
        plt.grid(True)
        plt.plot(xx, yy)

if __name__ == '__main__':
    main()
