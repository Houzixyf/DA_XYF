# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pickle
import numpy as np

path = 'E:\Yifan_Xue\DA\Codes\Data\without_Refsol_Brockett\Brockett_e2\delta_x_0.01'

get_u_list = True
if get_u_list:
    u1_place = open(path + '\\U1_data_dt_0.01.plk', 'rb')
    u1_data = pickle.load(u1_place)
    u1_place.close()

    u2_place = open(path + '\\U2_data_dt_0.01.plk', 'rb')
    u2_data = pickle.load(u2_place)
    u2_place.close()

    x1_place = open(path + '\\X1_data_dt_0.01.plk', 'rb')
    x1_data = pickle.load(x1_place)
    x1_place.close()

    x2_place = open(path + '\\X2_data_dt_0.01.plk', 'rb')
    x2_data = pickle.load(x2_place)
    x2_place.close()

    x3_place = open(path + '\\X3_data_dt_0.01.plk', 'rb')
    x3_data = pickle.load(x3_place)
    x3_place.close()

    t_place = open(path + '\\t_data_dt_0.01.plk', 'rb')
    t_data = pickle.load(t_place)
    t_place.close()

    k_place = open(path + '\\k_data_dt_0.01.plk', 'rb')
    k_data = pickle.load(k_place)
    k_place.close()

    U1_Umgebung = u1_data
    U2_Umgebung = u2_data
    X1_Umgebung = x1_data
    X2_Umgebung = x2_data
    X3_Umgebung = x3_data

    X1_Umgebung_array = np.array(X1_Umgebung)  # (13L, 11L) 13:x0的个数，11:时间点
    U1_Umgebung_array = np.array(U1_Umgebung)  # (13L, 11L)
    U2_Umgebung_array = np.array(U2_Umgebung)  # (13L, 11L)
    X2_Umgebung_array = np.array(X2_Umgebung)
    X3_Umgebung_array = np.array(X3_Umgebung)


    #    print X1_Umgebung_array[:,0]
    #    print len(t_data)


    #    from scipy.interpolate import interp1d
    #    f = interp1d(X1_Umgebung,U1_Umgebung_array)

    def find_place_x(x):
        if x in [round(i, 5) for i in X1_Umgebung_array[:, 0]]:
            # raise Exception("x is already in X0_List.")
            return [round(i, 5) for i in X1_Umgebung_array[:, 0]].index(x)
        else:
            assert X1_Umgebung_array[0, 0] < x < X1_Umgebung_array[-1, 0] or X1_Umgebung_array[
                                                                                 0, 0] > x > \
                                                                             X1_Umgebung_array[
                                                                                 -1, 0]
            if X1_Umgebung_array[0, 0] < X1_Umgebung_array[-1, 0]:  # 递增
                for i in range(len(X1_Umgebung_array[:, 0])):
                    if X1_Umgebung_array[i, 0] < x < X1_Umgebung_array[i + 1, 0]:
                        print ('x is between {} and {}'.format(X1_Umgebung_array[i, 0],
                                                               X1_Umgebung_array[i + 1, 0]))
                        return i
                    else:
                        continue
            elif X1_Umgebung_array[0, 0] > X1_Umgebung_array[-1, 0]:
                for i in range(len(X1_Umgebung_array[:, 0])):
                    if X1_Umgebung_array[i, 0] > x > X1_Umgebung_array[i + 1, 0]:
                        print ('x is between {} and {}'.format(X1_Umgebung_array[i, 0],
                                                               X1_Umgebung_array[i + 1, 0]))
                        return i
                    else:
                        continue


    def interpolate_u(x, index, t_data, U1_Umgebung_array, U2_Umgebung_array,
                      X1_Umgebung_array):  # index=place_x
        U1_new = []
        U2_new = []
        if x in [round(i, 5) for i in X1_Umgebung_array[:, 0]]:
            U1_new = U1_Umgebung_array[index]
            U2_new = U2_Umgebung_array[index]
        else:
            for i in range(len(t_data)):
                x_left = X1_Umgebung_array[index, 0]
                x_right = X1_Umgebung_array[index + 1, 0]
                u1_left = U1_Umgebung_array[index, i]
                u1_right = U1_Umgebung_array[index + 1, i]
                u2_left = U2_Umgebung_array[index, i]
                u2_right = U2_Umgebung_array[index + 1, i]

                slope_u1 = float(u1_right - u1_left) / (x_right - x_left)
                u1_new = slope_u1 * (x - x_left) + u1_left

                slope_u2 = float(u2_right - u2_left) / (x_right - x_left)
                u2_new = slope_u2 * (x - x_left) + u2_left

                U1_new.append(u1_new)
                U2_new.append(u2_new)
        U_new = [U1_new, U2_new]

        return U_new


    def f(x, u1, u2, k):  # Brockett_e2
        x1, x2, x3 = x
        u1 = u1
        u2 = u2
        ff = np.array([k * u1, k * u2, k * (x2 * u1 - x1 * u2)])
        return ff


    def euler(f, x0, T, u1_data_e, u2_data_e, k=1, t_data=t_data, dt=.01):  # simulate_x
        res = [x0]
        tt = [0]
        while round(tt[-1], 5) <= T:
            x_old = res[-1]
            t_index = t_data.index(round(tt[-1], 5))  #
            u1 = u1_data_e[t_index]
            u2 = u2_data_e[t_index]
            # u1 = u1_new[t_index]
            # u2 = u2_new[t_index]
            res.append(x_old + dt * f(x_old, u1, u2, k))
            tt.append(tt[-1] + dt)
        return np.array(res)# tt,


    def simulate(xx0, T, u1, u2, dt=0.01, k_use=1, t_data=t_data):
        xxn = euler(f, xx0, T, u1, u2, k_use, t_data=t_data, dt=dt)  #tt,
        return np.array(xxn)#tt,


    x_1 = np.linspace(0.02, 0.03, 11)
    x_1 = [round(i, 5) for i in x_1]
    index = [find_place_x(xi) for xi in x_1]
    u_new = np.array([interpolate_u(x_1[i], index[i], t_data, U1_Umgebung_array, U2_Umgebung_array,
                                    X1_Umgebung_array) for i in range(len(x_1))])
    u1_new = u_new[:, 0]
    u2_new = u_new[:, 1]
    k_use = [k_data[i][0] for i in index]

    t_data = [round(i, 5) for i in t_data]
    res_1 = [simulate(np.array([x_1[i],0.1,0.1]), t_data[-1], u1=u1_new[i], u2=u2_new[i], dt=round(t_data[-1] - t_data[-2], 5), k_use= k_use[i], t_data=t_data) for i in range(len(x_1))]# t_sim

    import matplotlib.pyplot as plt

    plt.figure(1)
    ax1 = plt.subplot(221)

    for i in range(len(res_1)):
        ax1.plot(t_data, u1_new[i])
    plt.show()
    plt.figure(2)
    #Todo:
    plt.plot(t_data, X3_Umgebung[index], 'b', t_data, res_1[:-1, 2], 'r', t_data,
             X3_Umgebung[index + 1], 'g')
    plt.show()



