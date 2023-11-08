from symtable import Symbol
import matplotlib as mt
import numpy as np
import scipy as sc
from sympy import *
from scipy.optimize import root
import matplotlib.pyplot as plt
import time

"Система уравнений описывающая стационарные процессы"

ksi_1 = 1000
ksi_2 = 1000
ksi_3 = 1000
ksi_dem = 1000
p_1 = 400000
p_2 = 300000
p_3 = 100000

# m_1 + m_2 - m_3 = 0
# p_4 - p_g = 0
# p_1 - p_4 - ksi_1 * m_1 ** 2 = 0
# p_4 - p_3 - ksi_3 * m_3 ** 2 = 0
# p_2 - p_4 - ksi_2 * m_2 ** 2 = 0
#
# "Перепишем систему уравнений"
#
# m_1 + m_2 - m_3 = 0
# p_4 - p_g = 0
# ksi_1 * m_1 ** 2 = p_1 - p_4
# ksi_3 * m_3 ** 2 = p_4 - p_3
# ksi_2 * m_2 ** 2 = p_2 - p_4

# p_4 = Symbol('p_4')
# m_1 = Symbol('m_1')
# m_2 = Symbol('m_2')
# m_3 = Symbol('m_3')
# p_g = Symbol('p_g')
# ssss = solve([m_1 + m_2 - m_3,
#               p_4 - p_g,
#               p_1 - p_4 - ksi_1 * m_1 ** 2,
#               p_4 - p_3 - ksi_3 * m_3 ** 2,
#               p_2 - p_4 - ksi_2 * m_2 ** 2], [p_4, m_1, m_2, m_3, p_g])
# print(ssss)

p_4_0 = 289442.71909999158785636694674925104941762473438446102897
m_1_0 = 10.51462224238267212051338169695753214570995864486684
m_2_0 = 3.24919696232906326155871412215134464954903471521475
m_3_0 = 13.76381920471173538207209581910887679525899336008159
p_g_0 = 289442.719_099_991_587_856_366_946_749_251_049_417_624_734_384_461_028_970

# p_4_0 = 289442.71909999
# m_1_0 = 10.5146222
# m_2_0 = 3.24919696
# m_3_0 = 13.7638192
# p_g_0 = 289442.71909999

# print(round(40000*sqrt(5) + 200000, 50))
# print(round((-10 + 6*sqrt(5))*sqrt(2*sqrt(5) + 5), 50))
# print(round((10 - 4*sqrt(5))*sqrt(2*sqrt(5) + 5), 50))
# print(round(sqrt(40*sqrt(5) + 100), 50))
# print(round(40000*sqrt(5) + 200000, 50))

l_1, l_2, l_3, l_dem = 1, 1, 1, 0.02
S_1, S_2, S_3, S_dem = 0.0025, 0.0025, 0.0025, 0.001
T_g_0 = 293
k = 1.4
R_gas = 287
R_liq = 287
ro_gas_0 = p_g_0 / R_gas / T_g_0 # Плотность газа
ro_liq_0 = 820 # Плотность жидкости
# print(ro_gas_0)
m_dem_0 = 0 # Это расход демпфера
V_dem_0 = 0 # Это расход демпфера
m_bes_tochki_dem_0 = 0 # Масса демпфера
V_bes_tochki_dem_0 = 0 # Объём демпфера

V_bes_tochki_gas_0 = 0.001 # Это объём газа !!!
m_bes_tochki_gas_0 = p_g_0 * V_bes_tochki_gas_0 / T_g_0 / R_gas  # Масса газа
# print(m_bes_tochki_gas_0)

V_liq_0 = 0.004 # Объём жидкости
m_liq_0 = V_liq_0 * ro_liq_0
# print(m_liq_0)
V = V_liq_0 + V_bes_tochki_gas_0 # Суммарный объём
C_liq = 1218 # Скорость звука в жидкости
alpha_0 = (V_liq_0 + V_bes_tochki_dem_0) / (C_liq ** 2)
p_4_1 = p_4_0

j_1 = l_1 / S_1
j_2 = l_2 / S_2
j_3 = l_3 / S_3
j_dem = l_dem / S_dem
# print(j_2, j_dem)

# j_1, j_2, j_3, j_dem = 0.01, 0.01, 0.01, 0.01

# t = Symbol('t')
# m_dem = symbols('m_dem', cls= Function)
# p_4 = symbols('p_4', cls= Function)
# m_3 = symbols('m_3', cls= Function)
# m_2 = symbols('m_2', cls= Function)
# m_1 = symbols('m_1', cls= Function)
# V_dem = symbols('V_dem', cls= Function)
#
# m_dem = Function('m_dem')
# p_4 = Function('p_4')
# m_3 = Function('m_3')
# m_2 = Function('m_2')
# m_1 = Function('m_1')
# V_dem = Function('V_dem')

# m_dem(0).diff(t) == 0
# ics = {m_dem(0): 0}
# Ny = m_dem(0) == 0, p_4(0) == p_4_0, m_3(0) == m_3_0, m_2(0) == m_2_0, m_1(0) == m_1_0, V_dem(0) == 0

# sistema = (Eq(m_dem(t).diff(t), (p_4(t) - p_2 - ksi_dem * m_dem(t) ** 2) / j_dem),
#            Eq(m_1(t).diff(t), (p_1 - p_4(t) - ksi_1 * m_1(t)) / j_1),
#            Eq(m_3(t).diff(t), (p_4(t) - p_3 - ksi_3 * m_3(t)) / j_3),
#            Eq(m_2(t).diff(t), (p_2 - p_4(t) - ksi_2 * m_2(t)) / j_2),
#            Eq(p_4(t).diff(t), (m_1(t) + m_2(t) - m_3(t) - m_dem(t)) / alpha),
#            Eq(V_dem(t).diff(t), m_dem(t) / ro)
#            )
# sss = dsolve([sistema], [m_dem(0) == 0, p_4(0) == p_4_0, m_3(0) == m_3_0, m_2(0) == m_2_0, m_1(0) == m_1_0, V_dem(0) == 0])
# print(sss)

# ics = {m_dem(0) : 0, p_4(0) : p_4_0, m_3(0) : m_3_0, m_2(0) : m_2_0, m_1(0) : m_1_0, V_dem(0) : 0}

# eqns = [diff(m_dem(t), t) - (p_4(t) - p_2 - ksi_dem * (m_dem(t)) ** 2) / j_dem,
#         diff(m_1(t), t) - (p_1 - p_4(t) - ksi_1 * m_1(t)) / j_1,
#         diff(m_3(t), t) - (p_4(t) - p_3 - ksi_3 * m_3(t)) / j_3,
#         diff(m_2(t), t) - (p_2 - p_4(t) - ksi_2 * m_2(t)) / j_2,
#         diff(p_4(t), t) - (m_1(t) + m_2(t) - m_3(t) - m_dem(t)) / alpha,
#         diff(V_dem(t), t) - m_dem(t) / ro]

#
# sss = dsolve((diff(m_dem(t), t) - (p_4_0 - p_2 - ksi_dem * (m_dem(t)) ** 2) / j_dem,
#               diff(m_1(t), t) - (p_1 - p_4_0 - ksi_1 * m_1(t)) / j_1), ics = {m_dem(0): 0, m_1(0): m_1_0})

# print(sss)

# x = Function('x')
# p = Function('p')
# print(dsolve((diff(m_dem(t), t) - (p_4(t) - p_2 - ksi_dem * (m_dem(t)) ** 2) / j_dem,
#               diff(m_1(t), t) - (p_1 - p_4(t) - ksi_1 * m_1(t)) / j_1,
#               )))
#
# dsolve(
#     (diff(m_dem(t), t) - (p_4_0 - p_2 - ksi_dem * (m_dem(t)) ** 2) / j_dem,
#      diff(m_1(t), t) - (p_1 - p_4_0 - ksi_1 * m_1(t)) / j_1), hint='all', ics={m_dem(0): 0, m_1(0): m_1_0})

t_start = 0
t_end = 0.5
dt = 0.00001
T = 0.1

V_dem   = np.array(())
V_liq   = np.array(())
V_bes_tochki_gas = np.array(())
m_dem   = np.array(())
m_1     = np.array(())
m_3     = np.array(())
m_2     = np.array(())
p_4     = np.array(())
T_g     = np.array(())
t       = np.array(())
p_g     = np.array(())
ksi_1_t = np.array(())
delta_V_dem_1 = np.array(())
delta_m_1_1 =  np.array(())

for i in np.arange(t_start, t_end, dt):

    def ksi_1_t_t(t):
        return ksi_1 - 0.4 * ksi_1 * (np.sin(2 * np.pi * (t - T / 4) / T) + 1)

    if i <= T / 2:
        ksi_1_f = ksi_1_t_t(i)
    else:
        ksi_1_f = ksi_1_t_t(T / 2)

    T_g_0    = T_g_0 * (p_4_0 / p_4_1) ** ((k - 1) / k)
    p_g_0    = m_bes_tochki_gas_0 / V_bes_tochki_gas_0 * R_gas * T_g_0

    dV_dem_1 = dt * (m_dem_0 / ro_liq_0)
    dm_dem_1 = dt * (p_4_0 - p_g_0 - ksi_dem * m_dem_0 * abs(m_dem_0)) / j_dem
    # dm_1_1   = dt * (p_1 - p_4_0 - ksi_1 * m_1_0 * abs(m_1_0)) / j_1
    dm_1_1   = dt * (p_1 - p_4_0 - ksi_1_f * m_1_0 * abs(m_1_0)) / j_1
    dm_3_1   = dt * (p_4_0 - p_3 - ksi_3 * m_3_0 * abs(m_3_0)) / j_3
    dm_2_1   = dt * (p_2 - p_4_0 - ksi_2 * m_2_0 * abs(m_2_0)) / j_2
    dp_4_1   = dt * (m_1_0 + m_2_0 - m_3_0 - m_dem_0) / alpha_0

    dV_dem_2 = dt * ((m_dem_0 + dm_dem_1 / 2) / ro_liq_0)
    dm_dem_2 = dt * (p_4_0 + dp_4_1 / 2 - p_g_0 - ksi_dem * (m_dem_0 + dm_dem_1 / 2) * abs(m_dem_0 + dm_dem_1 / 2)) / j_dem
    # dm_1_2   = dt * (p_1 - (p_4_0 + dp_4_1 / 2) - ksi_1 * (m_1_0 + dm_1_1 / 2) * abs(m_1_0 + dm_1_1 / 2)) / j_1
    dm_1_2   = dt * (p_1 - (p_4_0 + dp_4_1 / 2) - ksi_1_f * (m_1_0 + dm_1_1 / 2) * abs(m_1_0 + dm_1_1 / 2)) / j_1
    dm_3_2   = dt * (p_4_0 + dp_4_1 / 2 - p_3 - ksi_3 * (m_3_0 + dm_3_1 / 2) * abs(m_3_0 + dm_3_1 / 2)) / j_3
    dm_2_2   = dt * (p_2 - (p_4_0 + dp_4_1 / 2) - ksi_2 * (m_2_0 + dm_2_1 / 2) * abs(m_2_0 + dm_2_1 / 2)) / j_2
    dp_4_2   = dt * (m_1_0 + dm_1_1 / 2 + m_2_0 + dm_2_1 / 2 - (m_3_0 + dm_3_1 / 2) - (m_dem_0 + dm_dem_1 / 2)) / alpha_0

    dV_dem_3 = dt * ((m_dem_0 + dm_dem_2 / 2) / ro_liq_0)
    dm_dem_3 = dt * (p_4_0 + dp_4_2 / 2 - p_g_0 - ksi_dem * (m_dem_0 + dm_dem_2 / 2) * abs(m_dem_0 + dm_dem_2 / 2)) / j_dem
    # dm_1_3   = dt * (p_1 - (p_4_0 + dp_4_2 / 2) - ksi_1 * (m_1_0 + dm_1_2 / 2) * abs(m_1_0 + dm_1_2 / 2)) / j_1
    dm_1_3   = dt * (p_1 - (p_4_0 + dp_4_2 / 2) - ksi_1_f * (m_1_0 + dm_1_2 / 2) * abs(m_1_0 + dm_1_2 / 2)) / j_1
    dm_3_3   = dt * (p_4_0 + dp_4_2 / 2 - p_3 - ksi_3 * (m_3_0 + dm_3_2 / 2) * abs(m_3_0 + dm_3_2 / 2)) / j_3
    dm_2_3   = dt * (p_2 - (p_4_0 + dp_4_2 / 2) - ksi_2 * (m_2_0 + dm_2_2 / 2) * abs(m_2_0 + dm_2_2 / 2)) / j_2
    dp_4_3   = dt * (m_1_0 + dm_1_2 / 2 + m_2_0 + dm_2_2 / 2 - (m_3_0 + dm_3_2 / 2) - (m_dem_0 + dm_dem_2 / 2)) / alpha_0

    dV_dem_4 = dt * ((m_dem_0 + dm_dem_3) / ro_liq_0)
    dm_dem_4 = dt * (p_4_0 + dp_4_3 - p_g_0 - ksi_dem * (m_dem_0 + dm_dem_3) * abs(m_dem_0 + dm_dem_3)) / j_dem
    # dm_1_4   = dt * (p_1 - (p_4_0 + dp_4_3) - ksi_1 * (m_1_0 + dm_1_3) * abs(m_1_0 + dm_1_3)) / j_1
    dm_1_4   = dt * (p_1 - (p_4_0 + dp_4_3) - ksi_1_f * (m_1_0 + dm_1_3) * abs(m_1_0 + dm_1_3)) / j_1
    dm_3_4   = dt * (p_4_0 + dp_4_3 - p_3 - ksi_3 * (m_3_0 + dm_3_3) * abs(m_3_0 + dm_3_3)) / j_3
    dm_2_4   = dt * (p_2 - (p_4_0 + dp_4_3) - ksi_2 * (m_2_0 + dm_2_3) * abs(m_2_0 + dm_2_3)) / j_2
    dp_4_4   = dt * (m_1_0 + dm_1_3 + m_2_0 + dm_2_3 - (m_3_0 + dm_3_3) - (m_dem_0 + dm_dem_3)) / alpha_0

    delta_V_dem = (dV_dem_1 + 2 * dV_dem_2 + 2 * dV_dem_3 + dV_dem_4) / 6
    delta_m_dem = (dm_dem_1 + 2 * dm_dem_2 + 2 * dm_dem_3 + dm_dem_4) / 6
    delta_m_1   = (dm_1_1 + 2 * dm_1_2 + 2 * dm_1_3 + dm_1_4) / 6
    delta_m_3   = (dm_3_1 + 2 * dm_3_2 + 2 * dm_3_3 + dm_3_4) / 6
    delta_m_2   = (dm_2_1 + 2 * dm_2_2 + 2 * dm_2_3 + dm_2_4) / 6
    delta_p_4   = (dp_4_1 + 2 * dp_4_2 + 2 * dp_4_3 + dp_4_4) / 6

    p_4_1    = p_4_0
    V_dem_0  = V_dem_0 + delta_V_dem
    V_liq_i  = V_liq_0 + V_dem_0
    V_bes_tochki_gas_0 = V - V_liq_i
    alpha_0  = V_liq_i / (C_liq ** 2)
    m_dem_0  = m_dem_0 + delta_m_dem
    m_1_0    = m_1_0 + delta_m_1
    m_3_0    = m_3_0 + delta_m_3
    m_2_0    = m_2_0 + delta_m_2
    p_4_0    = p_4_0 + delta_p_4

    delta_V_dem_1 = np.append(delta_V_dem_1, delta_V_dem)
    p_g    = np.append(p_g, p_g_0)
    V_dem  = np.append(V_dem, V_dem_0)
    V_liq  = np.append(V_liq, V_liq_i)
    V_bes_tochki_gas = np.append(V_bes_tochki_gas, V_bes_tochki_gas_0)
    m_dem  = np.append(m_dem, m_dem_0)
    m_1    = np.append(m_1, m_1_0)
    m_3    = np.append(m_3, m_3_0)
    m_2    = np.append(m_2, m_2_0)
    p_4    = np.append(p_4, p_4_0)
    T_g    = np.append(T_g, T_g_0)
    t      = np.append(t, i)
    ksi_1_t= np.append(ksi_1_t, ksi_1_f)
    delta_m_1_1 = np.append(delta_m_1_1, delta_m_1)

    # print(m_dem_0)
# print(p_4)

# for i in np.arange(t_start, t_end, dt):
#
#     def ksi_1_t_t(t):
#         return ksi_1 + 0.4 * ksi_1 * (np.sin(2 * np.pi * (t - T / 4) / T) + 1)
#
#     def fun_alpha(x):
#         return (V_liq_0 + x) / C_liq ** 2
#
#     T_g_0 = T_g_0 * (p_4_0 / p_4_1) ** ((k - 1) / k)
#
#     def fun_T_g_0(x):
#         return T_g_0 * ((p_4_0 + x) / p_4_1) ** ((k - 1) / k)
#
#     if i <= T / 2:
#         ksi_1_f = ksi_1_t_t(i)
#     else:
#         ksi_1_f = ksi_1_t_t(T / 2)
#
#     # p_g_0    = m_bes_tochki_gas_0 / V_bes_tochki_gas_0 * R_gas * T_g_0
#
#     dV_dem_1 = dt * (m_dem_0 / ro_liq_0)
#     dm_dem_1 = dt * (p_4_0 - (m_bes_tochki_gas_0 / (V - (V_liq_0)) * R_gas * fun_T_g_0(0)) - ksi_dem * m_dem_0 * abs(m_dem_0)) / j_dem
#     # dm_1_1   = dt * (p_1 - p_4_0 - ksi_1 * m_1_0 * abs(m_1_0)) / j_1
#     dm_1_1   = dt * (p_1 - p_4_0 - ksi_1_f * m_1_0 * abs(m_1_0)) / j_1
#     dm_3_1   = dt * (p_4_0 - p_3 - ksi_3 * m_3_0 * abs(m_3_0)) / j_3
#     dm_2_1   = dt * (p_2 - p_4_0 - ksi_2 * m_2_0 * abs(m_2_0)) / j_2
#     dp_4_1   = dt * (m_1_0 + m_2_0 - m_3_0 - m_dem_0) / fun_alpha(0)
#
#     dV_dem_2 = dt * ((m_dem_0 + dm_dem_1 / 2) / ro_liq_0)
#     dm_dem_2 = dt * (p_4_0 + dp_4_1 / 2 - (m_bes_tochki_gas_0 / (V - (V_liq_0 + dV_dem_1 / 2)) * R_gas * fun_T_g_0(dp_4_1 / 2)) - ksi_dem *
#                      (m_dem_0 + dm_dem_1 / 2) * abs(m_dem_0 + dm_dem_1 / 2)) / j_dem
#     # dm_1_2   = dt * (p_1 - p_4_0 + dp_4_1 / 2 - ksi_1 * (m_1_0 + dm_1_1 / 2) * abs(m_1_0 + dm_1_1 / 2)) / j_1
#     dm_1_2   = dt * (p_1 - (p_4_0 + dp_4_1 / 2) - ksi_1_f * (m_1_0 + dm_1_1 / 2) * abs(m_1_0 + dm_1_1 / 2)) / j_1
#     dm_3_2   = dt * (p_4_0 + dp_4_1 / 2 - p_3 - ksi_3 * (m_3_0 + dm_3_1 / 2) * abs(m_3_0 + dm_3_1 / 2)) / j_3
#     dm_2_2   = dt * (p_2 - (p_4_0 + dp_4_1 / 2) - ksi_2 * (m_2_0 + dm_2_1 / 2) * abs(m_2_0 + dm_2_1 / 2)) / j_2
#     dp_4_2   = dt * (m_1_0 + dm_1_1 / 2 + m_2_0 + dm_2_1 / 2 - (m_3_0 + dm_3_1 / 2) - (m_dem_0 + dm_dem_1 / 2)) / fun_alpha(dV_dem_1 / 2)
#
#     dV_dem_3 = dt * ((m_dem_0 + dm_dem_2 / 2) / ro_liq_0)
#     dm_dem_3 = dt * (p_4_0 + dp_4_2 / 2 - (m_bes_tochki_gas_0 / (V - (V_liq_0 + dV_dem_2 / 2)) * R_gas * fun_T_g_0(dp_4_2 / 2)) - ksi_dem
#                      * (m_dem_0 + dm_dem_2 / 2) * abs(m_dem_0 + dm_dem_2 / 2)) / j_dem
#     # dm_1_3   = dt * (p_1 - p_4_0 + dp_4_2 / 2 - ksi_1 * (m_1_0 + dm_1_2 / 2) * abs(m_1_0 + dm_1_2 / 2)) / j_1
#     dm_1_3   = dt * (p_1 - (p_4_0 + dp_4_2 / 2) - ksi_1_f * (m_1_0 + dm_1_2 / 2) * abs(m_1_0 + dm_1_2 / 2)) / j_1
#     dm_3_3   = dt * (p_4_0 + dp_4_2 / 2 - p_3 - ksi_3 * (m_3_0 + dm_3_2 / 2) * abs(m_3_0 + dm_3_2 / 2)) / j_3
#     dm_2_3   = dt * (p_2 - (p_4_0 + dp_4_2 / 2) - ksi_2 * (m_2_0 + dm_2_2 / 2) * abs(m_2_0 + dm_2_2 / 2)) / j_2
#     dp_4_3   = dt * (m_1_0 + dm_1_2 / 2 + m_2_0 + dm_2_2 / 2 - (m_3_0 + dm_3_2 / 2) - (m_dem_0 + dm_dem_2 / 2)) / fun_alpha(dV_dem_2 / 2)
#
#     dV_dem_4 = dt * ((m_dem_0 + dm_dem_3) / ro_liq_0)
#     dm_dem_4 = dt * (p_4_0 + dp_4_3 - (m_bes_tochki_gas_0 / (V - (V_liq_0 + dV_dem_3)) * R_gas * fun_T_g_0(dp_4_3)) - ksi_dem
#                      * (m_dem_0 + dm_dem_3) * abs(m_dem_0 + dm_dem_3)) / j_dem
#     # dm_1_4   = dt * (p_1 - p_4_0 + dp_4_3 - ksi_1 * (m_1_0 + dm_1_3) * abs(m_1_0 + dm_1_3)) / j_1
#     dm_1_4   = dt * (p_1 - (p_4_0 + dp_4_3) - ksi_1_f * (m_1_0 + dm_1_3) * abs(m_1_0 + dm_1_3)) / j_1
#     dm_3_4   = dt * (p_4_0 + dp_4_3 - p_3 - ksi_3 * (m_3_0 + dm_3_3) * abs(m_3_0 + dm_3_3)) / j_3
#     dm_2_4   = dt * (p_2 - (p_4_0 + dp_4_3) - ksi_2 * (m_2_0 + dm_2_3) * abs(m_2_0 + dm_2_3)) / j_2
#     dp_4_4   = dt * (m_1_0 + dm_1_3 + m_2_0 + dm_2_3 - (m_3_0 + dm_3_3) - (m_dem_0 + dm_dem_3)) / fun_alpha(dV_dem_3)
#
#     delta_V_dem = (dV_dem_1 + 2 * dV_dem_2 + 2 * dV_dem_3 + dV_dem_4) / 6
#     delta_m_dem = (dm_dem_1 + 2 * dm_dem_2 + 2 * dm_dem_3 + dm_dem_4) / 6
#     delta_m_1   = (dm_1_1 + 2 * dm_1_2 + 2 * dm_1_3 + dm_1_4) / 6
#     delta_m_3   = (dm_3_1 + 2 * dm_3_2 + 2 * dm_3_3 + dm_3_4) / 6
#     delta_m_2   = (dm_2_1 + 2 * dm_2_2 + 2 * dm_2_3 + dm_2_4) / 6
#     delta_p_4   = (dp_4_1 + 2 * dp_4_2 + 2 * dp_4_3 + dp_4_4) / 6
#
#     p_4_1    = p_4_0
#     V_dem_0  = V_dem_0 + delta_V_dem
#     V_liq_0  = V_liq_0 + V_dem_0
#     V_bes_tochki_gas_0 = V - V_liq_0
#     # alpha_0  = V_liq_i / (C_liq ** 2)
#     m_dem_0  = m_dem_0 + delta_m_dem
#     m_1_0    = m_1_0 + delta_m_1
#     m_3_0    = m_3_0 + delta_m_3
#     m_2_0    = m_2_0 + delta_m_2
#     p_4_0    = p_4_0 + delta_p_4
#
#     delta_V_dem_1 = np.append(delta_V_dem_1, delta_p_4)
#     p_g    = np.append(p_g, p_g_0)
#     V_dem  = np.append(V_dem, V_dem_0)
#     V_liq  = np.append(V_liq, V_liq_0)
#     V_bes_tochki_gas = np.append(V_bes_tochki_gas, V_bes_tochki_gas_0)
#     m_dem  = np.append(m_dem, m_dem_0)
#     m_1    = np.append(m_1, m_1_0)
#     m_3    = np.append(m_3, m_3_0)
#     m_2    = np.append(m_2, m_2_0)
#     p_4    = np.append(p_4, p_4_0)
#     T_g    = np.append(T_g, T_g_0)
#     t      = np.append(t, i)
#     ksi_1_t= np.append(ksi_1_t, ksi_1_f)
#     delta_m_1_1 = np.append(delta_m_1_1, delta_m_1)
#
#     # print(m_dem_0)
# # print(p_4)


name='m_2'
plt.figure(name)
plt.xlabel('t, с') # Ось х
plt.ylabel('m_2, кг/с') # Ось у
# plt.plot(m_dem_0,a_1, label=r"Левая часть уравнения",color="red") # Кривая 1
plt.plot(t, m_2,color="green") # Кривая 2
# plt.legend(loc='best') # Где сделать надписи
plt.grid() # Добавить сетку
plt.show()

name='m_1'
plt.figure(name)
plt.xlabel('t, с') # Ось х
plt.ylabel('m_1, кг/с') # Ось у
# plt.plot(m_dem_0,a_1, label=r"Левая часть уравнения",color="red") # Кривая 1
plt.plot(t, m_1,color="red") # Кривая 2
# plt.legend(loc='best') # Где сделать надписи
plt.grid() # Добавить сетку
plt.show()

name='V_liq'
plt.figure(name)
plt.xlabel('t, с') # Ось х
plt.ylabel('V_liq, кг/м^3') # Ось у
# plt.plot(m_dem_0,a_1, label=r"Левая часть уравнения",color="red") # Кривая 1
plt.plot(t, V_liq,color="red") # Кривая 2
# plt.legend(loc='best') # Где сделать надписи
plt.grid() # Добавить сетку
plt.show()

name='p_g'
plt.figure(name)
plt.xlabel('t, с') # Ось х
plt.ylabel('p_g, Па') # Ось у
# plt.plot(m_dem_0,a_1, label=r"Левая часть уравнения",color="red") # Кривая 1
plt.plot(t, p_g,color="violet") # Кривая 2
# plt.legend(loc='best') # Где сделать надписи
plt.grid() # Добавить сетку
plt.show()

name='V_dem'
plt.figure(name)
plt.xlabel('t, с') # Ось х
plt.ylabel('V_dem, кг/(м^3 * c)') # Ось у
# plt.plot(m_dem_0,a_1, label=r"Левая часть уравнения",color="red") # Кривая 1
plt.plot(t, V_dem,color="blue") # Кривая 2
# plt.legend(loc='best') # Где сделать надписи
plt.grid() # Добавить сетку
plt.show()

name='V_bes_tochki_gas'
plt.figure(name)
plt.xlabel('t, с') # Ось х
plt.ylabel('V_bes_tochki_gas, м3') # Ось у
# plt.plot(m_dem_0,a_1, label=r"Левая часть уравнения",color="red") # Кривая 1
plt.plot(t, V_bes_tochki_gas,color="blue") # Кривая 2
# plt.legend(loc='best') # Где сделать надписи
plt.grid() # Добавить сетку
plt.show()

name='T_g'
plt.figure(name)
plt.xlabel('t, с') # Ось х
plt.ylabel('T_g, К') # Ось у
# plt.plot(m_dem_0,a_1, label=r"Левая часть уравнения",color="red") # Кривая 1
plt.plot(t, T_g,color="blue") # Кривая 2
# plt.legend(loc='best') # Где сделать надписи
plt.grid() # Добавить сетку
plt.show()

name='p_4'
plt.figure(name)
plt.xlabel('t, с') # Ось х
plt.ylabel('p_4, Па') # Ось у
# plt.plot(m_dem_0,a_1, label=r"Левая часть уравнения",color="red") # Кривая 1
plt.plot(t, p_4,color="blue") # Кривая 2
# plt.legend(loc='best') # Где сделать надписи
plt.grid() # Добавить сетку
plt.show()

name='ksi_1_t'
plt.figure(name)
plt.xlabel('t, с') # Ось х
plt.ylabel('ksi_1_t') # Ось у
# plt.plot(m_dem_0,a_1, label=r"Левая часть уравнения",color="red") # Кривая 1
plt.plot(t, ksi_1_t,color="green") # Кривая 2
# plt.legend(loc='best') # Где сделать надписи
plt.grid() # Добавить сетку
plt.show()

name='m_3'
plt.figure(name)
plt.xlabel('t, с') # Ось х
plt.ylabel('m_3, кг/с') # Ось у
# plt.plot(m_dem_0,a_1, label=r"Левая часть уравнения",color="red") # Кривая 1
plt.plot(t, m_3,color="green") # Кривая 2
# plt.legend(loc='best') # Где сделать надписи
plt.grid() # Добавить сетку
plt.show()

name='m_dem'
plt.figure(name)
plt.xlabel('t, с') # Ось х
plt.ylabel('m_dem, кг/с') # Ось у
# plt.plot(m_dem_0,a_1, label=r"Левая часть уравнения",color="red") # Кривая 1
plt.plot(t, m_dem,color="green") # Кривая 2
# plt.legend(loc='best') # Где сделать надписи
plt.grid() # Добавить сетку
plt.show()