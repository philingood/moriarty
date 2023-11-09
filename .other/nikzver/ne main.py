from sympy import *
import numpy as np
import matplotlib.pyplot as plt
import main as mn

"Система уравнений описывающая стационарные процессы без демпфера"

ksi_1, ksi_2, ksi_3 = 1000, mn.ksi_2, mn.ksi_3
p_1, p_2, p_3 = mn.p_1, mn.p_2, mn.p_3

# m_1 + m_2 - m_3 = 0
# ksi_1 * m_1 ** 2 = p_1 - p_4
# ksi_3 * m_3 ** 2 = p_4 - p_3
# ksi_2 * m_2 ** 2 = p_2 - p_4

# p_4 = Symbol('p_4')
# m_1 = Symbol('m_1')
# m_2 = Symbol('m_2')
# m_3 = Symbol('m_3')
# ssss = solve([m_1 + m_2 - m_3,
#               p_1 - p_4 - ksi_1 * m_1 ** 2,
#               p_4 - p_3 - ksi_3 * m_3 ** 2,
#               p_2 - p_4 - ksi_2 * m_2 ** 2], [p_4, m_1, m_2, m_3])
# print(ssss)

# print(round(40000*sqrt(5) + 200000, 50))
# print(round((-10 + 6*sqrt(5))*sqrt(2*sqrt(5) + 5), 50))
# print(round((10 - 4*sqrt(5))*sqrt(2*sqrt(5) + 5), 50))
# print(round(2*sqrt(10*sqrt(5) + 25), 50))

p_4_0 = 289442.71909999158785636694674925104941762473438446102897
m_1_0 = 10.51462224238267212051338169695753214570995864486684
m_2_0 = 3.24919696232906326155871412215134464954903471521475
m_3_0 = 13.76381920471173538207209581910887679525899336008159

# p_4_0, m_1_0, m_2_0, m_3_0 = mn.p_4_0, mn.m_1_0, mn.m_2_0, mn.m_3_0

j_1, j_2, j_3 = mn.j_1, mn.j_2, mn.j_3
C_liq   = mn.C_liq
V_liq   = mn.V
alpha_0 = V_liq / (C_liq ** 2)

t_start = mn.dt
t_end   = mn.t_end
dt      = mn.dt
T       = mn.T

m_1     = np.array(())
m_3     = np.array(())
m_2     = np.array(())
p_4     = np.array(())
t       = np.array(())
ksi_1_t = np.array(())
p_1_t_t = np.array(())

for i in np.arange(t_start, t_end, dt):

    def ksi_1_t_t(t):
        return ksi_1 + 0.4 * ksi_1 * (np.sin(2 * np.pi * (t - T / 4) / T) + 1)

    if i <= T / 2:
        ksi_1_f = ksi_1_t_t(i)
    else:
        ksi_1_f = ksi_1_t_t(T / 2)

    # def p_1_t(t):
    #     return p_1 + 0.5 * p_1 * (1 - np.cos(np.pi * t / T))
    #
    # if i <= T:
    #     p_1_t_1 = p_1_t(i)
    # else:
    #     p_1_t_1 = p_1_t(T)

    # dm_1_1 = dt * (p_1_t_1 - p_4_0 - ksi_1 * m_1_0 * abs(m_1_0)) / j_1
    # dm_1_1   = dt * (p_1 - p_4_0 - ksi_1 * m_1_0 * abs(m_1_0)) / j_1
    dm_1_1   = dt * (p_1 - p_4_0 - ksi_1_f * m_1_0 * abs(m_1_0)) / j_1
    dm_3_1   = dt * (p_4_0 - p_3 - ksi_3 * m_3_0 * abs(m_3_0)) / j_3
    dm_2_1   = dt * (p_2 - p_4_0 - ksi_2 * m_2_0 * abs(m_2_0)) / j_2
    dp_4_1   = dt * (m_1_0 + m_2_0 - m_3_0) / alpha_0

    # dm_1_2 = dt * (p_1_t_1 - p_4_0 + dp_4_1 / 2 - ksi_1 * (m_1_0 + dm_1_1 / 2) * abs(m_1_0 + dm_1_1 / 2)) / j_1
    # dm_1_2   = dt * (p_1 - p_4_0 + dp_4_1 / 2 - ksi_1 * (m_1_0 + dm_1_1 / 2) * abs(m_1_0 + dm_1_1 / 2)) / j_1
    dm_1_2   = dt * (p_1 - (p_4_0 + dp_4_1 / 2) - ksi_1_f * (m_1_0 + dm_1_1 / 2) * abs(m_1_0 + dm_1_1 / 2)) / j_1
    dm_3_2   = dt * (p_4_0 + dp_4_1 / 2 - p_3 - ksi_3 * (m_3_0 + dm_3_1 / 2) * abs(m_3_0 + dm_3_1 / 2)) / j_3
    dm_2_2   = dt * (p_2 - (p_4_0 + dp_4_1 / 2) - ksi_2 * (m_2_0 + dm_2_1 / 2) * abs(m_2_0 + dm_2_1 / 2)) / j_2
    dp_4_2   = dt * (m_1_0 + dm_1_1 / 2 + m_2_0 + dm_2_1 / 2 - (m_3_0 + dm_3_1 / 2)) / alpha_0

    # dm_1_3 = dt * (p_1_t_1 - p_4_0 + dp_4_2 / 2 - ksi_1 * (m_1_0 + dm_1_2 / 2) * abs(m_1_0 + dm_1_2 / 2)) / j_1
    # dm_1_3   = dt * (p_1 - p_4_0 + dp_4_2 / 2 - ksi_1 * (m_1_0 + dm_1_2 / 2) * abs(m_1_0 + dm_1_2 / 2)) / j_1
    dm_1_3   = dt * (p_1 - (p_4_0 + dp_4_2 / 2) - ksi_1_f * (m_1_0 + dm_1_2 / 2) * abs(m_1_0 + dm_1_2 / 2)) / j_1
    dm_3_3   = dt * (p_4_0 + dp_4_2 / 2 - p_3 - ksi_3 * (m_3_0 + dm_3_2 / 2) * abs(m_3_0 + dm_3_2 / 2)) / j_3
    dm_2_3   = dt * (p_2 - (p_4_0 + dp_4_2 / 2) - ksi_2 * (m_2_0 + dm_2_2 / 2) * abs(m_2_0 + dm_2_2 / 2)) / j_2
    dp_4_3   = dt * (m_1_0 + dm_1_2 / 2 + m_2_0 + dm_2_2 / 2 - (m_3_0 + dm_3_2 / 2)) / alpha_0

    # dm_1_4 = dt * (p_1_t_1 - p_4_0 + dp_4_3 - ksi_1 * (m_1_0 + dm_1_3) * abs(m_1_0 + dm_1_3)) / j_1
    # dm_1_4   = dt * (p_1 - p_4_0 + dp_4_3 - ksi_1 * (m_1_0 + dm_1_3) * abs(m_1_0 + dm_1_3)) / j_1
    dm_1_4   = dt * (p_1 - (p_4_0 + dp_4_3) - ksi_1_f * (m_1_0 + dm_1_3) * abs(m_1_0 + dm_1_3)) / j_1
    dm_3_4   = dt * (p_4_0 + dp_4_3 - p_3 - ksi_3 * (m_3_0 + dm_3_3) * abs(m_3_0 + dm_3_3)) / j_3
    dm_2_4   = dt * (p_2 - (p_4_0 + dp_4_3) - ksi_2 * (m_2_0 + dm_2_3) * abs(m_2_0 + dm_2_3)) / j_2
    dp_4_4   = dt * (m_1_0 + dm_1_3 + m_2_0 + dm_2_3 - (m_3_0 + dm_3_3)) / alpha_0

    delta_m_1   = (dm_1_1 + 2 * dm_1_2 + 2 * dm_1_3 + dm_1_4) / 6
    delta_m_3   = (dm_3_1 + 2 * dm_3_2 + 2 * dm_3_3 + dm_3_4) / 6
    delta_m_2   = (dm_2_1 + 2 * dm_2_2 + 2 * dm_2_3 + dm_2_4) / 6
    delta_p_4   = (dp_4_1 + 2 * dp_4_2 + 2 * dp_4_3 + dp_4_4) / 6

    m_1_0  = m_1_0 + delta_m_1
    m_3_0  = m_3_0 + delta_m_3
    m_2_0  = m_2_0 + delta_m_2
    p_4_0  = p_4_0 + delta_p_4

    m_1    = np.append(m_1, m_1_0)
    m_3    = np.append(m_3, m_3_0)
    m_2    = np.append(m_2, m_2_0)
    p_4    = np.append(p_4, p_4_0)
    t      = np.append(t, i)
    ksi_1_t= np.append(ksi_1_t, ksi_1_f)
    # p_1_t_t = np.append(p_1_t_t, p_1_t_1)

    # print(m_dem_0)
# print(p_4)

name='m'
plt.figure(name)
plt.xlabel('t, с') # Ось х
plt.ylabel('m, кг/с') # Ось у
plt.plot(t, m_3, label=r"m_3",color="blue") # Кривая 1
plt.plot(t, m_1, label=r"m_1",color="red") # Кривая 1
plt.plot(t, m_2, label=r"m_2",color="green") # Кривая 2
plt.legend(loc='right') # Где сделать надписи
plt.grid() # Добавить сетку
plt.show()

name='p_4'
plt.figure(name)
plt.xlabel('t, с') # Ось х
plt.ylabel('p_4, Па') # Ось у
# plt.plot(t,p_1_t_t, label=r"p_1", color="red") # Кривая 1
plt.plot(t, p_4, label=r"p_4", color="violet") # Кривая 2
# plt.legend(loc='best') # Где сделать надписи
plt.grid() # Добавить сетку
plt.show()
#
# name='ksi_1_t'
# plt.figure(name)
# plt.xlabel('t, с') # Ось х
# plt.ylabel('ksi_1_t') # Ось у
# # plt.plot(m_dem_0,a_1, label=r"Левая часть уравнения",color="red") # Кривая 1
# plt.plot(t, ksi_1_t,color="green") # Кривая 2
# # plt.legend(loc='best') # Где сделать надписи
# plt.grid() # Добавить сетку
# plt.show()
