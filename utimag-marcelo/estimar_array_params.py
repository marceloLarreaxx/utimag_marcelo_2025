import numpy as np
import matplotlib.pyplot as plt
import Simul_campo.helper_funcs as sim

plt.ion()

h1, h2 = 10, 5
c1, c2 = 1.48, 5.9
freq = 5
w = 0.7  # ancho del elemento
pitch = 0.9
nel = 16  # nro de elementos
lda = c1/freq  # longitud de onda en agua
w_total = pitch*nel

# angulo critico onda longitudinal
ang_crit = np.arcsin(c1/c2)

# x2: a donde va a parar el rayo a una prfundidad h2 bajo la interfaz plana
x_crit = np.tan(ang_crit) * h1
x = np.arange(0, x_crit, 0.1)
aux = np.abs(x-x_crit)
i_crit = np.argmin(aux)

sin1 = x/np.hypot(h1, x)
sin2 = (c2/c1)*sin1
cos2 = np.sqrt(1 - sin2**2)
tan2 = sin2/cos2
x2 = x + tan2*h2
x2_crit = x2[i_crit]

# factores de difraccion e interferencia
ang = np.arange(0, 2*ang_crit, 0.001)
ang_grad = 180*ang/np.pi
a1 = sim.factor_difraccion(ang, 0, w, lda)
a2 = sim.factor_interferencia(ang, 0, nel, pitch, lda)

# calcular angulo mitad de altura de lobulo principal, -3 dB
aux = np.abs(a1 - 0.5)
i_ang_hw = np.argmin(aux)
ang_hw = ang[i_ang_hw]
ang_hw_grad = ang_grad[i_ang_hw]
x_hw = np.tan(ang_hw) * h1
# refractar
ang2_hw_grad = np.arcsin((c2/c1)*np.sin(ang_hw))
x2_hw = x_hw + np.tan(ang2_hw_grad) * h2

fig1, ax1 = plt.subplots()
ax1.plot(x2, np.arcsin(sin1)*180/np.pi)
ax1.axhline(ang_hw_grad)
ax1.axvline(w_total)
ax1.grid()
ax1.set_title('profundidad ' + str(h2) + ' mm')
ax1.set_xlabel('x (mm)')
ax1.set_ylabel('deflexión en agua (º)')

fig2, ax2 = plt.subplots()
ax2.plot(ang_grad, a1, color='k')
# ax2.plot(ang_grad, a2)
ax2.plot(ang_grad, a1*a2, color='k')
ax2.axvline(ang_crit*180/np.pi, label='angulo crítico')
ax2.axvline(ang_hw_grad, color='r', label='half width')
ax2.axhline(0.5, color='k')
ax2.grid()
ax2.legend()
ax2.set_xlabel('deflexión en agua (º)')

fig3, ax3 = plt.subplots()
ax3.axhline(-h1, color='k')
ax3.axhline(-h1 - h2, color='k')
ax3.plot([0, x_crit, x2_crit], [0, -h1, -h1-h2], label='angulo crítico')
ax3.plot([0, x_hw, x2_hw], [0, -h1, -h1-h2], color='r', label='half width')
ax3.axvline(w_total, label='ancho apertura', color='g')
ax3.grid()
ax3.legend()