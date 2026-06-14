import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy.interpolate import interp1d
from scipy.integrate import odeint

# Parámetros fijos del mejor ajuste fraccionario 
BETA_FIT   = 4.56     
I0_FIT     = 2878     
P_REPORT   = 0.0069     
ALPHA_FIT  = 0.3455     

# Constantes epidemiológicas 
GAMMA = 0.875
KAPPA = 0.5
E0_FIJO = 0.0
N = 8_800_000 * (1 - 0.824)            

#  Solvers 
def seir_rhs(t, Y, beta, gamma, kappa, N):
    S, E, I, R = Y
    dS = -beta * S * I / N
    dE =  beta * S * I / N - kappa * E
    dI =  kappa * E - gamma * I
    dR =  gamma * I
    return np.array([dS, dE, dI, dR])

def solver_fraccionario(f, alpha, y0, T, h=0.05):
    n_steps = int(round(T / h))
    t = np.linspace(0, T, n_steps + 1)
    dim = len(y0)
    y = np.zeros((n_steps + 1, dim))
    fvals = np.zeros((n_steps + 1, dim))
    y[0] = y0
    fvals[0] = f(t[0], y[0])
    gamma_alpha1 = math.gamma(alpha + 1)
    gamma_alpha2 = math.gamma(alpha + 2)
    h_alpha = h ** alpha

    for n in range(1, n_steps + 1):
        b = np.array([(n - j) ** alpha - (n - j - 1) ** alpha for j in range(n)])
        y_pred = y[0] + (h_alpha / gamma_alpha1) * np.dot(b, fvals[:n])
        y_pred = np.maximum(y_pred, 0)
        f_pred = f(t[n], y_pred)

        a = np.zeros(n + 1)
        a[0] = (n - 1) ** (alpha + 1) - (n - 1 - alpha) * n ** alpha
        for j in range(1, n):
            a[j] = ((n - j + 1) ** (alpha + 1) + (n - j - 1) ** (alpha + 1)
                    - 2 * (n - j) ** (alpha + 1))
        a[n] = 1.0
        y[n] = y[0] + (h_alpha / gamma_alpha2) * (np.dot(a[:n], fvals[:n]) + a[n] * f_pred)
        y[n] = np.maximum(y[n], 0)
        fvals[n] = f(t[n], y[n])

    return t, y

def solve_seir_frac(beta, I0, alpha, t_eval, h=0.1):
    S0 = N - E0_FIJO - I0
    y0 = np.array([S0, E0_FIJO, I0, 0.0])
    T = t_eval[-1]
    t_fine, sol = solver_fraccionario(
        lambda t, Y: seir_rhs(t, Y, beta, GAMMA, KAPPA, N),
        alpha, y0, T, h
    )
    interp = interp1d(t_fine, sol, axis=0, kind='linear',
                      bounds_error=False, fill_value=(sol[0], sol[-1]))
    return interp(t_eval)

def solve_seir_classic(beta, I0, t_eval):
    S0 = N - E0_FIJO - I0
    y0 = [S0, E0_FIJO, I0, 0.0]
    def deriv(y, t):
        S, E, I, R = y
        dS = -beta * S * I / N
        dE =  beta * S * I / N - KAPPA * E
        dI =  KAPPA * E - GAMMA * I
        dR =  GAMMA * I
        return [dS, dE, dI, dR]
    return odeint(deriv, y0, t_eval)


y_obs = np.array([172, 190, 231, 425, 495,
                      563, 746, 520, 520, 474, 273, 310, 153, 269, 171, 223, 180, 138, 98, 70])
t_data = np.arange(0, len(y_obs) + 1, 1)   # 0,1,2,...,20

# ---------- Interfaz ----------
st.set_page_config(page_title="Sarampión – SEIR Fraccionario", layout="wide")
st.title(" Sarampión Jalisco: efecto de la memoria (α)")
st.markdown("Desliza α para ver cómo cambia la curva de casos predicha (con subreporte). Los demás parámetros se mantienen en los valores del mejor ajuste fraccionario.")

alpha = st.slider("Orden fraccionario α", 0.10, 1.00, ALPHA_FIT, 0.01)
p_usuario = st.slider("Fracción de reporte (p)", 0.0001, 0.5, P_REPORT, 0.001, format="%.4f")
# Calcular solución con α elegido (usando beta e I0 fijos)
with st.spinner("Calculando..."):
    sol_frac = solve_seir_frac(BETA_FIT, I0_FIT, alpha, t_data)
    # Incidencia: disminución de susceptibles
    inc_frac = np.maximum(sol_frac[:-1, 0] - sol_frac[1:, 0], 0)
    pred_frac = P_REPORT * inc_frac

# Referencia clásica (α=1) con los mismos beta e I0
sol_clas = solve_seir_classic(BETA_FIT, I0_FIT, t_data)
inc_clas = np.maximum(sol_clas[:-1, 0] - sol_clas[1:, 0], 0)
pred_clas = P_REPORT * inc_clas

# Referencia con el α óptimo (para comparar)
sol_opt = solve_seir_frac(BETA_FIT, I0_FIT, ALPHA_FIT, t_data)
inc_opt = np.maximum(sol_opt[:-1, 0] - sol_opt[1:, 0], 0)
pred_opt = P_REPORT * inc_opt

# Métricas rápidas
col1, col2 = st.columns(2)
col1.metric("Pico predicho (casos)", f"{np.max(pred_frac):.1f}")
col2.metric("Semana del pico", f"{np.argmax(pred_frac)}")

# Gráfica
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(y_obs, 'ko', markersize=3, label='Datos observados')
ax.plot(pred_opt, '--', color='gray', linewidth=2, label=f'α óptimo ({ALPHA_FIT:.2f})')
ax.plot(pred_clas, ':', color='blue', linewidth=2, label='Clásico (α=1)')
ax.plot(pred_frac, color='red', linewidth=3, label=f'α = {alpha:.2f}')
ax.set_xlabel("Semana")
ax.set_ylabel("Casos reportados")
ax.legend()
ax.grid(alpha=0.3)
st.pyplot(fig)

# Explicación del efecto de α
if alpha > 0.8:
    st.success(" Memoria débil: curva más cercana a la clásica (pico temprano, descenso rápido).")
elif alpha > 0.5:
    st.warning(" Memoria moderada: cierto retardo y cola más larga.")
else:
    st.error(" Memoria fuerte: pico más tardío y cola muy prolongada (efectos duraderos).")
