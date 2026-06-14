import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.interpolate import interp1d

# ---------- Parámetros fijos del mejor ajuste ----------
BETA_FIT   = 4.56
I0_FIT     = 2878
ALPHA_FIT  = 0.3455

GAMMA = 0.875
KAPPA = 0.5
E0_FIJO = 0.0
N = 8_800_000 * (1 - 0.824)   # población susceptible total

# ---------- Datos reales (incrustados) ----------
y_obs = np.array([172, 190, 231, 425, 495, 563, 746, 520, 520,
                  474, 273, 310, 153, 269, 171, 223, 180, 138, 98, 70])
t_data = np.arange(0, len(y_obs) + 1, 1)   # 0,1,2,...,20

# ---------- Solvers ----------
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

def solve_seir_frac(beta, I0, alpha, t_eval, h=0.05):
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

# ---------- Interfaz ----------
st.set_page_config(page_title="Sarampión – SEIR Fraccionario", layout="wide")
st.title("Sarampión Jalisco: efecto de la memoria (α) y subreporte (p)")
st.markdown("""
Mueve los deslizadores para cambiar **α** (orden fraccionario) y **p** (fracción de casos reportados).  
La línea roja muestra los casos predichos con esos valores, comparada con los datos reales (puntos negros).
""")

alpha = st.slider("Orden fraccionario α", 0.10, 1.00, ALPHA_FIT, 0.01)
p_usuario = st.slider("Fracción de reporte (p)", 0.0001, 0.5, 0.0069, 0.0001, format="%.4f")

# Calcular solución con α y p elegidos
with st.spinner("Calculando..."):
    sol_frac = solve_seir_frac(BETA_FIT, I0_FIT, alpha, t_data)
    inc_frac = np.maximum(sol_frac[:-1, 0] - sol_frac[1:, 0], 0)   # longitud = len(t_data)-1
    pred_frac = p_usuario * inc_frac

# Métricas
col1, col2 = st.columns(2)
col1.metric("Pico predicho (casos)", f"{np.max(pred_frac):.1f}")
col2.metric("Semana del pico", f"{np.argmax(pred_frac) + 1}")   # +1 porque el pico está en semana 1..20

# Gráfica
fig, ax = plt.subplots(figsize=(10, 5))
# Datos observados en semanas 1..20 (la incidencia corresponde a cada intervalo)
ax.plot(np.arange(1, len(y_obs)+1), y_obs, 'ko', markersize=4, label='Datos observados')
# Predicción en las mismas semanas
ax.plot(np.arange(1, len(y_obs)+1), pred_frac, 'r-', linewidth=3, label=f'α={alpha:.2f}, p={p_usuario:.4f}')
ax.set_xlabel("Semana")
ax.set_ylabel("Casos reportados")
ax.legend()
ax.grid(alpha=0.3)
st.pyplot(fig)

# Contexto sobre el efecto de α
if alpha > 0.8:
    st.success(" Memoria débil – dinámica casi clásica, pico temprano y descenso rápido.")
elif alpha > 0.5:
    st.warning(" Memoria moderada – cierto retraso y cola más larga.")
else:
    st.error(" Memoria fuerte – el pico aparece más tarde y la epidemia se prolonga.")
