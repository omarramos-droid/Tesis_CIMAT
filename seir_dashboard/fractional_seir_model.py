import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import math

# ---------- CONFIGURACIÓN DE LA PÁGINA (debe ir al principio) ----------
st.set_page_config(page_title="SEIR Fraccionario", layout="wide")

# ---------- SOLVER FRACCIONARIO (ABM) ----------
def solver_fraccionario(f, alpha, y0, T, h):
    N = int(T / h)
    t_fine = np.linspace(0, T, N+1)
    dim = len(y0)
    y = np.zeros((N+1, dim))
    fvals = np.zeros((N+1, dim))
    y[0] = y0
    fvals[0] = f(t_fine[0], y[0])

    gamma1 = math.gamma(alpha + 1)
    gamma2 = math.gamma(alpha + 2)

    for n in range(1, N+1):
        b = [(n-j)**alpha - (n-j-1)**alpha for j in range(n)]
        sum_pred = np.zeros(dim)
        for j in range(n):
            sum_pred += b[j] * fvals[j]
        y_pred = y[0] + (h**alpha/gamma1)*sum_pred
        f_pred = f(t_fine[n], y_pred)

        a = [(n-j+1)**(alpha+1) - 2*(n-j)**(alpha+1) + (n-j-1)**(alpha+1) for j in range(n)] + [1.0]
        sum_corr = np.zeros(dim)
        for j in range(n):
            sum_corr += a[j]*fvals[j]
        sum_corr += a[n]*f_pred

        y[n] = y[0] + (h**alpha/gamma2)*sum_corr
        y[n] = np.maximum(y[n], 0)
        fvals[n] = f(t_fine[n], y[n])

    return t_fine, y

# ---------- MODELO SEIR ----------
def seir_rhs(t, Y, beta, gamma, kappa, N):
    S, E, I, R = Y
    dS = -beta*S*I/N
    dE = beta*S*I/N - kappa*E
    dI = kappa*E - gamma*I
    dR = gamma*I
    return np.array([dS, dE, dI, dR])

# ---------- PARÁMETROS FIJOS ----------
beta = 2.5
gamma = 0.1
kappa = 0.5
N = 2000
I0 = 10
E0 = 2
S0 = N - I0 - E0
y0 = [S0, E0, I0, 0]
h = 0.1
T = 52

# ---------- SOLUCIÓN CLÁSICA (α=1) CACHEADA ----------
@st.cache_data
def solucion_clasica():
    t, sol = solver_fraccionario(
        lambda t, Y: seir_rhs(t, Y, beta, gamma, kappa, N),
        1.0, y0, T, h
    )
    return t, sol[:, 2]  # solo infectados

t_classic, I_classic = solucion_clasica()

# ---------- INTERFAZ ----------
st.title("📈 Memoria en un modelo SEIR fraccionario")
st.markdown("Mueve el deslizador para cambiar el orden fraccionario **α** y observa cómo se comportan los infectados.")

alpha = st.slider("Orden fraccionario α", 0.10, 1.00, 0.80, 0.01)

# Calcular solo la solución para el α elegido
with st.spinner("Calculando trayectoria fraccionaria..."):
    t_frac, sol_frac = solver_fraccionario(
        lambda t, Y: seir_rhs(t, Y, beta, gamma, kappa, N),
        alpha, y0, T, h
    )
    I_frac = sol_frac[:, 2]

# Métricas
col1, col2 = st.columns(2)
col1.metric("Pico de infectados", f"{np.max(I_frac):.0f}")
col2.metric("Tiempo del pico", f"{t_frac[np.argmax(I_frac)]:.1f}")

# Gráfica
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(t_classic, I_classic, '--', linewidth=2, label='α = 1 (clásico)')
ax.plot(t_frac, I_frac, linewidth=3, label=f'α = {alpha:.2f}')
ax.set_xlabel("Tiempo")
ax.set_ylabel("Infectados")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Indicador de memoria
if alpha > 0.8:
    st.success("🧠 Memoria débil – el sistema olvida rápido")
elif alpha > 0.5:
    st.warning("⏳ Memoria moderada")
else:
    st.error("🔁 Memoria fuerte – los efectos persisten más tiempo")