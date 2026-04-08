# Sim Mittag-Leffler 

rmittag_leffler <- function(n, nu, mu) {
  if (nu == 1) {
    return(rexp(n, rate = mu))
  }
  U1 <- runif(n)
  U2 <- runif(n)
  U3 <- runif(n)
  
  sin_pi_U2 <- sin(pi * U2)
  term1 <- sin(nu * pi * U2)
  term2 <- sin((1 - nu) * pi * U2) ^ (1/nu - 1)
  term3 <- sin_pi_U2 ^ (1/nu)
  term4 <- (-log(U3)) ^ (1/nu - 1)
  S_nu <- (term1 * term2) / (term3 * term4)
  
  T <- ((-log(U1)) ^ (1/nu)) / (mu ^ (1/nu)) * S_nu
  return(T)
}

library(MittagLeffleR)

# Función de supervivencia teórica: S(t) = E_nu(-mu * t^nu)
survival_teorica <- function(t, nu, mu) {
  # Usar mlf(z, a, b) donde:
  #   z = -mu * t^nu
  #   a = nu (el parámetro α)
  #   b = 1 (el parámetro β)
  z <- -mu * t^nu
  Re(mlf(z, nu, 1))
}

# Función de distribución (CDF) teórica: F(t) = 1 - S(t)
cdf_teorica <- function(t, nu, mu) {
  1 - survival_teorica(t, nu, mu)
}


#Validación

nu <- 0.5
mu <- 1
set.seed(12+1)
muestras <- rmittag_leffler(10000, nu, mu)

# CDF empírica
cdf_emp <- ecdf(muestras)

# Malla de tiempos ()
t_vals <- seq(0.01, quantile(muestras, 0.95), length.out = 200)
cdf_emp_vals <- cdf_emp(t_vals)
cdf_teo_vals <- sapply(t_vals, function(t) cdf_teorica(t, nu, mu))

# Gráfica
plot(t_vals, cdf_emp_vals, type = "l", col = "blue", lwd = 2,
     xlab = "t", ylab = "F(t) = P(T ≤ t)",
     main = paste0("Mittag-Leffler: ν = ", nu, ", μ = ", mu))
lines(t_vals, cdf_teo_vals, col = "red", lty = 2, lwd = 2)
legend("bottomright", legend = c("Empírica", "Teórica (MittagLeffleR)"),
       col = c("blue", "red"), lty = c(1,2), lwd = 2)

# 4. Validación  ν = 0.7, μ = 0.5

nu2 <- 0.7
mu2 <- 0.5
set.seed(123)
muestras2 <- rmittag_leffler(5000, nu2, mu2)

cdf_emp2 <- ecdf(muestras2)
t_vals2 <- seq(0.01, quantile(muestras2, 0.95), length.out = 200)
cdf_emp_vals2 <- cdf_emp2(t_vals2)
cdf_teo_vals2 <- sapply(t_vals2, function(t) cdf_teorica(t, nu2, mu2))

plot(t_vals2, cdf_emp_vals2, type = "l", col = "blue", lwd = 2,
     xlab = "t", ylab = "F(t)", main = paste0("ν = ", nu2, ", μ = ", mu2))
lines(t_vals2, cdf_teo_vals2, col = "red", lty = 2, lwd = 2)
legend("bottomright", legend = c("Empírica", "Teórica"),
       col = c("blue", "red"), lty = c(1,2), lwd = 2)

