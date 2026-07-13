functions {
  // Ecuaciones diferenciales SEIR
  vector seir_rhs(real t, vector Y, real beta_t, real gamma, real kappa, real N) {
    vector[4] dY;
    real S = Y[1]; 
    real E = Y[2]; 
    real I = Y[3];
    
    dY[1] = -beta_t * S * I / N;
    dY[2] =  beta_t * S * I / N - kappa * E;
    dY[3] =  kappa * E - gamma * I;
    dY[4] =  gamma * I;
    return dY;
  }

  //  SOLVER FRACCIONARIO (Predictor-Corrector)
  matrix solve_fractional_seir(real alpha, real beta0, real delta,
                                real gamma, real kappa,
                                real N, real I0, real E0, 
                                int K, real h) {
    
    real G1 = tgamma(alpha + 1.0);
    real G2 = tgamma(alpha + 2.0);
    real h_alpha = pow(h, alpha);
    
    matrix[K + 1, 4] y;
    array[K + 1] vector[4] f_vals;
    
    // Condiciones iniciales: S = N - I0 - E0, E = E0, I = I0, R = 0
    y[1, 1] = N - I0 - E0;
    y[1, 2] = E0;
    y[1, 3] = I0;
    y[1, 4] = 0.0;
    
    // Asegurar que la suma sea N (por redondeo)
    real total0 = sum(to_vector(y[1]));
    if (total0 > 0) {
      y[1] = y[1] * (N / total0);
    }
    
    real beta_t0 = beta0 * exp(-delta * 0.0);
    f_vals[1] = seir_rhs(0.0, to_vector(y[1]), beta_t0, gamma, kappa, N);
    
    // Método Predictor-Corrector de Adams-Bashforth-Moulton
    for (n in 2:(K + 1)) {
      real t_current = (n - 1) * h;
      real beta_current = beta0 * exp(-delta * t_current);
      
      // - PREDICTOR -
      vector[4] sum_pred = rep_vector(0.0, 4);
      for (j in 1:(n - 1)) {
        real b = pow(n - j * 1.0, alpha) - pow(n - j - 1.0, alpha);
        sum_pred += b * f_vals[j];
      }
      vector[4] y_pred = to_vector(y[1]) + (h_alpha / G1) * sum_pred;
      
      // Restricciones físicas
      for (k in 1:4) y_pred[k] = fmax(y_pred[k], 0.0);
      real total_pred = sum(y_pred);
      total_pred = fmax(total_pred, 1e-12);
      y_pred = y_pred * (N / total_pred);
      
      // Evaluar f en el predictor
      vector[4] f_pred = seir_rhs(t_current, y_pred, beta_current, gamma, kappa, N);
      
      // -- CORRECTOR ---
      vector[4] sum_corr = rep_vector(0.0, 4);
      
      // Término especial para j=1
      real a0 = pow(n - 2.0, alpha + 1.0) - (n - 2.0 - alpha) * pow(n - 1.0, alpha);
      sum_corr += a0 * f_vals[1];
      
      // Términos para j=2,...,n-1
      for (j in 2:(n - 1)) {
        real a = pow(n - j + 1.0, alpha + 1.0) 
                 - 2.0 * pow(n - j * 1.0, alpha + 1.0) 
                 + pow(n - j - 1.0, alpha + 1.0);
        sum_corr += a * f_vals[j];
      }
      
      vector[4] y_corr = to_vector(y[1]) + (h_alpha / G2) * (sum_corr + f_pred);
      
      // Restricciones físicas
      for (k in 1:4) y_corr[k] = fmax(y_corr[k], 0.0);
      real total_corr = sum(y_corr);
      total_corr = fmax(total_corr, 1e-12);
      y_corr = y_corr * (N / total_corr);
      
      // Almacenar solución
      y[n] = to_row_vector(y_corr);
      f_vals[n] = seir_rhs(t_current, y_corr, beta_current, gamma, kappa, N);
    }
    
    return y;
  }

  // INCIDENCIA SEMANAL DE LA SOLUCIÓN
  array[] real extraer_incidencia(matrix sol, int steps_per_week, int T_total, real p) {
    array[T_total] real incidence;
    array[T_total] real mu;
    
    for (t in 1:T_total) {
      int left  = (t - 1) * steps_per_week + 1;
      int right = t * steps_per_week + 1;
      real diff = sol[left, 1] - sol[right, 1];
      incidence[t] = fmax(diff, 1e-6);
      mu[t] = p * incidence[t];
    }
    
    return mu;
  }
}

// DATA
data {
  // --- Datos de entrenamiento ---
  int<lower=1> T;                      // Número de semanas observadas
  array[T] int<lower=0> y;             // Casos observados (entrenamiento)
  
  // --- Datos de validación si se tienn(opcional) ---
  int<lower=0> T_new;                  // Semanas a predecir
  array[T_new] int<lower=0> y_new;     // Datos reales futuros (para evaluación)
  
  // --- Parámetros del modelo ---
  real<lower=0> N0;                    // Población total
  real<lower=0> h;                     // Paso de tiempo (fracción de semana)
}
// TRANSFORMED DATA
transformed data {
  int steps_per_week = to_int(round(1.0 / h));
  int T_total = T + T_new;              // Horizonte total
  int K_total = T_total * steps_per_week;
  real h_solver = 1.0 / steps_per_week;
}
// PARAMETERS
parameters {
  // Parámetros en escala logarítmica para mejor muestreo
  real log_beta0;          // Tasa de transmisión inicial
  real log_delta;          // Tasa de decaimiento de beta
  real log_I0;             // Infectados iniciales
  real log_p;              // Tasa de detección
  real log_theta;          // Sobredispersión (Negative Binomial)
  
  // Parámetros con soporte restringido
  real<lower=0, upper=1> alpha;    // Orden fraccionario
  real<lower=0> gamma;             // Tasa de recuperación (1/días)
  real<lower=0> kappa;             // Tasa de latencia (1/días)
}

// TRANSFORMED PARAMETERS
transformed parameters {
  // --- Parámetros en escala natural ---
  real beta0 = exp(log_beta0);
  real delta = exp(log_delta);
  real I0    = exp(log_I0);
  real p     = exp(log_p);
  real theta = exp(log_theta);
  
  // SOLUCIÓN DEL SISTEMA PARA TODO EL HORIZONTE [0, T+T_new]
  //   1. El likelihood solo usa [1:T]
  //   2. La solución en [T+1:T+T_new] es una consecuencia 
  //      determinista de los parámetros y condiciones iniciales
  //   3. El cálculo fraccionario requiere conocer toda la historia
  //      para evaluar correctamente la derivada en cada punto
  matrix[K_total + 1, 4] sol_total =
      solve_fractional_seir(
          alpha,beta0,delta,gamma,kappa,N0,
          I0,0.0,        // E0 = 0 (asumimos que todos los expuestos iniciales son 0)
          K_total,h_solver);
  // --- Extraer incidencia y media para todas las semanas ---
  array[T_total] real mu;
  array[T_total] real incidence;
  
  for (t in 1:T_total) {
    int left  = (t - 1) * steps_per_week + 1;
    int right = t * steps_per_week + 1;
    real diff = sol_total[left, 1] - sol_total[right, 1];
    incidence[t] = fmax(diff, 1e-6);
    mu[t] = p * incidence[t];
  }
  
  // --- Beta semanal (para monitoreo) ---
  array[T_total] real beta_week;
  for (t in 1:T_total) {
    beta_week[t] = beta0 * exp(-delta * (t - 1));
  }
}

// MODEL (LIKELIHOOD)
model {
  log_beta0 ~ normal(log(2.0), 0.5);
  log_delta ~ normal(log(0.09), 0.5);
  log_I0   ~ normal(log(100), 1);
  log_p    ~ normal(-1.8444, 0.8984);   // Basado en estudios de subdetección
  log_theta~ normal(log(16), 0.5);
  alpha    ~ beta(1, 1);                 // Uniforme en [0,1]
  gamma    ~ lognormal(log(0.875), 0.05); // ~1/7 días
  kappa    ~ lognormal(log(0.5), 0.05);   // ~1/3.5 días
  
  //  LIKELIHOOD, SOLO PARA DATOS DE ENTRENAMIENTO
  // p(y_1:T | θ) = ∏_{t=1}^T NB(y_t | μ_t(θ), θ_disp)
  // donde μ_t(θ) = p * Incidencia_t(θ)
  for (t in 1:T) {
    y[t] ~ neg_binomial_2(mu[t], theta);
  }
}

// GENERATED QUANTITIES
generated quantities {
  //  PARÁMETROS DE INTERÉS
  real beta0_gq = exp(log_beta0);
  real delta_gq = exp(log_delta);
  
  // Número reproductivo básico (R0)
  real R0_initial = beta0_gq / gamma;
  real R0_final   = beta0_gq * exp(-delta_gq * (T_total - 1)) / gamma;
  real R0_mean    = beta0_gq * (1 - exp(-delta_gq * T_total)) / (delta_gq * T_total * gamma);
  
  // DISTRIBUCIÓN PREDICTIVA POSTERIOR (PPC)
  // La distribución predictiva posterior se define como:
  // p(y_rep_{1:T_total} | y_{1:T}) = 
  //     ∫ p(y_rep_{1:T_total} | μ(θ), θ_disp) p(θ | y_{1:T}) dθ
  //
  // Donde:
  //   - p(θ | y_{1:T}) es la distribución posterior, muestras de MCMC
  //   - p(y_rep | μ(θ), θ_disp) = NB(y_rep | μ(θ), θ_disp)
  //paraa cada muestra posterior θ^(s):
  //   1. Calculamos μ^(s) = μ(θ^(s)) para todo t=1:T_total
  //   2. Simulamos y_rep_t^(s) ~ NB(μ_t^(s), θ_disp^(s))
  //
  // El conjunto {y_rep^(s)}_{s=1}^{S} aproxima la distribución marginal
  array[T_total] int y_rep;           // Predicciones para todo el horizonte
  
  // Separación por período para facilitar el análisis
  array[T] int y_rep_train;           // Predicciones en entrenamiento
  array[T_new] int y_rep_future;      // Predicciones en futuro
  
  for (t in 1:T_total) {
    // Muestreo de la Negative Binomial con la media μ_t y sobredispersión θ
    y_rep[t] = neg_binomial_2_rng(mu[t], theta);
  }
  
  // Separar para análisis
  for (t in 1:T) {
    y_rep_train[t] = y_rep[t];
  }
  
  for (t in 1:T_new) {
    y_rep_future[t] = y_rep[T + t];
  }
  
  // LOG-VEROSIMILITUD PARA EVALUACIÓN
  array[T] real log_lik_train;
  for (t in 1:T) {
    log_lik_train[t] = neg_binomial_2_lpmf(y[t] | mu[t], theta);
  }
  
  // Log-verosimilitud en futuro para evaluar predicciones
  array[T_new] real log_lik_future;
  if (T_new > 0) {
    for (t in 1:T_new) {
      log_lik_future[t] = neg_binomial_2_lpmf(y_new[t] | mu[T + t], theta);
    }
  }
  // Para evaluación de la calidad predictiva
  real log_lik_total = sum(log_lik_train);
  
  if (T_new > 0) {
    real log_lik_future_total = sum(log_lik_future);
  }
}

