functions {
  vector seir_rhs(real t, vector Y, real beta, real gamma, real kappa, real N) {
    vector[4] dY;
    real S = Y[1]; 
    real E = Y[2]; 
    real I = Y[3];
    
    dY[1] = -beta * S * I / N;
    dY[2] =  beta * S * I / N - kappa * E;
    dY[3] =  kappa * E - gamma * I;
    dY[4] =  gamma * I;
    return dY;
  }

  matrix solve_fractional(real alpha, real beta, real gamma, real kappa,
                          real N, real I0, real E0, int K, real h) {
    real G1 = tgamma(alpha + 1.0);
    real G2 = tgamma(alpha + 2.0);
    real h_alpha = pow(h, alpha);
    
    matrix[K + 1, 4] y;
    array[K + 1] vector[4] f_vals;
    
    // Inicialización
    y[1, 1] = N - I0 - E0;
    y[1, 2] = E0;
    y[1, 3] = I0;
    y[1, 4] = 0.0;
    f_vals[1] = seir_rhs(0.0, to_vector(y[1]), beta, gamma, kappa, N);
    
    // Precalcular coeficientes 
    array[K] real b_coeffs;
    array[K + 1] real a_coeffs;
    
    for (n in 2:(K + 1)) {
      // Predictor
      vector[4] sum_pred = rep_vector(0.0, 4);
      for (j in 1:(n - 1)) {
        real b = pow(n - j * 1.0, alpha) - pow(n - j - 1.0, alpha);
        sum_pred += b * f_vals[j];
      }
      vector[4] y_pred = to_vector(y[1]) + (h_alpha / G1) * sum_pred;
      
      // Corrección de positividad
      for (k in 1:4) y_pred[k] = fmax(y_pred[k], 0.0);
      real total = sum(y_pred);
      y_pred = y_pred * (N / fmax(total, 1e-12));
      
      vector[4] f_pred = seir_rhs((n - 1) * h, y_pred, beta, gamma, kappa, N);
      
      // Corrector
      vector[4] sum_corr = rep_vector(0.0, 4);
      real a0 = pow(n - 2.0, alpha + 1.0) - (n - 2.0 - alpha) * pow(n - 1.0, alpha);
      sum_corr += a0 * f_vals[1];
      
      for (j in 2:(n - 1)) {
        real a = pow(n - j + 1.0, alpha + 1.0) - 2.0 * pow(n - j * 1.0, alpha + 1.0) + pow(n - j - 1.0, alpha + 1.0);
        sum_corr += a * f_vals[j];
      }
      
      vector[4] y_corr = to_vector(y[1]) + (h_alpha / G2) * (sum_corr + f_pred);
      
      // Positividad
      for (k in 1:4) y_corr[k] = fmax(y_corr[k], 0.0);
      total = sum(y_corr);
      y_corr = y_corr * (N / fmax(total, 1e-12));
      
      y[n] = to_row_vector(y_corr);
      f_vals[n] = seir_rhs((n - 1) * h, y_corr, beta, gamma, kappa, N);
    }
    return y;
  }
}

data {
  int<lower=1> T;
  array[T] int<lower=0> y;
  real<lower=0> N0;
  real<lower=0> h;
}

transformed data {
  int steps_per_week = to_int(round(1.0 / h));
  int K = T * steps_per_week;
  real h_solver = 1.0 / steps_per_week;
}

parameters {
  real log_beta;
  real log_I0;
  real log_p;
  real log_theta;
  real<lower=0, upper=1> alpha;  
  real<lower=0> gamma;
  real<lower=0> kappa;
}

transformed parameters {
  real beta  = exp(log_beta);
  real I0    = exp(log_I0);
  real p     = exp(log_p);
  real theta = exp(log_theta);

  matrix[K + 1, 4] sol_fine =
      solve_fractional(
          alpha,
          beta,
          gamma,
          kappa,
          N0,
          I0,
          0.0,
          K,
          h_solver);

  array[T] real incidence;
  array[T] real mu;

  for (t in 1:T) {
    int left  = (t - 1) * steps_per_week + 1;
    int right = t * steps_per_week + 1;
    real diff = sol_fine[left, 1] - sol_fine[right, 1];
    
    incidence[t] = fmax(diff, 1e-6);
    
    mu[t] = p * incidence[t];
  }
}

model {
  // Priors lognormales para parámetros positivos
  log_beta ~ normal(log(10), 0.5);    // beta ~ LogNormal(log(10), 0.5)
  log_I0 ~ normal(log(500), 1.5);     // I0 ~ LogNormal(log(500), 1.5)
  log_p ~ normal(log(0.1), 0.5);    // p ~ LogNormal(log(0.1), 0.5)
  log_theta ~ normal(log(30), 0.5);  // theta ~ LogNormal(log(30), 0.5)
  alpha ~ beta(1,1);  // Distribución Uniforme 
  // Priors lognormales para gamma y kappa
  gamma ~ lognormal(log(0.8), 0.1);  // gamma ~ LogNormal(log(0.8), 0.1)
  kappa ~ lognormal(log(0.4), 0.2);  // kappa ~ LogNormal(log(0.4), 0.2)
  
  // Likelihood
  for (t in 1:T)
    y[t] ~ neg_binomial_2(mu[t], theta);
}

generated quantities {
  real beta_gq = exp(log_beta);
  real R0 = beta_gq / gamma;
  array[T] int y_rep;

  for (t in 1:T)
    y_rep[t] = neg_binomial_2_rng(mu[t], theta);
}
