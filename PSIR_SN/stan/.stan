functions {
  vector sir_rhs(real t, vector Y, real beta, real gamma, real N) {
    vector[3] dY;
    real S = Y[1];
    real I = Y[2];
    dY[1] = -beta * S * I / N;
    dY[2] =  beta * S * I / N - gamma * I;
    dY[3] =  gamma * I;
    return dY;
  }

  matrix solve_fractional(real alpha, real beta, real gamma,
                          real N, real I0, int K, real h_solver) {
    real gamma_a1 = tgamma(alpha + 1);
    real gamma_a2 = tgamma(alpha + 2);
    real h_alpha  = pow(h_solver, alpha);

    matrix[K+1, 3] y;
    array[K+1] vector[3] f_vals;

    // initial condition
    y[1] = [N - I0, I0, 0.0];
    f_vals[1] = sir_rhs(0, to_vector(y[1]), beta, gamma, N);

    for (n in 2:(K+1)) {
      real tn = (n - 1) * h_solver;

      // ----- predictor (Adams-Bashforth) -----
      vector[3] sum_pred = rep_vector(0.0, 3);
      for (j in 1:(n-1)) {
        real b = pow(n - j + 1, alpha) - pow(n - j, alpha);
        sum_pred += b * f_vals[j];
      }
      vector[3] y_pred = to_vector(y[1]) + (h_alpha / gamma_a1) * sum_pred;
      vector[3] f_pred = sir_rhs(tn, y_pred, beta, gamma, N);

      // ----- corrector (Adams-Moulton) -----
      vector[3] corr_sum = rep_vector(0.0, 3);
      // a0
      corr_sum += (pow(n - 2, alpha + 1) - (n - 2 - alpha) * pow(n - 1, alpha))
                  * f_vals[1];
      // a1 .. a_{n-2}
      for (j in 2:(n-1)) {
        real aj = pow(n - j + 1, alpha + 1) +
                  pow(n - j - 1, alpha + 1) -
                  2 * pow(n - j, alpha + 1);
        corr_sum += aj * f_vals[j];
      }
      // a_{n-1} = 1
      corr_sum += f_pred;

      y[n] = to_row_vector(to_vector(y[1]) + (h_alpha / gamma_a2) * corr_sum);
      f_vals[n] = sir_rhs(tn, to_vector(y[n]), beta, gamma, N);
    }
    return y;
  }
}

data {
  int<lower=1> T;               // número de días observados (ej. 20)
  array[T] int y;               // incidencia diaria
  real N;                       // población total
  real<lower=0> h;              // paso del solver fraccionario (ej. 0.01)
}

transformed data {
  int steps_per_day = to_int(1.0 / h);   // 1/h debe ser entero exacto (ej. 100)
  int K = T * steps_per_day;
  real h_solver = 1.0 / steps_per_day;   // coincide con h original
}

parameters {
  real<lower=0> beta;
  real<lower=0> gamma;
  real<lower=1> I0;
  real<lower=0.2, upper=1> alpha;
  real<lower=0.01> phi;
}

transformed parameters {
  matrix[K+1, 3] sol_fine = solve_fractional(alpha, beta, gamma,
                                              N, I0, K, h_solver);
  array[T] real incidence;
  for (t in 1:T) {
    int idx = (t - 1) * steps_per_day + 1;
    incidence[t] = fmax(sol_fine[idx, 1] - sol_fine[idx + steps_per_day, 1], 1e-6);
  }
}

model {
  beta  ~ lognormal(0.5, 0.3);
  gamma ~ lognormal(-0.2, 0.2);
  I0    ~ normal(5, 1);
  alpha ~ beta(16, 4);
  phi   ~ gamma(100, 3.33);

  y ~ neg_binomial_2(incidence, phi);
}
