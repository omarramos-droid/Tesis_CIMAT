functions {
  vector seir_rhs(real t, vector Y,
                  real beta, real sigma, real gamma,
                  real Lambda, real mu) {
    vector[4] dY;
    real S = Y[1]; real E = Y[2]; real I = Y[3]; real R = Y[4];
    real N = S + E + I + R;
    dY[1] = Lambda - beta * S * I / N - mu * S;
    dY[2] = beta * S * I / N - (sigma + mu) * E;
    dY[3] = sigma * E - (gamma + mu) * I;
    dY[4] = gamma * I - mu * R;
    return dY;
  }

  matrix solve_fractional(real alpha, real beta, real sigma, real gamma,
                          real Lambda, real mu, real N0, real E0, real I0,
                          int K, real h) {
    real G1 = tgamma(alpha + 1);
    real G2 = tgamma(alpha + 2);
    real h_alpha = pow(h, alpha);
    matrix[K + 1, 4] y;
    array[K + 1] vector[4] f_vals;

    y[1,1] = N0 - E0 - I0; y[1,2] = E0; y[1,3] = I0; y[1,4] = 0.0;
    f_vals[1] = seir_rhs(0, to_vector(y[1]), beta, sigma, gamma, Lambda, mu);

    for (n in 1:K) {
      vector[4] sum_pred = rep_vector(0.0, 4);
      vector[4] sum_corr = rep_vector(0.0, 4);

      for (j in 1:n) {
        real b = pow(n - j + 1, alpha) - pow(n - j, alpha);
        sum_pred += b * f_vals[j];
      }
      vector[4] y_pred = to_vector(y[1]) + (h_alpha / G1) * sum_pred;
      vector[4] f_pred = seir_rhs(n * h, y_pred, beta, sigma, gamma, Lambda, mu);

      {
        real a0 = pow(n, alpha + 1) - (n - alpha) * pow(n + 1, alpha);
        sum_corr += a0 * f_vals[1];
      }
      if (n > 1) {
        for (j in 2:n) {
          real m = n - j + 1;
          real a = pow(m + 1, alpha + 1) - 2*pow(m, alpha + 1) + pow(m - 1, alpha + 1);
          sum_corr += a * f_vals[j];
        }
      }
      sum_corr += f_pred;
      y[n + 1] = to_row_vector(to_vector(y[1]) + (h_alpha / G2) * sum_corr);

      y[n + 1,1] = fmax(y[n + 1,1], 1e-9);
      y[n + 1,2] = fmax(y[n + 1,2], 1e-9);
      y[n + 1,3] = fmax(y[n + 1,3], 1e-9);
      y[n + 1,4] = fmax(y[n + 1,4], 0.0);

      f_vals[n + 1] = seir_rhs(n * h, to_vector(y[n + 1]), beta, sigma, gamma, Lambda, mu);
    }
    return y;
  }
}

data {
  int<lower=1> T;
  array[T] int<lower=0> y;
  real<lower=0> E0;
  real<lower=0> N0;
  real<lower=0> h;
}

transformed data {
  int steps_per_day = to_int(round(1.0 / h));
  int K = T * steps_per_day;
  real h_solver = 1.0 / steps_per_day;
}

parameters {
  real<lower=0> beta;
  real<lower=0> sigma;
  real<lower=0> gamma;
  real<lower=0> Lambda;
  real<lower=0> mu;
  real<lower=1> I0;
  real<lower=0.1, upper=1> alpha;
  real<lower=0.01> phi;
}

transformed parameters {
  matrix[K + 1, 4] sol_fine = solve_fractional(
    alpha, beta, sigma, gamma, Lambda, mu, N0, E0, I0, K, h_solver
  );

  // INCIDENCIA DIFERENCIA DE SUSCEPTIBLES 
  array[T] real incidence;
  for (t in 1:T) {
    int left  = (t - 1) * steps_per_day + 1;
    int right = t * steps_per_day + 1;
    incidence[t] = fmax(sol_fine[left, 1] - sol_fine[right, 1], 1e-9);
  }
}

model {
  beta   ~ lognormal(log(1.5), 0.3);
  sigma  ~ lognormal(log(1.2), 0.3);
  gamma  ~ lognormal(log(0.8), 0.2);
  Lambda ~ lognormal(log(30), 0.5);
  mu     ~ lognormal(log(0.001), 0.5);
  I0     ~ lognormal(log(15), 2);
  alpha  ~ beta(2, 2);
  phi    ~ lognormal(log(30), 2);

  y ~ neg_binomial_2(incidence, phi);
}

generated quantities {
  real R0 = (beta * sigma * Lambda) / ((sigma + mu) * (gamma + mu) *(mu) );
  array[T] int y_rep;
  for (t in 1:T)
    y_rep[t] = neg_binomial_2_rng(incidence[t], phi);
}
