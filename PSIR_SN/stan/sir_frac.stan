functions {

  
  // RHS del modelo SIR
  
  vector sir_rhs(real t,
                 vector Y,
                 real beta,
                 real gamma,
                 real N) {

    vector[3] dY;

    real S = Y[1];
    real I = Y[2];

    dY[1] = -beta * S * I / N;
    dY[2] =  beta * S * I / N - gamma * I;
    dY[3] =  gamma * I;

    return dY;
  }


  //==========================================================
  // Adams-Bashforth-Moulton fraccionario
  // Orden Caputo: 0 < alpha <= 1
  //==========================================================
  matrix solve_fractional(real alpha,
                          real beta,
                          real gamma,
                          real N,
                          real I0,
                          int K,
                          real h) {

    real G1 = tgamma(alpha + 1);
    real G2 = tgamma(alpha + 2);

    real h_alpha = pow(h, alpha);

    matrix[K + 1, 3] y;
    array[K + 1] vector[3] f_vals;

    //--------------------------------------------------------
    // Condición inicial
    //--------------------------------------------------------
    y[1,1] = N - I0;
    y[1,2] = I0;
    y[1,3] = 0.0;

    f_vals[1] = sir_rhs(
      0.0,
      to_vector(y[1]),
      beta,
      gamma,
      N
    );

    //--------------------------------------------------------
    // Bucle principal
    //--------------------------------------------------------
    for (n in 1:K) {

      vector[3] sum_pred = rep_vector(0.0, 3);
      vector[3] sum_corr = rep_vector(0.0, 3);

      //------------------------------------------------------
      // PREDICTOR
      //------------------------------------------------------
      for (j in 1:n) {

        real b;

        b = pow(n - j + 1, alpha)
          - pow(n - j, alpha);

        sum_pred += b * f_vals[j];
      }

      vector[3] y_pred;

      y_pred =
        to_vector(y[1])
        + (h_alpha / G1) * sum_pred;

      vector[3] f_pred;

      f_pred = sir_rhs(
        n * h,
        y_pred,
        beta,
        gamma,
        N
      );

      //------------------------------------------------------
      // CORRECTOR
      //------------------------------------------------------

      // coeficiente a0
      {
        real a0;

        a0 =
          pow(n, alpha + 1)
          - (n - alpha) * pow(n + 1, alpha);

        sum_corr += a0 * f_vals[1];
      }

      // coeficientes interiores
      if (n > 1) {

        for (j in 2:n) {

          real m;
          real a;

          m = n - j + 1;

          a =
            pow(m + 1, alpha + 1)
            - 2.0 * pow(m, alpha + 1)
            + pow(m - 1, alpha + 1);

          sum_corr += a * f_vals[j];
        }
      }

      // término predictor
      sum_corr += f_pred;

      //------------------------------------------------------
      // Actualización
      //------------------------------------------------------
      y[n + 1] =
        to_row_vector(
          to_vector(y[1])
          + (h_alpha / G2) * sum_corr
        );

      //------------------------------------------------------
      // Protección numérica
      //------------------------------------------------------
      y[n + 1, 1] = fmax(y[n + 1, 1], 1e-9);
      y[n + 1, 2] = fmax(y[n + 1, 2], 1e-9);
      y[n + 1, 3] = fmax(y[n + 1, 3], 0.0);

      //------------------------------------------------------
      // Recalcular RHS
      //------------------------------------------------------
      f_vals[n + 1] =
        sir_rhs(
          n * h,
          to_vector(y[n + 1]),
          beta,
          gamma,
          N
        );
    }

    return y;
  }
}



data {

  // Datos observados
  int<lower=1> T;

  array[T] int<lower=0> y;

  // Población
  real<lower=0> N;

  // Paso del solver

  real<lower=0> h;
}



transformed data {

  int steps_per_day;
  int K;

  real h_solver;

  steps_per_day = to_int(round(1.0 / h));

  K = T * steps_per_day;

  h_solver = 1.0 / steps_per_day;
}



parameters {

  
  // Parámetros epidemiológicos

  real<lower=0> beta;

  real<lower=0> gamma;

  real<lower=1> I0;

  
  // Orden fraccionario
  real<lower=0.1, upper=1> alpha;

  
  // Sobredispersión
  real<lower=0.01> phi;
}



transformed parameters {

  // Solución numérica

  matrix[K + 1, 3] sol_fine;

  // Incidencia diaria

  array[T] real incidence;

  sol_fine =
    solve_fractional(
      alpha,
      beta,
      gamma,
      N,
      I0,
      K,
      h_solver
    );

  // Casos nuevos:
  // S(t-1) - S(t)
  for (t in 1:T) {

    int left;
    int right;

    left  = (t - 1) * steps_per_day + 1;
    right = t * steps_per_day + 1;

    incidence[t] =
      sol_fine[left, 1]
      - sol_fine[right, 1];

    // Protección numérica
    incidence[t] =
      fmax(incidence[t], 1e-9);
  }
}



model {

  // Priors
  beta ~ lognormal(log(1.7), 0.3);

  gamma ~ lognormal(log(0.8), 0.2);

  I0 ~ lognormal(log(6), 2);

  alpha ~ beta(2,2);

  phi ~ lognormal(log(30),2 );

  // Likelihood

  y ~ neg_binomial_2(incidence, phi);
}



generated quantities {

  // Reproducción básica
  real R0;

  // Posterior predictive
  array[T] int y_rep;

  R0 = beta / gamma;

  for (t in 1:T) {

    y_rep[t] =
      neg_binomial_2_rng(
        incidence[t],
        phi
      );
  }
}
