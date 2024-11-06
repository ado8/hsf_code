import pickle

import stan


def mix_model(input_dict, iters=5000, jobs=7):
    stan_code = """
    data {
                // number of SNe
                int<lower=1> N;
                vector[N] mag_diff ;
                vector[N] e_mag_diff ;
                real<lower=0> core_width ;
                real<lower=0> outl_width ;
                real f_outl ;
                real f_outl_width ;
    }
    parameters {
                real<lower=-0.2, upper=0.2> mu_in;
                real mu_out;
                ordered[2] sigma;
                real<lower=0, upper=1> theta;
    }
    model {
        mu_in ~ normal(0, 1);
        mu_out ~ normal(0, 2);
        sigma[1] ~ normal(core_width, 1);
        sigma[2] ~ normal(outl_width, 3);
        theta ~ beta(N*f_outl+1, N*(1-f_outl)+1);
        for (n in 1:N)
            target += log_mix(theta,
                normal_lpdf(mag_diff[n] | mu_out, sigma[2]),
                normal_lpdf(mag_diff[n] | mu_in, sigma[1]));
    }
    """
    with open("foo.stancode", "w") as f:
        f.write(stan_code)
    pickle_path = f"{constants.STATIC_DIR}/background_subtraction.stan"

    try:
        sm, stxt = pickle.load(open(pickle_path, "rb"))
        if stxt != stan_code:
            raise Exception("Code does not match")
    except (FileNotFoundError, Exception) as e:
        if isinstance(e, FileNotFoundError) or str(e) == "Code does not match":
            sm = stan.StanModel(
                model_code=stan_code, extra_compile_args=["-pthread", "-DSTAN_THREADS"]
            )
            pickle.dump((sm, stan_code), open(pickle_path, "wb"))
        else:
            raise
    fit = sm.sampling(
        data=input_dict,
        iter=iters,
        chains=jobs,
        n_jobs=4 * jobs,
        sample_file="chains/stan.dat",
        # control={'max_treedepth':12},
    )
    # arviz.plot_trace(fit.extract())
    # plt.show()
    return fit
