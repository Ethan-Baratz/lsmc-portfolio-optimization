import yfinance as yf
import numpy as np
import pandas as pd
import cvxpy as cp
from sklearn.covariance import LedoitWolf
from arch import arch_model
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import random
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from hmmlearn import hmm
import logging
import warnings
from arch.utility.exceptions import ConvergenceWarning
from scipy.optimize import OptimizeWarning

logging.getLogger("hmmlearn").setLevel(logging.ERROR)

_hmm_cache = {'model': None}

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=OptimizeWarning)
warnings.filterwarnings("ignore", message=".*Iteration limit reached.*")


def getMultipleStocks_logReturns(tickers, start_date, end_date):
    stock_returns = []

    for ticker in tickers:
        
        his = ticker.history(start=start_date, end=end_date)
        
        his.index = his.index.tz_localize(None)
        his = his[~his.index.duplicated(keep='first')]
        close_prices = his['Close']

        # Gets us the daily log returns
        daily_log_returns = np.log(close_prices / close_prices.shift(1)).dropna()


        stock_returns.append(daily_log_returns)

    # doing axis = 1 puts all the dates in the rows for us and then each stock return in the columns
    # if we did axis = 0 we would be stacking the date and stock return.s next to each other
    # converts into data frame so can use .cov()
    stock_returns_df = pd.concat(stock_returns, axis=1)
    stock_returns_df.columns = [t.ticker for t in tickers] # Gets us ticker name, t.ticker = 'AAPL' for ex

    # Checks for holes in the data
    print("Data Shape:", stock_returns_df.shape)
    print("Null Values:\n", stock_returns_df.isnull().sum())
    print("First 5 rows:\n", stock_returns_df.head())


    stock_returns_df = stock_returns_df.dropna() # Remove any rows with NaN values after concatenation

    return stock_returns_df




def getMultiQuadProgRegimeSwitching(mus, r_annual, num_stocks, gamma, D,
                                     std_residuals, c_star, W_t,
                                     W_prev=None, returns_history=None,
                                     n_regimes=2, refit_hmm=False):

    T          = 10
    time_steps = 120
    dt         = T / time_steps
    mus        = np.asarray(mus)

    bear_risk_mult = 5.0
    lw = LedoitWolf().fit(std_residuals)
    annual_cov = D @ (lw.covariance_ / np.outer(np.sqrt(np.diag(lw.covariance_)), np.sqrt(np.diag(lw.covariance_)))) @ D

    try:

      if returns_history is None:
          returns_history = std_residuals # proxy if raw returns not passed

      # Fit or reuse the HMM
      min_obs   = max(num_stocks * 5, 60)
      hmm_model = None if refit_hmm else _hmm_cache['model']

      if hmm_model is None and len(returns_history) >= min_obs:
          port_ret    = returns_history.mean(axis=1, keepdims=True)
          rolling_vol = np.array([
              returns_history[max(0, t - 20):t + 1].std(axis=0).mean()
              for t in range(len(returns_history))
          ]).reshape(-1, 1)
          features  = np.hstack([port_ret, rolling_vol])  # (T, 2)
          features += np.random.normal(0, 1e-8, features.shape)

          try:
            hmm_model = hmm.GaussianHMM(
                n_components=n_regimes, covariance_type="full",
                n_iter=500, random_state=1, tol=1e-4, min_covar=1e-4
            )
            hmm_model.fit(features)
          except (ValueError, np.linalg.LinAlgError):
            hmm_model = hmm.GaussianHMM(
                n_components=n_regimes, covariance_type="diag",
                n_iter=500, random_state=1, tol=1e-4, min_covar=1e-3
            )
            hmm_model.fit(features)

          _hmm_cache['model'] = hmm_model

      # Regime detection + blended mu and cov
      if hmm_model is not None and len(returns_history) >= min_obs:
          port_ret    = returns_history.mean(axis=1, keepdims=True)
          rolling_vol = np.array([
              returns_history[max(0, t - 20):t + 1].std(axis=0).mean()
              for t in range(len(returns_history))
          ]).reshape(-1, 1)
          features        = np.hstack([port_ret, rolling_vol])
          regime_sequence = hmm_model.predict(features)
          regime_probs    = hmm_model.predict_proba(features)[-1] # current timestep

          # Bear state = whichever HMM state has the lowest mean return
          bear_regime = int(np.argmin(hmm_model.means_[:, 0]))
          bear_prob   = float(regime_probs[bear_regime])

          # Per regime mu + LW covariance (fall back to full sample if too few obs)
          regime_means, regime_covs = [], []
          for r in range(n_regimes):
              idx = np.where(regime_sequence == r)[0]
              if len(idx) < num_stocks + 5:
                  idx = np.arange(len(returns_history))
              subset = returns_history[idx]
              regime_means.append(subset.mean(axis=0))
              regime_covs.append(LedoitWolf().fit(subset).covariance_)

          # Soft blend: law of total variance
          blended_mu  = sum(p * m for p, m in zip(regime_probs, regime_means))
          blended_cov = (
              sum(p * (C + np.outer(m, m))
                  for p, m, C in zip(regime_probs, regime_means, regime_covs))
              - np.outer(blended_mu, blended_mu)
          )

          blended_cov = (blended_cov + blended_cov.T) / 2 # Ensure perfectly symmetric
          vals, vecs = np.linalg.eigh(blended_cov) # Get eigenvalues/vectors
          blended_cov = vecs @ np.diag(np.maximum(vals, 1e-8)) @ vecs.T

          if np.all(np.isfinite(blended_mu)):
              mus = blended_mu * 252
          annual_cov = blended_cov * 252

          # CVaR multiplier: 1.0 (fully bull) to 20.0 (fully bear)
          # regime-switching equivalent of LSMC lambda_global
          bear_risk_mult = 1.0 + 19.0 * bear_prob


      else:
          # Not enough history, plain LW fallback, no regime adjustment
          bear_risk_mult = 1.0
          lw         = LedoitWolf().fit(std_residuals)
          resid_cov  = lw.covariance_
          d_resid    = np.sqrt(np.diag(resid_cov))
          corr       = resid_cov / np.outer(d_resid, d_resid)
          annual_cov = D @ corr @ D
          hmm_model  = None

    except Exception as e:
      bear_risk_mult = 5.0

    annual_cov  = 0.5 * (annual_cov + annual_cov.T)
    annual_cov += np.eye(num_stocks) * 1e-6

    min_ev = np.min(np.linalg.eigvals(annual_cov).real)
    if min_ev <= 1e-8:
        annual_cov += (abs(min_ev) + 1e-6) * np.eye(num_stocks)

    if np.linalg.cond(annual_cov) > 1e8:
        annual_cov += 1e-5 * np.eye(num_stocks)

    try:
        np.linalg.cholesky(annual_cov)
    except np.linalg.LinAlgError:
        ev, evec   = np.linalg.eigh(annual_cov)
        ev         = np.maximum(ev, 1e-6)
        annual_cov = evec @ np.diag(ev) @ evec.T

    # CVaR scenario generation

    N_scenarios = std_residuals.shape[0]
    R_scenarios = (mus * dt) + (std_residuals @ D) * np.sqrt(dt)
    alpha_cvar  = 0.95

    excess_returns      = mus - r_annual
    p                   = cp.psd_wrap(annual_cov * dt)
    x                   = cp.Variable(num_stocks)
    q                   = -excess_returns * dt
    t_var               = cp.Variable()
    u_slack             = cp.Variable(N_scenarios)
    losses              = -R_scenarios @ x
    cvar                = t_var + (1 / ((1 - alpha_cvar) * N_scenarios)) * cp.sum(u_slack)
    cvar                = cvar / 20.0
    diversification_penalty = 0.03

    obj_exp = (
        0.5 * cp.quad_form(x, p)
        + q.T @ x
        + diversification_penalty * cp.sum_squares(x)
        + bear_risk_mult * cvar# regime-driven scaling, replaces lambda_global
    )

    if W_prev is not None:
        obj_exp += 0.002 * cp.norm(x - W_prev, 1)

    constraints = [
        x >= 0, x <= 1,
        cp.sum(x) == 1,
        u_slack >= 0,
        u_slack >= losses - t_var,
    ]

    prob = cp.Problem(cp.Minimize(obj_exp), constraints)

    for solver in [cp.CLARABEL, cp.ECOS, cp.SCS]:
        try:
            prob.solve(solver=solver, verbose=False)
            if prob.status in ('optimal', 'optimal_inaccurate') and x.value is not None:
                break
        except Exception:
            continue

    if (prob.status not in ('optimal', 'optimal_inaccurate')
            or x.value is None or np.allclose(x.value, 0)):
        print(f"RS-MV solver failed (status={prob.status}), using equal weights")
        pi_star = np.ones(num_stocks) / num_stocks
    else:
        pi_star  = np.clip(x.value, 0, 1)
        pi_star /= pi_star.sum()

    return pi_star, annual_cov, hmm_model, bear_risk_mult


def getMultiQuadProg_MV(mus, r_annual, num_stocks, gamma, D, std_residuals, c_star, W_t, W_prev=None, kappa=0.002):
    # Keep T and dt identical to your main solver for fair comparison
    T = 10
    time_steps = 120
    dt = T / time_steps

    # 1. Covariance Construction (Ledoit-Wolf + CCC logic)
    lw = LedoitWolf().fit(std_residuals)
    resid_cov = lw.covariance_
    d_resid = np.sqrt(np.diag(resid_cov))
    corr = resid_cov / np.outer(d_resid, d_resid)
    annual_cov = D @ corr @ D
    annual_cov = 0.5 * (annual_cov + annual_cov.T)
    annual_cov += np.eye(num_stocks) * 1e-6 # Regularization

    # 2. Objective Setup
    excess_returns = np.array(mus) - r_annual
    p = cp.psd_wrap(annual_cov * dt)
    q = -excess_returns * dt
    x = cp.Variable(num_stocks)

    N_scenarios = std_residuals.shape[0]
    R_scenarios = (mus * dt) + (std_residuals @ D) * np.sqrt(dt)

    t_var = cp.Variable()
    u_slack = cp.Variable(N_scenarios)
    alpha_cvar = .95

    losses = -R_scenarios @ x
    cvar = t_var + (1 / ((1- alpha_cvar) * N_scenarios)) * cp.sum(u_slack)
    cvar = cvar / 20.0


    # Standard Diversification Penalty
    diversification_penalty = 0.03

    # Standard MV Objective: (0.5 * Risk) - Return + Diversification
    # Note: No 'lambda_global' here. Risk weight is constant.
    obj_exp = 0.5 * cp.quad_form(x, p) + q.T @ x + diversification_penalty * cp.sum_squares(x) + (cvar)

    # Transaction Costs
    if W_prev is not None:
        obj_exp += kappa * cp.norm(x - W_prev, 1)

    # 3. Solve
    constraints = [x >= 0, x <= 1, cp.sum(x) == 1, u_slack >= 0, u_slack >= losses - t_var]
    prob = cp.Problem(cp.Minimize(obj_exp), constraints)

    try:
        prob.solve(solver=cp.CLARABEL, verbose=False)
    except:
        prob.solve(solver=cp.SCS, verbose=False)

    if x.value is None:
        pi_star = np.ones(num_stocks) / num_stocks
    else:
        pi_star = np.clip(x.value, 0, 1)
        pi_star /= pi_star.sum()

    return pi_star, annual_cov


def getMultiQuadProg(mus, r_annual, num_stocks, gamma, D, std_residuals, c_star, W_t, W_prev = None, lambda_global = None, phi_val = None):
    # * IF YOU CHANGE TIME HORIZON - CHANGE THIS AS WELL
    T = 10 # time horizon for portfolio
    time_steps = 120

    dt = T / time_steps

    mus = np.array(mus) # converts to array so we can use later (can't do vector math on lists - so change to numpy)

    # doing axis = 1 puts all the dates in the rows for us and then each stock return in the columns
    # if we did axis = 0 we would be stacking the date and stock returns next to each other
    #stock_returns = pd.concat(stock_returns, axis=1) # converts into data frame so can use .cov()

    # Use LW
    lw = LedoitWolf().fit(std_residuals)

    # remember - cov. matrix looks like
    # [var(A)  cov(A,B)]
    # [cov(B,A)  var(B)]
    resid_cov = lw.covariance_
    d_resid = np.sqrt(np.diag(resid_cov))

    # cov(x,y) = corr(x,y) * sigma_x * sigma_y -> solve for corr(x,y)
    corr = resid_cov / np.outer(d_resid, d_resid)

    # https://econ.uzh.ch/apps/workingpapers/wp/econwp231.pdf
    # CCC (Constant Conditional Correlation) - vol changes very fast (clusters) but correlation is relatively slower to change
    annual_cov = D @ corr @ D

    # cov. matrix needs to be symmetric (A = A^T) but floating point error might make it A12 = .500000001 and A21 = .500000000
    # so I just take average of them to make sure they are identical - avoids error
    annual_cov = 0.5 * (annual_cov + annual_cov.T)

    # Tikhonov Regularization
    # If two stocks are perfectly correlated we cannot invert it - so add small noise (QP solver inverts internally - Px=q => x = P^-1q)
    # If the two stocks are perfectly correlated determinant = 0 and then inverse formula = A^-1 = 1/ determinant... so error since 1/0
    # np.eye adds 1e-6 to the diagonal (variance) of every stock
    annual_cov += np.eye(num_stocks) * 1e-6

    # Check conditioning
    # returns an array (list) of pos, zero, or negative (zero means unstable here and negative means data is wrong)
    # finds variance of the data (portfolio) in a specific direction (steepness of curve)
    eigenvals = np.linalg.eigvals(annual_cov) # different than just checking diagonal since eignvalues tell us variance of factor 1, factor 2, etc (independent forces driving the market)
    min_eigenval = np.min(eigenvals) # find smallest eignvalue

    # If smallest eigenvalue is negative (matrix is broke) so we push all values up so smallest one becomes positive
    # * When we create an identity matrix here and do += and times a number by the identity matrix, we only increase diagonal values
    # which is the variance and we do not touch the correlations (the zeros here)

    if min_eigenval <= 1e-8: # want to avoid negative eigenvalue since that means variance is negative which isn't possible
        annual_cov += (abs(min_eigenval) + 1e-6) * np.eye(num_stocks) # np.eye creates identity matrix and then add 1e-6 on diagonals to shift everythig up
        print(f"Added regularization: min eigenval was {min_eigenval:.2e}")

    # Check condition number
    cond_num = np.linalg.cond(annual_cov) # finds ratio of max eigenvalue / min eigenvalue
    if cond_num > 1e8:  # Lower threshold since don't want it ill conditioned
        print(f"High condition number ({cond_num:.2e}), adding more regularization")
        annual_cov += 1e-5 * np.eye(num_stocks)

    # Verify Cholesky works
    try:
        # try this test to see if matrix is broken
        chol_test = np.linalg.cholesky(annual_cov) # must be positive definite (variance > 0)
    except np.linalg.LinAlgError:
        print("Cholesky failed, using eigenvalue decomposition fallback")
        # np.linalg.eigh breaks matrix into eigenvals (list of variance amounts) and eigenvecs (direction of variance)
        eigenvals, eigenvecs = np.linalg.eigh(annual_cov)
        eigenvals = np.maximum(eigenvals, 1e-6)  # turn any negative values into positive

        # Cov = Q * Lambda * Q^T : where Q = vectors and Lambda = np.diag and are positive values
        annual_cov = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T # use dot product


    N_scenarios = std_residuals.shape[0] # Monte Carlo paths (m=1,..,M)
    R_scenarios = (mus * dt) + (std_residuals @ D) * np.sqrt(dt)
    alpha_cvar = .95

    t_var = cp.Variable() # VaR canidate at alpha level
    u_slack = cp.Variable(N_scenarios) # captures how far losses exceed the VaR canidate

    # expected returns = excess returns + risk free rate => excess returns = expected returns - risk free rate
    # bank pays 5% rf and risky stock pays 5% why would we ever buy the stock - take risk for zero reward (excess = 0)
    # If stock pays 12% - first 5% is matching what bank gives us then remaining 7% is excess reward for taking vol risk
    excess_returns = mus - r_annual # need to make sure excess_return is not a list and is an np array

    # cp.psd_wrap is so CVXPY is required to make it positive semidefinite
    p = cp.psd_wrap(annual_cov * dt) # multiply by dt so on same scale (monthly)
    x = cp.Variable(num_stocks) # represents vector of length num_stocks - unknown until we solve
    q = -excess_returns * dt # flip sign to maximize profits, not minimize. If didn't want to flip signs, use cp.Maximize

    losess = -R_scenarios @ x # Represents the loss in scenario m
    cvar = t_var + (1 / ((1 - alpha_cvar) * N_scenarios)) * cp.sum(u_slack) # represents CVaR objective
    cvar = cvar / 20.0

    diversification_penalty = .03
    k = 20 

    # Gx <= h
    constraint = [x >= 0, x <= 1] # first constraint that we denote as a list

    # Ax = b
    constraint += [cp.sum(x) == 1] # append x > 0 and x <= 1 constraint into this constraint here

    constraint += [u_slack >= 0, u_slack >= losess -t_var] # comes from CVaR paper


    if lambda_global is not None:
      
      risk_multiplier = 1.0 + (lambda_global - 1.0) * (k * diversification_penalty)
      
    else:
      risk_multiplier = 1.0


    # Formula is (1/2)x^T * Px + q^Tx + r (not for x^T * Px - would do (x^T * P)*x)
    obj_exp = (.5) * cp.quad_form(x, p) + q.T @ x + diversification_penalty * cp.sum_squares(x) + (risk_multiplier * cvar)


    # Transaction cost:
    # with this, the formula is now ... + r + kappa * ||x - W_prev||_1
    if W_prev is not None:
      kappa = .002 # 20 basis points
      obj_exp += kappa * cp.norm(x-W_prev, 1) # use cp.norm since using cvxpy - cp.norm here is same as np.sum(np.abs())


    objective = cp.Minimize(obj_exp)
    opt = cp.Problem(objective, constraint) # defines problem into singular object


    try:
        opt.solve(solver=cp.CLARABEL, verbose=False) # takes der. to solve
    except:
        try:
            opt.solve(solver=cp.ECOS, verbose=False)
        except:
            opt.solve(solver=cp.SCS, verbose=False)

    # Check if solved successfully
    if opt.status not in ['optimal', 'optimal_inaccurate']: # if solver fails
        print(f'QP solver status: {opt.status}')
        if lambda_global is not None:

          # why did solver fail - print statements:
          print(f"Solver failed. Lambda (Penalty Strength) was: {lambda_global:.2f}")
          print(f"Current Wealth: {W_t:.2f}, Target Floor: {0.97:.2f}")
        pi_star = sample_control_random(num_stocks)
    elif x.value is None or np.allclose(x.value, 0): # first part says: solver 'succeeded' but sol. is empty; second part says: solver 'succeed' but sol. is all zeros
        pi_star = sample_control_random(num_stocks)
    else: # if successful:
        pi_star = x.value
        pi_star = np.clip(pi_star, 0, 1)
        pi_star /= pi_star.sum()


    return pi_star, annual_cov


def sample_control_random(num_stocks):
    while True:
      Pi_star = np.random.rand(num_stocks) # uniform [0,1)
      Pi_star /= Pi_star.sum() # normalize to sum <= 1
      if np.all(Pi_star <= .7):
        return Pi_star


def transformed_regression(V_t):
  V_t_safe = np.clip(V_t,-100, 100)
  return np.arcsinh(V_t_safe)

def inverse_transformed_regression(z):
  z_safe = np.clip(z, -100, 100)
  return np.sinh(z_safe)



#The bad outcomes lower utility significantly
#The good outcomes raise utility only slightly
#The net effect = lower expected CRRA utility -> JENSONS INEQUALITY E[U(W)] < U(E[W])

# Note CRRA utility is concave -> U^ll(W) < 0
def crra_utility(x, gamma):
    # CRRA utility: if eta == 1 -> log, else power
    eps = 1e-12
    if gamma == 1.0:
        return np.log(np.maximum(x, eps))
    else:
        return (np.maximum(x, eps)**(1.0 - gamma)) / (1.0 - gamma)


def MCsimMulti(initial_wealth, r_annual, num_stocks, gamma, rho, train_data = None):
    np.random.seed(1)
    random.seed(1)
    policy_brain = {}


    stock_returns_df = train_data
    T = len(train_data) / 252
    time_steps = int(T * 12)


    # Note if want to change to longer time_steps, would make 240 and rebalance_interval_steps_forward = 2
    #time_steps = 120 # changed from 200 to 120 since 10 * 12 = 120, dt = 10 / 120 = 1/12 = 1 month per step
    simulations = 500

    c_star = .03 # Constant consumption to start with

    dt = T / time_steps

    days_per_step = 21 # (monthly - approx 21 days per month)

    beta = np.exp(-rho * dt)

    # extract first 252 days in each column (every column is different ticker)
    # So pretty much gets every tickers, 252 days of info (or each row ig)
    window = 252

    # allows us to store all pi* paths - necessary for PFI (realized value)
    # start at time_step 0 and go to 2000

    # Note for these arrays, axis = 0 means collapse time, keep simulations : axis = 1 means collapse simulations, keep time
    # * If we want 1 number per sim, axis = 0
    # * If we want 1 number per time, axis = 1
    pi_star = np.zeros((time_steps + 1, simulations, num_stocks))
    chol = np.zeros((time_steps + 1, num_stocks, num_stocks)) # might not need sim here and replace with num_stocks?
    mu_storage = np.zeros((time_steps, num_stocks))

    W = np.zeros((time_steps + 1, simulations))
    W[0, :] = initial_wealth

    current_pi = np.zeros(num_stocks) # buying first portfolio is our first transaction - starting weight of 0 (or all cash)
    kappa = .002 # 20 basis points for QP - cost per dollar traded (fee rate)


    # Add random shocks outside for loop - pre-generate so we can re-use it in backwards loop
    Z = np.random.normal(0,1,size=(time_steps, num_stocks, simulations))

    # 60% of time we use a random control - allows backward pass to see variation
    # allowing it to be non deterministic
    p_explore = .6 

    rebalancing_timesteps_forward = []
    rebalance_interval_steps_forward = 1 
    t_next_forward = 0
    while t_next_forward < time_steps:
        rebalancing_timesteps_forward.append(t_next_forward)
        t_next_forward += rebalance_interval_steps_forward
    rebalancing_timesteps_forward = set(rebalancing_timesteps_forward)


    sim_ret = np.zeros((time_steps, num_stocks, simulations))

    count_randm = 0
    count_opt = 0


    for t in range(time_steps):
        dB = Z[t, :, :] * np.sqrt(dt)

        if t > 0: # makes it so keeps same portfolio weights when not on rebalancing step
            pi_star[t, :, :] = pi_star[t-1, :, :]
        if t in rebalancing_timesteps_forward:

            end_idx = window + t * days_per_step
            end_idx = min(end_idx, len(stock_returns_df))
            start_idx = end_idx - window

            current_window_stock_returns = stock_returns_df.iloc[start_idx:end_idx] # get starting window for returns


            rolling_vols = []
            rolling_std_resids = []

            for col in current_window_stock_returns.columns:
              series = current_window_stock_returns[col]

              try:

                # dist = t to account for fat tails (kurtosis) - allowing for multiple std dev away
                # egarch prevents negative variance and spikes vol more when returns are negative than when they are positive (leverage effect)
                # multiply by 100 since garch struggles with smaller numbers
                # p = yesterdays returns (p=1) squared - did market move a lot yesterday, o = yesterdays sign (o=1) - did market crash (neg) or rally (pos)
                # q = yesterdays predicted vol (q=1) - think of as was I already panicked yesterday - if vol was then stays rel high td unless market goes quite
                model = arch_model(series * 1000, vol = 'GARCH', p = 1, q = 1, dist = 't')
                results = model.fit(disp = 'off')

                if results.convergence_flag !=0:
                  raise ValueError("Didn't converge")

                # vector of length daily_log_returns
                current_vols_t = results.conditional_volatility.iloc[-1] / 1000
                z = series / (results.conditional_volatility / 1000) # units are daily here but doesnt matter since finding residuals which are unitless

              except:
                simple_std = series.std()
                current_vols_t = simple_std
                z = series / simple_std

              rolling_vols.append(current_vols_t * np.sqrt(252)) # annual vol
              rolling_std_resids.append(z.values)


            D = np.diag(rolling_vols) # take last day in window - grabs last row (recent date) across all columns (all stocks)
            current_window_std_residuals = np.column_stack(rolling_std_resids) # get window for std residuals

            # .values converts to numpy array
            rolling_mus = current_window_stock_returns.mean(axis=0).values * 252
            current_blended_mus = rolling_mus

            potential_pi_star, annual_cov_forward = getMultiQuadProg(current_blended_mus, r_annual, num_stocks, gamma, D, current_window_std_residuals, c_star, W[t, :], W_prev = current_pi)

            if np.random.rand() < p_explore:
                pi_tm = sample_control_random(num_stocks) # added to reduce bias in forward path and explore
                count_randm += 1
            else:
                pi_tm = potential_pi_star
                count_opt += 1

            # transaction costs:
            turnover = np.sum(np.abs(pi_tm - current_pi)) # how much we trade (buy) (in portfolio weight "terms")
            W[t, :] -= W[t,:] * turnover * kappa # where turnover = amt we buy and kappa = price per unit

            # update:
            current_pi = pi_tm

            # stores pi* at every simulation - allows for low variance (don't want to much noise)
            pi_star[t, :, :] = np.tile(pi_tm, (simulations, 1)) # computes optimal pi* and then applies it to all mc sims


            chol_tm = np.linalg.cholesky(annual_cov_forward)

        mu_storage[t, :] = current_blended_mus
        chol[t, :] = chol_tm

        # Note in paper it says theta = n_t * S_t -> so subbing in theta for those two
        # we get dW_t = r(W_t - theta_t)dt + theta_t * dS_t/S_t - c_t dt and note instead of working
        # with n_t (shares) we work directly with portfolio weights

        # return of all assets at this time step
        # note dR = dS / S
        dR = (chol_tm @ dB) + current_blended_mus.reshape(-1,1) * dt
        sim_ret[t, :, :] = dR # stores each simulation return

        # reshape here so both are 2d
        # Need to update optimal weights continously - dollar amount in each asset at time t
        # reshape creates column of stocks (num_stocks, 1) where W[t,:] is a row of simulations (1, simulations)
        # multiplying them gives us 2d matrix with shape (num_stocks, simulations)
        Pi_star_t = pi_tm.reshape(-1,1) * W[t, :] # pi_star_t = theta_t: Note we skip over n_t here (amt in each asset)

        # profit / loss from dollar amount in each asset (multiply dollars in each asset by returns)
        # axis = 0 collapses means we want 1 # per simulation (collapse stocks)
        wealth_in_asset = np.sum(Pi_star_t * dR, axis = 0)

        # Cash that goes towards risk-less assets (dollars that aren't invested in risky assets).
        # Wealth - portfolio value = cash (Wealth in bank)
        cash = W[t, :] - np.sum(Pi_star_t, axis = 0) # cash currently = 0 since portfolio weights sum to 1

        # Optimal consumption at each step
        c_star_t = c_star * W[t, :]
        consumption = c_star_t * dt

        # updates Wealth
        W[t + 1, :] = W[t, :] + (r_annual * cash * dt) + wealth_in_asset - consumption
        W[t + 1, :] = np.maximum(W[t + 1, :], 1e-6) # did bc optimizer was failing with 7 assets


    # Val func for cont. values:
    V = np.zeros((time_steps + 1, simulations))
    V[time_steps, :] = crra_utility(W[time_steps,:], gamma)

    rebalancing_timesteps = []
    rebalance_interval_steps = 1
    t_next = 0
    while t_next < time_steps:
        rebalancing_timesteps.append(t_next)
        t_next += rebalance_interval_steps
    rebalancing_timesteps = set(rebalancing_timesteps)

    # for smoothing - Crossetts suggestion
    alpha_coef = .3
    coef_next = None
    #intercept_next = None


    dynamic_c_star_matrix = np.zeros((time_steps + 1, simulations)) # used for consumption

    W_tilde = W.copy()
    for t in range(time_steps-1, -1, -1):
        eps = 1e-10


        # Note to self: wealth at time t is known whereas shock (Z) is not since it corresponds to t > t+1
        W_t = W_tilde[t, :]

        pi_star_tm = pi_star[t, :, :].copy()

        mu_t = mu_storage[t,:]

        chol_t = chol[t, :, :]

        sim = len(W_t) # need to broadcast scalars here

        W_t = np.clip(W_t, 1, 1e10)  # avoid zero/negative issues
        log_W_t = np.log(np.clip(W_t,1e-6,1e10) / initial_wealth)

        pi_star_tm = np.nan_to_num(pi_star_tm, nan=0.0) # clears NaNs

        recent_wealth_change = np.log(W_t / W_tilde[max(0, t-5), :])

        # vol feature
        # formula for realized vol: sqrt(sum s=1 to t of (delta log W_s)^2)
        if t == 0:
          cum_vol = np.zeros(sim) # no history when t=0
        else:
          log_ret = np.diff(np.log(W_tilde[:t+1, :]), axis=0)
          cum_vol = np.sqrt(np.sum(log_ret**2, axis=0))

        W_history = W_tilde[:t+1, :]  # shape: (t+1, sim) - all paths from 0 to t

        # cumulative max for drawdown
        log_W_history = np.log(np.clip(W_history, 1e-6, 1e10))
        running_max_log = np.maximum.accumulate(log_W_history, axis=0) # keep track of highest wealth so far per sim
        log_drawdowns = running_max_log - log_W_history # finds diff between max vs history
        max_dd = np.max(log_drawdowns, axis=0) # figures out for each sim what was worst drop

        exp_smooth_wealth = np.zeros(sim)
        alpha = 0.3
        for k in range(5):
          weight = alpha * (1-alpha)**k
          W_past = W_tilde[max(0, t-k), :]
          log_W_past = np.log(np.clip(W_past,1e-6, 1e10) / initial_wealth)
          norm_W_past = log_W_past
          exp_smooth_wealth += weight * norm_W_past

        start = max(0, t-5)
        if start == t:
          past_shock_signal = np.zeros(sim) # at t=0 the mean of an empty slice = NaN so do this to prevent bug
        else:
          past_shock_signal = np.mean(Z[start:t, :, :], axis=(0,1)) # stop at t-1 since that is only information available before trading. t + 1 includes info after trade

        if t ==0:
          pi_prev = np.tile(np.ones(num_stocks) / num_stocks, (sim, 1)) # Set to equal weight so it doesn't crash
        else:
          pi_prev = pi_star[t-1, :, :]

        L = np.column_stack([
            log_W_t, # works
            log_W_t**2, # works
            log_W_t**3,
            log_W_t * np.max(pi_prev, axis=1), # works
            np.full(sim, t / time_steps), # works (and works at t=0)
            recent_wealth_change, # works
            cum_vol, # works
            past_shock_signal,
            max_dd, # works
            exp_smooth_wealth # works
        ])

        L = np.column_stack([L, np.random.normal(0, 0.01 * np.std(log_W_t), size=sim)]) # works

        # Assign zero-variance to be equal to 0
        # get std. of all columns (axis=0)
        feature_stds = np.std(L, axis=0)
        L[:, feature_stds < 1e-10] = 0.0


        scaler = StandardScaler()
        L_asset = scaler.fit_transform(L)


        V_transformed = transformed_regression(beta * V[t + 1, :]) # V + 1 since 'target' is future wealth

        if np.isnan(L_asset).any() or np.isnan(V_transformed).any():
          print(f"NaNs detected at t={t}")
          print("NaNs in L_asset columns:", np.any(np.isnan(L_asset), axis=0))
          print("NaNs in V_transformed:", np.any(np.isnan(V_transformed)))
          break

        L_asset = np.nan_to_num(L_asset, nan=0.0, posinf=1e6, neginf=-1e6)
        V_transformed = np.nan_to_num(V_transformed, nan=0.0, posinf=1e6, neginf=-1e6)


        reg = Ridge(alpha = 1e-6, fit_intercept=True).fit(L_asset, V_transformed)

        coef_hat = reg.coef_.copy() # shape (n_features,) - betas
        intercept_hat = reg.intercept_ # scalar - y-intercept

        if coef_next is None:
          #first time in loop: t = time_steps - 1
          coef_smooth = coef_hat
          intercept_smooth = intercept_hat
        else:
          if coef_smooth.shape == coef_hat.shape:
            coef_smooth = alpha_coef * coef_hat + (1 - alpha_coef) * coef_smooth
            intercept_smooth = alpha_coef * intercept_hat + (1 - alpha_coef) * intercept_smooth
          else: # if shapes don't match, just assign coef_hat to coef_smooth
            coef_smooth = coef_hat
            intercept_smooth = alpha_coef * intercept_hat + (1 - alpha_coef) * intercept_smooth

        # update next iterations
        coef_next = coef_smooth
        #intercept_next = intercept_smooth

        # this does the reg.predict(L_asset) but for smoothed coefficients
        phi_t = L_asset @ coef_smooth + intercept_smooth

        residuals = V_transformed - phi_t # residuals for smearing estimate

        # Heteroskedackicity control in Andreasson paper (appendix 2 has good explanation):

        resid_sq = residuals**2
        y_var = np.log(resid_sq + eps)

        var_reg = Ridge(alpha = 1e-3, fit_intercept=True).fit(L_asset, y_var)

        log_psi = var_reg.predict(L_asset) # this is X^T * betahat (in notes)
        psi = np.exp(np.clip(log_psi, -20, 20))

        psi = np.maximum(psi, eps)

        # psi_t / psi_m in Andreasson paper
        psi_ratio = np.sqrt(psi[:, None] / psi[None, :])

        scaled_residuals = residuals[None, :] * psi_ratio

        phi_t_smearing = np.mean(inverse_transformed_regression(phi_t[:, None] + scaled_residuals), axis=1)
        phi_t_smearing = np.nan_to_num(phi_t_smearing, nan=np.nanmean(phi_t_smearing), posinf=1e10, neginf=-1e10) # added so doesn't blow up over big or small returns

        policy_brain[t] = {
          'model': reg, # The Ridge model
          'scaler': scaler, # The StandardScaler
          'bin_policies': {} # Empty for now, filled if rebalancing
        }

        if t in rebalancing_timesteps:

            end_idx = window + t * days_per_step
            end_idx = min(end_idx, len(stock_returns_df))
            start_idx = end_idx - window


            current_window_stock_returns = stock_returns_df.iloc[start_idx:end_idx]


            rolling_vols = []
            rolling_std_resids = []

            for col in current_window_stock_returns.columns:
              series = current_window_stock_returns[col]
              try:
                model = arch_model(series * 1000, vol = 'GARCH', p =1,q = 1, dist = 't')
                results = model.fit(disp = 'off')

                if results.convergence_flag !=0:
                  raise ValueError("Didn't converge")

                current_vols_t = results.conditional_volatility.iloc[-1] / 1000
                z = series / (results.conditional_volatility / 1000)
              except:
                simple_Std = series.std()
                current_vols_t = simple_Std
                z = series / simple_Std

              rolling_vols.append(current_vols_t * np.sqrt(252))
              rolling_std_resids.append(z.values)


            D = np.diag(rolling_vols)
            current_window_std_residuals = np.column_stack(rolling_std_resids)

            rolling_mus = current_window_stock_returns.mean(axis=0).values * 252

            current_blended_mus = rolling_mus


            phi_mean_t = np.mean(phi_t_smearing)
            phi_std_t = np.std(phi_t_smearing) + 1e-10 # no divide by zero error

            phi_normalized = (phi_t_smearing - phi_mean_t) / phi_std_t # z-score

            phi_normalized = np.clip(phi_normalized, -3, 3) # prevents #s from becoming to big


            # use percentiles to create 5 bins (and 6 "walls") that adapt to current wealth levels
            wealth_percentiles = [0, 20, 40, 60, 80, 100]
            wealth_thresholds = np.percentile(W_t, wealth_percentiles) # gets us the threshold for each wealth path. So bin 0 might be 500-644, bin 1: 644-786, etc

            # make sure the threshold "walls" go up
            # for ex: at start, all thresholds might be 1000 (starting wealth)
            # so if thresholds[1] = 1000 and thresholds[0] = 1000, then thresholds[1] = 1000 + 1e-6
            # thresholds[2] = 1000 and thresholds[1] = 1000 + 1e-6, then thresholds[2] = thresholds[1] + 1e-6
            for i in range(1, len(wealth_thresholds)):
              if wealth_thresholds[i] <= wealth_thresholds[i-1]:
                wealth_thresholds[i] = wealth_thresholds[i-1] + 1e-6

            W_reference = np.median(W_t) # compute reference wealth (median)


            policy_brain[t]['wealth_thresholds'] = wealth_thresholds.copy()
            policy_brain[t]['W_reference'] = W_reference

            # assign paths to wealt bins
            # Looks at W_t values and the wealth thresholds and determines what index wealth is at
            # ex: [500, 700, 900] and w_t = 800, then return 2 (past 500 and 700). 2 - 1 = index 1
            bin_assignments = np.digitize(W_t, wealth_thresholds) - 1 # subtract 1 since digitize starts at index 1,2, etc not index 0

            # ex: if we have 6 "walls" (0, 20, ..., 100), then we have 5 bins. Index starts at 0, so bins labeled 0-4. Highest index = 4
            # 6 - 2 = 4 which is why we have the - 2 there
            # if we have a path that say is lower than 500 (look at abv ex with [500, 700, 900]), we get index -1. Clipping it
            # makes it index 0. If we end up with more than 900, than it would be assigned to bin 1 (since would be 3 - 2)
            bin_assignments = np.clip(bin_assignments, 0, len(wealth_thresholds) - 2)

            num_bins = len(wealth_thresholds) - 1 # tells us how many bins we have


            # loop through bin_assignments
            for b in range(num_bins):
              # Find paths in this bin
              paths_in_bin = np.where(bin_assignments == b)[0] # returns singular array (thats why we do [0])

              if len(paths_in_bin) == 0:
                continue

              # use median since one path could get lucky so by using median we ignore outliers
              W_representative = np.median(W_t[paths_in_bin]) # find median wealth of that specific bin group


              # compare total median wealth to the current bin median wealth
              # if current bin median wealth < total median wealth -> penalty grows
              # current bin median wealth > total median wealth -> penalty shrinks
              ratio = W_representative / W_reference
              penalty_base = 2.5 * np.exp(-.002 * (ratio - 1.0) * 100) # ORIGINAL VALUE - OTHERS ARE FOR TEST SENSITIVITY ANALYSIS
              
              # take mean here instead of median since we want "expected value" of the future
              # using mean allows penalty to react if a few paths in bin look low and allow it to not ignore tail risk
              # while with median it would ignore the lower values
              avg_phi_normalized = np.mean(phi_normalized[paths_in_bin]) # avg of normalized phi

              # smaller phi = worse future and makes # grow (phi < 1) -> bigger phi = better future and makes # shrink (phi > 1)
              phi_multiplier = np.exp(-1.5 * avg_phi_normalized) # ORIGINAL VALUE
              
              penalty_final = penalty_base * phi_multiplier # multiplying allows us to amplify or cancel each other out
              penalty_final = np.clip(penalty_final, 1.0, 20.0) # ORIGINAL VALUE
              
              c_min, c_max = .01, .06
              # clip consumption to be at max "twice as rich" even if you might be 100x richer than median
              # if you are broke, we treat you as "half as rich"
              ratio = np.clip(W_representative / W_reference, 0.5, 2.0) # rel. wealth -> clipped between .5 and 2

              # normalize to 0 and 1
              normalize_c = (ratio - 0.5) / (2.0 - 0.5)
              c_base = c_min + (c_max - c_min) * normalize_c # pick # between min and max based on normalize_c -> richer you are, more you consume

              c_adjustment = 1.0 + 0.15 * avg_phi_normalized # higher phi -> increase in consumption (strength controlled by .15)
              c_adjustment = np.clip(c_adjustment, 0.5, 1.5)
              c_final = c_base * c_adjustment
              c_final = np.clip(c_final, c_min, c_max)


              p_prev = np.median(pi_star[max(0, t-1), paths_in_bin, :], axis=0)

              pi_for_bin, annual_cov_backward = getMultiQuadProg(
                current_blended_mus, r_annual, num_stocks, gamma, D,
                current_window_std_residuals, c_final, W_representative,
                W_prev=p_prev, lambda_global=penalty_final, phi_val = avg_phi_normalized
              )

              policy_brain[t]['bin_policies'][b] = {
                'consumption': c_final,
                'penalty': penalty_final,
                'W_representative': W_representative,
                'weights': pi_for_bin.copy(),
                'mus_at_t': current_blended_mus.copy(),
                'cov_at_t': annual_cov_backward.copy()
              }

              # assigns same portfolio weights and consumption to people who fall in same bin
              for idx in paths_in_bin:
                pi_star[t, idx, :] = pi_for_bin
                dynamic_c_star_matrix[t, idx] = c_final

                # makes sure that we use the new weights above in any non rebalancing time steps
                # ex: if rebalance happens at t =10, 13, 16 -> start at t =10, curr_t = 11, 11 not in 10,13,16
                # loop runs and sets pi_star[11] = pi_star[10], pi_star[12] = pi_star[10], when t = 13, stops.
                curr_t = t + 1
                while curr_t < time_steps and curr_t not in rebalancing_timesteps:
                  pi_star[curr_t, idx, :] = pi_for_bin
                  curr_t += 1

            # outside loop since market condition at t+1 shouldn't use market conditions
            # at t. Compared to pi* weights that keep weights until next rebalance step
            # *Also worth noting that chol and mu_storage should be same numbers from forward pass
            # pi* is path dependent (different for every sim) and a decision needs propagation
            # chol and mu_storage are time dependent (same for all simulations at time t) and market states (already recorded in forward pass)
            chol_tm = np.linalg.cholesky(annual_cov_backward)

            chol[t, :, :] = chol_tm # update cholesky with optimizted version
            mu_storage[t, :] = current_blended_mus

        V_realized = np.zeros(simulations)

        for m in range(simulations):
            cum_reward = 0
            discount = 1

            Wtemp = W_tilde[t,m]

            p_prev = pi_star[max(0, t-1), m, :]
            for j in range(t, time_steps):
                pi_star_j = pi_star[j, m, :]

                # only have transaction costs when at rebalancing step.
                # if not at rebalance time - then we skip transaction costs
                if j in rebalancing_timesteps:
                  turnover = np.sum(np.abs(pi_star_j - p_prev))
                  cost = Wtemp * turnover * kappa
                  Wtemp -= cost
                  p_prev = pi_star_j

                chol_j = chol[j, :, :]

                mu_j = mu_storage[j, :]

                dB_j = Z[j, :, m] * np.sqrt(dt)

                dR_j = (chol_j @ dB_j) + mu_j * dt
                
                c_j = dynamic_c_star_matrix[j, m] * Wtemp

                cum_reward += discount * crra_utility(c_j * dt, gamma)

                cash = Wtemp - np.sum(pi_star_j * Wtemp) # np.sum(...) is portfolio_value

                Wtemp = Wtemp + (r_annual * cash *dt) + np.sum(pi_star_j * Wtemp * dR_j) - c_j * dt

                W_tilde[j + 1, m] = Wtemp

                discount *= beta

            cum_reward += discount * crra_utility(Wtemp, gamma)
            V_realized[m] = cum_reward
        V[t, :] = V_realized


    print('In sample training: ')

    # Terminal wealth statistics
    mean_wealth = np.mean(W_tilde[-1, :])
    median_wealth = np.median(W_tilde[-1, :])
    std_wealth = np.std(W_tilde[-1, :])

    print(f"\nTerminal Wealth After {T} Years:")
    print(f"Initial Wealth: ${initial_wealth:,.2f}")

    print(f"Mean Wealth in forward loop: ${np.mean(W[-1,:]):,.2f}")
    print(f"Mean Final Wealth: ${mean_wealth:,.2f}")

    print(f"Median Wealth in forward loop: ${np.median(W[-1,:]):,.2f}")
    print(f"Median Final Wealth: ${median_wealth:,.2f}")

    print(f"Std Dev Wealth in forward loop: ${np.std(W[-1,:]):,.2f}")
    print(f"Final Std Dev: ${std_wealth:,.2f}")


    # Sharpe ratio

    # formula for Sharpe is S = (E(R_p) - R_f) / sigma_p
    # Sharpe treats all volatility as risk. If portfolio jumps up 20%, sharpe penalizes you as much if it jumped down 20%


    # Annual Sharpe:
    annual_returns = (W_tilde[-1, :] / initial_wealth)**(1 / T) - 1
    excess_annual = annual_returns - r_annual
    sharpe_annual = (
      excess_annual.mean() / excess_annual.std()
      if excess_annual.std() > 0 else 0.0
    )

    print(f"Sharpe Ratio (annualized) after backward loop: {sharpe_annual:.3f}")

    # Value at Risk (5%) and Expected Shortfall (Conditional VaR)
    VaR_5 = np.percentile(W_tilde[-1, :], 5)
    ES_5 = np.mean(W_tilde[-1, W_tilde[-1, :] <= VaR_5]) # boolean indexing - only picks simulation where outcoe was in worst 5%
    print(f"5% VaR: ${VaR_5:,.2f}, 5% Expected Shortfall: ${ES_5:,.2f}")

    print(f"\n" + "-"*60)

    return policy_brain


def backtest_policy(initial_wealth, test_returns_df, policy_brain, r_annual, num_stocks, gamma, tickers, rho_disc):
    np.random.seed(1)
    random.seed(1)
    
    current_wealth = initial_wealth

    penalty_history = []
    init_weights = None # t =0

    # keep a history of log wealth to calculate specific features
    wealth_history = [initial_wealth]
    log_W_history = [np.log(initial_wealth / initial_wealth)] # Starts at 0
    consumption_history = []

    predicted_phi_radar = []

    weight_history = []

    current_weights = np.zeros(num_stocks)
    days_per_step = 21
    window = 252
    kappa = 0.002

    snap_idx = {0, 29, 59}   # t=0, mid, end for 60 steps

    snapshots = {
        "t": [],
        "wealth": [],
        "penalty": [],
        "weights": []
    }

    print('Out of sample:')
    print()
    print("\nLSMC SNAPSHOTS")
    print("t\tWealth\t\tPenalty")


    for t in range(60):
        # STEP 1: NEED TO REPLICATE THE 11-FEATURE STATE VECTOR (L)
        W_t = current_wealth
        log_W_t = np.log(np.maximum(W_t / initial_wealth, 1e-10))

        # Feature 4: log_W * max(pi_prev)
        feat_4 = log_W_t * np.max(current_weights)

        # Feature 5: Time Progress (In training this was np.arange(sim)/sim,
        # but for a single live path, we use the time index t/12)
        feat_5 = t / 60

        # Feature 6: recent_wealth_change (last 5 months)
        if len(log_W_history) >= 5:
            feat_6 = log_W_t - log_W_history[-5]
        else:
            feat_6 = log_W_t - log_W_history[0]

        # Feature 7: cum_vol (Realized volatility of the log wealth path)
        if len(log_W_history) > 1:
            feat_7 = np.sqrt(np.sum(np.diff(log_W_history)**2))
        else:
            feat_7 = 0.0

        # Feature 8: past_shock_signal (Training used Z mean, backtest uses 0)
        feat_8 = 0.0 # not observable during out of sample training

        # Feature 9: max_dd (Maximum Drawdown of log wealth)
        log_W_history_arr = np.array(log_W_history)
        running_peak = np.maximum.accumulate(log_W_history_arr)
        drawdown_series = running_peak - log_W_history_arr
        feat_9 = np.max(drawdown_series) # The worst drop experienced so far

        # Feature 10: exp_smooth_wealth (EMA)
        alpha_ema = 0.3
        ema_W = 0.0
        for k in range(min(5, len(log_W_history))):
            ema_W += (alpha_ema * (1-alpha_ema)**k) * log_W_history[-(k+1)]
        feat_10 = ema_W

        # Construct the 11-feature vector (Matches your L exactly)
        state = np.array([
            log_W_t, log_W_t**2, log_W_t**3, # 0, 1, 2
            feat_4, feat_5, feat_6,          # 3, 4, 5
            feat_7, feat_8, feat_9,          # 6, 7, 8
            feat_10, 0.0                     # 9, 10 (10 is noise=0)
        ]).reshape(1, -1)

        # STEP 2: CONSULT THE POLICY (BIN LOGIC)
        rule = policy_brain[t]

        # Use the rule's scaler to transform the state
        state_scaled = rule['scaler'].transform(state)


        # Determine which Bin the current wealth falls into
        wealth_thresholds = rule['wealth_thresholds']
        bin_idx = np.digitize([current_wealth], wealth_thresholds)[0] - 1
        bin_idx = np.clip(bin_idx, 0, len(wealth_thresholds) - 2)

        # Get bin-specific parameters
        policy = rule['bin_policies'].get(bin_idx, {'consumption': 0.03, 'penalty': 5.0}) # if bin is missing somehow, fallback
        c_star_t = policy['consumption']
        penalty_t = policy['penalty']

        penalty_history.append(penalty_t)

        consumption_history.append(c_star_t)

        phi_t = rule['model'].predict(state_scaled)[0]
        predicted_phi_radar.append(phi_t)

        # STEP 3: MARKET ESTIMATION
        end_idx = window + (t * days_per_step)

        hist_available = test_returns_df.iloc[:end_idx]
        hist_window = hist_available.iloc[-window:]

        # Internal mu calculation
        rolling_mus = hist_window.mean(axis=0).values * 252

        # GARCH Vol (Using 1000 multiplier to match in sample - and for stability)
        vols, resids = [], []
        for col in hist_window.columns:
          series = hist_window[col]
          try:
            res = arch_model(hist_window[col]*1000, vol='GARCH', p=1, q=1).fit(disp='off')

            if res.convergence_flag != 0:
              raise ValueError("Didn't converge")

            vols.append((res.conditional_volatility.iloc[-1]/1000) * np.sqrt(252))
            resids.append((hist_window[col] / (res.conditional_volatility/1000)).values)
          except:
            simple_vol = series.std()
            vols.append(simple_vol * np.sqrt(252))
            resids.append((series / simple_vol).values)


        # STEP 4: SOLVE QP & APPLY REAL RETURNS
        new_weights, _ = getMultiQuadProg(rolling_mus, r_annual, num_stocks, gamma,
                                          np.diag(vols), np.column_stack(resids),
                                          c_star_t, current_wealth, W_prev=current_weights,
                                          lambda_global=penalty_t)

        weight_history.append(new_weights)


        if t==0:
          init_weights = new_weights.copy()

        # Turnover Costs
        turnover = np.sum(np.abs(new_weights - current_weights))
        current_wealth *= (1 - turnover * kappa)
        # *ONLY UNCOMMENT OUT THE BELOW LINE IF YOU WANT DYNAMIC CONSUMPTION
        # *LINE IS PURPOSELY COMMENTED OUT TO TEST STRATEGY AGAINST OTHER MODELS
        # *USING A FAIR COMPARISON OF CONSUMPTION, i.e, 3%
        #current_wealth -= current_wealth * c_star_t * (1/12) # consumption
        
        # *CONSUMPTION OF 3% TO MATCH OTHER STRATEGIES
        current_wealth -= current_wealth * .03 * (1/12)

        # Monthly return application
        actual_returns = test_returns_df.iloc[end_idx : end_idx + days_per_step]
        limit = min(end_idx + days_per_step, len(test_returns_df))
        actual_returns = test_returns_df.iloc[end_idx : limit]

        daily_port_ret = actual_returns.values @ new_weights
        portfolio_return = daily_port_ret.sum()
        current_wealth *= np.exp(portfolio_return)


        # Update path history for the next month's features
        current_weights = new_weights
        wealth_history.append(current_wealth)
        log_W_history.append(np.log(current_wealth / initial_wealth))


        if t in snap_idx:
            snapshots["t"].append(t)
            snapshots["wealth"].append(current_wealth)
            snapshots["penalty"].append(penalty_t)
            snapshots["weights"].append(new_weights.copy())
        
            print(f"Snapshot t={t}, Wealth=${current_wealth:.0f}, Pen={penalty_t:.2f}")


    return current_wealth, wealth_history, consumption_history, penalty_history, init_weights, np.array(weight_history)




def backtest_policy_CPPI(initial_wealth, test_returns_df, r_annual, num_stocks):
    multiplier = 3.0
    floor = initial_wealth * 0.85 # Protect 85% of initial wealth

    current_wealth = initial_wealth
    wealth_history = [initial_wealth]
    consumption_history = [0.03] * 60
    days_per_step = 21
    window = 252
    kappa = .002
    eq_weights = np.ones(num_stocks) / num_stocks
    prev_risky_weight = 0.0


    for t in range(60):
        # 1. Calculate Cushion
        cushion = max(0, current_wealth - floor)

        # 2. Calculate Exposure to Risky Assets
        risky_exposure_dollars = cushion * multiplier

        # Limit exposure to 100% of wealth (no leverage)
        risky_weight = min(1.0, risky_exposure_dollars / current_wealth)
        safe_weight = 1.0 - risky_weight

        # 3. Apply Returns
        end_idx = window + (t * days_per_step)
        actual_returns = test_returns_df.iloc[end_idx : min(end_idx + days_per_step, len(test_returns_df))]

        # Risky Part (Equal weight portfolio)
        risky_ret = (actual_returns.values @ eq_weights).sum()
        # Safe Part (Risk free rate)
        safe_ret = np.log(1 + r_annual * (1/12))

        total_ret = (risky_weight * risky_ret) + (safe_weight * safe_ret)

        # transaction costs:
        turnover = abs(risky_weight - prev_risky_weight)  # scalar since CPPI has 1 risky bucket
        current_wealth *= (1 - turnover * kappa)

        prev_risky_weight = risky_weight

        current_wealth -= (current_wealth * 0.03 * (1/12)) # Consumption
        current_wealth *= np.exp(total_ret)
        wealth_history.append(current_wealth)

    return current_wealth, wealth_history, consumption_history


def get_HRP_weights(returns_window):
    cov = returns_window.cov().values
    corr = returns_window.corr().values

    # 1. Clustering
    d_mat = np.sqrt(np.clip(0.5 * (1 - corr), 0, 1))
    dist = squareform(d_mat, checks=False)
    link = linkage(dist, 'single')

    def get_quasi_diag(link):
        link = link.astype(int)
        sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
        num_items = link[-1, 3]
        while sort_ix.max() >= num_items:
            sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
            df0 = sort_ix[sort_ix >= num_items]
            i = df0.index
            j = df0.values - num_items
            sort_ix[i] = link[j, 0]
            df0 = pd.Series(link[j, 1], index=i + 1)
            sort_ix = pd.concat([sort_ix, df0])
            sort_ix = sort_ix.sort_index()
            sort_ix.index = range(sort_ix.shape[0])
        return sort_ix.tolist()

    sort_ix = get_quasi_diag(link)

    def get_cluster_var(cov, cluster_indices):
        sub_cov = cov[np.ix_(cluster_indices, cluster_indices)]
        inv_diag = 1.0 / np.diag(sub_cov)
        w = inv_diag / inv_diag.sum()
        return float(w.T @ sub_cov @ w)

    #  recursive bisection
    weights = pd.Series(1.0, index=sort_ix)
    clusters = [sort_ix]

    while clusters:
        new_clusters = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            mid = len(cluster) // 2
            c0 = cluster[:mid]
            c1 = cluster[mid:]

            v0 = get_cluster_var(cov, c0)
            v1 = get_cluster_var(cov, c1)

            alpha = 1 - v0 / (v0 + v1)  # allocate LESS to the higher-variance cluster
            weights[c0] *= alpha
            weights[c1] *= (1 - alpha)

            new_clusters.extend([c0, c1])
        clusters = new_clusters

    # Return in original stock order
    final_weights = np.zeros(len(sort_ix))
    for idx, w in weights.items():
        final_weights[idx] = w
    return final_weights



def calculate_backtest_stats(final_wealths, histories, consumption_histories, initial_wealth, r_annual, label):
    final_wealths = np.array(final_wealths)
    histories = np.array(histories)

    print(f"\n" + "="*50)
    print(f" STATS SUMMARY FOR: {label}")
    print("="*50)

    # Terminal Wealth Statistics
    print(f"\n" + "!"*30)
    print("FINAL BACKTEST PERFORMANCE SUMMARY")
    print("!"*30)
    print(f"Mean Final Wealth: ${np.mean(final_wealths):,.2f}")
    print(f"Median Final Wealth: ${np.median(final_wealths):,.2f}")
    print(f"Std Dev of Wealth: ${np.std(final_wealths):,.2f}")

    # Sharpe (Calculating monthly returns from all test windows)
    all_monthly_returns = []
    for path in histories:
        # path is a list of 13 wealth points (Month 0 to Month 12)
        returns = np.diff(path) / path[:-1]
        all_monthly_returns.extend(returns)


    all_monthly_returns = np.array(all_monthly_returns)
    rf_monthly = (1 + r_annual)**(1/12) - 1
    excess_returns = all_monthly_returns - rf_monthly


    # Sharpe
    sharpe = (np.mean(excess_returns) / np.std(excess_returns)) * np.sqrt(12)

    print(f"Backtest Sharpe Ratio: {sharpe:.3f}")

    VaR_5 = np.percentile(final_wealths, 5)

    # ES: The average of all outcomes that fell below the VaR
    below_var = final_wealths[final_wealths <= VaR_5]
    ES_5 = np.mean(below_var) if len(below_var) > 0 else VaR_5

    print(f"\nTail Risk Metrics (Across Test Windows):")
    print(f"5% Value at Risk (VaR): ${VaR_5:,.2f}")
    print(f"5% Expected Shortfall (ES): ${ES_5:,.2f}")

    # Drawdown
    max_dds = []
    for path in histories:
        path = np.array(path)
        peak = np.maximum.accumulate(path)
        dd = (peak - path) / peak
        max_dds.append(np.max(dd))

    print(f"Mean Maximum Drawdown: {np.mean(max_dds)*100:.2f}%")



def backtest_policy_RS_MV(initial_wealth, test_returns_df, r_annual, num_stocks, gamma, tickers):
    current_wealth   = initial_wealth
    wealth_history   = [initial_wealth]
    consumption_history = [0.03] * 60
    current_weights  = np.zeros(num_stocks)
    days_per_step    = 21
    window           = 252
    kappa            = 0.002
    c_star_constant  = 0.03

    snap_idx = {0, 29, 59}   # t=0, mid, end for 60 steps

    snapshots = {
        "t": [],
        "wealth": [],
        "penalty": [],
        "weights": []
    }
    
    print("\nHMM SNAPSHOTS")
    print("t\tWealth\t\tPenalty")
    for t in range(60):
        end_idx     = window + (t * days_per_step)
        hist_window = test_returns_df.iloc[:end_idx].iloc[-window:]

        rolling_mus = hist_window.mean(axis=0).values * 252

        # GARCH vol — identical to backtest_policy_MV
        vols, resids = [], []
        for col in hist_window.columns:
            series = hist_window[col]
            try:
                res = arch_model(hist_window[col] * 1000, vol='GARCH', p=1, q=1).fit(disp='off')
                if res.convergence_flag != 0:
                    raise ValueError("didn't converge")
                vols.append((res.conditional_volatility.iloc[-1] / 1000) * np.sqrt(252))
                resids.append((hist_window[col] / (res.conditional_volatility / 1000)).values)
            except Exception:
                simple_vol = series.std()
                vols.append(simple_vol * np.sqrt(252))
                resids.append((series / simple_vol).values)

        std_residuals   = np.column_stack(resids)
        returns_history = hist_window.values # *** raw returns for HMM

        # *Refit HMM every 3 months (every 3 steps) to pick up regime shifts
        refit = (t % 3 == 0)

        # *Call regime-switching solver instead of standard MV solver
        new_weights, _, _, penalty_t = getMultiQuadProgRegimeSwitching(
            mus             = rolling_mus,
            r_annual        = r_annual,
            num_stocks      = num_stocks,
            gamma           = gamma,
            D               = np.diag(vols),
            std_residuals   = std_residuals,
            c_star          = c_star_constant,
            W_t             = current_wealth,
            W_prev          = current_weights,
            returns_history = returns_history,
            n_regimes       = 2,
            refit_hmm       = refit,
        )

        # Apply returns & turnover — identical to backtest_policy_MV
        current_wealth *= (1 - np.sum(np.abs(new_weights - current_weights)) * kappa)
        current_wealth -= (current_wealth * c_star_constant * (1 / 12))
        actual_returns  = test_returns_df.iloc[end_idx : min(end_idx + days_per_step, len(test_returns_df))]
        current_wealth *= np.exp((actual_returns.values @ new_weights).sum())

        current_weights = new_weights
        wealth_history.append(current_wealth)

        if t in snap_idx:
            snapshots["t"].append(t)
            snapshots["wealth"].append(current_wealth)
            snapshots["penalty"].append(penalty_t)
            snapshots["weights"].append(new_weights.copy())
            print(f"Snapshot t={t}, Wealth=${current_wealth:.0f}, Pen={penalty_t:.2f}")
            
        
    return current_wealth, wealth_history, consumption_history



def walk_forward_master2(tickers, r_annual, num_stocks, gamma, rho, initial_wealth, start_date, end_date):
    np.random.seed(1)
    random.seed(1)
    _hmm_cache['model'] = None
    
    full_returns = getMultipleStocks_logReturns(tickers, start_date, end_date)
    train_days = 5 * 252
    test_days = 5 * 252

    # 1. LSMC Storage
    lsmc_w_list, lsmc_h_list, lsmc_c_list, lsmc_penalties = [], [], [], []

    # 2. MV Storage
    mv_w_list, mv_h_list, mv_c_list = [], [], []

    # 3. EW Storage
    ew_w_list, ew_h_list, ew_c_list = [], [], []

    #4. buy&hold
    b_w, b_h, b_c = [], [], []

    # 5. CPPI
    cppi_w_list, cppi_h_list, cppi_c_list = [], [], [] 

    # S&P 500
    spy_w_list, spy_h_list = [], []

    rs_w_list, rs_h_list, rs_c_list = [], [], []

    hrp_w_list, hrp_h_list, hrp_c_list = [], [], [] 

    kappa = .002
    window = 252

    step_size = 252 # makes window slide by 1 year instead of 5 years
    for start in range(0, len(full_returns) - (train_days + test_days), step_size):
        _hmm_cache['model'] = None
        train_slice = full_returns.iloc[start : start + train_days]
        test_slice = full_returns.iloc[start + train_days - 252 : start + train_days + test_days]
        
        print(f"\n" + "-"*60)
        print("WALK-FORWARD WINDOW")
        print(f"Training: {train_slice.index[0].date()} to {train_slice.index[-1].date()}")
        print(f"Testing: {full_returns.index[start + train_days].date()} to {full_returns.index[start + train_days + test_days - 1].date()}")
        print("-"*60)

        # --- A. LSMC ---
        brain = MCsimMulti(initial_wealth, r_annual, num_stocks, gamma, rho, train_slice)
        w, h, c, penalties, init_weights, weights_hist = backtest_policy(initial_wealth, test_slice, brain, r_annual, num_stocks, gamma, tickers, rho)
        lsmc_w_list.append(w); lsmc_h_list.append(h); lsmc_c_list.append(c), lsmc_penalties.extend(penalties)

        # --- B. Standard MV ---
        w_m, h_m, c_m = backtest_policy_MV(initial_wealth, test_slice, r_annual, num_stocks, gamma, tickers)
        mv_w_list.append(w_m); mv_h_list.append(h_m); mv_c_list.append(c_m)


        # C. Regime-Switching MV
        w_rs, h_rs, c_rs = backtest_policy_RS_MV(initial_wealth, test_slice, r_annual, num_stocks, gamma, tickers)
        rs_w_list.append(w_rs); rs_h_list.append(h_rs); rs_c_list.append(c_rs)

        # --- C. Equal-Weight (1/N) ---
        current_w_ew = initial_wealth
        h_ew = [initial_wealth]
        c_ew = [0.03] * 60 # Fixed 3% to match benchmark
        eq_weights = np.ones(num_stocks) / num_stocks

        for m in range(60):
            # Same math as the other backtesters
            end_idx = window + (m * 21)
            limit = min(end_idx + 21, len(test_slice))
            month_ret = np.exp((test_slice.iloc[end_idx : limit].values @ eq_weights).sum())
            current_w_ew -= current_w_ew * 0.03 * (1/12) # Subtract consumption
            current_w_ew *= month_ret
            h_ew.append(current_w_ew)

        ew_w_list.append(current_w_ew); ew_h_list.append(h_ew); ew_c_list.append(c_ew)


        # D. Static Baseline (Uses Month 1 LSMC weights and never changes them)
        w_bh = initial_wealth; h_bh = [initial_wealth]; c_bh = [0.03] * 60
        for m in range(60):
            end_idx = window + (m * 21)
            limit = min(end_idx + 21, len(test_slice))
            m_ret = np.exp((test_slice.iloc[end_idx : limit].values @ init_weights).sum())
            w_bh -= (w_bh * 0.03 * (1/12)) # Static consumption
            w_bh *= m_ret
            h_bh.append(w_bh)
        b_w.append(w_bh); b_h.append(h_bh); b_c.append(c_bh)


        # --- E. CPPI ---
        w_cp, h_cp, c_cp = backtest_policy_CPPI(initial_wealth, test_slice, r_annual, num_stocks)
        cppi_w_list.append(w_cp); cppi_h_list.append(h_cp); cppi_c_list.append(c_cp)


        # --- F. S&P 500 ---

        test_start = test_slice.index[0]
        spy_launch = pd.Timestamp("1993-01-29")

        if test_start >= spy_launch:
          benchmark_ticker = "SPY"
        else:
          benchmark_ticker = "^GSPC" # fallback

        spy = yf.Ticker(benchmark_ticker).history(start=test_start, end=test_slice.index[-1])
        # Calculate monthly simple returns for SPY
        spy_prices = spy['Close'].tz_localize(None).reindex(test_slice.index, method='ffill')
        spy_m_rets = np.log(spy_prices / spy_prices.shift(1)).fillna(0).values

        

        current_w_spy = initial_wealth
        h_spy = [initial_wealth]

        for m in range(60):
          end_idx = window + (m * 21)

          if end_idx >=len(spy_m_rets):
            break

          limit = min(end_idx + 21, len(spy_m_rets))
          r = spy_m_rets[end_idx:limit].sum()
          current_w_spy -= current_w_spy * 0.03 * (1/12)
          current_w_spy *= np.exp(r)

          h_spy.append(current_w_spy)


        spy_w_list.append(current_w_spy)
        spy_h_list.append(h_spy)


        curr_w_hrp = initial_wealth
        h_hrp = [initial_wealth]
        c_hrp = [0.03] * 60 # Fixed 3% to match benchmarks

        prev_w_hrp = np.zeros(num_stocks)
        for m in range(60):
            end_idx = window + (m * 21)
            hist_window = test_slice.iloc[end_idx-window : end_idx]
            actual_m_rets = test_slice.iloc[end_idx : min(end_idx+21, len(test_slice))]

            # --- HRP ---
            w_hrp = get_HRP_weights(hist_window)

            turnover_hrp = np.sum(np.abs(w_hrp - prev_w_hrp)) # transaction costs
            curr_w_hrp *= (1 - turnover_hrp * kappa)
            curr_w_hrp -= curr_w_hrp * 0.03 * (1/12) # Consumption
            curr_w_hrp *= np.exp((actual_m_rets.values @ w_hrp).sum())

            prev_w_hrp = w_hrp
            h_hrp.append(curr_w_hrp)


        hrp_w_list.append(curr_w_hrp); hrp_h_list.append(h_hrp); hrp_c_list.append(c_hrp)

    run_phi_forecasting_test(lsmc_penalties, lsmc_h_list)




    return lsmc_w_list, lsmc_h_list, lsmc_c_list, \
           mv_w_list, mv_h_list, mv_c_list, \
           rs_w_list,   rs_h_list,   rs_c_list, \
           ew_w_list, ew_h_list, ew_c_list, \
           b_w, b_h, b_c, \
           cppi_w_list, cppi_h_list, cppi_c_list, brain, weights_hist, lsmc_penalties, spy_h_list, hrp_w_list, hrp_h_list, hrp_c_list


def backtest_policy_MV(initial_wealth, test_returns_df, r_annual, num_stocks, gamma, tickers):
    current_wealth = initial_wealth
    wealth_history = [initial_wealth]
    consumption_history = [.03] * 60
    current_weights = np.zeros(num_stocks)
    days_per_step = 21
    window = 252
    kappa = 0.002
    c_star_constant = 0.03 # Standard consumption for MV
    
    snap_idx = {0, 29, 59}   # t=0, mid, end for 60 steps

    snapshots = {
        "t": [],
        "wealth": [],
        "weights": []
    }

    print("\nMV SNAPSHOTS")
    print("t\tWealth\t")

    for t in range(60):
        # 1. Causal Market Estimation (End at Today logic)
        end_idx = window + (t * days_per_step)
        hist_window = test_returns_df.iloc[:end_idx].iloc[-window:]

        rolling_mus = hist_window.mean(axis=0).values * 252

        # GARCH Vol (Using 1000 multiplier to stabalize / scale)
        vols, resids = [], []
        for col in hist_window.columns:
            series = hist_window[col]
            try:
              res = arch_model(hist_window[col]*1000, vol='GARCH', p=1, q=1).fit(disp='off')

              if res.convergence_flag != 0:
                raise ValueError("didn't converge")

              vols.append((res.conditional_volatility.iloc[-1]/1000) * np.sqrt(252))
              resids.append((hist_window[col] / (res.conditional_volatility/1000)).values)

            except:
              simple_vol = series.std()
              vols.append(simple_vol * np.sqrt(252))
              resids.append((series / simple_vol).values)
        # Solve QP (Using the MV solver)
        new_weights, _ = getMultiQuadProg_MV(rolling_mus, r_annual, num_stocks, gamma,
                                             np.diag(vols), np.column_stack(resids),
                                             c_star_constant, current_wealth, W_prev=current_weights)

        # Apply Returns & Turnover
        current_wealth *= (1 - np.sum(np.abs(new_weights - current_weights)) * kappa)
        current_wealth -= (current_wealth * c_star_constant * (1/12))
        actual_returns = test_returns_df.iloc[end_idx : min(end_idx + days_per_step, len(test_returns_df))]
        current_wealth *= np.exp((actual_returns.values @ new_weights).sum())

        current_weights = new_weights
        wealth_history.append(current_wealth)

        if t in snap_idx:
            snapshots["t"].append(t)
            snapshots["wealth"].append(current_wealth)
            snapshots["weights"].append(new_weights.copy())
            
            print(f"Snapshot t={t}, Wealth=${current_wealth:.0f}")
            

    return current_wealth, wealth_history, consumption_history


def crisis_validation_master(tickers, r_annual, num_stocks, gamma, rho, initial_wealth, start_date, end_date):
    np.random.seed(1)
    random.seed(1)
    _hmm_cache['model'] = None
    full_returns = getMultipleStocks_logReturns(tickers, start_date, end_date)

    # We want to target specific start years for the TEST period
    # To test 2000, we must start training in 1995 (5 years prior) - training: 1995-1999, test: 2000-2004
    # To test 2008, we must start training in 2003 (5 years prior) - training: 2003-2007, test: 2008-2012
    target_test_years = [1999, 2000, 2001, 2002, 2007, 2008, 2009]

    # Storage
    lsmc_w_list, lsmc_h_list, lsmc_c_list, lsmc_penalties = [], [], [], []
    mv_w_list, mv_h_list, mv_c_list = [], [], []
    ew_w_list, ew_h_list, ew_c_list = [], [], []
    b_w, b_h, b_c = [], [], []
    cppi_w_list, cppi_h_list, cppi_c_list = [], [], [] 
    spy_w_list, spy_h_list = [], []

    rs_w_list, rs_h_list, rs_c_list = [], [], []

    hrp_w_list, hrp_h_list, hrp_c_list = [], [], [] 

    kappa = .002
    window = 252


    for year in target_test_years:
        _hmm_cache['model'] = None
        # Find the index where the training slice should begin (5 years before the crisis)
        train_start_date = pd.Timestamp(year=year-5, month=1, day=1)


        if train_start_date < full_returns.index[0]:
            print(f"Skipping {year}: Data only starts at {full_returns.index[0].date()}")
            continue

        # Find the closest actual trading day in our data
        start_idx = full_returns.index.get_indexer([train_start_date], method='pad')[0]

        train_days = 5 * 252
        test_days = 5 * 252

        train_slice = full_returns.iloc[start_idx : start_idx + train_days]
        test_slice = full_returns.iloc[start_idx + train_days - 252 : start_idx + train_days + test_days]

        print(f"\n" + "!"*60)
        print(f"CRISIS WINDOW DETECTED: Testing {year} onwards")
        print(f"Training: {train_slice.index[0].date()} to {train_slice.index[-1].date()}")
        print(f"Testing: {full_returns.index[start_idx + train_days].date()} to {full_returns.index[start_idx + train_days + test_days - 1].date()}")
        print("!"*60)

        # --- A. LSMC ---
        brain = MCsimMulti(initial_wealth, r_annual, num_stocks, gamma, rho, train_slice)
        w, h, c, penalties, init_weights, weights_hist = backtest_policy(initial_wealth, test_slice, brain, r_annual, num_stocks, gamma, tickers, rho)
        lsmc_w_list.append(w); lsmc_h_list.append(h); lsmc_c_list.append(c), lsmc_penalties.extend(penalties)

        # --- B. Standard MV ---
        w_m, h_m, c_m = backtest_policy_MV(initial_wealth, test_slice, r_annual, num_stocks, gamma, tickers)
        mv_w_list.append(w_m); mv_h_list.append(h_m); mv_c_list.append(c_m)

        # C. Regime-Switching MV
        w_rs, h_rs, c_rs = backtest_policy_RS_MV(initial_wealth, test_slice, r_annual, num_stocks, gamma, tickers)
        rs_w_list.append(w_rs); rs_h_list.append(h_rs); rs_c_list.append(c_rs)

        # --- C. Equal-Weight (1/N) ---
        current_w_ew = initial_wealth
        h_ew = [initial_wealth]
        c_ew = [0.03] * 60
        eq_weights = np.ones(num_stocks) / num_stocks

        for m in range(60):
            end_idx = window + (m * 21)
            limit = min(end_idx + 21, len(test_slice))
            month_ret = np.exp((test_slice.iloc[end_idx : limit].values @ eq_weights).sum())
            current_w_ew -= current_w_ew * 0.03 * (1/12)
            current_w_ew *= month_ret
            h_ew.append(current_w_ew)

        ew_w_list.append(current_w_ew); ew_h_list.append(h_ew); ew_c_list.append(c_ew)


        # D. Static Baseline (Uses Month 1 LSMC weights and never change them)

        w_bh = initial_wealth; h_bh = [initial_wealth]; c_bh = [0.03] * 60
        for m in range(60):
            end_idx = window + (m * 21)
            limit = min(end_idx + 21, len(test_slice))
            m_ret = np.exp((test_slice.iloc[end_idx : limit].values @ init_weights).sum())
            w_bh -= (w_bh * 0.03 * (1/12)) # Static consumption
            w_bh *= m_ret
            h_bh.append(w_bh)
        b_w.append(w_bh); b_h.append(h_bh); b_c.append(c_bh)

        # --- E. CPPI ---
        w_cp, h_cp, c_cp = backtest_policy_CPPI(initial_wealth, test_slice, r_annual, num_stocks)
        cppi_w_list.append(w_cp); cppi_h_list.append(h_cp); cppi_c_list.append(c_cp)

        # --- F. S&P 500 ---
        test_start = test_slice.index[0]
        spy_launch = pd.Timestamp("1993-01-29")
        if test_start >= spy_launch:
          benchmark_ticker = "SPY"
        else:
          benchmark_ticker = "^GSPC" # fallback: SP500 index

        spy = yf.Ticker(benchmark_ticker).history(start=test_start, end=test_slice.index[-1])
        spy_prices = spy['Close'].tz_localize(None).reindex(test_slice.index, method='ffill')

        # Calculate monthly returns for SPY
        spy_m_rets = np.log(spy_prices / spy_prices.shift(1)).fillna(0).values

        current_w_spy = initial_wealth
        h_spy = [initial_wealth]

        for m in range(60):
          end_idx = window + (m * 21)
          if end_idx >=len(spy_m_rets):
            break

          limit = min(end_idx + 21, len(spy_m_rets))

          r = spy_m_rets[end_idx:limit].sum()

          current_w_spy -= current_w_spy * 0.03 * (1/12)
          current_w_spy *= np.exp(r)

          h_spy.append(current_w_spy)

        spy_w_list.append(current_w_spy)
        spy_h_list.append(h_spy)


        curr_w_hrp = initial_wealth
        h_hrp = [initial_wealth]
        c_hrp = [0.03] * 60 # Fixed 3% to match benchmarks

        prev_w_hrp = np.zeros(num_stocks)
        for m in range(60):
            end_idx = window + (m * 21)
            hist_window = test_slice.iloc[end_idx-window : end_idx]
            actual_m_rets = test_slice.iloc[end_idx : min(end_idx+21, len(test_slice))]

            # --- HRP ---
            w_hrp = get_HRP_weights(hist_window)

            turnover_hrp = np.sum(np.abs(w_hrp - prev_w_hrp)) # transaction costs
            curr_w_hrp *= (1 - turnover_hrp * kappa)
            curr_w_hrp *= (1 - 0.03 * (1/12)) # Consumption
            curr_w_hrp *= np.exp((actual_m_rets.values @ w_hrp).sum())
            prev_w_hrp = w_hrp
            h_hrp.append(curr_w_hrp)


        hrp_w_list.append(curr_w_hrp); hrp_h_list.append(h_hrp); hrp_c_list.append(c_hrp)

    # Run the forecasting test before returning
    run_phi_forecasting_test(lsmc_penalties, lsmc_h_list)

    return lsmc_w_list, lsmc_h_list, lsmc_c_list, \
           mv_w_list, mv_h_list, mv_c_list, \
           rs_w_list,   rs_h_list,   rs_c_list, \
           ew_w_list, ew_h_list, ew_c_list, \
           b_w, b_h, b_c, \
           cppi_w_list, cppi_h_list, cppi_c_list, brain, weights_hist, lsmc_penalties, spy_h_list, hrp_w_list, hrp_h_list, hrp_c_list



def run_phi_forecasting_test(penalty_list, wealth_history_list):

    # Per-window drawdowns
    all_drawdowns = []
    for h in wealth_history_list:
        path = np.array(h[1:])
        peak = np.maximum.accumulate(path)
        dd = (peak - path) / peak
        all_drawdowns.extend(dd)

    drawdowns = np.array(all_drawdowns)
    penalty_array = np.array(penalty_list)

    # Per-window returns (59 per window since diff reduces by 1)
    all_returns = []
    for h in wealth_history_list:
        path = np.array(h[1:])
        r = np.diff(path) / path[:-1]
        all_returns.extend(r)
    all_returns = np.array(all_returns)

    # Trim penalty to match returns length (59 per window, not 60)
    all_penalty_trimmed = []
    for i in range(len(wealth_history_list)):
        all_penalty_trimmed.extend(penalty_list[i*60 : i*60 + 59])
    all_penalty_trimmed = np.array(all_penalty_trimmed)

    # Boolean masks for drawdown comparison (uses full 60-per-window penalty)
    high_penalty_mask = penalty_array > 10
    low_penalty_mask  = penalty_array < 5

    dd_high = drawdowns[high_penalty_mask]
    dd_low  = drawdowns[low_penalty_mask]

    # Index arrays for protection test (uses trimmed 59-per-window penalty)
    high_penalty_idx = np.where(all_penalty_trimmed > 10)[0]
    low_penalty_idx  = np.where(all_penalty_trimmed < 5)[0]

    print("\n" + "="*60)
    print("DEFENSIVE RESPONSE TEST")
    print("="*60)
    print(f"Total months analyzed: {len(penalty_array)}")
    print(f"High penalty months (>10): {np.sum(high_penalty_mask)}")
    print(f"Low penalty months (<5):   {np.sum(low_penalty_mask)}")

    if len(dd_high) > 0:
        print(f"\nAverage drawdown when penalty HIGH (>10): {np.mean(dd_high)*100:.2f}%")
    else:
        print(f"\nAverage drawdown when penalty HIGH (>10): N/A")

    if len(dd_low) > 0:
        print(f"Average drawdown when penalty LOW  (<5):  {np.mean(dd_low)*100:.2f}%")
    else:
        print(f"Average drawdown when penalty LOW  (<5):  N/A")

    if len(dd_high) > 0 and len(dd_low) > 0:
        print(f"\nDifference: {(np.mean(dd_high) - np.mean(dd_low))*100:.2f}%")

    # T-test
    if len(dd_high) > 1 and len(dd_low) > 1:
        t_stat, p_val = stats.ttest_ind(dd_high, dd_low, equal_var=False)
        print(f"\nT-statistic: {t_stat:.3f}")
        print(f"P-value:     {p_val:.4f}")

        if p_val < 0.05:
            print("\n RESULT: Penalty successfully identifies crisis periods!")
            print("High penalties significantly coincide with drawdowns.")
        else:
            print("\n RESULT: Penalty not significantly associated with drawdowns")
    else:
        print("\n WARNING: Insufficient data for t-test")

    # Protection test
    if len(high_penalty_idx) > 0 and len(low_penalty_idx) > 0:
        ret_high = all_returns[high_penalty_idx]
        ret_low  = all_returns[low_penalty_idx]

        print(f"\n--- PROTECTION TEST ---")
        print(f"Next month return when penalty HIGH: {np.mean(ret_high)*100:.2f}%")
        print(f"Next month return when penalty LOW: {np.mean(ret_low)*100:.2f}%")
        print(f"Difference: {(np.mean(ret_high) - np.mean(ret_low))*100:.2f}%")

        if np.mean(ret_high) > np.mean(ret_low):
            print("High penalty periods have BETTER subsequent returns!")
        else:
            print("High penalty periods have WORSE subsequent returns")

        if len(ret_high) > 1 and len(ret_low) > 1:
          t_stat_ret, p_val_ret = stats.ttest_ind(ret_high, ret_low, equal_var=False)
          print(f"\nT-statistic for returns: {t_stat_ret:.3f}")
          print(f"P-value for returns: {p_val_ret:.4f}")

    # Penalty distribution
    print(f"\n--- PENALTY DISTRIBUTION ---")
    print(f"Mean: {np.mean(penalty_array):.2f}")
    print(f"Median: {np.median(penalty_array):.2f}")
    print(f"Min: {np.min(penalty_array):.2f}")
    print(f"Max: {np.max(penalty_array):.2f}")
    print(f"Std: {np.std(penalty_array):.2f}")
    print("="*60 + "\n")



def plot_backtest_comparison(all_results, r_annual, tickers, lsmc_weights_history, lsmc_penalties):
    sns.set_theme(style="whitegrid")
    # Increased to 4x2 grid to accommodate Asset Allocation
    #fig, axes = plt.subplots(5, 2, figsize=(18, 35))
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    style_map = {
        'LSMC': {'color': 'red', 'marker': 'o', 'lw': 3},
        'STANDARD MEAN-VARIANCE': {'color': 'blue', 'marker': 's', 'lw': 1.5},
        'REGIME SWITCHING HMM': {'color': 'teal', 'marker': 'p', 'lw': 1.5},
        'EQUAL-WEIGHT BASELINE': {'color': 'green', 'marker': '^', 'lw': 1.5},
        'STATIC BUY-AND-HOLD': {'color': 'gray', 'marker': 'x', 'lw': 1.5},
        'CPPI (FLOOR PROTECTION)': {'color': 'orange', 'marker': '*', 'lw': 1.5},
        'HIERARCHICAL RISK PARITY': {'color': 'olive', 'marker': 'v', 'lw': 1.5},
        'S&P 500 INDEX': {'color': 'black', 'marker': 'P', 'lw': 2, 'ls': '--'}
    }

    metrics = {'Sharpe': [], 'Sortino': [], 'Names': []}
    vols, rets, names = [], [], []

    # Cumulative wealth 
    #ax1 = axes[0, 0]
    for label, histories in all_results.items():
        mean_history = np.mean(histories, axis=0)
        #ax1.plot(mean_history, label=label, color=style_map[label]['color'], linewidth=style_map[label]['lw'])


        m_rets = np.diff(mean_history) / mean_history[:-1]
        ann_ret = np.mean(m_rets) * 12
        ann_vol = np.std(m_rets) * np.sqrt(12)

        downside_rets = m_rets[m_rets < 0]
        downside_vol = np.std(downside_rets) * np.sqrt(12) if len(downside_rets) > 0 else 1e-6

        # Stats for Frontier
        rets.append(ann_ret)
        vols.append(ann_vol)
        names.append(label)


        # Store for Bar Chart
        metrics['Sharpe'].append((ann_ret - r_annual) / (ann_vol + 1e-6))
        metrics['Sortino'].append((ann_ret - r_annual) / (downside_vol + 1e-6))
        metrics['Names'].append(label)

    #ax1.set_title("Average Wealth Trajectory", fontsize=14, fontweight='bold')
    #ax1.set_ylabel("Wealth in $ Amount")
    #ax1.set_xlabel("Months")
    #ax1.legend()

    # efficient frontier
    #ax2 = axes[0, 1]
    ax2 = axes[0, 0]
    for i in range(len(names)):
        ax2.scatter(vols[i], rets[i], color=style_map[names[i]]['color'],
                    marker=style_map[names[i]]['marker'], s=200, zorder=5)


        if names[i] == 'LSMC':
            # VOLATILITY SAVINGS ARROW (Horizontal Gap)
            bh_idx = names.index('STANDARD MEAN-VARIANCE')
            ax2.annotate('', xy=(vols[i], rets[i]), xytext=(vols[bh_idx], rets[i]),
                         arrowprops=dict(arrowstyle='<->', color='black', lw=2))
            ax2.text((vols[i] + vols[bh_idx])/2, rets[i] + 0.005, 'Volatility Savings',
                     ha='center', fontweight='bold')

    ax2.axhline(y=r_annual, color='black', linestyle='-', alpha=0.5, label='Risk-Free Rate')
    ax2.set_title("Realized Risk-Return", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Volatility")
    ax2.set_ylabel("Return")
    ax2.set_xlim(0, max(vols)*1.2)
    ax2.legend(fontsize=8)


    # --- PLOT 4: RETURN BOXPLOT ---
    #ax4 = axes[1, 1]
    ax4 = axes[0, 1]
    all_rets_data = []
    for label in names:
        window_rets = []
        for h in all_results[label]:
            window_rets.extend(np.diff(h) / h[:-1])
        all_rets_data.append(window_rets)

    sns.boxplot(data=all_rets_data, ax=ax4, palette=[style_map[l]['color'] for l in names])
    ax4.set_xticks(np.arange(len(names)))
    ax4.set_xticklabels(['LSMC', 'MV', 'HMM', 'EW', 'B&H', 'CPPI', 'HRP', 'S&P 500'], rotation=45)
    ax4.set_title("Monthly Return Distribution", fontsize=14, fontweight='bold')
    ax4.set_xlabel("Portfolio Model", fontsize=12)
    ax4.set_ylabel("Monthly Return (%)", fontsize=12)

    # Asset allocation
    # This shows what the LSMC was doing over time
    ax5 = axes[1, 0]

    ticker_names = [t.ticker for t in tickers]

    # lsmc_weights_history is shape (60, num_stocks)
    # We want heatmap: rows=assets, cols=months
    weights_matrix = lsmc_weights_history.T  # shape (num_stocks, 60)

    im = ax5.imshow(
      weights_matrix,
      aspect='auto',
      cmap='YlOrRd',
      vmin=0,
      vmax=weights_matrix.max(),
      interpolation='nearest'
    )

    ax5.set_title("LSMC Asset Allocation Over Time", fontsize=14, fontweight='bold')
    ax5.set_xlabel("Months")
    ax5.set_ylabel("Asset")
    ax5.set_yticks(np.arange(len(ticker_names)))
    ax5.set_yticklabels(ticker_names, fontsize=9)

    # Add month labels every 10 months
    ax5.set_xticks(np.arange(0, 60, 10))
    ax5.set_xticklabels(np.arange(0, 60, 10))

    plt.colorbar(im, ax=ax5, label='Portfolio Weight', shrink=0.6)


    # ---  ADaptive risk pen (brain) -
    #ax6 = axes[2, 1]
    ax6 = axes[1, 1]
    # Penalties are often long, so we average them to match the 60 month timeframe
    penalty_series = np.array(lsmc_penalties)
    # If the penalty list is from multiple windows, just take the first 60
    ax6.plot(penalty_series[:60], color='red', linewidth=2, label='Risk Penalty (Lambda)')
    ax6.fill_between(range(len(penalty_series[:60])), penalty_series[:60], color='red', alpha=0.1)
    ax6.set_title("LSMC Adaptive Risk Penalty", fontsize=14, fontweight='bold')
    ax6.set_ylabel("Penalty Strength (High = Defensive)")
    ax6.set_xlabel("Months")


    spy_path = np.mean(all_results['S&P 500 INDEX'], axis=0)
    r_m = np.diff(spy_path) / spy_path[:-1]
    ann_m = np.mean(r_m) * 12


    print(f"\n" + " " * 20 + "RISK-ADJUSTED ALPHA")
    print(f"="*85)
    print(f"{'Model Name':<28} | {'Beta':<8} | {'Alpha (Avg)':<12} | {'Alpha (Std)'}")
    print("-"*85)

    for label in all_results.keys():
        if label == 'S&P 500 INDEX': continue

        path_alphas, path_betas = [], []

        for path in all_results[label]:
            r_p = np.diff(path) / path[:-1]
            ann_p = np.mean(r_p) * 12

            # Beta & Alpha
            cov_mat = np.cov(r_p, r_m)
            beta = cov_mat[0, 1] / cov_mat[1, 1]
            alpha = ann_p - (r_annual + beta * (ann_m - r_annual))


            path_alphas.append(alpha); path_betas.append(beta)
            
        # Print Table 
        m_alpha, s_alpha, m_beta = np.mean(path_alphas), np.std(path_alphas), np.mean(path_betas)
        print(f"{label:<28} | {m_beta:>8.3f} | {m_alpha:>11.2%} | {s_alpha:>10.2%}")


    plt.tight_layout()
    plt.show()




def main():
    # IN PAPER, I CHOSE 3
    # Coefficient of relative risk aversion (CRRA) - risk tolerance
    # Higher gamma = less risky investments
    # Lower Gamma - risker investments
    gamma = 3.0
    
    # IN PAPER, I CHOSE 1000.0
    # Initial portfolio value
    initial_wealth = 1000.0

    r_annual = .0425 # risk free interest rate
    
    # FOR PAPER I CHOSE 4%
    rho_disc = 0.04
        
    # *ASSET UNIVERSE SELECTION*
    # Change this to 1,2, or 3 to reproduce specific paper results
    universe_selection = 2
        
    universes = {
        1: ['VTI', 'QQQ', 'TLT', 'GLD'],
        2: ['MSFT', 'AAPL', 'JPM', 'WMT', 'PG', 'XOM', 'ORCL', 'VUSTX', 'SU'],
        3: ['MSFT', 'AAPL', 'JPM', 'WMT', 'PG', 'XOM', 'ORCL', 'VUSTX', 'SU', 
            'NEM', 'PFE', 'GE', 'BAC', 'T']
    }
    
    # *Set to *True* to run crisis walk-forward. In paper: *UNIVERSE* 2, table 3 in the paper and *UNIVERSE* 3, table 5*
    # *Set to *False* to run walk-forward. In paper: *UNIVERSE* 1, table 1 in the paper and *UNIVERSE* 2, table 2*
    run_crisis_test = True
    
    ticker_symbols = universes[universe_selection]
    num_stocks = len(ticker_symbols)
    tickers = [yf.Ticker(s) for s in ticker_symbols]
    
    print(f"Universe {universe_selection}: {ticker_symbols}")
    print(f"Parameters: Gamma= {gamma}, Rho= {rho_disc}, Risk Free={r_annual}, Initial Wealth= ${initial_wealth}")


    if run_crisis_test:
        CRISIS_START = "1992-01-01"
        CRISIS_END   = "2025-12-31"
        print(f"RUNNING CRISIS VALIDATION (Universe {universe_selection})")
        l_w, l_h, l_c, m_w, m_h, m_c, rs_w, rs_h, rs_c, e_w, e_h, e_c, b_w, b_h, b_c, cppi_w, cppi_h, cppi_c, brain, l_weights, penalty, spy_h, hrp_w, hrp_h, hrp_c = crisis_validation_master(
        tickers, r_annual, num_stocks, gamma, rho_disc, initial_wealth, CRISIS_START, CRISIS_END)
    else:
        WALK_FORWARD_START = "2004-01-01"
        WALK_FORWARD_END   = "2025-12-31"
        print(f"RUNNING STANDARD WALK-FORWARD (Universe {universe_selection})")
        l_w, l_h, l_c, m_w, m_h, m_c, rs_w, rs_h, rs_c, e_w, e_h, e_c, b_w, b_h, b_c, cppi_w, cppi_h, cppi_c, brain, l_weights, penalty, spy_h, hrp_w, hrp_h, hrp_c = walk_forward_master2(tickers, r_annual, num_stocks, gamma, rho_disc, initial_wealth, WALK_FORWARD_START, WALK_FORWARD_END)

    calculate_backtest_stats(l_w, l_h, l_c, initial_wealth, r_annual, "LSMC")

    calculate_backtest_stats(m_w, m_h, m_c, initial_wealth, r_annual, "STANDARD MEAN-VARIANCE")

    calculate_backtest_stats(rs_w, rs_h, rs_c, initial_wealth, r_annual, "REGIME SWITCHING HIDDEN MARKOV MODEL")

    calculate_backtest_stats(e_w, e_h, e_c, initial_wealth, r_annual, "EQUAL-WEIGHT BASELINE")

    calculate_backtest_stats(b_w, b_h, b_c, initial_wealth, r_annual, "STATIC BUY-AND-HOLD")

    calculate_backtest_stats(cppi_w, cppi_h, cppi_c, initial_wealth, r_annual, "CPPI (FLOOR PROTECTION)")

    calculate_backtest_stats(hrp_w, hrp_h, hrp_c, initial_wealth, r_annual, "HIERARCHICAL RISK PARITY")

    # Organize data for plotting
    results_dict = {
        'LSMC': l_h,
        'STANDARD MEAN-VARIANCE': m_h,
        'REGIME SWITCHING HMM': rs_h,
        'EQUAL-WEIGHT BASELINE': e_h,
        'STATIC BUY-AND-HOLD': b_h,
        'CPPI (FLOOR PROTECTION)': cppi_h,
        'HIERARCHICAL RISK PARITY': hrp_h,
        'S&P 500 INDEX': spy_h
    }

    # Call the plotter
    plot_backtest_comparison(results_dict, r_annual, tickers, l_weights, penalty)


if __name__ == '__main__':
    main()