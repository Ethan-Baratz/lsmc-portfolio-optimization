# lsmc-portfolio-optimization
Supplemental code for "Least Squares Monte Carlo Guided Portfolio Optimization under Dynamic Risk Sensitivity" - submitted to SIURO
README supplemental Code for:
"Least Squares Monte Carlo Guided Portfolio Optimization
under Dynamic Risk Sensitivity"
Author: Ethan Baratz, West Chester University
Project Advisors: Dr. Chaun Li and Dr Jiatian Xu
Submitted to SIAM Undergraduate Research Online (SIURO)

============================================================
OVERVIEW
============================================================
This file contains the Python implementation used to produce
all numerical results in the paper. The code runs the LSMC
portfolio strategy and six benchmark strategies across
user-selected asset universes and testing regimes.

============================================================
REQUIREMENTS
============================================================

Required packages:
    yfinance==0.2.58
    numpy==2.1.3
    pandas==2.2.3
    cvxpy==1.7.2
    scikit-learn==1.6.1
    arch==8.0.0
    scipy==1.15.3
    matplotlib==3.10.3
    seaborn==0.13.2
    hmmlearn==0.3.3
    statsmodels==0.14.4

Install all dependencies with:
    pip install yfinance numpy pandas cvxpy scikit-learn arch
                scipy matplotlib seaborn hmmlearn statsmodels

============================================================
HOW TO REPRODUCE PAPER RESULTS
============================================================
All parameters are set at the top of the main() function.
To reproduce a specific table, change two variables:

    universe_selection = 1, 2, or 3
    run_crisis_test    = True or False

Table mapping:
    Table 1  (Universe 1, walk-forward):
        universe_selection = 1
        run_crisis_test    = False

    Table 2  (Universe 2, walk-forward):
        universe_selection = 2
        run_crisis_test    = False

    Table 3  (Universe 2, crisis windows):
        universe_selection = 2
        run_crisis_test    = True

    Table 4  (Sensitivity analysis):
        Uses the same run as Table 3.
        Results are produced by manually changing the
        penalty parameters in MCsimMulti() as noted
        in the table caption.

	*Note the variables in the code are called*:
	penalty_base = 2.5 * np.exp(-.002 * (ratio - 1.0) * 100)
	phi_multiplier = np.exp(-1.5 * avg_phi_normalized)
	penalty_final = np.clip(penalty_final, 1.0, 20.0)


    Table 5  (Universe 3, crisis windows):
        universe_selection = 3
        run_crisis_test    = True

All other parameters match the paper:
    gamma          = 3.0    (risk aversion)
    initial_wealth = 1000.0
    rho_disc       = 0.04   (discount rate)
    M              = 500    (Monte Carlo paths)

============================================================
RUNNING THE CODE
============================================================
From a terminal:
    python portfolio_lsmc.py

The code will prompt for no input. All parameters are
hardcoded in main() as described above.

Expected runtime per walk-forward window: approximately
10 to 30 minutes depending on hardware, due to GARCH
estimation and Monte Carlo simulation at each rebalancing
step. The crisis validation test runs seven windows and
will take a slightly shorter amount of time.

============================================================
OUTPUT
============================================================
The code prints performance statistics to the terminal
for each strategy (Mean Wealth, Sharpe ratio, VaR, ES,
Max Drawdown) and produces four figures matching the
paper's figures 1 through 4.

Snapshot outputs at months t=0, t=29, and t=59 are
printed for LSMC, MV, and HMM strategies to allow
progress monitoring during long runs.

============================================================
NOTES ON REPRODUCIBILITY
============================================================
Random seeds are fixed (numpy.random.seed(1) and
random.seed(1)) inside MCsimMulti() and backtest_policy().
Results should be consistent across runs on the same
machine and Python version.

Small numerical differences may occur across different
operating systems or package versions due to differences
in floating point behavior in GARCH and convex solver
routines. The paper's conclusions are based on patterns
that are stable across these small variations.

Price data is downloaded live from Yahoo Finance via
yfinance at the time of execution. Results may differ
slightly from the paper if Yahoo Finance revises
historical data or if the code is run significantly
after the paper's submission date.

============================================================
FILE
============================================================
CodeBase.py   Main implementation file containing
                    all functions and the main() entry point.
