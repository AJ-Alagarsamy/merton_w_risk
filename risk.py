"""
merton_risk.py - Complete Merton Jump-Diffusion Model with Risk Metrics
Includes:
- European/American option pricing
- Value-at-Risk (VaR)
- Expected Shortfall (ES)
- Jump risk analysis
"""

import numpy as np
from scipy.stats import norm, poisson
from dataclasses import dataclass
from typing import Optional

@dataclass
class MertonConfig:
    S0: float = .9       # Spot price
    K: float = .95        # Strike price
    T: float = 0.5342465753 # Time to maturity (years)
    r: float = 0.04351         # Risk-free rate
    sigma: float = 0.3      # Diffusion volatility
    lamb: float = 0.8       # Jump intensity
    mu_j: float = -0.1     # Mean jump size (log)
    sigma_j: float = 0.3    # Jump volatility

class MertonModel:
    """Complete Merton Jump-Diffusion implementation with risk metrics"""
    
    def __init__(self, config: MertonConfig):
        self.c = config
        self._validate()
        
    def _validate(self):
        """Parameter validation"""
        assert all(v > 0 for v in [self.c.S0, self.c.K, self.c.T]), "S0, K, T must be > 0"
        assert all(v >= 0 for v in [self.c.sigma, self.c.lamb, self.c.sigma_j]), "Volatilities and lambda must be >= 0"

    # ======================
    # Option Pricing Methods
    # ======================
    def european_call(self, max_jumps=20):
        """Closed-form solution for European calls"""
        price = 0.0
        for n in range(max_jumps + 1):
            lambda_p = self.c.lamb * np.exp(self.c.mu_j + 0.5*self.c.sigma_j**2)
            prob = poisson.pmf(n, lambda_p * self.c.T)
            sigma_n = np.sqrt(self.c.sigma**2 + n*self.c.sigma_j**2/self.c.T)
            S_n = self.c.S0 * np.exp(n*self.c.mu_j + 0.5*n*self.c.sigma_j**2)
            
            d1 = (np.log(S_n/self.c.K) + (self.c.r + 0.5*sigma_n**2)*self.c.T) / (sigma_n*np.sqrt(self.c.T))
            d2 = d1 - sigma_n*np.sqrt(self.c.T)
            bs_price = S_n*norm.cdf(d1) - self.c.K*np.exp(-self.c.r*self.c.T)*norm.cdf(d2)
            
            price += prob * bs_price
        return price

    def _simulate_paths(self, n_paths: int, n_steps: int) -> np.ndarray:
        """Generate Merton jump-diffusion paths"""
        dt = self.c.T / n_steps
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = self.c.S0
        
        for t in range(1, n_steps + 1):
            # Diffusion component
            Z = np.random.normal(0, 1, n_paths)
            diffusion = (self.c.r - 0.5*self.c.sigma**2 - self.c.lamb*self.c.mu_j)*dt 
            diffusion += self.c.sigma * np.sqrt(dt) * Z
            
            # Jump component
            N = np.random.poisson(self.c.lamb*dt, n_paths)
            jumps = np.where(N > 0,
                           np.exp(self.c.mu_j + self.c.sigma_j*np.random.normal(0, 1, n_paths)) - 1,
                           0)
            
            paths[:, t] = paths[:, t-1] * np.exp(diffusion + jumps)
        return paths

    def american_call(self, n_paths=100000, n_steps=252, degree=2):
        """Price American calls using Least Squares Monte Carlo"""
        paths = self._simulate_paths(n_paths, n_steps)
        dt = self.c.T / n_steps
        payoffs = np.maximum(paths - self.c.K, 0)
        values = np.zeros_like(payoffs)
        values[:, -1] = payoffs[:, -1]
        
        for t in range(n_steps-1, 0, -1):
            in_the_money = payoffs[:, t] > 0
            if np.sum(in_the_money) < degree + 2:
                values[:, t] = values[:, t+1] * np.exp(-self.c.r*dt)
                continue
                
            S_t = paths[in_the_money, t]
            Y = values[in_the_money, t+1] * np.exp(-self.c.r*dt)
            
            # Robust regression
            try:
                X = np.column_stack([S_t**i for i in range(degree+1)])
                coeffs = np.linalg.lstsq(X, Y, rcond=None)[0]
                continuation = X @ coeffs
            except:
                continuation = Y
                
            exercise = payoffs[in_the_money, t] > continuation
            values[in_the_money, t] = np.where(exercise, payoffs[in_the_money, t], Y)
            values[~in_the_money, t] = values[~in_the_money, t+1] * np.exp(-self.c.r*dt)
        
        return np.mean(values[:, 1] * np.exp(-self.c.r*dt))

    # ======================
    # Risk Metrics
    # ======================
    def value_at_risk(self, position_size=1_000_000, time_horizon=1/252, alpha=0.05, n_simulations=100_000):     #requires size input
        """Compute VaR using full Merton dynamics"""
        dt = time_horizon
        Z = np.random.normal(0, 1, n_simulations)
        diffusion = (self.c.r - 0.5*self.c.sigma**2 - self.c.lamb*self.c.mu_j)*dt + self.c.sigma*np.sqrt(dt)*Z
        
        N = np.random.poisson(self.c.lamb*dt, n_simulations)
        jumps = np.where(N > 0,
                       np.exp(self.c.mu_j + self.c.sigma_j*np.random.normal(0, 1, n_simulations)) - 1,
                       0)
        
        returns = np.exp(diffusion + jumps) - 1
        pnl = position_size * returns
        return -np.percentile(pnl, alpha * 100)

    def expected_shortfall(self, position_size=1_000_000, time_horizon=1/252, alpha=0.05, n_simulations=100_000):        #requires size input
        """Compute ES (CVaR) - average loss beyond VaR"""
        var = self.value_at_risk(position_size, time_horizon, alpha, n_simulations)
        dt = time_horizon
        
        # Resimulate to get tail distribution
        Z = np.random.normal(0, 1, n_simulations)
        diffusion = (self.c.r - 0.5*self.c.sigma**2 - self.c.lamb*self.c.mu_j)*dt + self.c.sigma*np.sqrt(dt)*Z
        
        N = np.random.poisson(self.c.lamb*dt, n_simulations)
        jumps = np.where(N > 0,
                       np.exp(self.c.mu_j + self.c.sigma_j*np.random.normal(0, 1, n_simulations)) - 1,
                       0)
        
        returns = np.exp(diffusion + jumps) - 1
        pnl = position_size * returns
        return -np.mean(pnl[pnl <= -var])

    def jump_risk_impact(self, position_size=1_000_000, time_horizon=1/252, alpha=0.05):
        """Quantify how jumps affect risk metrics"""
        original_lamb = self.c.lamb
        
        # With jumps
        var_with_jumps = self.value_at_risk(position_size, time_horizon, alpha)
        es_with_jumps = self.expected_shortfall(position_size, time_horizon, alpha)
        
        # Without jumps
        self.c.lamb = 0
        var_no_jumps = self.value_at_risk(position_size, time_horizon, alpha)
        es_no_jumps = self.expected_shortfall(position_size, time_horizon, alpha)
        self.c.lamb = original_lamb
        
        return {
            'var': {'with_jumps': var_with_jumps, 'no_jumps': var_no_jumps,
                   'difference': var_with_jumps - var_no_jumps},
            'es': {'with_jumps': es_with_jumps, 'no_jumps': es_no_jumps,
                  'difference': es_with_jumps - es_no_jumps}
        }

if __name__ == "__main__":
    # Example Usage
    config = MertonConfig(S0=100, K=100, T=1, r=0.05, sigma=0.2, lamb=0.5, mu_j=-0.05, sigma_j=0.2)
    model = MertonModel(config)
    
    print("=== Option Pricing ===")
    print(f"European Call: {model.european_call():.2f}")
    print(f"American Call: {model.american_call():.2f}")
    
    print("\n=== Risk Metrics ===")
    var = model.value_at_risk(position_size=1_000_000, time_horizon=1/252, alpha=0.05)
    es = model.expected_shortfall(position_size=1_000_000, time_horizon=1/252, alpha=0.05)
    print(f"1-day 95% VaR: ${var:,.2f}")
    print(f"1-day 95% ES: ${es:,.2f}")
    
    jump_impact = model.jump_risk_impact()
    print("\n=== Jump Impact ===")
    print(f"VaR with jumps: ${jump_impact['var']['with_jumps']:,.2f}")
    print(f"VaR without jumps: ${jump_impact['var']['no_jumps']:,.2f}")
    print(f"Additional risk from jumps: ${jump_impact['var']['difference']:,.2f}")