"""
merton_jump_enhanced.py - Advanced Merton Jump-Diffusion Model with Powerful Jump Effects

Features:
- Highly visible jump effects in simulations
- Comprehensive jump risk analysis
- Dynamic visualization tools
- Scenario comparison capabilities
- Detailed risk metric reporting
"""

import numpy as np
import dataclasses
from scipy.stats import norm, poisson
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, Tuple, List
import seaborn as sns

@dataclass
class MertonConfig:
    """Configuration for Merton Jump-Diffusion Model with impactful defaults"""
    S0: float = 96.0          # Spot price
    K: float = 95.0           # Strike price
    T: float = .510             # Time to maturity (years)
    r: float = 0.04311            # Risk-free rate
    sigma: float = 0.6775         # Diffusion volatility
    lamb: float = 5.0          # Jump intensity (higher for more frequent jumps)
    mu_j: float = -0.10        # Mean jump size (log) - more negative for bigger drops
    sigma_j: float = 0.3       # Jump volatility (higher for more variable jumps)
    jump_impact_factor: float = 1.5  # Multiplier to amplify jump effects

class MertonJumpModel:
    """Enhanced Merton model with powerful jump effects and advanced analytics"""
    
    def __init__(self, config: MertonConfig):
        self.c = config
        self._validate()
        self._setup_style()
        
    def _validate(self):
        """Parameter validation with meaningful error messages"""
        if not all(v > 0 for v in [self.c.S0, self.c.K, self.c.T]):
            raise ValueError("S0, K, T must be positive")
        if not all(v >= 0 for v in [self.c.sigma, self.c.lamb, self.c.sigma_j]):
            raise ValueError("Volatilities and lambda must be non-negative")
        if self.c.jump_impact_factor <= 0:
            raise ValueError("Jump impact factor must be positive")
    
    def _setup_style(self):
        """Configure visualization style"""
        sns.set(style='whitegrid')
        plt.rcParams['figure.figsize'] = [12, 6]
        plt.rcParams['font.size'] = 12
    
    # ======================
    # Core Simulation Engine
    # ======================
    
    def simulate_paths(self, n_paths: int = 1000, n_steps: int = 252) -> np.ndarray:
        """
        Generate asset paths with pronounced jump effects
        Returns:
            np.ndarray: Simulated paths (n_paths × n_steps+1)
        """
        dt = self.c.T / n_steps
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = self.c.S0
        
        # Pre-calculate jump parameters
        drift_adj = (self.c.r - 0.5*self.c.sigma**2 - 
                    self.c.lamb*(np.exp(self.c.mu_j + 0.5*self.c.sigma_j**2)-1))
        
        for t in range(1, n_steps + 1):
            # Diffusion component (standard Brownian motion)
            Z = np.random.normal(0, 1, n_paths)
            diffusion = drift_adj*dt + self.c.sigma * np.sqrt(dt) * Z
            
            # Enhanced jump component with impact factor
            N = np.random.poisson(self.c.lamb*dt * self.c.jump_impact_factor, n_paths)
            jumps = np.zeros(n_paths)
            
            # Apply jumps where they occur
            jump_idx = N > 0
            if np.any(jump_idx):
                jump_counts = N[jump_idx]
                jump_sizes = np.array([
                    np.sum(np.exp(self.c.mu_j + self.c.sigma_j*np.random.normal(0, 1, count)))
                    for count in jump_counts
                ])
                jumps[jump_idx] = jump_sizes - jump_counts  # Compensate for E[e^J]
            
            paths[:, t] = paths[:, t-1] * np.exp(diffusion + jumps)
            
        return paths
    
    # ======================
    # Visualization Tools
    # ======================
    
    def plot_paths_with_jumps(self, n_paths: int = 5, n_steps: int = 252):
        """Visualize simulated paths with jumps highlighted"""
        paths = self.simulate_paths(n_paths, n_steps)
        plt.figure(figsize=(14, 7))
        
        for i in range(n_paths):
            # Calculate daily returns to identify jumps
            returns = np.diff(paths[i]) / paths[i, :-1]
            jump_threshold = 3 * self.c.sigma * np.sqrt(1/n_steps)  # 3σ threshold
            jump_points = np.where(np.abs(returns) > jump_threshold)[0] + 1
            
            # Plot the path
            line, = plt.plot(paths[i], alpha=0.7, lw=2)
            color = line.get_color()
            
            # Mark jump points
            if len(jump_points) > 0:
                plt.scatter(jump_points, paths[i, jump_points], color=color,
                           s=100, edgecolors='black', zorder=10,
                           label=f'Path {i+1} jumps (n={len(jump_points)})')
        
        plt.title(f'Merton Jump-Diffusion Paths\n'
                 f'λ={self.c.lamb:.1f}, μ_j={self.c.mu_j:.2f}, σ_j={self.c.sigma_j:.2f}, '
                 f'Impact Factor={self.c.jump_impact_factor:.1f}')
        plt.xlabel('Time Steps')
        plt.ylabel('Asset Price')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def plot_jump_size_distribution(self, n_simulations: int = 10000):
        """Visualize the distribution of jump sizes"""
        # Simulate jump sizes
        jump_sizes = np.exp(self.c.mu_j + self.c.sigma_j * np.random.normal(0, 1, n_simulations))
        
        plt.figure(figsize=(12, 6))
        sns.histplot(jump_sizes, bins=50, kde=True, stat='density')
        plt.title(f'Jump Size Distribution\nμ_j={self.c.mu_j:.2f}, σ_j={self.c.sigma_j:.2f}')
        plt.xlabel('Jump Multiplier (e.g., 0.9 = 10% drop)')
        plt.ylabel('Density')
        plt.axvline(1.0, color='red', linestyle='--', label='No change')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    # ======================
    # Risk Analytics
    # ======================
    
    def calculate_var_es(self, position_size: float = 1_000_000,
                       time_horizon: float = 1/252, alpha: float = 0.05,
                       n_simulations: int = 100_000) -> Dict[str, float]:
        """Compute Value-at-Risk and Expected Shortfall with jump effects"""
        dt = time_horizon
        # Simulate daily returns with jumps
        Z = np.random.normal(0, 1, n_simulations)
        diffusion = (self.c.r - 0.5*self.c.sigma**2 - 
                    self.c.lamb*(np.exp(self.c.mu_j + 0.5*self.c.sigma_j**2)-1))*dt
        diffusion += self.c.sigma * np.sqrt(dt) * Z
        
        N = np.random.poisson(self.c.lamb*dt * self.c.jump_impact_factor, n_simulations)
        jumps = np.zeros(n_simulations)
        jump_idx = N > 0
        if np.any(jump_idx):
            jump_counts = N[jump_idx]
            jump_sizes = np.array([
                np.sum(np.exp(self.c.mu_j + self.c.sigma_j*np.random.normal(0, 1, count)))
                for count in jump_counts
            ])
            jumps[jump_idx] = jump_sizes - jump_counts
        
        returns = np.exp(diffusion + jumps) - 1
        pnl = position_size * returns
        
        # Calculate VaR and ES
        var = -np.percentile(pnl, alpha * 100)
        es = -np.mean(pnl[pnl <= -var])
        
        return {
            'Value_at_Risk': var,
            'Expected_Shortfall': es,
            'Worst_Loss': -np.min(pnl),
            'Jump_Probability': np.mean(N > 0),
            'Average_Jump_Impact': np.mean(jumps[jump_idx]) if np.any(jump_idx) else 0
        }
    
    def jump_scenario_analysis(self, position_size: float = 1_000_000) -> Dict[str, Dict]:
        """Compare risk metrics across different jump scenarios"""
        scenarios = {
            'Current': {},
            'No_Jumps': {'lamb': 0},
            'Frequent_Small_Jumps': {'lamb': 10, 'mu_j': -0.05, 'sigma_j': 0.1},
            'Rare_Large_Jumps': {'lamb': 1, 'mu_j': -0.3, 'sigma_j': 0.4},
            'Extreme_Events': {'lamb': 0.5, 'mu_j': -0.5, 'sigma_j': 0.5, 'jump_impact_factor': 2.0}
        }
        
        results = {}
        original_config = {f.name: getattr(self.c, f.name) for f in dataclasses.fields(self.c)}
        
        for name, params in scenarios.items():
            # Update model parameters
            for param, value in params.items():
                setattr(self.c, param, value)
            
            # Calculate risk metrics
            results[name] = self.calculate_var_es(position_size)
            
            # Store scenario parameters
            results[name]['parameters'] = {k: getattr(self.c, k) 
                                         for k in ['lamb', 'mu_j', 'sigma_j', 'jump_impact_factor']}
        
        # Restore original configuration
        for param, value in original_config.items():
            setattr(self.c, param, value)
        
        return results
    
    # ======================
    # Pricing Methods
    # ======================
    
    def european_option_price(self, call_put: str = 'call', max_jumps: int = 30) -> float:
        """Price European options using Merton's closed-form solution"""
        price = 0.0
        call_put = call_put.lower()
        
        for n in range(max_jumps + 1):
            # Probability of n jumps occurring
            lambda_p = self.c.lamb * np.exp(self.c.mu_j + 0.5*self.c.sigma_j**2)
            prob = poisson.pmf(n, lambda_p * self.c.T)
            
            # Adjusted volatility and spot for n jumps
            sigma_n = np.sqrt(self.c.sigma**2 + n*self.c.sigma_j**2/self.c.T)
            r_n = self.c.r - self.c.lamb*(np.exp(self.c.mu_j + 0.5*self.c.sigma_j**2)-1) + n*(self.c.mu_j + 0.5*self.c.sigma_j**2)/self.c.T
            S_n = self.c.S0 * np.exp(n*self.c.mu_j + 0.5*n*self.c.sigma_j**2)
            
            # Black-Scholes components
            d1 = (np.log(S_n/self.c.K) + (r_n + 0.5*sigma_n**2)*self.c.T) / (sigma_n*np.sqrt(self.c.T))
            d2 = d1 - sigma_n*np.sqrt(self.c.T)
            
            if call_put == 'call':
                option_price = S_n*norm.cdf(d1) - self.c.K*np.exp(-r_n*self.c.T)*norm.cdf(d2)
            else:
                option_price = self.c.K*np.exp(-r_n*self.c.T)*norm.cdf(-d2) - S_n*norm.cdf(-d1)
            
            price += prob * option_price
        
        return price


# Example Usage
if __name__ == "__main__":
    print("=== Enhanced Merton Jump-Diffusion Model ===")
    
    # Create model with impactful jump parameters
    config = MertonConfig(
        S0=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2,
        lamb=5.0, mu_j=-0.10, sigma_j=0.3, jump_impact_factor=1.5
    )
    model = MertonJumpModel(config)
    
    # Visualization
    print("\nVisualizing simulated paths with jumps...")
    model.plot_paths_with_jumps(n_paths=5)
    model.plot_jump_size_distribution()
    
    # Risk analysis
    print("\nCalculating risk metrics...")
    risk_metrics = model.calculate_var_es(position_size=1_000_000)
    print("\n1-Day 95% Risk Metrics:")
    for metric, value in risk_metrics.items():
        print(f"{metric.replace('_', ' '):<25}: ${value:,.2f}" if isinstance(value, float) else 
              f"{metric.replace('_', ' '):<25}: {value:.2%}" if 'Probability' in metric else
              f"{metric.replace('_', ' '):<25}: {value:.4f}")
    
    # Scenario analysis
    print("\nRunning jump scenario analysis...")
    scenarios = model.jump_scenario_analysis()
    
    print("\n=== Scenario Comparison ===")
    print(f"{'Scenario':<20} {'VaR':>12} {'ES':>12} {'Jump Prob':>12} {'Avg Jump':>12}")
    for name, data in scenarios.items():
        print(f"{name:<20} ${data['Value_at_Risk']:>11,.2f} ${data['Expected_Shortfall']:>11,.2f} "
              f"{data['Jump_Probability']:>11.2%} {data['Average_Jump_Impact']:>11.4f}")
    
    # Option pricing
    print("\n=== Option Pricing ===")
    call_price = model.european_option_price('call')
    put_price = model.european_option_price('put')
    print(f"European Call Price: {call_price:.2f}")
    print(f"European Put Price: {put_price:.2f}")