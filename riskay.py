import numpy as np
import dataclasses
from scipy.stats import norm, poisson
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict
import seaborn as sns
import os

# Configure matplotlib for non-interactive backend
plt.switch_backend('Agg')

@dataclass
class MertonConfig:
    """Configuration for Merton Jump-Diffusion Model"""
    S0: float = .960          # Spot price
    K: float = .950           # Strike price
    T: float = .51320             # Time to maturity (years)
    r: float = 0.04311           # Risk-free rate
    sigma: float = 0.6775         # Diffusion volatility
    lamb: float = 5.0          # Jump intensity
    mu_j: float = -0.10        # Mean jump size (log)
    sigma_j: float = 0.3       # Jump volatility
    jump_impact_factor: float = 1.5  # Multiplier to amplify jumps

class MertonJumpModel:
    """Merton model with visualization saving for Codespaces"""
    
    def __init__(self, config: MertonConfig):
        self.c = config
        self._validate()
        self._setup_style()
        self.output_dir = "merton_visualizations"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def _validate(self):
        """Parameter validation"""
        if not all(v > 0 for v in [self.c.S0, self.c.K, self.c.T]):
            raise ValueError("S0, K, T must be positive")
        if not all(v >= 0 for v in [self.c.sigma, self.c.lamb, self.c.sigma_j]):
            raise ValueError("Volatilities and lambda must be non-negative")
    
    def _setup_style(self):
        """Configure visualization style"""
        sns.set(style='whitegrid')
        plt.rcParams['figure.figsize'] = [12, 6]
        plt.rcParams['font.size'] = 12
    
    def _save_plot(self, fig, filename: str) -> str:
        """Save plot to file and return path"""
        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
        return path
    
    def simulate_paths(self, n_paths: int = 1000, n_steps: int = 252) -> np.ndarray:
        """Generate asset paths with jumps"""
        dt = self.c.T / n_steps
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = self.c.S0
        
        drift_adj = (self.c.r - 0.5*self.c.sigma**2 - 
                    self.c.lamb*(np.exp(self.c.mu_j + 0.5*self.c.sigma_j**2)-1))
        
        for t in range(1, n_steps + 1):
            Z = np.random.normal(0, 1, n_paths)
            diffusion = drift_adj*dt + self.c.sigma * np.sqrt(dt) * Z
            
            N = np.random.poisson(self.c.lamb*dt * self.c.jump_impact_factor, n_paths)
            jumps = np.zeros(n_paths)
            jump_idx = N > 0
            if np.any(jump_idx):
                jump_counts = N[jump_idx]
                jump_sizes = np.array([
                    np.sum(np.exp(self.c.mu_j + self.c.sigma_j*np.random.normal(0, 1, count)))
                    for count in jump_counts
                ])
                jumps[jump_idx] = jump_sizes - jump_counts
            
            paths[:, t] = paths[:, t-1] * np.exp(diffusion + jumps)
            
        return paths
    
    def plot_paths_with_jumps(self, n_paths: int = 5, n_steps: int = 252) -> str:
        """Visualize paths with jumps and save to file"""
        paths = self.simulate_paths(n_paths, n_steps)
        fig, ax = plt.subplots(figsize=(14, 7))
        
        for i in range(n_paths):
            returns = np.diff(paths[i]) / paths[i, :-1]
            jump_threshold = 3 * self.c.sigma * np.sqrt(1/n_steps)
            jump_points = np.where(np.abs(returns) > jump_threshold)[0] + 1
            
            line, = ax.plot(paths[i], alpha=0.7, lw=2)
            if len(jump_points) > 0:
                ax.scatter(jump_points, paths[i, jump_points], color=line.get_color(),
                         s=100, edgecolors='black', zorder=10,
                         label=f'Path {i+1} jumps (n={len(jump_points)})')
        
        ax.set_title(f'Merton Jump-Diffusion Paths\n'
                   f'λ={self.c.lamb:.1f}, μ_j={self.c.mu_j:.2f}, σ_j={self.c.sigma_j:.2f}')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Asset Price')
        ax.legend()
        ax.grid(True)
        
        return self._save_plot(fig, 'jump_paths.png')
    
    def plot_jump_size_distribution(self, n_simulations: int = 10000) -> str:
        """Visualize jump size distribution and save to file"""
        jump_sizes = np.exp(self.c.mu_j + self.c.sigma_j * np.random.normal(0, 1, n_simulations))
        
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.histplot(jump_sizes, bins=50, kde=True, stat='density', ax=ax)
        ax.set_title(f'Jump Size Distribution\nμ_j={self.c.mu_j:.2f}, σ_j={self.c.sigma_j:.2f}')
        ax.set_xlabel('Jump Multiplier (e.g., 0.9 = 10% drop)')
        ax.set_ylabel('Density')
        ax.axvline(1.0, color='red', linestyle='--', label='No change')
        ax.legend()
        ax.grid(True)
        
        return self._save_plot(fig, 'jump_distribution.png')
    
    def calculate_var_es(self, position_size: float = 4_500_000,
                        time_horizon: float = 1/252, alpha: float = 0.05,
                        n_simulations: int = 100_000) -> Dict[str, float]:
        """Compute Value-at-Risk and Expected Shortfall"""
        dt = time_horizon
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
        
        var = -np.percentile(pnl, alpha * 100)
        es = -np.mean(pnl[pnl <= -var])
        
        return {
            'Value_at_Risk': var,
            'Expected_Shortfall': es,
            'Worst_Loss': -np.min(pnl),
            'Jump_Probability': np.mean(N > 0),
            'Average_Jump_Impact': np.mean(jumps[jump_idx]) if np.any(jump_idx) else 0
        }
    
    def jump_scenario_analysis(self, position_size: float = 4_500_000) -> Dict[str, Dict]:
        """Compare risk metrics across jump scenarios"""
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
            for param, value in params.items():
                setattr(self.c, param, value)
            results[name] = self.calculate_var_es(position_size)
            results[name]['parameters'] = {k: getattr(self.c, k) 
                                         for k in ['lamb', 'mu_j', 'sigma_j', 'jump_impact_factor']}
        
        for param, value in original_config.items():
            setattr(self.c, param, value)
        
        return results

if __name__ == "__main__":
    print("=== Merton Jump-Diffusion Model ===")
    
    # Create model with impactful parameters
    config = MertonConfig(
        S0=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2,
        lamb=5.0, mu_j=-0.10, sigma_j=0.3, jump_impact_factor=1.5
    )
    model = MertonJumpModel(config)
    
    # Generate and save visualizations
    paths_plot_path = model.plot_paths_with_jumps()
    jump_dist_path = model.plot_jump_size_distribution()
    
    print(f"Saved path visualization to: {paths_plot_path}")
    print(f"Saved jump distribution to: {jump_dist_path}\n")
    
    # Risk analysis
    risk_metrics = model.calculate_var_es(position_size=1_000_000)
    print("1-Day 95% Risk Metrics:")
    for metric, value in risk_metrics.items():
        print(f"{metric.replace('_', ' '):<25}: ${value:,.2f}" if isinstance(value, float) else 
              f"{metric.replace('_', ' '):<25}: {value:.2%}" if 'Probability' in metric else
              f"{metric.replace('_', ' '):<25}: {value:.4f}")
    
    # Scenario analysis
    print("\nRunning jump scenario analysis")
    scenarios = model.jump_scenario_analysis()
    
    print("\n=== Scenario Comparison ===")
    print(f"{'Scenario':<20} {'VaR':>12} {'ES':>12} {'Jump Prob':>12} {'Avg Jump':>12}")
    for name, data in scenarios.items():
        print(f"{name:<20} ${data['Value_at_Risk']:>11,.2f} ${data['Expected_Shortfall']:>11,.2f} "
              f"{data['Jump_Probability']:>11.2%} {data['Average_Jump_Impact']:>11.4f}")