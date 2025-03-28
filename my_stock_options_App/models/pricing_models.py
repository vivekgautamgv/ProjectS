# models/pricing_models.py
from scipy.stats import norm
import numpy as np

def black_scholes(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    if option_type == 'call':
        price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    else:
        price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
    return price

def monte_carlo_option_price(S, K, T, r, sigma, option_type='call', simulations=100000):
    np.random.seed(42)
    returns = np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*np.random.normal(0,1,simulations))
    ST = S * returns
    
    if option_type == 'call':
        payoff = np.maximum(ST - K, 0)
    else:
        payoff = np.maximum(K - ST, 0)
    
    return np.exp(-r*T) * np.mean(payoff)


# Update models/pricing_models.py
def generate_mc_paths(S, T, r, sigma, n_paths=1000, n_steps=252):
    """Generate Monte Carlo paths for visualization"""
    dt = T/n_steps
    paths = np.zeros((n_paths, n_steps+1))
    paths[:, 0] = S
    
    for t in range(1, n_steps+1):
        rand = np.random.standard_normal(n_paths)
        paths[:, t] = paths[:, t-1] * np.exp((r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*rand)
    
    return paths

# Add Greeks calculation
def black_scholes_greeks(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    delta = norm.cdf(d1) if option_type == 'call' else -norm.cdf(-d1)
    gamma = norm.pdf(d1)/(S*sigma*np.sqrt(T))
    vega = S*norm.pdf(d1)*np.sqrt(T)
    theta = (-(S*norm.pdf(d1)*sigma)/(2*np.sqrt(T)) - r*K*np.exp(-r*T)*norm.cdf(d2))/365
    
    return {
        'delta': delta,
        'gamma': gamma,
        'vega': vega,
        'theta': theta
    }