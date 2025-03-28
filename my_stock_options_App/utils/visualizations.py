# Update utils/visualization.py (new file)
import plotly.graph_objects as go
import numpy as np

def create_vol_surface_plot():
    """Create 3D volatility surface visualization"""
    strikes = np.linspace(50, 150, 25)
    expiries = np.linspace(0.1, 2.0, 25)
    X, Y = np.meshgrid(strikes, expiries)
    Z = np.zeros_like(X)
    
    # Example volatility surface calculation
    for i in range(len(expiries)):
        Z[i,:] = 0.2 + 0.1*np.sin(X[i,:]/50) + 0.05*np.exp(-Y[i,:])
    
    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y)])
    fig.update_layout(title='Volatility Surface',
                    scene=dict(xaxis_title='Strike',
                              yaxis_title='Time to Expiry',
                              zaxis_title='Volatility'),
                    height=800)
    return fig

# Update utils/visualization.py
def plot_greek_surface(S_range, K_range, T_range, greek_type='delta'):
    """Create 3D surface plot for Greek analysis"""
    X, Y = np.meshgrid(K_range, T_range)
    Z = np.zeros_like(X)
    
    for i in range(len(T_range)):
        for j in range(len(K_range)):
            greeks = black_scholes_greeks(S_range, K_range[j], T_range[i], 
                                        risk_free_rate, 0.2, 'call')
            Z[i,j] = greeks[greek_type]
    
    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y)])
    fig.update_layout(title=f'{greek_type.capitalize()} Surface',
                    scene=dict(xaxis_title='Strike',
                              yaxis_title='Time',
                              zaxis_title=greek_type.capitalize()),
                    height=800)
    return fig