import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

def get_weights(d: float, size: int, threshold: float = 1e-5) -> np.ndarray:
    """
    Compute binomial weights: w_0=1, w_k = -w_{k-1}*(d-k+1)/k.
    Stop when |w_k| < threshold. Return array of length min(size, k).
    """
    w = [1.0]
    for k in range(1, size):
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < threshold:
            break
        w.append(w_k)
    return np.array(w[::-1]).reshape(-1, 1) # Reverse to match past to present in convolution

def get_weights_ffd(d: float, threshold: float = 1e-5) -> np.ndarray:
    """
    Fixed-width: compute weights until |w_k| < threshold. Return all.
    """
    w = [1.0]
    k = 1
    while True:
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < threshold:
            break
        w.append(w_k)
        k += 1
    return np.array(w[::-1]).reshape(-1, 1)

def frac_diff_ffd(series: pd.Series, d: float, threshold: float = 1e-5) -> pd.Series:
    """
    Apply FFD weights to the series using a dot product on a rolling window.
    Width = len(get_weights_ffd(d, threshold)).
    Drop initial NaN rows (width - 1 lost).
    """
    w = get_weights_ffd(d, threshold)
    width = len(w)
    
    df = {}
    
    # We can use rolling apply, or just numpy convolution or strided arrays.
    # A simple and reasonably fast way for Pandas Series:
    series_np = series.fillna(method='ffill').dropna().values
    
    # Ensure series is long enough
    if len(series_np) < width:
        return pd.Series(index=series.index, dtype=float)
        
    res = np.convolve(series_np, w.flatten(), mode='valid')
    
    # mode='valid' means the output length is len(series) - width + 1
    # which corresponds to indices from width - 1 to end
    
    out_index = series.index[series.shape[0] - len(res):]
    return pd.Series(res, index=out_index)

def find_min_d(series: pd.Series, d_range: np.ndarray = np.arange(0, 1.05, 0.05), threshold: float = 1e-5) -> float:
    """
    For each d, compute FFD, run ADF test.
    Return smallest d where ADF p-value < 0.05.
    """
    min_d = None
    
    for d in d_range:
        df_ = frac_diff_ffd(series, d, threshold)
        if len(df_) < 10:
            continue
            
        res = adfuller(df_, maxlag=1, regression='c', autolag=None)
        p_value = res[1]
        
        if p_value < 0.05:
            min_d = d
            break
            
    return min_d if min_d is not None else 1.0

def plot_min_ffd(series: pd.Series):
    """
    Dual y-axis: left = ADF statistic, right = correlation with original.
    Mark 1% and 5% critical values for ADF.
    Mark the chosen d* with a vertical line.
    """
    out = pd.DataFrame(columns=['adfStat', 'pVal', 'corr'])
    
    d_range = np.arange(0, 1.05, 0.05)
    for d in d_range:
        df_ = frac_diff_ffd(series, d, threshold=1e-5)
        if len(df_) < 10:
            continue
            
        res = adfuller(df_, maxlag=1, regression='c', autolag=None)
        
        # Calculate correlation with original series (aligned index)
        corr = df_.corr(series.loc[df_.index])
        
        out.loc[d] = [res[0], res[1], corr]
        
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Find min_d
    min_d = out[out['pVal'] < 0.05].index.min()
    if pd.isna(min_d):
        min_d = 1.0
        
    # Plot ADF statistic
    ax1.plot(out.index, out['adfStat'], color='blue', label='ADF Stat')
    ax1.set_xlabel('d Value')
    ax1.set_ylabel('ADF Statistic', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # ADF critical values
    # These depend on sample size, but approx values are:
    ax1.axhline(y=-3.432, color='red', linestyle='--', label='1% Critical Value')
    ax1.axhline(y=-2.862, color='orange', linestyle='--', label='5% Critical Value')
    
    # Plot Correlation
    ax2 = ax1.twinx()
    ax2.plot(out.index, out['corr'], color='green', label='Correlation')
    ax2.set_ylabel('Correlation with Original Series', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    
    # Mark optimal d
    ax1.axvline(x=min_d, color='black', linestyle=':', label=f'Optimal d*={min_d:.2f}')
    
    fig.tight_layout()
    
    # Combine legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper right')
    
    plt.title('Fractional Differentiation: Stationarity vs Memory')
    return fig
