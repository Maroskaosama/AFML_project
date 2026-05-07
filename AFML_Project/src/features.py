import pandas as pd
import numpy as np
from scipy import stats

def compute_momentum_features(close: pd.Series) -> pd.DataFrame:
    features = pd.DataFrame(index=close.index)
    
    # Rolling returns
    for d in [5, 10, 20, 60]:
        features[f'ret_{d}d'] = np.log(close / close.shift(d))
        
    # Momentum 12-1 (252-day minus 21-day)
    ret_252 = np.log(close / close.shift(252))
    ret_21 = np.log(close / close.shift(21))
    features['momentum_12_1'] = ret_252 - ret_21
    
    # 14-day RSI (Wilder's Smoothing)
    delta = close.diff()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    down = down.abs()
    
    roll_up = up.ewm(alpha=1/14, adjust=False).mean()
    roll_down = down.ewm(alpha=1/14, adjust=False).mean()
    rs = roll_up / roll_down
    features['rsi_14'] = 100.0 - (100.0 / (1.0 + rs))
    
    return features

def compute_volatility_features(close: pd.Series) -> pd.DataFrame:
    features = pd.DataFrame(index=close.index)
    log_ret = np.log(close / close.shift(1))
    
    for d in [20, 50]:
        features[f'vol_{d}d'] = log_ret.rolling(window=d).std() * np.sqrt(252)
        
    return features

def compute_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)
    
    features['log_dollar_volume'] = np.log((df['Adj Close'] * df['Volume']).replace(0, np.nan))
    features['volume_ratio'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
    
    return features

def compute_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)
    
    high = df['High']
    low = df['Low']
    close = df['Adj Close']
    
    # Corwin-Schultz spread
    H_L = np.log(high / low) ** 2
    H_L_prev = H_L.shift(1)
    
    max_H = np.maximum(high, high.shift(1))
    min_L = np.minimum(low, low.shift(1))
    gamma = np.log(max_H / min_L) ** 2
    
    beta = H_L + H_L_prev
    alpha_num = np.sqrt(2 * beta) - np.sqrt(beta)
    alpha_den = 3 - 2 * np.sqrt(2)
    alpha = alpha_num / alpha_den - np.sqrt(gamma / alpha_den)
    
    spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    features['corwin_schultz_spread'] = spread
    
    # Bekker-Parkinson vol
    features['bekker_parkinson_vol'] = (1.0 / (4 * np.log(2))) * H_L
    
    # Amihud illiquidity
    ret = np.abs(np.log(close / close.shift(1)))
    dollar_vol = close * df['Volume']
    amihud = ret / dollar_vol.replace(0, np.nan)
    features['amihud_illiquidity'] = amihud.rolling(window=20).mean()
    
    # Roll spread
    dp = close.diff()
    dp_prev = dp.shift(1)
    
    cov_dp = dp.rolling(window=20).cov(dp_prev)
    features['roll_spread'] = 2 * np.sqrt(np.maximum(0, -cov_dp))
    
    return features

def lempel_ziv_complexity(binary_sequence):
    """Calculate Lempel-Ziv complexity for a binary sequence."""
    if len(binary_sequence) == 0:
        return 0
    s = ''.join(binary_sequence.astype(str))
    i, k, l = 0, 1, 1
    k_max = 1
    n = len(s)
    complexity = 1
    while True:
        if i + k >= n:
            break
        if s[i + k] == s[l + k - 1]:
            k += 1
            if l + k > k_max:
                k_max = l + k
        else:
            i += 1
            if i == l:
                complexity += 1
                l += k_max
                i = 0
                k = 1
                k_max = 1
            else:
                k = 1
    return complexity

def compute_entropy_features(returns: pd.Series, window: int = 50) -> pd.DataFrame:
    features = pd.DataFrame(index=returns.index)
    
    # Rolling Shannon entropy
    def shannon_entropy(x):
        if len(x) == 0: return np.nan
        hist, _ = np.histogram(x, bins=10)
        p = hist / np.sum(hist)
        p = p[p > 0]
        return -np.sum(p * np.log(p))
        
    features['shannon_entropy'] = returns.rolling(window=window).apply(shannon_entropy, raw=True)
    
    # Lempel-Ziv complexity
    # Binary encode sign of returns
    signs = np.where(returns > 0, 1, 0)
    signs_series = pd.Series(signs, index=returns.index)
    
    features['lempel_ziv_complexity'] = signs_series.rolling(window=window).apply(
        lambda x: lempel_ziv_complexity(x.astype(int)), raw=True
    )
    
    return features

def build_feature_matrix(df: pd.DataFrame, fracdiff: pd.Series, events: pd.DataFrame, labels: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    close = df['Adj Close']
    ret = np.log(close / close.shift(1))
    
    # Compute feature groups
    mom_features = compute_momentum_features(close)
    vol_features = compute_volatility_features(close)
    volume_features = compute_volume_features(df)
    micro_features = compute_microstructure_features(df)
    ent_features = compute_entropy_features(ret)
    
    # Combine features
    all_features = pd.concat([
        mom_features, vol_features, volume_features, micro_features, ent_features
    ], axis=1)
    
    # fracdiff should be passed as a DataFrame or Series. Assume fracdiff is a Series or a DataFrame with one column
    if isinstance(fracdiff, pd.Series):
        all_features['fracdiff'] = fracdiff
    else:
        all_features['fracdiff'] = fracdiff.iloc[:, 0]
    
    # Align to event timestamps from labels
    dataset = all_features.loc[labels.index].copy()
    
    # Merge with labels and weights
    dataset['label'] = labels['bin']
    dataset['weight'] = weights['weight'] if 'weight' in weights else weights.iloc[:, 0]
    
    # Also include the t1 exit time (used later for purged CV)
    if 't1' in events.columns:
        dataset['t1'] = events['t1']
    elif 't1' in labels.columns:
        dataset['t1'] = labels['t1']
        
    # Drop rows with any NaN
    dataset.dropna(inplace=True)
    
    return dataset
