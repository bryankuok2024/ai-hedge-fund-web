import math
import pandas as pd
import numpy as np
from typing import Any # Add Any for type hinting

from langchain_core.messages import HumanMessage

from graph.state import AgentState, show_agent_reasoning

import json

from tools.api import get_prices, prices_to_df
from utils.progress import progress

# Helper function to safely get the last value or NaN
def safe_iloc_float(series: pd.Series) -> float:
    """Safely get the last value of a Series as float, return NaN if fails."""
    if series is None or series.empty:
        return math.nan
    try:
        val = series.iloc[-1]
        # Check for pandas _NA_ or numpy nan
        if pd.isna(val):
             return math.nan
        return float(val)
    except (IndexError, TypeError):
        return math.nan


##### Technical Analyst #####
def technical_analyst_agent(state: AgentState):
    """
    Sophisticated technical analysis system that combines multiple trading strategies for multiple tickers:
    1. Trend Following
    2. Mean Reversion
    3. Momentum
    4. Volatility Analysis
    5. Statistical Arbitrage Signals
    """
    data = state["data"]
    start_date = data["start_date"]
    end_date = data["end_date"]
    tickers = data["tickers"]

    # Initialize analysis for each ticker
    technical_analysis = {}

    for ticker in tickers:
        progress.update_status("technical_analyst_agent", ticker, "Analyzing price data")

        # Get the historical price data
        prices = get_prices(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
        )

        if not prices:
            progress.update_status("technical_analyst_agent", ticker, "Failed: No price data found")
            # Assign default neutral/NaN result if no data
            technical_analysis[ticker] = {
                "signal": "neutral",
                "confidence": 50,
                "strategy_signals": {
                    "trend_following": {"signal": "neutral", "confidence": 50, "metrics": {}},
                    "mean_reversion": {"signal": "neutral", "confidence": 50, "metrics": {}},
                    "momentum": {"signal": "neutral", "confidence": 50, "metrics": {}},
                    "volatility": {"signal": "neutral", "confidence": 50, "metrics": {}},
                    "statistical_arbitrage": {"signal": "neutral", "confidence": 50, "metrics": {}},
                }
            }
            continue # Skip to next ticker

        # Convert prices to a DataFrame
        prices_df = prices_to_df(prices)

        # Check if DataFrame is empty after conversion (e.g., API returned empty list)
        if prices_df.empty:
            progress.update_status("technical_analyst_agent", ticker, "Failed: Empty DataFrame after conversion")
            # Assign default neutral/NaN result
            technical_analysis[ticker] = {
                "signal": "neutral",
                "confidence": 50,
                "strategy_signals": {
                    "trend_following": {"signal": "neutral", "confidence": 50, "metrics": {}},
                    "mean_reversion": {"signal": "neutral", "confidence": 50, "metrics": {}},
                    "momentum": {"signal": "neutral", "confidence": 50, "metrics": {}},
                    "volatility": {"signal": "neutral", "confidence": 50, "metrics": {}},
                    "statistical_arbitrage": {"signal": "neutral", "confidence": 50, "metrics": {}},
                }
            }
            continue # Skip to next ticker


        progress.update_status("technical_analyst_agent", ticker, "Calculating trend signals")
        trend_signals = calculate_trend_signals(prices_df)

        progress.update_status("technical_analyst_agent", ticker, "Calculating mean reversion")
        mean_reversion_signals = calculate_mean_reversion_signals(prices_df)

        progress.update_status("technical_analyst_agent", ticker, "Calculating momentum")
        momentum_signals = calculate_momentum_signals(prices_df)

        progress.update_status("technical_analyst_agent", ticker, "Analyzing volatility")
        volatility_signals = calculate_volatility_signals(prices_df)

        progress.update_status("technical_analyst_agent", ticker, "Statistical analysis")
        stat_arb_signals = calculate_stat_arb_signals(prices_df)

        # Combine all signals using a weighted ensemble approach
        strategy_weights = {
            "trend": 0.25,
            "mean_reversion": 0.20,
            "momentum": 0.25,
            "volatility": 0.15,
            "stat_arb": 0.15,
        }

        progress.update_status("technical_analyst_agent", ticker, "Combining signals")
        combined_signal = weighted_signal_combination(
            {
                "trend": trend_signals,
                "mean_reversion": mean_reversion_signals,
                "momentum": momentum_signals,
                "volatility": volatility_signals,
                "stat_arb": stat_arb_signals,
            },
            strategy_weights,
        )

        # Generate detailed analysis report for this ticker
        technical_analysis[ticker] = {
            "signal": combined_signal["signal"],
            "confidence": round(combined_signal["confidence"] * 100),
            "strategy_signals": {
                "trend_following": {
                    "signal": trend_signals["signal"],
                    "confidence": round(trend_signals["confidence"] * 100),
                    # Use normalize_pandas which now handles NaN/inf
                    "metrics": normalize_pandas(trend_signals["metrics"]),
                },
                "mean_reversion": {
                    "signal": mean_reversion_signals["signal"],
                    "confidence": round(mean_reversion_signals["confidence"] * 100),
                    "metrics": normalize_pandas(mean_reversion_signals["metrics"]),
                },
                "momentum": {
                    "signal": momentum_signals["signal"],
                    "confidence": round(momentum_signals["confidence"] * 100),
                    "metrics": normalize_pandas(momentum_signals["metrics"]),
                },
                "volatility": {
                    "signal": volatility_signals["signal"],
                    "confidence": round(volatility_signals["confidence"] * 100),
                    "metrics": normalize_pandas(volatility_signals["metrics"]),
                },
                "statistical_arbitrage": {
                    "signal": stat_arb_signals["signal"],
                    "confidence": round(stat_arb_signals["confidence"] * 100),
                    "metrics": normalize_pandas(stat_arb_signals["metrics"]),
                },
            },
        }
        progress.update_status("technical_analyst_agent", ticker, "Done")

    # Create the technical analyst message
    # Ensure the content is valid JSON even if metrics contain None (from NaN)
    message_content = json.dumps(technical_analysis)
    message = HumanMessage(
        content=message_content,
        name="technical_analyst_agent",
    )

    if state["metadata"]["show_reasoning"]:
        # Use the already prepared dict for reasoning display
        show_agent_reasoning(technical_analysis, "Technical Analyst")

    # Add the signal to the analyst_signals list
    state["data"]["analyst_signals"]["technical_analyst_agent"] = technical_analysis

    return {
        "messages": state["messages"] + [message],
        "data": data,
    }

def calculate_trend_signals(prices_df):
    """Advanced trend following strategy using multiple timeframes and indicators"""
    if prices_df.empty or len(prices_df) < 55: # Check if enough data for longest EMA
        return {"signal": "neutral", "confidence": 0.5, "metrics": {"adx": math.nan, "trend_strength": math.nan}}

    ema_8 = calculate_ema(prices_df, 8)
    ema_21 = calculate_ema(prices_df, 21)
    ema_55 = calculate_ema(prices_df, 55)
    adx = calculate_adx(prices_df, 14)

    last_ema_8 = safe_iloc_float(ema_8)
    last_ema_21 = safe_iloc_float(ema_21)
    last_ema_55 = safe_iloc_float(ema_55)
    last_adx = safe_iloc_float(adx["adx"])

    # Determine trend direction and strength
    trend_strength = last_adx / 100.0 if not math.isnan(last_adx) else 0.0 # Default strength 0 if NaN

    short_trend_valid = not (math.isnan(last_ema_8) or math.isnan(last_ema_21))
    medium_trend_valid = not (math.isnan(last_ema_21) or math.isnan(last_ema_55))

    short_trend = last_ema_8 > last_ema_21 if short_trend_valid else False
    medium_trend = last_ema_21 > last_ema_55 if medium_trend_valid else False

    if short_trend_valid and medium_trend_valid and short_trend and medium_trend:
        signal = "bullish"
        confidence = trend_strength
    elif short_trend_valid and medium_trend_valid and not short_trend and not medium_trend:
        signal = "bearish"
        confidence = trend_strength
    else:
        signal = "neutral"
        confidence = 0.5 # Neutral confidence if trends are mixed or invalid

    return {
        "signal": signal,
        "confidence": confidence if not math.isnan(confidence) else 0.5,
        "metrics": {
            "adx": last_adx, # Keep NaN if it was NaN
            "trend_strength": trend_strength if not math.isnan(trend_strength) else math.nan,
        },
    }

def calculate_mean_reversion_signals(prices_df):
    """Mean reversion strategy using statistical measures and Bollinger Bands"""
    if prices_df.empty or len(prices_df) < 50: # Need enough data for 50-day MA/Std
         return {"signal": "neutral", "confidence": 0.5, "metrics": {"z_score": math.nan, "price_vs_bb": math.nan, "rsi_14": math.nan, "rsi_28": math.nan}}

    ma_50 = prices_df["close"].rolling(window=50).mean()
    std_50 = prices_df["close"].rolling(window=50).std()

    # Avoid division by zero or NaN std dev
    last_std_50 = safe_iloc_float(std_50)
    last_ma_50 = safe_iloc_float(ma_50)
    last_close_price = safe_iloc_float(prices_df["close"])

    if math.isnan(last_std_50) or last_std_50 == 0 or math.isnan(last_ma_50) or math.isnan(last_close_price):
        z_score_val = math.nan
    else:
        z_score_val = (last_close_price - last_ma_50) / last_std_50

    bb_upper, bb_lower = calculate_bollinger_bands(prices_df)
    rsi_14 = calculate_rsi(prices_df, 14)
    rsi_28 = calculate_rsi(prices_df, 28)

    last_bb_upper = safe_iloc_float(bb_upper)
    last_bb_lower = safe_iloc_float(bb_lower)
    last_rsi_14 = safe_iloc_float(rsi_14)
    last_rsi_28 = safe_iloc_float(rsi_28)

    # Calculate price_vs_bb safely
    bb_range = last_bb_upper - last_bb_lower
    if math.isnan(last_close_price) or math.isnan(last_bb_lower) or math.isnan(bb_range) or bb_range == 0:
        price_vs_bb_val = math.nan
    else:
        price_vs_bb_val = (last_close_price - last_bb_lower) / bb_range

    # Combine signals
    signal = "neutral"
    confidence = 0.5
    if not math.isnan(z_score_val) and not math.isnan(price_vs_bb_val):
        if z_score_val < -2 and price_vs_bb_val < 0.2:
            signal = "bullish"
            confidence = min(abs(z_score_val) / 3, 1.0) # Adjusted divisor slightly
        elif z_score_val > 2 and price_vs_bb_val > 0.8:
            signal = "bearish"
            confidence = min(abs(z_score_val) / 3, 1.0) # Adjusted divisor slightly

    return {
        "signal": signal,
        "confidence": confidence if not math.isnan(confidence) else 0.5,
        "metrics": {
            "z_score": z_score_val,
            "price_vs_bb": price_vs_bb_val,
            "rsi_14": last_rsi_14,
            "rsi_28": last_rsi_28,
        },
    }

def calculate_momentum_signals(prices_df):
    """Multi-factor momentum strategy"""
    if prices_df.empty or len(prices_df) < 126: # Need enough data for longest lookback
        return {"signal": "neutral", "confidence": 0.5, "metrics": {"momentum_1m": math.nan, "momentum_3m": math.nan, "momentum_6m": math.nan, "volume_momentum": math.nan}}

    returns = prices_df["close"].pct_change()
    mom_1m = returns.rolling(21).sum()
    mom_3m = returns.rolling(63).sum()
    mom_6m = returns.rolling(126).sum()

    # Volume momentum
    volume_ma = prices_df["volume"].rolling(21).mean()
    # Avoid division by zero or NaN MA
    volume_momentum = prices_df["volume"] / volume_ma.replace(0, np.nan) # Replace 0 with NaN before division

    last_mom_1m = safe_iloc_float(mom_1m)
    last_mom_3m = safe_iloc_float(mom_3m)
    last_mom_6m = safe_iloc_float(mom_6m)
    last_vol_mom = safe_iloc_float(volume_momentum)

    # Calculate momentum score safely
    if math.isnan(last_mom_1m) or math.isnan(last_mom_3m) or math.isnan(last_mom_6m):
        momentum_score = math.nan
    else:
        momentum_score = (0.4 * last_mom_1m + 0.3 * last_mom_3m + 0.3 * last_mom_6m)

    # Volume confirmation safely
    volume_confirmation = False
    if not math.isnan(last_vol_mom):
        volume_confirmation = last_vol_mom > 1.0

    # Signal logic
    signal = "neutral"
    confidence = 0.5
    if not math.isnan(momentum_score):
        if momentum_score > 0.05 and volume_confirmation:
            signal = "bullish"
            confidence = min(abs(momentum_score) * 8, 1.0) # Adjusted multiplier
        elif momentum_score < -0.05 and volume_confirmation:
            signal = "bearish"
            confidence = min(abs(momentum_score) * 8, 1.0) # Adjusted multiplier

    return {
        "signal": signal,
        "confidence": confidence if not math.isnan(confidence) else 0.5,
        "metrics": {
            "momentum_1m": last_mom_1m,
            "momentum_3m": last_mom_3m,
            "momentum_6m": last_mom_6m,
            "volume_momentum": last_vol_mom,
        },
    }

def calculate_volatility_signals(prices_df):
    """Volatility analysis using historical volatility and ATR"""
    if prices_df.empty or len(prices_df) < 21: # Need data for 21-day vol
        return {"signal": "neutral", "confidence": 0.5, "metrics": {"historical_volatility": math.nan, "volatility_regime": math.nan, "volatility_z_score": math.nan, "atr_ratio": math.nan}}

    log_returns = np.log(prices_df["close"] / prices_df["close"].shift(1))
    hist_vol_21 = log_returns.rolling(window=21).std() * np.sqrt(252) # Annualized
    atr = calculate_atr(prices_df, 14)

    last_hist_vol = safe_iloc_float(hist_vol_21)
    last_atr = safe_iloc_float(atr)
    last_close = safe_iloc_float(prices_df["close"])

    # Volatility regime (simple comparison to rolling mean)
    vol_ma_63 = hist_vol_21.rolling(window=63).mean()
    last_vol_ma = safe_iloc_float(vol_ma_63)
    volatility_regime = math.nan
    if not math.isnan(last_hist_vol) and not math.isnan(last_vol_ma):
        volatility_regime = 1 if last_hist_vol > last_vol_ma else 0 # 1: High, 0: Low

    # Volatility z-score
    vol_std_63 = hist_vol_21.rolling(window=63).std()
    last_vol_std = safe_iloc_float(vol_std_63)
    volatility_z_score = math.nan
    if not math.isnan(last_hist_vol) and not math.isnan(last_vol_ma) and not math.isnan(last_vol_std) and last_vol_std != 0:
        volatility_z_score = (last_hist_vol - last_vol_ma) / last_vol_std

    # ATR ratio
    atr_ratio = math.nan
    if not math.isnan(last_atr) and not math.isnan(last_close) and last_close != 0:
         atr_ratio = last_atr / last_close

    # Volatility signals (Example: High vol might be bearish)
    signal = "neutral"
    confidence = 0.5
    if not math.isnan(volatility_regime):
        if volatility_regime == 1: # High vol
            signal = "bearish"
            confidence = 0.6 # Example confidence
        else: # Low vol
            signal = "bullish"
            confidence = 0.6 # Example confidence

    return {
        "signal": signal,
        "confidence": confidence,
        "metrics": {
            "historical_volatility": last_hist_vol,
            "volatility_regime": volatility_regime,
            "volatility_z_score": volatility_z_score,
            "atr_ratio": atr_ratio,
        },
    }

def calculate_stat_arb_signals(prices_df):
    """
    Statistical arbitrage signals (e.g., Hurst exponent, skewness, kurtosis)
    """
    if prices_df.empty or len(prices_df) < 21: # Basic check
         return {"signal": "neutral", "confidence": 0.5, "metrics": {"hurst_exponent": math.nan, "skewness": math.nan, "kurtosis": math.nan}}

    close_prices = prices_df["close"].dropna()
    if close_prices.empty or len(close_prices) < 2:
         return {"signal": "neutral", "confidence": 0.5, "metrics": {"hurst_exponent": math.nan, "skewness": math.nan, "kurtosis": math.nan}}

    # Hurst exponent - requires more data typically
    hurst = calculate_hurst_exponent(close_prices) if len(close_prices) > 20 else math.nan

    # Skewness and Kurtosis
    log_returns = np.log(close_prices / close_prices.shift(1)).dropna()
    skewness = log_returns.skew() if not log_returns.empty else math.nan
    kurtosis = log_returns.kurtosis() if not log_returns.empty else math.nan # Fisher's definition (excess kurtosis)

    # Example signal: Mean-reverting if Hurst < 0.5
    signal = "neutral"
    confidence = 0.5
    if not math.isnan(hurst):
        if hurst < 0.45: # Threshold for mean-reversion indication
            signal = "neutral" # Might imply mean reversion strategy is applicable
            confidence = 0.6
        elif hurst > 0.55: # Threshold for trending indication
             signal = "neutral" # Might imply trend following is applicable
             confidence = 0.6

    return {
        "signal": signal,
        "confidence": confidence,
        "metrics": {
            "hurst_exponent": hurst,
            "skewness": float(skewness) if not pd.isna(skewness) else math.nan,
            "kurtosis": float(kurtosis) if not pd.isna(kurtosis) else math.nan,
        },
    }

def weighted_signal_combination(signals, weights):
    """
    Combines signals from different strategies using weighted averaging.
    Handles potential NaN confidence values.
    """
    combined_score = 0
    total_weight = 0

    for strategy, result in signals.items():
        weight = weights.get(strategy, 0)
        signal_map = {"bullish": 1, "neutral": 0, "bearish": -1}
        signal_value = signal_map.get(result["signal"], 0)
        confidence = result["confidence"]

        # Skip if confidence is NaN or invalid
        if math.isnan(confidence) or not (0 <= confidence <= 1):
             print(f"Warning: Skipping strategy {strategy} due to invalid confidence: {confidence}")
             continue

        combined_score += signal_value * confidence * weight
        total_weight += weight

    # Avoid division by zero if all weights/confidences were invalid
    if total_weight == 0:
         final_signal = "neutral"
         final_confidence = 0.5
    else:
         # Normalize score to be between -1 and 1
         normalized_score = combined_score / total_weight

         # Convert score back to signal and confidence
         if normalized_score > 0.2:  # Threshold for bullish
             final_signal = "bullish"
             final_confidence = min(normalized_score * 1.5, 1.0) # Scale up confidence
         elif normalized_score < -0.2: # Threshold for bearish
             final_signal = "bearish"
             final_confidence = min(abs(normalized_score) * 1.5, 1.0) # Scale up confidence
         else:
             final_signal = "neutral"
             final_confidence = 0.5 # Default confidence for neutral zone

    return {"signal": final_signal, "confidence": final_confidence}


def normalize_pandas(obj: Any) -> Any:
    """Recursively converts pandas/numpy types, NaNs, and infs in nested objects to JSON serializable formats."""
    if isinstance(obj, (pd.Series, np.ndarray)):
        # Convert Series/arrays, replacing NaN/inf with None
        return [normalize_pandas(x) for x in obj.tolist()]
    elif isinstance(obj, (pd.Timestamp, pd.Timedelta)):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: normalize_pandas(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [normalize_pandas(x) for x in obj]
    elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        # Check for NaN before converting (though less common for numpy ints)
        if pd.isna(obj):
             return None
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16, float)):
        # Convert NaN and inf to None (JSON doesn't support NaN/inf)
        if math.isnan(obj) or math.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif pd.isna(obj): # Catch pandas NA types like pd.NA
         return None
    # Add other type checks if necessary
    return obj # Return object as is if no conversion needed

# --- Helper Calculation Functions (ensure they handle edge cases) ---

def calculate_rsi(prices_df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate RSI, returning NaN if not enough data."""
    if len(prices_df) < period + 1:
        return pd.Series([math.nan] * len(prices_df), index=prices_df.index)
    delta = prices_df["close"].diff()
    gain = delta.where(delta > 0, 0).fillna(0)
    loss = -delta.where(delta < 0, 0).fillna(0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.bfill().fillna(50) # Use bfill() directly

def calculate_bollinger_bands(prices_df: pd.DataFrame, window: int = 20) -> tuple[pd.Series, pd.Series]:
    """Calculate Bollinger Bands, returning NaNs if not enough data."""
    if len(prices_df) < window:
         nan_series = pd.Series([math.nan] * len(prices_df), index=prices_df.index)
         return nan_series, nan_series
    ma = prices_df["close"].rolling(window=window).mean()
    std = prices_df["close"].rolling(window=window).std().fillna(0) # Fill NaN std with 0
    upper_band = ma + (std * 2)
    lower_band = ma - (std * 2)
    return upper_band, lower_band

def calculate_ema(df: pd.DataFrame, window: int) -> pd.Series:
    """Calculate EMA, handling potential NaNs in input."""
    if len(df) < window:
        return pd.Series([math.nan] * len(df), index=df.index)
    return df["close"].ffill().ewm(span=window, adjust=False, min_periods=window).mean() # Use ffill() directly

def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Calculate ADX, handling potential NaNs and zero division."""
    if len(df) < period * 2: # ADX typically needs more data
        return pd.DataFrame({'adx': [math.nan] * len(df)}, index=df.index)

    df = df.copy() # Work on a copy
    df['H-L'] = df['high'] - df['low']
    df['H-PC'] = abs(df['high'] - df['close'].shift(1))
    df['L-PC'] = abs(df['low'] - df['close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1).fillna(0)

    df['DMplus'] = np.where((df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low']), df['high'] - df['high'].shift(1), 0)
    df['DMplus'] = np.where(df['DMplus'] < 0, 0, df['DMplus'])
    df['DMminus'] = np.where((df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1)), df['low'].shift(1) - df['low'], 0)
    df['DMminus'] = np.where(df['DMminus'] < 0, 0, df['DMminus'])

    TRn = df['TR'].ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    DMplusN = df['DMplus'].ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    DMminusN = df['DMminus'].ewm(alpha=1/period, adjust=False, min_periods=period).mean()

    DIplus = (100 * DMplusN / TRn.replace(0, np.nan)).fillna(0)
    DIminus = (100 * DMminusN / TRn.replace(0, np.nan)).fillna(0)
    DIdiff = abs(DIplus - DIminus)
    DIsum = (DIplus + DIminus).replace(0, np.nan) # Avoid zero division

    DX = (100 * DIdiff / DIsum).fillna(0)
    ADX = DX.ewm(alpha=1/period, adjust=False, min_periods=period).mean()

    return pd.DataFrame({'adx': ADX})

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate ATR, handling NaNs."""
    if len(df) < period:
        return pd.Series([math.nan] * len(df), index=df.index)

    df = df.copy()
    df['H-L'] = df['high'] - df['low']
    df['H-PC'] = abs(df['high'] - df['close'].shift(1))
    df['L-PC'] = abs(df['low'] - df['close'].shift(1))
    # Use fillna(0) for TR calculation components
    tr = df[['H-L', 'H-PC', 'L-PC']].max(axis=1).fillna(0)
    # Use EWM for ATR calculation
    atr = tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    return atr

def calculate_hurst_exponent(price_series: pd.Series, max_lag: int = 100) -> float:
    """Calculate Hurst Exponent, requires sufficient non-NaN data."""
    # Ensure the series is clean and has enough data
    price_series = price_series.dropna()
    if len(price_series) < max_lag + 1:
        return math.nan

    lags = range(2, max_lag + 1)
    # Calculate log returns carefully
    log_returns = np.log(price_series / price_series.shift(1)).dropna()
    if len(log_returns) < max_lag: # Check again after dropna
        return math.nan

    # Calculate rescaled range
    tau = [np.std(log_returns[i:i+lag]) for lag in lags for i in range(len(log_returns) - lag + 1)]
    # Check if tau contains valid standard deviations
    if not tau or np.isnan(tau).any() or np.isinf(tau).any() or np.any(np.array(tau) <= 0):
        return math.nan

    # Calculate R/S statistic for each lag size n
    m = [np.mean(log_returns[i:i+lag]) for lag in lags for i in range(len(log_returns) - lag + 1)]
    y = [log_returns[i:i+lag] - m[k] for k, lag in enumerate(lags) for i in range(len(log_returns) - lag + 1)]
    z = [np.cumsum(y[k]) for k in range(len(y))]
    rs = [(np.max(z[k]) - np.min(z[k])) / tau[k] if tau[k] > 0 else 0 for k in range(len(z))]

    # Average R/S values for each lag size
    rs_avg = [np.mean([rs[k] for k, lag_val in enumerate(lags) if lag_val == lag]) for lag in lags]

    # Ensure we have valid R/S averages
    valid_lags_indices = [i for i, rs_val in enumerate(rs_avg) if rs_val > 0]
    if len(valid_lags_indices) < 2: # Need at least 2 points for regression
        return math.nan

    valid_lags = np.array(lags)[valid_lags_indices]
    valid_rs_avg = np.array(rs_avg)[valid_lags_indices]

    # Fit line to log-log plot
    try:
        log_lags = np.log(valid_lags)
        log_rs = np.log(valid_rs_avg)
        coeffs = np.polyfit(log_lags, log_rs, 1)
        hurst = coeffs[0]
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        hurst = math.nan # Handle potential errors during fitting

    return hurst if not (math.isnan(hurst) or math.isinf(hurst)) else math.nan
