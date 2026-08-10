from __future__ import annotations
from pathlib import Path
import os
import numpy as np
import pandas as pd
CACHE = Path(os.environ.get('SVJMCMC_DATA_CACHE', 'data/data_cache'))
PHASE1_TICKERS = ['AAPL', 'TSLA', 'SPY']
PHASE1_START, PHASE1_END = ('2018-01-01', '2024-12-31')

def _cache_path(ticker: str, start: str, end: str) -> Path:
    return CACHE / f'{ticker}_{start}_{end}.parquet'

def trim_leading_stale(r: np.ndarray, win: int=20, zthresh: float=0.2, min_block: int=20, max_frac: float=0.7):
    r = np.asarray(r, float)
    z = np.abs(r) < 1e-09
    n = z.size
    if n < win * 3 or z[:win].mean() <= zthresh:
        return (r, 0)
    cap = int(max_frac * n)
    for i in range(1, cap):
        if z[i:i + win].mean() <= zthresh:
            return (r[i:], i) if i >= min_block else (r, 0)
    return (r, 0)

def load_returns(ticker: str, start: str=PHASE1_START, end: str=PHASE1_END, refresh: bool=False, trim_stale: bool=True) -> np.ndarray:
    p = _cache_path(ticker, start, end)
    if p.exists() and (not refresh):
        ret = pd.read_parquet(p)['ret_pct'].to_numpy()
    else:
        import yfinance as yf
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        close = df['Close'].dropna().to_numpy().ravel()
        ret = 100.0 * np.diff(np.log(close))
        out = pd.DataFrame({'ret_pct': ret})
        CACHE.mkdir(parents=True, exist_ok=True)
        out.to_parquet(p)
    if trim_stale:
        ret, _ = trim_leading_stale(ret)
    return ret

def load_phase1(refresh: bool=False) -> dict[str, np.ndarray]:
    return {t: load_returns(t, refresh=refresh) for t in PHASE1_TICKERS}
UNIVERSE = ['AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'GOOG', 'META', 'TSLA', 'AVGO', 'BRK-B', 'JPM', 'V', 'MA', 'UNH', 'HD', 'PG', 'JNJ', 'COST', 'ABBV', 'MRK', 'CVX', 'PEP', 'KO', 'ADBE', 'WMT', 'CRM', 'BAC', 'ACN', 'MCD', 'NFLX', 'LIN', 'TMO', 'ABT', 'CSCO', 'AMD', 'ORCL', 'DIS', 'WFC', 'DHR', 'TXN', 'VZ', 'INTC', 'QCOM', 'PM', 'INTU', 'CAT', 'IBM', 'GE', 'AMGN', 'NOW', 'UNP', 'SPGI', 'HON', 'LOW', 'GS', 'ISRG', 'BKNG', 'AMAT', 'NKE', 'PFE', 'T', 'RTX', 'AXP', 'ELV', 'PGR', 'SYK', 'BSX', 'TJX', 'BLK', 'MS', 'C', 'MDT', 'LMT', 'VRTX', 'ADP', 'GILD', 'DE', 'REGN', 'MU', 'CB', 'ETN', 'PLD', 'ADI', 'MMC', 'BMY', 'LRCX', 'SBUX', 'SCHW', 'KLAC', 'CI', 'SO', 'ZTS', 'DUK', 'MO', 'BX', 'CME', 'ICE', 'EQIX', 'SHW', 'ITW', 'WM', 'PYPL', 'AON', 'CL', 'MCK', 'GD', 'TGT', 'FCX', 'NOC', 'EMR', 'CSX', 'PNC', 'USB', 'APD', 'FDX', 'MAR', 'MMM', 'ORLY', 'NSC', 'HUM', 'PH', 'ROP', 'ECL', 'CDNS', 'PSX', 'MPC', 'APH', 'MSI', 'AJG', 'CARR', 'PCAR', 'SNPS', 'HCA', 'OXY', 'WELL', 'F', 'GM', 'VLO', 'TT', 'EOG', 'COF', 'TFC', 'DXCM', 'AEP', 'SLB', 'MET', 'NXPI', 'SRE', 'TDG', 'AFL', 'FTNT', 'PSA', 'O', 'AIG', 'D', 'KMB', 'GIS', 'CTAS', 'MCHP', 'EW', 'IDXX', 'STZ', 'AMP', 'ROST', 'DOW', 'HLT', 'KHC', 'JCI', 'FIS', 'EXC', 'CMG', 'ADM', 'BK', 'PRU', 'ALL', 'TEL', 'TRV', 'A', 'WMB', 'CNC', 'PAYX', 'DD', 'MNST', 'YUM', 'OKE', 'GWW', 'AME', 'VRSK', 'BIIB', 'CTSH', 'XEL', 'ODFL', 'KR', 'GEHC', 'FAST', 'KMI', 'PCG', 'CMI', 'EA', 'DG', 'VICI', 'ED', 'HSY', 'OTIS', 'DLTR', 'CSGP', 'PPG', 'PEG', 'RSG', 'CCI', 'ON', 'DVN', 'MPWR', 'GLW', 'FANG', 'HES', 'EFX', 'WEC', 'KDP', 'DLR', 'ANSS', 'ZBH', 'AVB', 'TROW', 'WST', 'ULTA', 'HPQ', 'FTV', 'WBD', 'EBAY', 'DAL', 'STT', 'MTD', 'GPN', 'VMC', 'MLM', 'ES', 'NUE', 'EIX', 'AWK', 'CHD', 'PWR', 'WTW', 'RMD', 'DFS', 'HIG', 'TSCO', 'BR', 'EXR', 'APTV', 'NDAQ', 'ROK', 'AEE', 'K', 'IFF', 'WY', 'CBRE', 'GPC', 'LH', 'STE', 'MKC', 'CAH', 'IT', 'DTE', 'VTR', 'CLX', 'ETR', 'HBAN', 'SPY', 'QQQ']
_SEALED_500 = Path(os.environ.get('SVJMCMC_UNIVERSE', 'universe_sp500.txt'))

def sealed_universe() -> list[str]:
    if _SEALED_500.exists():
        return [ln.strip() for ln in _SEALED_500.read_text().splitlines() if ln.strip() and (not ln.startswith('#'))]
    return list(UNIVERSE)

def load_universe(refresh: bool=False, min_len: int=1759, tickers: list[str] | None=None, full_history_only: bool=True) -> dict[str, np.ndarray]:
    names = tickers if tickers is not None else sealed_universe()
    raw = {}
    for t in names:
        try:
            r = load_returns(t, refresh=refresh)
            if len(r) >= min_len and np.isfinite(r).all():
                raw[t] = r
        except Exception:
            continue
    if full_history_only and raw:
        Tmax = max((len(v) for v in raw.values()))
        raw = {k: v for k, v in raw.items() if len(v) == Tmax}
    return raw
