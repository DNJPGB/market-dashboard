import json
import time
import datetime
import sys
import csv
import xml.etree.ElementTree as ET
from pathlib import Path
from io import StringIO

try:
    import yfinance as yf
except ImportError:
    print("Installing yfinance...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance", "requests"])
    import yfinance as yf

import requests

# ── DEFAULT TICKERS (overridden by tickers.json if present) ────────────────────
ETF_MAIN    = ['SPY','QQQ','DIA','IWM']
SUBMARKET   = ['IVW','IVE','IJK','IJJ','IJT','IJS','MGK','VUG','VTV']
SECTOR      = ['XLK','XLV','XLF','XLE','XLY','XLI','XLB','XLU','XLRE','XLC','XLP']
SECTOR_EW   = ['RSPG','RSPT','RSPF','RSPN','RSPD','RSP','RSPU','RSPM','RSPH','RSPR','RSPS','RSPC']
THEMATIC    = ['BOTZ','HACK','SOXX','ICLN','SKYY','XBI','ITA','FINX','ARKG','URA',
               'AIQ','CIBR','ROBO','ARKK','DRIV','OGIG','ACES','PAVE','HERO','CLOU']
COUNTRY     = ['GREK','ARGT','EWS','EWP','EUFN','MCHI','EWZ','EWI','EWY','EWH',
               'ECH','EWC','EWL','EWQ','EWA','IEV','IEUR','INDA','EWG','EWW',
               'EZU','EEM','EFA','EWD','TUR','EZA','ACWI','KSA','EIDO','EWJ','EWT','THD']
FUTURES     = ['ES=F','NQ=F','RTY=F','YM=F']
METALS      = ['GC=F','SI=F','HG=F','PL=F','PA=F']
ENERGY      = ['CL=F','NG=F']
GLOBAL_IDX  = ['^N225','^KS11','^NSEI','000001.SS','000300.SS','^HSI','^FTSE','^FCHI','^GDAXI']
YIELDS      = ['^TNX','^TYX']
DX_VIX      = ['DX-Y.NYB','^VIX']
CRYPTO_YF   = ['BTC-USD','ETH-USD','SOL-USD','XRP-USD']

# ── LOAD FROM tickers.json (single source of truth) ─────────────────────────────────
config_path = Path(__file__).parent / 'tickers.json'
if config_path.exists():
    with open(config_path) as f:
        CFG = json.load(f)
    ETF_MAIN   = CFG.get('etfmain',    ETF_MAIN)
    SUBMARKET  = CFG.get('submarket',  SUBMARKET)
    SECTOR     = CFG.get('sectors',    SECTOR)
    SECTOR_EW  = CFG.get('sectors_ew', SECTOR_EW)
    THEMATIC   = CFG.get('thematic',   THEMATIC)
    COUNTRY    = CFG.get('country',    COUNTRY)
    FUTURES    = CFG.get('futures',    FUTURES)
    METALS     = CFG.get('metals',     METALS)
    ENERGY     = CFG.get('energy',     ENERGY)
    GLOBAL_IDX = CFG.get('global',     GLOBAL_IDX)
    YIELDS     = CFG.get('yields',     YIELDS)
    DX_VIX     = CFG.get('dxvix',      DX_VIX)
    CRYPTO_YF  = CFG.get('crypto',     CRYPTO_YF)
    print(f"\u2713 Loaded tickers from tickers.json ({len(THEMATIC)} thematic, {len(COUNTRY)} country)")
else:
    print("\u26a0 tickers.json not found — using built-in defaults")

# ── TICKER REMAPS ──────────────────────────────────────────────────────────────────────────────
TICKER_REMAP = {
    'ES=F':'ES1!', 'NQ=F':'NQ1!', 'RTY=F':'RTY1!', 'YM=F':'YM1!',
    'GC=F':'GC1!', 'SI=F':'SI1!', 'HG=F':'HG1!', 'PL=F':'PL1!', 'PA=F':'PA1!',
    'CL=F':'CL1!', 'NG=F':'NG1!',
    '^TNX':'US10Y', '^TYX':'US30Y',
    'DX-Y.NYB':'DX-Y.NYB', '^VIX':'CBOE:VIX',
    'BTC-USD':'BTC','ETH-USD':'ETH','SOL-USD':'SOL','XRP-USD':'XRP',
}

# ── 2-YEAR TREASURY YIELD ─────────────────────────────────────────────────────────────────────────────
def fetch_treasury_2y():
    """Fetch 2-year Treasury yield. Tries FRED CSV first, then Treasury XML."""

    # Method 1: FRED public CSV (no API key needed)
    try:
        url = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS2'
        resp = requests.get(url, timeout=15, headers={'User-Agent': 'Mozilla/5.0'})
        resp.raise_for_status()
        reader = csv.reader(StringIO(resp.text))
        rows = list(reader)
        # Last row with a real value (skip "." missing values)
        rate = None
        for row in reversed(rows[1:]):  # skip header
            if len(row) == 2 and row[1] not in ('.', '', 'VALUE'):
                rate = float(row[1])
                break
        if rate is not None:
            print(f"  \u2713 US2Y = {rate}% (FRED)")
            return {
                'sym': 'US2Y', 'price': round(rate, 4),
                'd1': 0.0, 'w1': 0.0, 'hi52': 0.0, 'ytd': 0.0,
                'spark': [0.0, 0.0, 0.0, 0.0, 0.0],
            }
    except Exception as e:
        print(f"  FRED CSV failed: {e} — trying Treasury XML...")

    # Method 2: US Treasury XML feed
    try:
        now = datetime.datetime.utcnow()
        url = (
            "https://home.treasury.gov/resource-center/data-chart-center/"
            "interest-rates/pages/xml?data=daily_treasury_yield_curve"
            f"&field_tdr_date_value={now.strftime('%Y%m')}"
        )
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        ns_m = 'http://schemas.microsoft.com/ado/2007/08/dataservices/metadata'
        ns_d = 'http://schemas.microsoft.com/ado/2007/08/dataservices'
        root = ET.fromstring(resp.content)
        entries = root.findall(f'.//{{{ns_m}}}properties')
        if entries:
            last = entries[-1]
            val = last.find(f'{{{ns_d}}}BC_2YEAR')
            if val is not None and val.text:
                rate = float(val.text)
                print(f"  \u2713 US2Y = {rate}% (Treasury XML)")
                return {
                    'sym': 'US2Y', 'price': round(rate, 4),
                    'd1': 0.0, 'w1': 0.0, 'hi52': 0.0, 'ytd': 0.0,
                    'spark': [0.0, 0.0, 0.0, 0.0, 0.0],
                }
    except Exception as e:
        print(f"  Treasury XML also failed: {e}")

    print("  \u26a0 Could not fetch 2Y yield from any source")
    return None

# ── ETF HOLDINGS ──────────────────────────────────────────────────────────────────────────────────────
def fetch_etf_holdings(tickers):
    """
    Fetch top 10 holdings for each ETF using yfinance.
    Returns dict: { 'SLV': [{s, n, w}, ...], ... }
    Format matches HOLDINGS object in index.html: s=symbol, n=name, w=weight%
    """
    holdings_map = {}
    total = len(tickers)
    for i, sym in enumerate(tickers):
        print(f"  Holdings [{i+1}/{total}] {sym}...", end=' ')
        try:
            t = yf.Ticker(sym)
            rows = []

            # Try funds_data.top_holdings (most reliable for ETFs)
            try:
                fd = t.funds_data
                if fd is not None:
                    th = fd.top_holdings
                    if th is not None and hasattr(th, 'iterrows') and not th.empty:
                        for idx, row in th.head(10).iterrows():
                            # holdingName is the DataFrame INDEX, not a column
                            n = str(idx).strip() if idx and str(idx) != 'nan' else ''
                            s = ''
                            w = 0.0
                            for sym_col in ['symbol', 'Symbol', 'ticker']:
                                if sym_col in row.index:
                                    s = str(row[sym_col]).strip()
                                    break
                            for pct_col in ['holdingPercent', 'holdingpercent', '% Assets', 'weight']:
                                if pct_col in row.index:
                                    try:
                                        w = float(row[pct_col])
                                        w = round(w * 100, 2) if w <= 1.0 else round(w, 2)
                                    except Exception:
                                        pass
                                    break
                            if n or s:
                                rows.append({'s': s, 'n': n or s, 'w': w})
            except Exception:
                pass

            # Fallback: info['holdings']
            if not rows:
                try:
                    info = t.info
                    for h in info.get('holdings', [])[:10]:
                        s = h.get('symbol', '')
                        n = h.get('holdingName', s)
                        w = h.get('holdingPercent', 0)
                        if w <= 1.0:
                            w = round(w * 100, 2)
                        else:
                            w = round(w, 2)
                        rows.append({'s': str(s), 'n': str(n), 'w': w})
                except Exception:
                    pass

            if rows:
                holdings_map[sym] = rows
                print(f"\u2713 {len(rows)} holdings")
            else:
                print("\u2014")

        except Exception as e:
            print(f"\u2717 {e}")

        time.sleep(0.4)  # polite delay

    return holdings_map

# ── CORE METRIC EXTRACTION ─────────────────────────────────────────────────────────────────────────────────
def pct(new, old):
    if old and old != 0:
        return round((new - old) / abs(old) * 100, 2)
    return 0.0

def fetch_batch(tickers, retries=3):
    results = {}
    for attempt in range(retries):
        try:
            data = yf.download(
                tickers, period='1y', interval='1d',
                group_by='ticker', auto_adjust=True,
                progress=False, threads=True,
            )
            break
        except Exception as e:
            print(f"  Attempt {attempt+1} failed: {e}")
            time.sleep(5)
    else:
        print(f"  All retries failed for batch: {tickers[:3]}...")
        return results

    if len(tickers) == 1:
        sym = tickers[0]
        try:
            results[sym] = extract_metrics(data, sym)
        except Exception as e:
            print(f"  Error extracting {sym}: {e}")
        return results

    for sym in tickers:
        try:
            if sym in data.columns.get_level_values(0):
                df = data[sym].dropna()
            elif hasattr(data, 'columns') and sym in data:
                df = data[sym].dropna()
            else:
                continue
            results[sym] = extract_metrics(df, sym)
        except Exception as e:
            print(f"  Error extracting {sym}: {e}")
    return results

def extract_metrics(df, sym):
    df = df.dropna(subset=['Close'])
    if len(df) < 2:
        return None
    closes = df['Close'].values
    price  = float(closes[-1])
    d1 = pct(closes[-1], closes[-2]) if len(closes) >= 2 else 0.0
    w1 = pct(closes[-1], closes[-6]) if len(closes) >= 6 else 0.0
    hi52_price = float(df['High'].max()) if 'High' in df else price
    hi52_pct   = pct(price, hi52_price)
    this_year  = datetime.datetime.now().year
    ytd_df     = df[df.index.year == this_year]
    ytd = pct(price, float(ytd_df['Close'].iloc[0])) if len(ytd_df) > 0 else 0.0
    spark = []
    for i in range(max(1, len(closes)-5), len(closes)):
        spark.append(round(pct(closes[i], closes[i-1]), 2))
    while len(spark) < 5:
        spark.insert(0, 0.0)
    result = {
        'sym': TICKER_REMAP.get(sym, sym),
        'price': round(price, 4),
        'd1': d1, 'w1': w1, 'hi52': hi52_pct, 'ytd': ytd, 'spark': spark,
    }
    crypto_ids   = {'BTC-USD':'bitcoin','ETH-USD':'ethereum','SOL-USD':'solana','XRP-USD':'ripple'}
    crypto_names = {'BTC-USD':'Bitcoin','ETH-USD':'Ethereum','SOL-USD':'Solana','XRP-USD':'Ripple'}
    if sym in crypto_ids:
        result['id']   = crypto_ids[sym]
        result['name'] = crypto_names[sym]
    return result

# ── MAIN FETCH ───────────────────────────────────────────────────────────────────────────────────────────────────
def fetch_all():
    output = {
        'generated_at': datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
        'futures': [], 'dxvix': [], 'metals': [], 'commod': [],
        'yields': [], 'global': [], 'etfmain': [], 'submarket': [],
        'sector': [], 'sectorew': [], 'thematic': [], 'country': [],
        'crypto': [], 'holdings': {},
    }

    batches = [
        ('futures', FUTURES), ('etfmain', ETF_MAIN), ('submarket', SUBMARKET),
        ('sector', SECTOR), ('sectorew', SECTOR_EW), ('thematic', THEMATIC),
        ('country', COUNTRY), ('metals', METALS), ('commod', ENERGY),
        ('global', GLOBAL_IDX), ('yields', YIELDS), ('dxvix', DX_VIX),
        ('crypto', CRYPTO_YF),
    ]

    for key, tickers in batches:
        print(f"Fetching {key} ({len(tickers)} tickers)...")
        raw = fetch_batch(tickers)
        for yf_sym in tickers:
            rec = raw.get(yf_sym)
            if rec:
                if key == 'yields':
                    yield_map = {'^TNX': 'US10Y', '^TYX': 'US30Y'}
                    rec['sym'] = yield_map.get(yf_sym, rec['sym'])
                output[key].append(rec)
            else:
                print(f"  \u26a0 No data for {yf_sym}")
        time.sleep(1)

    # 2-Year Treasury yield (prepend so order is 2Y, 10Y, 30Y)
    print("Fetching 2Y Treasury yield...")
    rec_2y = fetch_treasury_2y()
    if rec_2y:
        output['yields'].insert(0, rec_2y)

    # Sort ranked tables by 1W desc
    for key in ('country', 'sector', 'sectorew', 'thematic', 'submarket'):
        output[key].sort(key=lambda x: x.get('w1', 0), reverse=True)

    # Fetch ETF holdings for all relevant categories
    holdings_tickers = list(dict.fromkeys(
        ETF_MAIN + SUBMARKET + SECTOR + SECTOR_EW + THEMATIC + COUNTRY
    ))
    print(f"\nFetching ETF holdings ({len(holdings_tickers)} ETFs)...")
    output['holdings'] = fetch_etf_holdings(holdings_tickers)
    print(f"\u2713 Holdings fetched for {len(output['holdings'])} ETFs")

    return output

if __name__ == '__main__':
    print("=== Market Dashboard Data Fetch ===")
    print(f"Time: {datetime.datetime.utcnow()} UTC\n")
    data = fetch_all()
    out_path = Path('data/data.json')
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(data, f, indent=2)
    total = sum(len(v) for v in data.values() if isinstance(v, list))
    print(f"\n\u2713 Wrote {total} records to {out_path}")
    print(f"  Yields: {[x['sym'] for x in data['yields']]}")
    print(f"  Thematic top 3: {[x['sym'] for x in data['thematic'][:3]]}")
    print(f"  Holdings for: {list(data['holdings'].keys())[:5]}...")
