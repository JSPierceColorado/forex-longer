import os
import json
import time
import logging
from typing import List, Dict, Any, Optional, Tuple

import requests
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials

# ------------------------
# Config & Globals
# ------------------------

OANDA_API_URLS = {
    "practice": "https://api-fxpractice.oanda.com/v3",
    "live": "https://api-fxtrade.oanda.com/v3",
}

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

logger = logging.getLogger(__name__)


# ========================
# Google Sheets helpers
# ========================

def get_gspread_client() -> gspread.Client:
    """Authenticate to Google Sheets using a service account JSON in env GOOGLE_CREDS_JSON."""
    creds_json = os.environ["GOOGLE_CREDS_JSON"]
    info = json.loads(creds_json)
    credentials = Credentials.from_service_account_info(info, scopes=SCOPES)
    client = gspread.authorize(credentials)
    return client


def write_dataframe_to_sheet(df: pd.DataFrame, sheet_name: str, tab_name: str) -> None:
    """
    Create/replace the screener tab data in columns starting at A ONLY.
    Columns to the right of the DataFrame are left untouched so user formulas & notes persist.
    """
    gc = get_gspread_client()
    sh = gc.open(sheet_name)

    try:
        ws = sh.worksheet(tab_name)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(
            title=tab_name,
            rows=str(len(df) + 100),
            cols="30",  # give some room for user columns (R+)
        )

    # Build values for our data columns
    header = df.columns.tolist()
    data_rows = df.fillna("").astype(str).values.tolist()
    values = [header] + data_rows

    num_data_rows = len(values)
    num_data_cols = len(header)

    # Make sure we cover all existing rows in our data columns so old data gets cleared
    max_rows = max(ws.row_count, num_data_rows)
    blank_grid = [[""] * num_data_cols for _ in range(max_rows)]

    # Overlay our header + data at the top of the blank grid
    for r, row in enumerate(values):
        for c, val in enumerate(row):
            blank_grid[r][c] = val

    # Update starting at A1, only for our data columns
    ws.update(range_name="A1", values=blank_grid)
    logger.info(
        "Updated sheet '%s' tab '%s' A:? with %d data rows",
        sheet_name,
        tab_name,
        len(df),
    )


def get_screener_rows(sheet_name: str, tab_name: str) -> Tuple[gspread.Worksheet, List[Tuple[int, str]]]:
    """
    Read the screener tab and return:
      - the worksheet object
      - a list of (row_number, pair_name) for non-empty rows.

    row_number is 1-based index in Sheets.
    """
    gc = get_gspread_client()
    sh = gc.open(sheet_name)
    ws = sh.worksheet(tab_name)

    records = ws.get_all_records()  # list of dicts, header = row 1
    rows: List[Tuple[int, str]] = []

    for i, rec in enumerate(records):
        pair = rec.get("pair")
        if pair:
            # +2 because get_all_records starts at row 2, and indices are 0-based
            row_number = i + 2
            rows.append((row_number, pair))

    return ws, rows


def delete_sheet_rows(ws: gspread.Worksheet, row_numbers: List[int]) -> None:
    """
    Delete the given row numbers from the worksheet.
    Delete from bottom to top so indices don't shift.
    """
    if not row_numbers:
        return

    for rn in sorted(row_numbers, reverse=True):
        logger.info("Deleting row %s from screener sheet", rn)
        ws.delete_rows(rn)


# ========================
# Oanda helpers
# ========================

def get_oanda_session():
    token = os.environ["OANDA_API_TOKEN"]
    env = os.getenv("OANDA_ENV", "practice").lower()
    base_url = OANDA_API_URLS.get(env, OANDA_API_URLS["practice"])

    session = requests.Session()
    session.headers.update({
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    })
    return session, base_url


def fetch_instruments(session: requests.Session, base_url: str, account_id: str) -> List[Dict[str, Any]]:
    """Fetch all tradable currency instruments for the account."""
    url = f"{base_url}/accounts/{account_id}/instruments"
    resp = session.get(url)
    resp.raise_for_status()
    data = resp.json()
    instruments = data.get("instruments", [])

    fx_instruments = [
        ins for ins in instruments
        if ins.get("type") == "CURRENCY" and ins.get("tradeable", True)
    ]
    logger.info("Fetched %d FX instruments from Oanda", len(fx_instruments))
    return fx_instruments


def fetch_candles(
    session: requests.Session,
    base_url: str,
    instrument: str,
    granularity: str,
    count: int,
) -> Tuple[List[float], List[float], List[float], List[int]]:
    """Fetch up to `count` candles for an instrument at a given granularity."""
    url = f"{base_url}/instruments/{instrument}/candles"
    params = {
        "granularity": granularity,
        "count": count,
        "price": "M",
    }
    resp = session.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()
    candles = [c for c in data.get("candles", []) if c.get("complete", False)]

    closes = [float(c["mid"]["c"]) for c in candles]
    highs = [float(c["mid"]["h"]) for c in candles]
    lows = [float(c["mid"]["l"]) for c in candles]
    volumes = [int(c.get("volume", 0)) for c in candles]

    return closes, highs, lows, volumes


def fetch_account_summary(session: requests.Session, base_url: str, account_id: str) -> Dict[str, Any]:
    """Fetch account summary (balance, NAV, marginAvailable, etc.)."""
    url = f"{base_url}/accounts/{account_id}/summary"
    resp = session.get(url)
    resp.raise_for_status()
    data = resp.json()
    return data.get("account", {})


def fetch_pricing(
    session: requests.Session,
    base_url: str,
    account_id: str,
    instruments: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Fetch current bid/ask/mid prices for a list of instruments.
    Returns dict: { "EUR_USD": {"bid": ..., "ask": ..., "mid": ...}, ... }
    """
    if not instruments:
        return {}

    url = f"{base_url}/accounts/{account_id}/pricing"
    params = {"instruments": ",".join(instruments)}
    resp = session.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()

    result: Dict[str, Dict[str, float]] = {}

    for p in data.get("prices", []):
        inst = p.get("instrument")
        bids = p.get("bids", [])
        asks = p.get("asks", [])
        if not inst:
            continue

        bid = float(bids[0]["price"]) if bids else None
        ask = float(asks[0]["price"]) if asks else None

        if bid is None and ask is None:
            continue

        if bid is not None and ask is not None:
            mid = (bid + ask) / 2.0
        elif bid is not None:
            mid = bid
        else:
            mid = ask  # type: ignore

        result[inst] = {
            "bid": bid if bid is not None else mid,
            "ask": ask if ask is not None else mid,
            "mid": mid,
        }

    logger.info("Fetched pricing for %d instruments", len(result))
    return result


def place_market_order(
    session: requests.Session,
    base_url: str,
    account_id: str,
    instrument: str,
    units: int,
) -> Dict[str, Any]:
    """
    Place a MARKET order (FOK) for the given instrument and units.
    Positive units = buy; negative = sell.
    """
    url = f"{base_url}/accounts/{account_id}/orders"
    order = {
        "order": {
            "units": str(units),
            "instrument": instrument,
            "timeInForce": "FOK",
            "type": "MARKET",
            "positionFill": "DEFAULT",
        }
    }
    resp = session.post(url, data=json.dumps(order))
    resp.raise_for_status()
    data = resp.json()
    return data


# ========================
# Pip helpers
# ========================

def build_pip_location_map(
    session: requests.Session,
    base_url: str,
    account_id: str,
) -> Dict[str, int]:
    """
    Build a dict: { "EUR_USD": -4, "USD_JPY": -2, ... }
    using Oanda's pipLocation from the instruments endpoint.
    """
    fx_instruments = fetch_instruments(session, base_url, account_id)
    pip_map: Dict[str, int] = {}

    for ins in fx_instruments:
        name = ins.get("name")
        pip_loc = ins.get("pipLocation")
        if name is None or pip_loc is None:
            continue
        try:
            pip_map[name] = int(pip_loc)
        except Exception:
            # Fallback if parsing fails
            pip_map[name] = -4

    return pip_map


def round_price_to_pip(price: float, pip_location: int) -> float:
    """
    Round a price to the nearest pip using pipLocation.
    Example: pipLocation=-4 => 4 decimal places => 0.0001 pip.
    """
    decimals = max(0, -pip_location)
    return round(price, decimals)


# ========================
# Indicator helpers (UT-style logic)
# ========================

def compute_atr_series(
    highs: List[float],
    lows: List[float],
    closes: List[float],
    period: int = 10,
) -> Optional[pd.Series]:
    """
    Compute ATR series using Wilder's smoothing (similar to Pine's ta.atr).
    Returns a pandas Series aligned with input lists, or None if not enough data.
    """
    if len(closes) < period + 1:
        return None

    h = pd.Series(highs, dtype=float)
    l = pd.Series(lows, dtype=float)
    c = pd.Series(closes, dtype=float)

    prev_close = c.shift(1)

    tr = pd.concat(
        [
            (h - l),
            (h - prev_close).abs(),
            (l - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    if len(tr) < period:
        return None

    atr = pd.Series(index=tr.index, dtype=float)

    # First ATR value is simple mean of first 'period' TR values
    atr.iloc[period - 1] = tr.iloc[:period].mean()

    # Wilder's smoothing for remaining values
    for i in range(period, len(tr)):
        atr.iloc[i] = (atr.iloc[i - 1] * (period - 1) + tr.iloc[i]) / period

    return atr


def compute_ut_trailing_stop(
    closes: List[float],
    atr_series: pd.Series,
    atr_mult: float = 1.2,
) -> Optional[pd.Series]:
    """
    Reproduce the SuperTrend-style trailing stop logic from the Pine script.

    - Long regime: stop trails up, never down
    - Short regime: stop trails down, never up
    - Flips when price crosses the stop
    """
    price = pd.Series(closes, dtype=float)
    atr = atr_series

    first_valid = atr.first_valid_index()
    if first_valid is None:
        return None

    stop = pd.Series(index=price.index, dtype=float)

    # Initialize stop same way as Pine: price - entryLoss
    entry_loss0 = atr_mult * atr.iloc[first_valid]
    prev_stop = price.iloc[first_valid] - entry_loss0
    stop.iloc[first_valid] = prev_stop

    for i in range(first_valid + 1, len(price)):
        p = price.iloc[i]
        p1 = price.iloc[i - 1]
        prev_stop = stop.iloc[i - 1]
        entry_loss = atr_mult * atr.iloc[i]

        if p > prev_stop and p1 > prev_stop:
            # Long regime: trail up, never down
            stop_val = max(prev_stop, p - entry_loss)
        elif p < prev_stop and p1 < prev_stop:
            # Short regime: trail down, never up
            stop_val = min(prev_stop, p + entry_loss)
        elif p > prev_stop:
            # Flip to long
            stop_val = p - entry_loss
        else:
            # Flip to short
            stop_val = p + entry_loss

        stop.iloc[i] = stop_val

    return stop


def compute_daily_sma(
    closes: List[float],
    length: int = 240,
) -> Optional[float]:
    """Compute simple moving average of daily closes with given length."""
    if len(closes) < length:
        return None
    s = pd.Series(closes, dtype=float)
    ma = s.rolling(window=length, min_periods=length).mean().iloc[-1]
    return float(ma) if pd.notna(ma) else None


def compute_buy_signal_and_metrics(
    h1_closes: List[float],
    h1_highs: List[float],
    h1_lows: List[float],
    daily_closes: List[float],
) -> Optional[Dict[str, Any]]:
    """
    Apply the UT-style H1 logic and daily SMA 240 filter.

    Returns a dict with metrics if conditions are met, otherwise None.
    """
    if len(h1_closes) < 20:  # just a minimal sanity check
        return None

    # 1) ATR & trailing stop on H1
    atr_series = compute_atr_series(h1_highs, h1_lows, h1_closes, period=10)
    if atr_series is None:
        return None

    stop_series = compute_ut_trailing_stop(h1_closes, atr_series, atr_mult=1.2)
    if stop_series is None:
        return None

    # We need at least 2 valid stop values to detect crossover
    if stop_series.isna().sum() > len(stop_series) - 2:
        return None

    last_idx = len(h1_closes) - 1

    # Ensure last two stops are not NaN
    if pd.isna(stop_series.iloc[last_idx]) or pd.isna(stop_series.iloc[last_idx - 1]):
        return None

    price_prev = h1_closes[last_idx - 1]
    price_last = h1_closes[last_idx]
    stop_prev = stop_series.iloc[last_idx - 1]
    stop_last = stop_series.iloc[last_idx]

    # Buy condition (equivalent to ta.crossover(price, stop) && price > stop)
    crossed_up = price_prev <= stop_prev and price_last > stop_last
    buy_signal = crossed_up and (price_last > stop_last)

    if not buy_signal:
        return None

    # 2) Daily SMA 240 as "buy zone"
    daily_ma = compute_daily_sma(daily_closes, length=240)
    if daily_ma is None or daily_ma <= 0:
        return None

    # Require price be BELOW the daily SMA
    below_ma = price_last < daily_ma
    if not below_ma:
        return None

    pct_below_ma = (daily_ma - price_last) / daily_ma * 100.0

    result = {
        "last_price": price_last,
        "h1_trailing_stop": float(stop_last),
        "h1_atr10": float(atr_series.iloc[last_idx]),
        "daily_ma240": float(daily_ma),
        "pct_below_ma": float(pct_below_ma),
        "buy_signal_h1": True,
    }
    return result


# ========================
# Screener logic
# ========================

def build_instrument_row(
    session: requests.Session,
    base_url: str,
    instrument_name: str,
) -> Optional[Dict[str, Any]]:
    """
    For a single FX instrument:
      - Get H1 data for UT-style trailing stop and buy signal.
      - Get daily data for 240-SMA.
      - Return a row ONLY if buy signal is present and price < SMA240.
    """
    # Daily data for SMA 240
    daily_closes, _, _, _ = fetch_candles(
        session, base_url, instrument_name, granularity="D", count=300
    )

    # H1 data for ATR/trailing stop
    h1_closes, h1_highs, h1_lows, _ = fetch_candles(
        session, base_url, instrument_name, granularity="H1", count=400
    )

    if not daily_closes or not h1_closes:
        logger.warning("Missing data for %s", instrument_name)
        return None

    metrics = compute_buy_signal_and_metrics(
        h1_closes=h1_closes,
        h1_highs=h1_highs,
        h1_lows=h1_lows,
        daily_closes=daily_closes,
    )

    # Only keep instruments that satisfy the UT buy + below SMA240
    if metrics is None:
        return None

    row = {
        "pair": instrument_name,
        "last_price": metrics["last_price"],
        "daily_MA240": metrics["daily_ma240"],
        "%_below_MA240": metrics["pct_below_ma"],
        "H1_ATR10": metrics["h1_atr10"],
        "H1_trailing_stop": metrics["h1_trailing_stop"],
        "buy_signal_H1": metrics["buy_signal_h1"],
        "updated_at": pd.Timestamp.utcnow().isoformat(),
    }

    return row


def run_screener_once():
    """
    Scan all tradable FX instruments on Oanda and
    write ONLY the pairs that:
      - have a UT-style buy signal on H1, and
      - are trading below the 240-day SMA.
    """
    session, base_url = get_oanda_session()
    account_id = os.environ["OANDA_ACCOUNT_ID"]

    instruments = fetch_instruments(session, base_url, account_id)

    rows: List[Dict[str, Any]] = []

    for ins in instruments:
        name = ins.get("name")
        if not name:
            continue
        try:
            logger.info("Processing %s", name)
            row = build_instrument_row(session, base_url, name)
            if row:
                rows.append(row)
        except Exception as exc:
            logger.exception("Failed to build row for %s: %s", name, exc)

    if not rows:
        logger.info("No instruments met UT-buy + below-MA240 conditions this run.")
        # Still write an empty sheet with just headers so you see it's working
        df = pd.DataFrame(columns=[
            "pair",
            "last_price",
            "daily_MA240",
            "%_below_MA240",
            "H1_ATR10",
            "H1_trailing_stop",
            "buy_signal_H1",
            "updated_at",
        ])
    else:
        df = pd.DataFrame(rows)

        # Stable column ordering
        columns = [
            "pair",
            "last_price",
            "daily_MA240",
            "%_below_MA240",
            "H1_ATR10",
            "H1_trailing_stop",
            "buy_signal_H1",
            "updated_at",
        ]
        df = df[columns]

    sheet_name = os.getenv("GOOGLE_SHEET_NAME", "Active-Investing")
    screener_tab = os.getenv("OANDA_UT_SCREENER_TAB", "Oanda-UT-Screener")

    write_dataframe_to_sheet(df, sheet_name, screener_tab)


# ========================
# Buyer logic
# ========================

def run_buyer_once():
    """
    - Read all pairs from the UT screener sheet.
    - For each pair:
        * Compute a notional = BUYER_ALLOCATION_PERCENT% of available funds (marginAvailable).
        * Convert that notional to units using the ask price (rounded to pip).
        * Place a MARKET buy order.
        * If successful, delete that row from the sheet.
    """
    session, base_url = get_oanda_session()
    account_id = os.environ["OANDA_ACCOUNT_ID"]

    # Screener sheet info (reuse same envs as screener)
    sheet_name = os.getenv("GOOGLE_SHEET_NAME", "Active-Investing")
    screener_tab = os.getenv("OANDA_UT_SCREENER_TAB", "Oanda-UT-Screener")

    # Read screener rows (pair + row number)
    try:
        ws, rows = get_screener_rows(sheet_name, screener_tab)
    except gspread.WorksheetNotFound:
        logger.warning("Worksheet '%s' not found in sheet '%s'", screener_tab, sheet_name)
        return

    if not rows:
        logger.info("No pairs in screener sheet; nothing to buy this run.")
        return

    logger.info("Found %d screener rows to process", len(rows))

    # Fetch pip locations once per run
    pip_map = build_pip_location_map(session, base_url, account_id)

    # Fetch account summary (use marginAvailable as 'available funds')
    account = fetch_account_summary(session, base_url, account_id)
    margin_available_str = account.get("marginAvailable") or account.get("NAV") or account.get("balance")
    margin_available = float(margin_available_str)

    allocation_percent = float(os.getenv("BUYER_ALLOCATION_PERCENT", "0.5"))  # default 0.5%
    alloc_fraction = allocation_percent / 100.0

    if margin_available <= 0 or alloc_fraction <= 0:
        logger.warning(
            "Non-positive margin_available=%.4f or allocation fraction=%.4f; skipping.",
            margin_available,
            alloc_fraction,
        )
        return

    notional_per_trade = margin_available * alloc_fraction
    logger.info(
        "marginAvailable=%.4f, allocation=%.4f%% => notional_per_trade=%.4f",
        margin_available,
        allocation_percent,
        notional_per_trade,
    )

    # Build list of instruments in sheet and fetch pricing in one call
    instruments = [pair for _, pair in rows]
    prices = fetch_pricing(session, base_url, account_id, instruments)

    rows_to_delete: List[int] = []

    for row_number, pair in rows:
        price_info = prices.get(pair)
        if not price_info:
            logger.warning("No pricing info for %s; skipping", pair)
            continue

        ask = price_info["ask"]
        pip_loc = pip_map.get(pair, -4)  # fallback pipLocation=-4

        rounded_price = round_price_to_pip(ask, pip_loc)
        if rounded_price <= 0:
            logger.warning("Rounded ask price for %s is non-positive (%.8f); skipping", pair, rounded_price)
            continue

        # allocation% of available funds per pair => notional / price = units
        units = int(round(notional_per_trade / rounded_price))

        # ---- 1-unit fallback ----
        if units <= 0 and notional_per_trade > 0:
            logger.info(
                "Computed units < 1 for %s (notional=%.4f, price=%.8f); "
                "using minimum 1 unit instead.",
                pair,
                notional_per_trade,
                rounded_price,
            )
            units = 1

        if units <= 0:
            logger.warning(
                "Computed units <= 0 for %s (notional=%.4f, price=%.8f); skipping",
                pair,
                notional_per_trade,
                rounded_price,
            )
            continue

        try:
            logger.info(
                "Placing BUY market order: pair=%s, units=%d, approx_price=%.8f (pipLocation=%d)",
                pair,
                units,
                rounded_price,
                pip_loc,
            )
            resp = place_market_order(session, base_url, account_id, pair, units)
            logger.info("Order response for %s: %s", pair, json.dumps(resp))

            # Only delete if order succeeded without raising
            rows_to_delete.append(row_number)

        except Exception as exc:
            logger.exception("Failed to place order for %s: %s", pair, exc)

    # Delete rows for which orders were placed successfully
    if rows_to_delete:
        delete_sheet_rows(ws, rows_to_delete)
        logger.info("Finished run: placed %d buys and deleted their rows.", len(rows_to_delete))
    else:
        logger.info("Finished run: no successful buys this time.")


# ========================
# Combined main loop
# ========================

if __name__ == "__main__":
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    screener_interval = int(os.getenv("UT_SCREENER_INTERVAL_SECONDS", "3600"))
    buyer_interval = int(os.getenv("BUYER_INTERVAL_SECONDS", "30"))

    logger.info(
        "Starting combined Oanda bot: screener every %ss, buyer every %ss (buys %.3f%% per pair)...",
        screener_interval,
        buyer_interval,
        float(os.getenv("BUYER_ALLOCATION_PERCENT", "0.5")),
    )

    next_screener_time = time.time()
    next_buyer_time = time.time()

    while True:
        now = time.time()

        # Run screener when due
        if now >= next_screener_time:
            try:
                logger.info("Running UT-style screener...")
                run_screener_once()
            except Exception as exc:
                logger.exception("Error in UT screener: %s", exc)
            finally:
                next_screener_time = now + screener_interval

        # Run buyer when due
        if now >= next_buyer_time:
            try:
                logger.info("Running buyer bot...")
                run_buyer_once()
            except Exception as exc:
                logger.exception("Error in buyer bot: %s", exc)
            finally:
                next_buyer_time = now + buyer_interval

        # Sleep until the next scheduled event (with a floor)
        sleep_for = max(1.0, min(next_screener_time, next_buyer_time) - time.time())
        logger.debug("Sleeping for %.2f seconds...", sleep_for)
        time.sleep(sleep_for)
