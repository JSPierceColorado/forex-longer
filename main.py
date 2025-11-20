import os
import json
import time
import logging
from typing import List, Dict, Any, Tuple

import requests
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


# ------------------------
# Google Sheets helpers
# ------------------------

def get_gspread_client() -> gspread.Client:
    """Authenticate to Google Sheets using a service account JSON in env GOOGLE_CREDS_JSON."""
    creds_json = os.environ["GOOGLE_CREDS_JSON"]
    info = json.loads(creds_json)
    credentials = Credentials.from_service_account_info(info, scopes=SCOPES)
    client = gspread.authorize(credentials)
    return client


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


# ------------------------
# Oanda helpers
# ------------------------

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


# ------------------------
# Pip helpers
# ------------------------

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


# ------------------------
# Core buyer logic
# ------------------------

def run_buyer_once():
    """
    - Read all pairs from the UT screener sheet.
    - For each pair:
        * Compute a notional = 0.5% of available funds (marginAvailable).
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

        # 0.5% of available funds per pair => notional / price = units
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
    

# ------------------------
# Main loop
# ------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    interval_seconds = int(os.getenv("BUYER_INTERVAL_SECONDS", "30"))  # runs fast & perpetually

    logger.info(
        "Starting Oanda buyer bot loop (%.1fs interval, buys %.3f%% per pair)...",
        interval_seconds,
        float(os.getenv("BUYER_ALLOCATION_PERCENT", "0.5")),
    )

    while True:
        try:
            run_buyer_once()
        except Exception as exc:
            logger.exception("Error in buyer bot loop: %s", exc)
        logger.info("Sleeping for %s seconds...", interval_seconds)
        time.sleep(interval_seconds)
