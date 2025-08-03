import sqlite3

def auto_tune_confidence_threshold(
    db_path="learning_log.db",
    rolling_window=20,
    base_threshold=5.0,
    low_winrate=0.45,
    high_winrate=0.7,
    min_threshold=1.0,
    max_threshold=10.0,
    risk_rule=False,
    filter_symbol=None,
    filter_tf=None
):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    # Filter by symbol/TF
    q = "SELECT pnl FROM learning_log"
    params = []
    if filter_symbol and filter_tf:
        q += " WHERE symbol=? AND tf=?"
        params = [filter_symbol, filter_tf]
    elif filter_symbol:
        q += " WHERE symbol=?"
        params = [filter_symbol]
    elif filter_tf:
        q += " WHERE tf=?"
        params = [filter_tf]
    q += " ORDER BY timestamp DESC LIMIT ?"
    params.append(rolling_window)
    c.execute(q, params)
    rows = c.fetchall()
    conn.close()
    if not rows or len(rows) < rolling_window:
        return base_threshold

    pnls = [r[0] for r in rows]
    winrate = sum(1 for p in pnls if p > 0) / rolling_window
    avg_pnl = sum(pnls) / rolling_window
    total_pnl = sum(pnls)
    profit_curve = [sum(pnls[:i+1]) for i in range(len(pnls))]
    max_dd = min(profit_curve) if profit_curve else 0

    new_threshold = base_threshold
    if winrate < low_winrate:
        new_threshold = base_threshold + 1.0
    elif winrate > high_winrate:
        new_threshold = base_threshold - 0.5
    if total_pnl < 0:
        new_threshold += 1.0
    if avg_pnl < 0.2:
        new_threshold += 0.5
    if max_dd < -2:
        new_threshold += 1.0
    if risk_rule:
        new_threshold = base_threshold + (0.55 - winrate) * 5

    new_threshold = max(min_threshold, min(new_threshold, max_threshold))
    return round(new_threshold, 3)

def auto_tune_min_distance_pips(
    db_path="learning_log.db",
    rolling_window=20,
    base_distance=4000,
    low_winrate=0.45,
    high_winrate=0.7,
    min_distance=1000,
    max_distance=10000,
    reward_risk_rule=True,
    filter_symbol=None,
    filter_tf=None
):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    # Filter by symbol/TF
    q = "SELECT pnl FROM learning_log"
    params = []
    if filter_symbol and filter_tf:
        q += " WHERE symbol=? AND tf=?"
        params = [filter_symbol, filter_tf]
    elif filter_symbol:
        q += " WHERE symbol=?"
        params = [filter_symbol]
    elif filter_tf:
        q += " WHERE tf=?"
        params = [filter_tf]
    q += " ORDER BY timestamp DESC LIMIT ?"
    params.append(rolling_window)
    c.execute(q, params)
    rows = c.fetchall()
    conn.close()
    if not rows or len(rows) < rolling_window:
        return base_distance

    pnls = [r[0] for r in rows]
    winrate = sum(1 for p in pnls if p > 0) / rolling_window
    avg_pnl = sum(pnls) / rolling_window
    rewards = [p for p in pnls if p > 0]
    risks = [abs(p) for p in pnls if p <= 0]
    reward_risk = (sum(rewards) / len(rewards)) / (sum(risks) / len(risks)) if rewards and risks and sum(risks)!=0 else 1.0

    new_distance = base_distance
    if winrate < low_winrate:
        new_distance += 1000
    elif winrate > high_winrate:
        new_distance -= 800
    if avg_pnl < 0.2:
        new_distance += 500
    if reward_risk_rule and reward_risk < 1.0:
        new_distance += 1000
    elif reward_risk_rule and reward_risk > 2.0:
        new_distance -= 600

    new_distance = int(max(min_distance, min(new_distance, max_distance)))
    return new_distance

def is_on_grid(entry_price: float, existing_prices: list, grid_pips: float, pip_size: float = 0.01) -> bool:
    grid_distance = grid_pips * pip_size
    for price in existing_prices:
        if abs(entry_price - price) < (grid_distance / 2):  # setengah grid = buffer
            return True
    return False
