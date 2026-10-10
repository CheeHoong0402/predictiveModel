"""Formatting utilities for Malaysian Ringgit, percentages, and action badges."""
from typing import Optional, Union


def format_myr(amount: Optional[Union[float, int]]) -> str:
    """Format monetary value as standard Malaysian Ringgit (RM 1,234.56)."""
    if amount is None:
        return "RM 0.00"
    try:
        val = float(amount)
        return f"RM {val:,.2f}"
    except (ValueError, TypeError):
        return "RM 0.00"


def format_pct(val: Optional[Union[float, int]], decimals: int = 1, show_sign: bool = False) -> str:
    """Format percentage value with standard precision (e.g. 44.5% or +5.2%)."""
    if val is None:
        return "0.0%"
    try:
        v = float(val)
        # If val is probability in [0, 1], scale to [0, 100] if requested
        if 0.0 <= v <= 1.0 and decimals == 1 and not show_sign and v < 1.0:
            v = v * 100.0
        sign_str = "+" if show_sign and v > 0 else ""
        return f"{sign_str}{v:.{decimals}f}%"
    except (ValueError, TypeError):
        return "0.0%"


def format_units(qty: Optional[Union[float, int]]) -> str:
    """Format quantity as rounded procurement integer (e.g. 144 units)."""
    if qty is None:
        return "0 units"
    try:
        val = int(round(float(qty)))
        return f"{val:,} units"
    except (ValueError, TypeError):
        return "0 units"


def format_days(days: Optional[Union[float, int]]) -> str:
    """Format stock coverage in days (e.g. 14.5 days)."""
    if days is None:
        return "0.0 days"
    try:
        val = float(days)
        return f"{val:.1f} days"
    except (ValueError, TypeError):
        return "0.0 days"


def get_action_badge(action: str) -> str:
    """Return HTML badge for recommendation action with semantic styling."""
    act = str(action).upper()
    color_map = {
        "URGENT_REORDER": ("#DC2626", "#FEE2E2"),       # Red
        "BUY_NOW": ("#2563EB", "#DBEAFE"),              # Blue
        "BUY_SMALLER_QUANTITY": ("#D97706", "#FEF3C7"), # Amber
        "WAIT_FOR_LOWER_PRICE": ("#059669", "#D1FAE5"), # Emerald
        "HOLD_CURRENT_STOCK": ("#4B5563", "#F3F4F6"),   # Gray
        "DO_NOT_REORDER": ("#6B7280", "#F3F4F6"),       # Gray
        "OVERSTOCK_WARNING": ("#EA580C", "#FFEDD5"),    # Orange
        "MONITOR_CLOSELY": ("#7C3AED", "#EDE9FE")       # Purple
    }
    text_color, bg_color = color_map.get(act, ("#1F2937", "#E5E7EB"))
    return (
        f'<span style="background-color: {bg_color}; color: {text_color}; '
        f'padding: 3px 8px; border-radius: 4px; font-weight: 600; font-size: 0.85rem; '
        f'border: 1px solid {text_color}33;">{act}</span>'
    )


def get_urgency_badge(urgency: str) -> str:
    """Return HTML badge for urgency level."""
    urg = str(urgency).upper()
    color_map = {
        "CRITICAL": ("#991B1B", "#FEE2E2"),
        "HIGH": ("#DC2626", "#FEE2E2"),
        "MEDIUM": ("#D97706", "#FEF3C7"),
        "LOW": ("#059669", "#D1FAE5"),
        "NONE": ("#6B7280", "#F3F4F6")
    }
    text_color, bg_color = color_map.get(urg, ("#1F2937", "#E5E7EB"))
    return (
        f'<span style="background-color: {bg_color}; color: {text_color}; '
        f'padding: 2px 7px; border-radius: 4px; font-weight: 600; font-size: 0.8rem;">{urg}</span>'
    )


def get_price_direction_badge(direction: str, change_pct: Optional[float] = None) -> str:
    """Return formatted direction badge with arrow and percentage or quarantine notice."""
    d = str(direction).upper()
    if d in ["QUARANTINED", "NOT_VALIDATED", "DISABLED", "NONE"]:
        return '<span style="color: #6B7280; font-weight: 600; font-size: 0.88rem; background: #F3F4F6; padding: 2px 6px; border-radius: 4px;">⚠️ NOT VALIDATED (DISABLED)</span>'
    
    pct_val = change_pct if change_pct is not None else 0.0
    if d == "UP":
        arrow = "▲"
        color = "#DC2626"
    elif d == "DOWN":
        arrow = "▼"
        color = "#059669"
    else:
        arrow = "▶"
        color = "#4B5563"
    return f'<span style="color: {color}; font-weight: bold;">{arrow} {pct_val:+.1f}% ({d})</span>'


def get_price_signal_badge(signal: str) -> str:
    """Return formatted HTML pill for price signal strength."""
    sig = str(signal).upper()
    style_map = {
        "STRONG_BUY_EARLY_SIGNAL": ("#065F46", "#D1FAE5", "🟢 STRONG BUY-EARLY (+5% Rise)"),
        "MODERATE_BUY_EARLY_SIGNAL": ("#1E40AF", "#DBEAFE", "🔵 MODERATE BUY-EARLY (+2% Rise)"),
        "NEUTRAL_PRICE_SIGNAL": ("#374151", "#F3F4F6", "⚪ NEUTRAL PRICE (Stable <2%)"),
        "MODERATE_WAIT_SIGNAL": ("#92400E", "#FEF3C7", "🟡 MODERATE WAIT (-2% Drop)"),
        "STRONG_WAIT_SIGNAL": ("#991B1B", "#FEE2E2", "🔴 STRONG WAIT (-5% Drop)"),
    }
    text_color, bg_color, label = style_map.get(sig, ("#1F2937", "#E5E7EB", sig))
    return (
        f'<span style="background-color: {bg_color}; color: {text_color}; '
        f'padding: 3px 8px; border-radius: 4px; font-weight: 600; font-size: 0.82rem; '
        f'border: 1px solid {text_color}33;">{label}</span>'
    )
