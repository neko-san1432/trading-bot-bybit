"""
Utility functions for understanding and calculating leverage in futures trading.
"""

def calculate_leverage_metrics(notional_value: float, margin_used: float) -> dict:
    """
    Calculate leverage metrics for a futures position.
    
    Args:
        notional_value: Total value of the position (quantity × price)
        margin_used: Amount of margin used to open the position
    
    Returns:
        Dictionary with leverage metrics
    """
    leverage = notional_value / margin_used if margin_used > 0 else 0
    
    return {
        'notional_value': notional_value,
        'margin_used': margin_used,
        'leverage': leverage,
        'leverage_ratio': f"{leverage:.1f}x",
        'margin_ratio': margin_used / notional_value if notional_value > 0 else 0
    }

def calculate_position_size(available_margin: float, leverage: float, price: float) -> dict:
    """
    Calculate position size based on available margin and desired leverage.
    
    Args:
        available_margin: Available margin in account
        leverage: Desired leverage ratio (e.g., 10.0 for 10x)
        price: Entry price of the asset
    
    Returns:
        Dictionary with position sizing information
    """
    notional_value = available_margin * leverage
    quantity = notional_value / price
    margin_used = notional_value / leverage
    
    return {
        'quantity': quantity,
        'notional_value': notional_value,
        'margin_used': margin_used,
        'leverage': leverage,
        'leverage_ratio': f"{leverage:.1f}x"
    }

def calculate_pnl_impact(entry_price: float, exit_price: float, quantity: float, 
                        direction: str, leverage: float) -> dict:
    """
    Calculate PnL impact considering leverage.
    
    Args:
        entry_price: Price when position was opened
        exit_price: Price when position was closed
        quantity: Position size
        direction: 'long' or 'short'
        leverage: Leverage used
    
    Returns:
        Dictionary with PnL information
    """
    notional_value = quantity * entry_price
    margin_used = notional_value / leverage
    
    if direction == 'long':
        price_change = exit_price - entry_price
    else:
        price_change = entry_price - exit_price
    
    # PnL is calculated on the full notional value
    pnl_absolute = price_change * quantity
    pnl_percentage = pnl_absolute / margin_used if margin_used > 0 else 0
    
    return {
        'pnl_absolute': pnl_absolute,
        'pnl_percentage': pnl_percentage,
        'pnl_percentage_formatted': f"{pnl_percentage:.2%}",
        'price_change': price_change,
        'price_change_percentage': price_change / entry_price,
        'notional_value': notional_value,
        'margin_used': margin_used,
        'leverage': leverage
    }

def display_leverage_summary(trade_data: dict) -> str:
    """
    Display a formatted summary of leverage information for a trade.
    
    Args:
        trade_data: Dictionary containing trade information with leverage data
    
    Returns:
        Formatted string with leverage summary
    """
    summary = []
    summary.append(f"Position: {trade_data.get('quantity', 0):.4f} @ ${trade_data.get('entry', 0):.2f}")
    summary.append(f"Leverage: {trade_data.get('leverage_ratio', 'N/A')}")
    summary.append(f"Notional Value: ${trade_data.get('notional_value', 0):.2f}")
    summary.append(f"Margin Used: ${trade_data.get('margin_used', 0):.2f}")
    
    if 'pnl' in trade_data:
        summary.append(f"PnL: ${trade_data['pnl']:.2f}")
    
    return " | ".join(summary)

# Example usage and testing
if __name__ == "__main__":
    # Example: Calculate position size for 10x leverage
    available_margin = 1000  # $1000 available
    leverage = 10.0  # 10x leverage
    price = 50000  # BTC at $50,000
    
    position_info = calculate_position_size(available_margin, leverage, price)
    print("Position Sizing Example:")
    print(f"Available Margin: ${available_margin}")
    print(f"Leverage: {leverage}x")
    print(f"Price: ${price}")
    print(f"Position Size: {position_info['quantity']:.6f} BTC")
    print(f"Notional Value: ${position_info['notional_value']:.2f}")
    print(f"Margin Used: ${position_info['margin_used']:.2f}")
    
    # Example: Calculate PnL impact
    entry_price = 50000
    exit_price = 51000  # 2% price increase
    quantity = 0.2  # 0.2 BTC
    direction = 'long'
    
    pnl_info = calculate_pnl_impact(entry_price, exit_price, quantity, direction, leverage)
    print("\nPnL Impact Example:")
    print(f"Price moved from ${entry_price} to ${exit_price}")
    print(f"PnL: ${pnl_info['pnl_absolute']:.2f}")
    print(f"PnL %: {pnl_info['pnl_percentage_formatted']}")
    print(f"Leverage amplified return by {leverage}x")
