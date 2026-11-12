"""The tools the pricing agent may call. Each call is traced as its own span by the agent loop."""

from __future__ import annotations

from statistics import median

from .simulator import Scenario, expected_demand, expected_revenue, expected_rooms_sold

# name -> one-line description shown to the model
TOOL_SPECS = {
    "get_hotel_state": "no args. Stay date, rooms left, current price, allowed price floor and ceiling, any local event.",
    "get_competitor_rates": "no args. Tonight's rates at four comparable hotels nearby.",
    "forecast_demand": 'args {"price": number}. Expected booking requests, rooms sold and revenue if we charge that price.',
    "set_price": 'args {"price": number, "reason": string}. FINAL action: publish the price for this stay date.',
    "close_out": 'args {"reason": string}. FINAL action: stop selling this stay date.',
}
FINAL_TOOLS = {"set_price", "close_out"}


class ToolError(Exception):
    pass


def forecast(s: Scenario, price: float) -> dict:
    return {
        "price": price,
        "expected_requests": round(expected_demand(s, price), 1),
        "expected_rooms_sold": round(expected_rooms_sold(s, price), 1),
        "expected_revenue": round(expected_revenue(s, price)),
    }


def call_tool(s: Scenario, name: str, args: dict) -> dict:
    if name == "get_hotel_state":
        return {
            "days_to_arrival": s.days_to_arrival,
            "capacity": s.capacity,
            "rooms_left": s.rooms_left,
            "current_price": s.current_price,
            "price_floor": s.floor,
            "price_ceiling": s.ceiling,
            "local_event": s.event,
        }
    if name == "get_competitor_rates":
        return {"rates": list(s.competitor_rates), "median": median(s.competitor_rates)}
    if name == "forecast_demand":
        prices = args.get("prices", args.get("price"))
        if not isinstance(prices, list):
            prices = [prices]
        try:
            prices = [float(p) for p in prices]
        except (TypeError, ValueError) as e:
            raise ToolError('forecast_demand needs {"prices": [number, ...]}') from e
        if not 1 <= len(prices) <= 6 or min(prices) <= 0:
            raise ToolError("give 1 to 6 positive prices")
        rows = [forecast(s, p) for p in prices]
        return {"forecasts": rows, "best_of_these": max(rows, key=lambda r: r["expected_revenue"])["price"]}
    raise ToolError(f"unknown tool {name!r}; use one of {', '.join(TOOL_SPECS)}")
