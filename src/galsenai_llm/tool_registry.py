from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

INVENTORY: list[dict[str, Any]] = [
    {"item_code": "ITEM-001", "name": "Apple", "cost": 1.13, "quantity": 4},
    {"item_code": "ITEM-002", "name": "Bottled Water", "cost": 1.04, "quantity": 20},
    {"item_code": "ITEM-003", "name": "Instant Ramen", "cost": 10.13, "quantity": 4},
]
FIXED_USD_TO_EUR = 0.92


def get_vector_sum(a: list[float], b: list[float]) -> list[float]:
    """Get the element-wise sum of two numeric vectors.

    Args:
        a: The first vector.
        b: The second vector.
    """
    if len(a) != len(b):
        raise ValueError("Vectors must be the same length.")
    return [left + right for left, right in zip(a, b, strict=True)]


def get_dot_product(a: list[float], b: list[float]) -> float:
    """Get the dot product of two numeric vectors.

    Args:
        a: The first vector.
        b: The second vector.
    """
    if len(a) != len(b):
        raise ValueError("Vectors must be the same length.")
    return float(sum(left * right for left, right in zip(a, b, strict=True)))


def get_all_items() -> list[dict[str, Any]]:
    """Return the full inventory catalog.

    Args:
        None: This function does not require arguments.
    """
    return INVENTORY


def fetch_item_by_name(item_name: str) -> dict[str, Any] | None:
    """Fetch a single inventory item by human-readable name.

    Args:
        item_name: The name of the requested item.
    """
    return next((item for item in INVENTORY if item["name"] == item_name), None)


def get_usd_to_euro_conversion_rate() -> float:
    """Return the deterministic USD to EUR conversion rate used in the sample benchmark.

    Args:
        None: This function does not require arguments.
    """
    return FIXED_USD_TO_EUR


def inventory_total(item_codes: list[str] | None, conversion_rate: float) -> float:
    """Compute the inventory total for the selected item codes using a conversion rate.

    Args:
        item_codes: The item codes to include. Use null or an empty list for all items.
        conversion_rate: The USD to EUR conversion rate to apply.
    """
    if not item_codes:
        selected = INVENTORY
    else:
        code_set = set(item_codes)
        selected = [item for item in INVENTORY if item["item_code"] in code_set]

    total = sum(item["cost"] * item["quantity"] for item in selected)
    return round(total * conversion_rate, 2)


TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "get_vector_sum": {
        "type": "function",
        "function": {
            "name": "get_vector_sum",
            "description": "Get the element-wise sum of two vectors.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "array", "items": {"type": "number"}},
                    "b": {"type": "array", "items": {"type": "number"}},
                },
                "required": ["a", "b"],
            },
        },
    },
    "get_dot_product": {
        "type": "function",
        "function": {
            "name": "get_dot_product",
            "description": "Get the dot product of two vectors.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "array", "items": {"type": "number"}},
                    "b": {"type": "array", "items": {"type": "number"}},
                },
                "required": ["a", "b"],
            },
        },
    },
    "get_all_items": {
        "type": "function",
        "function": {
            "name": "get_all_items",
            "description": "Return the full inventory catalog.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    "fetch_item_by_name": {
        "type": "function",
        "function": {
            "name": "fetch_item_by_name",
            "description": "Fetch a single inventory item by name.",
            "parameters": {
                "type": "object",
                "properties": {"item_name": {"type": "string"}},
                "required": ["item_name"],
            },
        },
    },
    "get_usd_to_euro_conversion_rate": {
        "type": "function",
        "function": {
            "name": "get_usd_to_euro_conversion_rate",
            "description": "Return the deterministic sample benchmark USD to EUR conversion rate.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    "inventory_total": {
        "type": "function",
        "function": {
            "name": "inventory_total",
            "description": (
                "Compute the inventory total for the selected items"
                " using a conversion rate."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "item_codes": {"type": "array", "items": {"type": "string"}},
                    "conversion_rate": {"type": "number"},
                },
                "required": ["item_codes", "conversion_rate"],
            },
        },
    },
}


TOOL_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "get_vector_sum": get_vector_sum,
    "get_dot_product": get_dot_product,
    "get_all_items": get_all_items,
    "fetch_item_by_name": fetch_item_by_name,
    "get_usd_to_euro_conversion_rate": get_usd_to_euro_conversion_rate,
    "inventory_total": inventory_total,
}


@dataclass(frozen=True)
class ToolSpec:
    name: str
    schema: dict[str, Any]
    function: Callable[..., Any]


def get_tool_schemas(names: Sequence[str]) -> list[dict[str, Any]]:
    return [TOOL_SCHEMAS[name] for name in names]


def get_tool_functions(names: Sequence[str]) -> list[Callable[..., Any]]:
    return [TOOL_FUNCTIONS[name] for name in names]


def tool_specs_for_names(names: Sequence[str]) -> list[ToolSpec]:
    return [
        ToolSpec(name=name, schema=TOOL_SCHEMAS[name], function=TOOL_FUNCTIONS[name])
        for name in names
    ]
