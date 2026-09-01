"""Minimal in-process agent: exposes domain operations as tool functions."""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from sumac import config, decide, ledger, models, paths, store
from sumac.errors import Rejected, SumacError


@dataclass(frozen=True, slots=True)
class SearchResult:
    """Result of searching inventory."""

    product_id: str
    product_name: str
    locations: list[LocationQuantity]

    def __str__(self) -> str:
        if not self.locations:
            return f"No {self.product_name} found"
        locs = ", ".join(f"{loc.amount} {loc.unit} at {loc.name}" for loc in self.locations)
        return f"{self.product_name}: {locs}"


@dataclass(frozen=True, slots=True)
class LocationQuantity:
    """A location and quantity."""

    id: str
    name: str
    amount: Decimal
    unit: str


@dataclass(frozen=True, slots=True)
class ToolSuccess:
    """Successful tool result with writes to apply."""

    message: str
    writes: list[decide.Write]


class AgentError(Exception):
    """Agent-specific error."""

    pass


def _near_matches(value: str, candidates: list[str], n: int = 1) -> list[str]:
    """Find near-matches for a value in candidates."""
    return difflib.get_close_matches(value, candidates, n=n, cutoff=0.6)


def search_inventory(
    query: str,
    *,
    inventory: ledger.Inventory,
    locations: dict[str, models.Location],
    products: dict[str, models.Product],
) -> SearchResult | AgentError:
    """Search for a product in inventory by name (substring match).

    Args:
        query: Product name to search for (substring)
        inventory: Current inventory
        locations: Location registry
        products: Product registry

    Returns:
        SearchResult with locations holding the product, or AgentError if ambiguous.
    """
    query_lower = query.lower()

    # Find matching products (substring match)
    matches = [
        pid
        for pid in products.keys()
        if query_lower in pid.lower() or query_lower in products[pid].name.lower()
    ]

    if not matches:
        suggestions = _near_matches(query, list(products.keys()), n=3)
        if suggestions:
            return AgentError(
                f"No product found matching {query!r}. Did you mean: {', '.join(suggestions)}?"
            )
        return AgentError(f"No product found matching {query!r}")

    if len(matches) > 1:
        products_list = ", ".join(f"{products[pid].name} ({pid})" for pid in sorted(matches))
        return AgentError(
            f"Ambiguous: {len(matches)} products match {query!r}: {products_list}. Please be more specific."
        )

    # Exactly one match
    product_id = matches[0]
    product = products[product_id]

    # Find all locations with this product
    locs_with_product: list[LocationQuantity] = []
    for location_id, holdings in inventory.by_location.items():
        if product_id in holdings:
            q = holdings[product_id]
            location = locations.get(location_id)
            if location:
                locs_with_product.append(
                    LocationQuantity(
                        id=location_id, name=location.name, amount=q.amount, unit=q.unit
                    )
                )

    # Sort by location name for consistent output
    locs_with_product.sort(key=lambda x: x.name)

    return SearchResult(
        product_id=product_id,
        product_name=product.name,
        locations=locs_with_product,
    )


def consume(
    product_query: str,
    amount: Decimal | str,
    *,
    inventory: ledger.Inventory,
    locations: dict[str, models.Location],
    products: dict[str, models.Product],
    cfg: config.Config,
) -> SearchResult | ToolSuccess | AgentError:
    """Consume a product from inventory.

    Searches for the product first. If found at exactly one location,
    consumes from there. If found at multiple locations, asks which one.

    Args:
        product_query: Product name to consume
        amount: Amount as a string (e.g. "1", "0.5")
        inventory: Current inventory
        locations: Location registry
        products: Product registry
        cfg: Config (for unit conversion)

    Returns:
        ToolResult on success, or AgentError/SearchResult for clarification.
    """
    if isinstance(amount, str):
        try:
            amount_decimal = Decimal(amount)
        except Exception:
            return AgentError(f"Invalid amount: {amount!r}. Use a decimal like '1' or '0.5'")
    else:
        amount_decimal = amount

    # Search for product
    search_result = search_inventory(
        product_query, inventory=inventory, locations=locations, products=products
    )
    if isinstance(search_result, AgentError):
        return search_result

    product_id = search_result.product_id
    product = products[product_id]

    # Check locations holding the product
    if not search_result.locations:
        return AgentError(f"No {product.name} currently in inventory to consume")

    if len(search_result.locations) > 1:
        locs = ", ".join(f"{l.name} ({l.id})" for l in search_result.locations)
        return AgentError(
            f"{product.name} found at multiple locations: {locs}. "
            f"Please specify which location or use 'move' first."
        )

    # Exactly one location
    from_location_id = search_result.locations[0].id

    # Actually perform the consume
    try:
        writes, messages = decide.decide_change(
            kind=models.ChangeKind.CONSUMPTION,
            product_id=product_id,
            amount=amount_decimal,
            unit=product.unit,
            from_location=from_location_id,
            to_location=None,
            actor=paths.current_user(),
            occurred_at=datetime.now(UTC),
            inventory=inventory,
            cfg=cfg,
        )
    except Rejected as e:
        return AgentError(f"Cannot consume: {e.detail}")

    message = f"Consumed {amount_decimal} {product.unit} {product.name}"
    if messages:
        message += " (" + "; ".join(messages) + ")"
    return ToolSuccess(message=message, writes=writes)


def move(
    product_query: str,
    to_location_query: str,
    amount: Decimal | str | None = None,
    *,
    inventory: ledger.Inventory,
    locations: dict[str, models.Location],
    products: dict[str, models.Product],
    cfg: config.Config,
) -> SearchResult | ToolSuccess | AgentError:
    """Move a product between locations.

    Searches for the product and the target location. If product is at multiple
    locations, asks which one.

    Args:
        product_query: Product name to move
        to_location_query: Target location name
        amount: Amount to move (optional; if not specified, moves all)
        inventory: Current inventory
        locations: Location registry
        products: Product registry
        cfg: Config

    Returns:
        ToolSuccess on success (with writes), or AgentError/SearchResult for clarification.
    """
    # Search for product
    search_result = search_inventory(
        product_query, inventory=inventory, locations=locations, products=products
    )
    if isinstance(search_result, AgentError):
        return search_result

    product_id = search_result.product_id
    product = products[product_id]

    # Check source locations
    if not search_result.locations:
        return AgentError(f"No {product.name} currently in inventory to move")

    if len(search_result.locations) > 1:
        locs = ", ".join(f"{l.name} ({l.id})" for l in search_result.locations)
        return AgentError(
            f"{product.name} found at multiple locations: {locs}. "
            f"Please specify which location to move from."
        )

    # Exactly one source location
    from_location_id = search_result.locations[0].id
    from_location_obj = locations[from_location_id]

    # Resolve target location
    to_location_lower = to_location_query.lower()
    to_matches = [
        lid
        for lid in cfg.active_locations
        if to_location_lower in lid.lower() or to_location_lower in locations[lid].name.lower()
    ]

    if not to_matches:
        suggestions = _near_matches(to_location_query, list(cfg.active_locations), n=3)
        if suggestions:
            return AgentError(
                f"Location {to_location_query!r} not found. Did you mean: {', '.join(locations[s].name for s in suggestions)}?"
            )
        return AgentError(f"Location {to_location_query!r} not found")

    if len(to_matches) > 1:
        locs = ", ".join(f"{locations[lid].name} ({lid})" for lid in sorted(to_matches))
        return AgentError(f"Ambiguous location: {len(to_matches)} locations match: {locs}")

    to_location_id = to_matches[0]

    # Sanity check: same location
    if from_location_id == to_location_id:
        return AgentError(f"Source and destination are the same: {from_location_obj.name}")

    # Determine amount to move
    if amount is None:
        # Move all
        move_amount = search_result.locations[0].amount
    else:
        if isinstance(amount, str):
            try:
                move_amount = Decimal(amount)
            except Exception:
                return AgentError(f"Invalid amount: {amount!r}")
        else:
            move_amount = amount

    # Actually perform the move
    try:
        writes, messages = decide.decide_change(
            kind=models.ChangeKind.MOVEMENT,
            product_id=product_id,
            amount=move_amount,
            unit=product.unit,
            from_location=from_location_id,
            to_location=to_location_id,
            actor=paths.current_user(),
            occurred_at=datetime.now(UTC),
            inventory=inventory,
            cfg=cfg,
        )
    except Rejected as e:
        return AgentError(f"Cannot move: {e.detail}")

    to_location_obj = locations[to_location_id]
    message = f"Moved {move_amount} {product.unit} {product.name} from {from_location_obj.name} to {to_location_obj.name}"
    if messages:
        message += " (" + "; ".join(messages) + ")"
    return ToolSuccess(message=message, writes=writes)


def process_tool_call(
    tool_name: str,
    tool_input: dict[str, Any],
    *,
    data_dir: Path,
    key: bytes,
) -> str:
    """Process a single tool call from the model.

    Loads current state, calls the tool, applies any writes to storage,
    and returns a string result for the model.

    Args:
        tool_name: Name of the tool to call
        tool_input: Tool arguments
        data_dir: Data directory
        key: Encryption key

    Returns:
        String result to return to the model.
    """
    try:
        # Load state
        inventory = ledger.build_inventory(data_dir, key)
        cfg = config.build_config(data_dir, key)
        locations = cfg.known_locations
        products = cfg.known_products

        result: SearchResult | ToolSuccess | AgentError

        if tool_name == "search_inventory":
            result = search_inventory(
                tool_input.get("query", ""),
                inventory=inventory,
                locations=locations,
                products=products,
            )
        elif tool_name == "consume":
            result = consume(
                tool_input.get("product_query", ""),
                tool_input.get("amount", "1"),
                inventory=inventory,
                locations=locations,
                products=products,
                cfg=cfg,
            )
        elif tool_name == "move":
            result = move(
                tool_input.get("product_query", ""),
                tool_input.get("to_location_query", ""),
                tool_input.get("amount"),
                inventory=inventory,
                locations=locations,
                products=products,
                cfg=cfg,
            )
        else:
            return f"Unknown tool: {tool_name}"

        # Handle AgentError
        if isinstance(result, AgentError):
            return f"Error: {str(result)}"

        # Handle SearchResult
        if isinstance(result, SearchResult):
            return str(result)

        # Handle ToolSuccess - apply writes to storage
        if isinstance(result, ToolSuccess):
            for write in result.writes:
                store.append(data_dir, key, write.stream, write.obj)
            return result.message

        return str(result)

    except SumacError as e:
        return f"Sumac error: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"
