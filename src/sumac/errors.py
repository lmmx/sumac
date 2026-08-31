"""Exception hierarchy the CLI maps to exit codes."""


class SumacError(Exception):
    """Base for all sumac errors."""


class ForeignStreamError(SumacError):
    """Attempted to append to a stream_id that isn't the current user's."""


class SchemaVersionError(SumacError):
    """A record's schema_version is newer than this build of sumac supports."""


class VaultExistsError(SumacError):
    """`init` was run against a data dir that already has a vault."""


class VaultNotFoundError(SumacError):
    """No vault.json in the data dir; run `sumac init` first."""


class UnknownLocationError(SumacError):
    """A command names a location id that has never been defined."""


class UnknownProductError(SumacError):
    """A command names a product id that has never been defined."""


class RetireNonemptyError(SumacError):
    """Attempted to retire a location that still holds stock."""


class Rejected(SumacError):
    """A write-time validation failure from `decide` (docs/journal §4's
    rejection catalogue). `reason` is the machine-readable code; `detail`
    carries whatever's relevant to that reason (field, value, suggestions)."""

    def __init__(self, reason: str, **detail: object) -> None:
        self.reason = reason
        self.detail = detail
        parts = ", ".join(f"{k}={v!r}" for k, v in detail.items())
        super().__init__(f"{reason} ({parts})" if parts else reason)
