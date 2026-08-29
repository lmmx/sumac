"""Exception hierarchy the CLI maps to exit codes."""


class SumacError(Exception):
    """Base for all sumac errors."""


class WrongPassphraseError(SumacError):
    """Passphrase does not match the vault verifier."""


class DecryptionError(SumacError):
    """A line failed to authenticate under its expected AAD."""


class ForeignStreamError(SumacError):
    """Attempted to append to a stream_id that isn't the current user's."""


class SchemaVersionError(SumacError):
    """A record's schema_version is newer than this build of sumac supports."""


class VaultExistsError(SumacError):
    """`init` was run against a data dir that already has a vault."""


class VaultNotFoundError(SumacError):
    """No vault.json in the data dir; run `sumac init` first."""
