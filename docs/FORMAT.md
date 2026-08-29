# On-disk format and threat model

## Threat model

Anyone holding the repo but not the passphrase must learn nothing about the home's layout or
its contents: no location names, no product names, no quantities. Accepted leakage: record
count, approximate record size, commit timestamps, and OS usernames.

Households share one passphrase. Ownership of a user's log is a convention
(`getpass.getuser()`), not enforced by file permissions — the design makes violations
*detectable* (`sumac verify`), not impossible.

## Layout

```
data/
  vault.json           # plaintext: format version, Argon2id params, salt, verifier
  config.jsonl.enc      # encrypted JSONL, append-only: location definitions
  log/<osuser>.jsonl     # encrypted JSONL, append-only, one per user: changes and snapshots
```

Every path component is a fixed literal or an OS username — never derived from a location or
product name.

## Sealed line

Each line is `base64(nonce‖ciphertext‖tag)`, sealed with XChaCha20-Poly1305 and a fresh random
24-byte nonce. Appending is a byte-append, so git packs the history well. This costs per-line
ciphertext overhead and leaks record count and approximate size — accepted per the threat model.

## AAD binding

Every sealed line is bound to the stream it belongs to via associated data:
`b"sumac/v1|" + stream_id`, where `stream_id` is `"config"` or `"log:<osuser>"`. A line copied
out of one stream into another fails to authenticate. This is what makes the ownership
convention auditable: it can't stop a user from truncating their own file, but it prevents
laundering a record into someone else's history.

## Key derivation

The key is derived from the shared passphrase via Argon2id, with a random salt and the KDF
params stored in `vault.json`. A `verifier` line — a known plaintext sealed under AAD
`sumac/v1|verifier` — lets `sumac` reject a wrong passphrase immediately with a clear error,
instead of producing garbage downstream.

## Versioning

Every record carries `schema_version`. A reader that encounters a record from a newer schema
than it understands raises an "upgrade sumac" error rather than guessing.

`data/**/*.jsonl` is declared `merge=union` in `.gitattributes`: correct for append-only
streams, and it keeps concurrent pushes from conflicting.
