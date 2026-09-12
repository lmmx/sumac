# On-disk format and threat model

## Threat model

Anyone holding the repo but not the passphrase must learn nothing about the home's layout or
its contents: no location names, no product names, no quantities. Accepted leakage: record
count, approximate record size, commit timestamps, branch names (each names a writer), commit
messages of the form `sumac: N records`, and OS usernames — leaked via git commit authorship,
not via any path.

Households share one passphrase. Ownership of a `writer/<id>` branch is a convention, trusted
rather than checked — no command compares the checked-out branch against anything (docs/journal
2026-09-11-branch-per-user-design.md §5) — not enforced by file permissions or ref protection at
this layer; the design makes violations *detectable* (`sumac verify`), not impossible.

## Layout

```
data/
  vault.json          # plaintext: format version, Argon2id params, salt, verifier — shared,
                      # from the repo's root commit, an ancestor of every writer branch
  config.jsonl.enc    # encrypted JSONL, append-only: this writer's location/product definitions
  log.jsonl.enc       # encrypted JSONL, append-only: this writer's changes and snapshots
```

Each `writer/<id>` branch carries its own `config.jsonl.enc` and `log.jsonl.enc`; only
`vault.json` is shared, via common ancestry rather than a shared file on any one branch. Every
path component is a fixed literal — never derived from a location, product, or writer name.

## Sealed line

The line-sealing primitive — AEAD, nonces, base64 framing, AAD, Argon2id key derivation, and
the wrong-passphrase verifier — is not sumac's own code. It's provided by
[`sealedlog`](https://pypi.org/project/sealedlog/), a standalone library extracted from an
earlier version of this app; see its `docs/FORMAT.md` for the full spec. Summary of what that
buys sumac:

Each line is `base64(nonce‖ciphertext‖tag)`, sealed with XChaCha20-Poly1305 and a fresh random
24-byte nonce. Appending is a byte-append, so git packs the history well. This costs per-line
ciphertext overhead and leaks record count and approximate size — accepted per the threat model.

## AAD binding

Every sealed line is bound to the stream it belongs to via associated data built from sumac's
namespace (`sumac.NAMESPACE`, `"sumac"`) and the `stream_id` (`"config:<id>"` or `"log:<id>"`,
where `<id>` is the same slug that names the `writer/<id>` branch) — see `sealedlog`'s AAD scheme
for the exact byte layout. A line copied out of one stream into another fails to authenticate.
This is what makes the ownership convention auditable: it can't stop a writer from truncating
their own file, but it prevents laundering a record into someone else's history.

## Key derivation

The key is derived from the shared passphrase via Argon2id (`sealedlog.Vault`), with a random
salt and the KDF params stored in `vault.json` alongside sumac's own `format_version`. A
`verifier` — a known plaintext sealed at vault-creation time — lets `sealedlog.Vault.unlock`
reject a wrong passphrase immediately with `WrongPassphraseError`, instead of producing garbage
downstream. `sumac.vault` wraps `Vault.create`/`Vault.unlock` with sumac's namespace baked in so
call sites can't typo it.

## Versioning

Every record carries `schema_version`. A reader that encounters a record from a newer schema
than it understands raises an "upgrade sumac" error rather than guessing.
