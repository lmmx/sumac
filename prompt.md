Make a Python app with encrypted event store to log the inventory of a home’s groceries in different locations.

Make it fully generic, reading the config for the locations from a file that is encrypted.

Encryption is with a universal passphrase that can be shared between users of the app. The goal of
encryption is for the grocery records to be encrypted at rest (the config of the location layout and the content of groceries at those locations).
The app must not leak these secrets through any files it creates (e.g. by using location names as
folder names).

The form of encryption will be per-line AEAD: each JSONL record is independently encrypted (XChaCha20-Poly1305, random nonce per line, base64'd), so appending a line is a byte-append and git packs it well. This costs you per-line ciphertext overhead and leaks record count and approximate size, but we accept that.

The users of the app will pull down the app and update the mutable parts.

The app users should expect some parts of the app to be read only and some parts to be mutable, and the app itself should be versioned.

Users cannot mutate another user's historical records; they correct or supersede them by appending new records.

Note that the read-only-ness is a convention and not achieved by file permissions. The name of the user can be obtained from the OS (`getpass.getuser()` in Python).

The datasets should be read-only for the given user, no user should be able to edit someone else’s data.

The program should be documented (but concisely, not narrated extensively). Use uv to manage the dependencies and include Claude Code hooks that tell an agent to run ruff format when editing the code.

Keep the data model of the objects stored standalone in a module so that it can be easily edited.

The product inventory should be able to carry extra metadata (such as might be provided by a grocery seller, or by users) beyond the core data model.

Opt for modules dedicated to each of the scopes of functionality, do not overburden any one module. Use Pydantic for validation at the boundaries where data is ingested, and frozen dataclasses elsewhere.

Use the rich library for the terminal app.

Use ty in a GitHub CI check on the software.

Use JSONL for the format of inventory logging, users will write that (users being the people living there).

Model inventory logging around two primitives: inventory snapshots and inventory changes.

A snapshot records the observed quantities of items at a particular location/sublocation at a specific point in time.

A change records a delta or transfer to/from inventory (e.g. purchase, consumption, waste, discovery, correction, or movement between locations).

Treat changes as the normal operational input and snapshots as explicit observations/reconciliation points.

The current inventory should be derivable from a snapshot plus subsequent changes.

Write concisely and avoid ‘slop’.

Initially, create a plan, do not go all in
