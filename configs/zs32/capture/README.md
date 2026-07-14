# ZS32 capture profiles

`hikvision.example.json` contains a complete acquisition schema, but its
values are still candidates until verified on the Linux camera station.
It is an authoring template, never a production CLI input. Copy approved Linux
values to `capture/acquisition/hikvision.json`, hash the exact bytes, and bind
that asset from capture gate policy schema v2. `zs32-capture` has no loose
`--camera-config` escape hatch.

Quality and registration profiles are deliberately not populated with guessed
thresholds. They must be generated from real 4K captures on Linux and kept
separately for left/right hand. Both schemas require:

- `schema_version`, `profile_id`, `product`, `hand`
- the exact `topology_id` and `topology_sha256`
- one explicit record for every topology view
- `profile_sha256`, computed over canonical compact JSON with sorted keys after
  removing only the `profile_sha256` field

Quality records require brightness mean min/max, brightness standard-deviation
minimum, dark/saturated pixel levels and maximum ratios, and Laplacian variance
minimum. Registration records require one full-frame PNG reference with SHA256
and dimensions plus explicit ECC affine limits. There is no production default
or disabled gate.

Loose profile or acquisition paths are never accepted by `zs32-capture`.
Publication parses both profiles, checks every profile-to-policy reference,
and verifies each PNG IHDR dimension before any asset can become trusted.
Before the first model release, publish topology, `capture/gates/policy.json`,
the policy-bound acquisition asset, both profile
files, and every reference through `zs32-publish-gate-policy`, then capture
with `--gate-publication`. Production capture uses `--release` instead. The
two sources are mutually exclusive and resolve to the same policy SHA256
contract later bound by the recipe and deployment release.

Production capture additionally requires `--operator-id`. Before each topology
round, a foreground TTY must enter the exact `CONFIRM <round_id>` token within
the configured timeout. This is the physical positioning/flip boundary; its
prompt, operator, and UTC timestamps are stored in the immutable capture
manifest. Cancellation, timeout, or an invalid token quarantines the partial
set as `RETAKE` and never triggers the next round.
