# Code-Only Release Checklist

This checklist prepares a release; it does not authorize publication. Professional counsel remains responsible for the patent and legal assessment.

## Non-Negotiable Gates

- [ ] A patent professional has reviewed the exact proposed source snapshot, the intended disclosure scope is covered, and a filing receipt with application number and filing date has been obtained.
- [ ] `python3 tools/assets/check_model_assets.py` passes in the private development tree.
- [ ] The generated source snapshot passes `check_model_assets.py --permit-missing --public-release` and contains no blocked asset.
- [ ] Release build, all CTest tests, schema modes, fixture dry-run, README links, license scan, secret scan, absolute-path scan, and large-file scan pass in a fresh clone of the snapshot.
- [ ] Every row in the README readiness matrix has been manually checked against evidence; unknown status remains “Not documented.”
- [ ] Realtime reference transport is either loopback/local-only in the release snapshot or has a separately reviewed authenticated protocol; supervisor-command HMAC must not be counted as reference-data authentication.
- [ ] The snapshot contains no paper, experiment result/protocol, patent material, private branch history, LFS object/history, key, model, motion data, label PDF, or precompiled third-party library.

Run the public snapshot/content scan before downloading dependencies or building.
It is intentionally strict: a post-build tree contains generated binaries and
must not be mistaken for the source-only artifact selected by
`create_public_snapshot.py`.

## Snapshot and Staging

```bash
python3 tools/release/create_public_snapshot.py --output /tmp/ActionGo
python3 tools/release/create_review_bundle.py --output /tmp/ActionGo-review.zip
```

Create the private `Uranus2234/ActionGo` staging repository from the audited snapshot with one fresh initial history. Never push the current `.git`, branches, tags, Git LFS history, or experiment remote.

Run CI in staging without LFS, resolve every gate, and only then change visibility after the patent filing gate is satisfied. Create immutable `v0.1.0` from that history. Do not rewrite it when the paper is finished.

## Release Metadata

- [ ] Apache-2.0, `NOTICE`, `THIRD_PARTY.md`, `SECURITY.md`, and community templates are present.
- [ ] `CITATION.cff` names the final public repository and software release.
- [ ] Archive the GitHub release with a software repository such as Zenodo and add the resulting DOI to `CITATION.cff` and release notes. A DOI cannot be generated locally.
- [ ] Attach only rights-cleared demo media.
- [ ] Add the paper URL, paper citation, and experiment artifact in a later tag/release; keep `v0.1.0` unchanged.
