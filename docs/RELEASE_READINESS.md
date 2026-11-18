# Release readiness assessment

Status: **ready for a local commit and subsequent public-release approval**.

Completed:

- Root-level multi-version layout with no maintained full upstream snapshot.
- Five exact SHA-256 input profiles retained.
- FLUX.2 1.3 and 1.4 patch semantics unchanged.
- Callback API v3 contract unchanged.
- Both qualified Windows wheels reproduced exactly twice.
- Compiled extensions preserved byte-for-byte.
- Archive, metadata, RECORD, and fail-closed tests passed.
- Apache-2.0 root license, NOTICE, provenance, compatibility, build,
  installation, rollback, migration, and draft release notes prepared.
- Generated wheels, binaries, caches, and build directories ignored.

Release boundaries:

- Only the exact Windows profiles have runtime qualification.
- Linux x86-64 and aarch64 claims remain archive-only.
- The upstream compiled binary is not source-reproducible by this repository.
- A newly created empty-environment dependency resolver transaction remains
  unqualified.
- No tag, release, wheel upload, push, or production migration is authorized by
  this preparation.

Recommended next repository tag: `v1.3.0`. Repository release numbering is
independent of the Nunchaku Python package version.
