# Security policy

Decibri-aec is a pure Rust library. It processes audio samples in memory and ships no binaries, no native addons, and no packages other than the crate itself on crates.io. That narrows its attack surface considerably, and the sections below set out what that means in practice and how to report anything you find.

## Supported versions

Only the latest published version receives security updates.

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |
| < 0.1   | Not applicable |

## Reporting a vulnerability

**Do not open a public issue for security vulnerabilities.**

Report privately through either channel:

- [GitHub's private vulnerability reporting](https://github.com/decibri/decibri-aec/security/advisories/new), which opens an advisory visible only to you and the maintainers, and handles coordinated disclosure end to end.
- Email [hello@decibri.com](mailto:hello@decibri.com) with "SECURITY - decibri-aec" in the subject line.

Include in your report:

- A description of the vulnerability and how it can be exploited
- The affected crate version
- Steps to reproduce, ideally with the input that triggers it
- Potential impact
- A suggested fix, if you have one
- Your preferred attribution, name and URL, or a request for anonymity

### What to expect

- Acknowledgement within 7 days
- Initial assessment within 14 days
- A coordinated disclosure timeline agreed with you
- Credit in the published advisory, unless you prefer anonymity

### What is worth reporting

For a library of this kind, the realistic issues are input handling rather than access control. Anything along these lines is worth a report:

- Input that causes a panic, an unbounded allocation, or an infinite loop
- Arithmetic overflow or non-finite values escaping into the output
- Memory growth that does not settle over a long stream
- Any behaviour that could be driven by a remote party who controls the far-end reference or the microphone signal

## Security-relevant design choices

- **No unsafe code.** The crate is built with `#![forbid(unsafe_code)]`, so the compiler rejects any `unsafe` block anywhere in the library.
- **No network access.** The library never opens a socket, resolves a name, or contacts any remote service, at any point.
- **No file access.** The library reads and writes no files. It takes sample slices and appends to a caller-owned buffer. The examples in this repository do read and write WAV files, but examples are not part of the published library.
- **No telemetry, analytics, or phone-home.** Nothing is collected and nothing is transmitted. The crate can emit `tracing` events, but only when the host opts in by enabling the off-by-default `tracing` cargo feature; a default build has no `tracing` dependency at all and the emit sites compile to nothing. Even with the feature on, the events are inert unless the host application installs a subscriber, and they go wherever that host directs them.
- **No elevated permissions and no hardware access.** The library does not open an audio device, request microphone permission, or need any privilege beyond running as the calling process.
- **A small dependency surface.** The runtime dependency tree is deliberately minimal, and the crate owns its own transform code rather than pulling in a third-party signal processing library.
- **Non-finite input is contained.** Input samples that are not finite are sanitised on entry rather than propagating through the filter state, and the adaptive filter carries a divergence guard.

## Supply chain and publishing integrity

- The crate is published only from GitHub Actions, from the `decibri/decibri-aec` repository, triggered by a release tag. It is not published from a developer machine.
- Publishing uses keyless [Trusted Publishing via OIDC](https://crates.io/docs/trusted-publishing). A short-lived, crate-specific token is issued per run by exchanging a GitHub OIDC token through `rust-lang/crates-io-auth-action`. No long-lived crates.io token is stored in the repository or in CI.
- The crate is configured to require trusted publishing, so publishing with an API token is rejected outright.
- The publish job runs in a protected GitHub environment restricted to release tags and requiring approval.
- The full workflow is open source and auditable in `.github/workflows/publish-crates.yml`.
- Published packages carry `.cargo_vcs_info.json`, recording the exact commit each version was built from.

## Dependency monitoring

- `cargo audit` runs on every pull request against the [RustSec](https://rustsec.org/) advisory database.
- Cargo and GitHub Actions dependencies are monitored by Dependabot for security advisories and version updates.
- `Cargo.lock` is committed, so CI builds from a fixed dependency resolution.

## Known limitations

- **Audio content is the caller's responsibility.** The library processes whatever samples it is given. It does not authenticate, validate provenance of, or make any trust decision about the audio passing through it.
- **Timing behaviour is not constant-time.** The engine's work varies with signal content, as any adaptive filter's does. It is not designed for use in a context where processing time must not correlate with input.

Neither is a vulnerability. Both are stated so the boundary is clear.

## CVE policy

For confirmed vulnerabilities we will request a CVE identifier where appropriate and publish a GitHub Security Advisory with the details, affected versions, and remediation steps. Advisories are visible at the [decibri-aec security advisories page](https://github.com/decibri/decibri-aec/security/advisories).

## Security best practices for users

- Keep the crate up to date and run `cargo audit` against your own dependency tree
- Install only from the official crates.io registry
- Enable two-factor authentication on your crates.io account
- Pin dependencies to specific versions in production
- Treat far-end reference audio from an untrusted source as untrusted input, as you would any other

## Reporting concerns about this policy

If you have questions about this policy itself, or suggestions for improving it, please open a regular issue. Those are not vulnerability reports and do not need private disclosure.

## Acknowledgments

Thank you to the researchers and community members who help keep Decibri users secure. If you report a valid vulnerability and would like public acknowledgment, we will credit you in the security advisory and the release notes.

## Contact

For security questions, email [hello@decibri.com](mailto:hello@decibri.com) with "SECURITY - decibri-aec" in the subject line.
