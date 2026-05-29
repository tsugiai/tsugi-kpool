# Security Policy

## Reporting a Vulnerability

Please report suspected security vulnerabilities privately to Tong Liu at
<tong@tsugicinema.com>. Do not open a public GitHub issue for vulnerability
reports.

Include as much detail as you can safely share:

- The affected package version, commit, or branch.
- Your operating system, Python version, and relevant dependency versions.
- A minimal reproduction or proof of concept.
- The expected impact and any known mitigations.

The maintainers aim to acknowledge security reports within five business days.
After triage, we will coordinate on remediation timing and public disclosure.

## Supported Versions

`tsugi-kpool` is pre-alpha software. The `0.1.x` line receives best-effort
security fixes while APIs stabilize.

## Disclosure

Please give the maintainers a reasonable opportunity to investigate and release
a fix before publishing details. Security fixes may be released as a patched
package, a GitHub advisory, release notes, or a combination of those channels.

## Scope

Security-sensitive reports include issues such as unsafe execution behavior,
credential exposure, dependency supply-chain risk, denial-of-service behavior,
and network-facing behavior that could affect a training deployment. General
bugs, documentation gaps, and feature requests can be filed as public issues.
