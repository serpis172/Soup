# Security Policy

> **Fork notice:** this is a fork of [MakazhanAlpamys/Soup](https://github.com/MakazhanAlpamys/Soup).
> This policy covers vulnerabilities in **this fork's own additions** (the Web UI's FastAPI
> backend and auth, the RAM-prefetch streaming, the compression pipeline, the multi-dataset
> loading changes — see [README-FORK.md](README-FORK.md)). For vulnerabilities in code shared
> with upstream (the core training/CLI logic this fork didn't change), please report to
> upstream directly via their own security process, linked from
> [their repository](https://github.com/MakazhanAlpamys/Soup).

## Supported Versions

This fork tracks upstream's `main` branch; there isn't yet a separate versioned release
line for it. Run the latest commit from this fork's repository.

## Reporting a Vulnerability

Please report security issues **privately** — do not open a public GitHub issue
for anything security-sensitive.

- Preferred: open a private report via
  [GitHub Security Advisories on this fork](https://github.com/serpis172/Soup/security/advisories/new).
- For issues in code shared with upstream (not specific to this fork's additions), also
  consider reporting via
  [upstream's GitHub Security Advisories](https://github.com/MakazhanAlpamys/Soup/security/advisories/new)
  so a fix reaches every fork, not just this one.

We aim to acknowledge reports within 5 business days and to ship a fix or
mitigation for confirmed, in-scope issues as promptly as is practical. When
reporting, please include:

- the affected version(s)/commit and platform,
- a minimal reproduction or proof of concept,
- the impact you observed.

## Scope

Soup is a local-first CLI for fine-tuning LLMs. The threat model assumes the
operator runs Soup on their own machine with their own data. Representative
in-scope issues:

- path traversal or arbitrary file read/write from user-supplied config,
  dataset, or artifact paths;
- SSRF in the synthetic-data providers, inference server, or hub/endpoint
  validators;
- command, Modelfile, Jinja chat-template, or systemd/launchd unit injection;
- secret leakage in logs, crash bundles, or generated artifacts;
- sandbox escape in the RLVR code-execution reward path;
- **fork-specific — Web UI (`soup ui`) auth**: the UI issues a bearer token embedded in
  the URL it prints (`http://127.0.0.1:<port>/?token=...`); every `/api/*` route (except
  static file serving) requires it via constant-time comparison. In-scope: any way to reach
  a `/api/*` route without the correct token, any way the token could leak (logs, referrer
  headers to a third-party origin, etc.), or any way `--public` (LAN exposure) weakens this
  beyond documented behavior. The banner that appears when no token is present is a UX aid,
  not a security control — it does not grant access, it only prompts for the token.

Out of scope:

- vulnerabilities in third-party model weights or datasets you choose to load;
- issues that require an already-compromised host;
- DNS and email configuration of `trysoup.dev` (upstream's domain, not this fork's) — a
  missing or permissive DMARC / SPF / DKIM record, and anything else established by a public
  DNS query. Not a finding in this fork's code.

**There is no bug bounty and no monetary reward.** We credit reporters by name
in the release notes, which is the whole of what we offer. Reports that open
with a request for payment get this paragraph as the reply.

## Disclosure

We practice coordinated disclosure. Once a fix is released we credit the
reporter in the release notes, unless anonymity is requested.


> A detailed, per-version log of historical security hardening previously lived
> in this file. It now lives in the project's git history and in the
> [GitHub Releases](https://github.com/MakazhanAlpamys/Soup/releases) notes.
