# 0001 · One `cortex` monorepo as a uv workspace

- **Status:** Accepted · **Date:** 2026-11-04 · **Deciders:** Ganeshan (owner), Claude Code (proposer) · **Related:** D-004, D-010

## Context
Cortex is four projects (serve, slm, retrieval, agent) that call each other: P1/P3/P4 use P2's gateway, and P2 serves P4's model. Each must also stand alone for recruiters, and the learning guide must be easy to follow.

## Options considered
1. **Monorepo + uv workspace**: one lockfile, shared tooling and CI, cross-package refactors in one PR. Downside: a recruiter landing on the repo sees more than one project.
2. **One repo per project**: cleaner individual showcases, but versions drift and cross-repo changes are painful.
3. Monorepo for code **and** the learning guide: mixes audiences.

## Decision
Option 1, with the learning guide in a **separate** public repo (`cortex-serve-guide`). uv is mandatory, Python 3.12 pinned, and each package keeps its own README/RESULTS/docs. `git subtree split` can publish a package as its own repo later.

## Consequences
+ One `make check`, one CI, one lockfile. + Shared `CLAUDE.md` conventions.
− The package README must carry the project on its own. − CI time grows with packages (path filters later).
