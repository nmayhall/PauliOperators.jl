# PauliOperators.jl — brand assets

The mark encodes the package's core idea: a Pauli string stored in the
**symplectic `[ z | x ]`** form — two sparse bit-grids (purple `z`, blue `x`)
split by a divider, with grey cells marking the zeros. Colors are drawn from
the Julia palette.

## Where the assets live

The files actually consumed by the repo are in `docs/src/assets/`, where
Documenter picks them up automatically (no `make.jl` changes needed):

| File | Use |
|------|-----|
| `docs/src/assets/logo.svg` | Docs sidebar logo, light theme (transparent) |
| `docs/src/assets/logo-dark.svg` | Docs sidebar logo, dark theme (lightened palette, white divider) |
| `docs/src/assets/favicon.ico` | Docs favicon (48/32/16 px, converted from the 64px icon) |
| `docs/src/assets/readme-banner.png` | GitHub README banner + repo social preview (1280×640) |

This directory holds the remaining vector sources:

| File | Use |
|------|-----|
| `icon-dark.svg` / `icon-light.svg` / `icon-purple.svg` | Square app icon / avatar, on each background |
| `logo-horizontal.svg` / `logo-horizontal-dark.svg` | Wordmark lockup — used in the repo README via a `<picture>` element that switches on `prefers-color-scheme` |
| `readme-banner.svg` | Vector source of the README banner |

## Palette

Light backgrounds (primary):

- `z` axis — Julia purple `#9558B2`
- `x` axis — blue `#2563EB`
- sparse / zero cells — `#CBCFD7`
- ink / divider — `#1B1F2A`

Dark backgrounds (`logo-dark.svg`, banner): `z` `#A56BC4`, `x` `#4C86F5`,
zeros `#4A5162`, divider `#FFFFFF`.

Wordmark: **Poppins** (700 / 600). Code/labels: **JetBrains Mono**.

## GitHub social preview

`docs/src/assets/readme-banner.png` is exactly GitHub's 2:1 social-card size.
Upload it manually under **Settings → General → Social preview** so shared
repo links show the banner.
