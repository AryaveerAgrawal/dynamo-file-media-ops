# XAF codec recovery

## Overview

The agent is given a corpus of files in XAF, an invented block-based lossy audio
container, each paired with the WAV its original decoder produced. `SPEC.md` documents the
20-byte file header and nothing else. The agent must recover the block encoding from the
paired evidence and decode five held-out files, writing `/app/out/h01.wav` through
`/app/out/h05.wav`.

## Approach

The unknowns are mutually dependent: the quantiser step table cannot be read without the
predictor coefficients, the coefficients cannot be fitted without knowing the block header
layout, and the packing order cannot be pinned without already decoding something. The
recovery path breaks that circle at the blocks where the system degenerates. Blocks whose
residual codes are all zero contribute nothing from the quantiser, so their output is pure
predictor recursion and each decoded sample gives one linear equation in two unknown
coefficients. Least squares over those blocks recovers each of the six predictor
coefficient pairs exactly. The recursion then inverts, exposing the residual behind every
code, and the step table and index-adjust rule follow by constraint propagation from the
step indices recorded in the block headers.

`solution/solve.py` performs this recovery from the shipped corpus at run time. It
hardcodes no coefficient, step value or threshold, and it verifies its recovered decoder
reproduces all ten corpus WAVs bit-exactly before decoding the held-out set.

## Environment

A single image built from the pre-approved digest-pinned `python:3.13-slim-bookworm`, with
`pytest` and `pytest-json-ctrf` baked in for the verifier. The corpus and held-out inputs
are copied to `/app/data`. The solution and tests are never copied into the image. The
decoder needs only the Python standard library.

The corpus is committed rather than generated at build time. `tools/gen_xaf_corpus.py`
reproduces it byte-identically from a fixed seed.

## Verification

The held-out inputs ship without their decoded counterparts, so the agent produces the
WAVs during its own run and the verifier reads finished artifacts only, never executing
agent code while the answer key is mounted. Each output must exist as a real file
resolving inside `/app/out` rather than a symlink aimed at the verifier's tree, carry
16-bit PCM, and match the channel count, sample rate and frame count declared in the
source `.xaf` header. Correctness is a byte-for-byte comparison of decoded PCM against
reference WAVs in `tests/expected/`, absent during the agent run.

No tolerance is used. The codec is integer arithmetic throughout, so each input has
exactly one correct output and a tolerance could only admit wrong answers. Two further
tests close off shortcuts: the five decodes must be mutually distinct, and the shipped
inputs are hash-pinned.
