# XAF container, partial specification

This is the only documentation that survives for the XAF audio container. It describes
the file header. It does not describe how audio is encoded inside a block, and that part
of the format has to be recovered from the paired corpus.

## File header

20 bytes, little-endian, at offset 0.

| Offset | Size | Field | Meaning |
|--------|------|-------|---------|
| 0 | 4 | magic | ASCII `XAF1` |
| 4 | 4 | total_frames | uint32, frames per channel in the decoded stream |
| 8 | 2 | channels | uint16, 1 or 2 |
| 10 | 4 | sample_rate | uint32, Hz |
| 14 | 2 | block_bytes | uint16, size of every block in bytes |
| 16 | 4 | block_count | uint32, number of blocks |

Blocks follow immediately at offset 20, contiguous, each exactly `block_bytes` long.
Every block carries the same number of frames, and `total_frames` is exactly
`block_count` times that number.

## What is not documented

The internal layout of a block. Each block starts with a per-channel header and is
followed by 4-bit residual codes, and beyond that nothing about the encoding is recorded
here: not the header fields or their order, not the predictor, not the quantiser, not the
tables either of them uses, not the packing order of the codes.

All of it is determined by the corpus. Each `.xaf` file in `corpus/` is paired with the
`.wav` file that the original decoder produced from it, and those pairs fix the encoding
uniquely.

## The one thing worth stating

The block header selects a predictor. That selector takes exactly **6** distinct values,
and every one of the 6 occurs somewhere in `corpus/`. Any decoder that handles all 6
handles the entire format.
