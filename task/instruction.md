A decoder for the XAF audio container has been lost. Rebuild it.

`/app/data/SPEC.md` documents the 20-byte file header and nothing else. The block
encoding is undocumented and you have to recover it from evidence.

`/app/data/corpus/` holds ten pairs, `c01.xaf` through `c10.xaf`, each next to the
`.wav` file the original decoder produced from it. Those pairs determine the encoding
uniquely. `/app/data/holdout/` holds five more files, `h01.xaf` through `h05.xaf`, with
no `.wav` beside them.

Decode all five holdout files and write the results to `/app/out/h01.wav`,
`/app/out/h02.wav`, `/app/out/h03.wav`, `/app/out/h04.wav` and `/app/out/h05.wav`.

Each output must be a RIFF WAVE file containing 16-bit signed PCM, little-endian, with
channels interleaved. Its channel count and sample rate must match the `channels` and
`sample_rate` fields of the corresponding `.xaf` header, and it must contain exactly
`total_frames` frames as declared in that same header.

Your output is compared sample for sample against the reference decode. Every sample of
every file must match exactly, so an approximation of the codec is not enough.

Do not modify anything in `/app/data/`.

You have 3600 seconds to complete this task. Do not cheat by using online solutions or hints specific to this task. 
