import os
import struct
import sys
import wave

HEADER_BYTES = 20
MAGIC = b"XAF1"
NUM_STEPS = 32
NUM_MAGS = 8


def ceil_div(a, b):
    return -((-a) // b)


def clamp16(v):
    if v < -32768:
        return -32768
    if v > 32767:
        return 32767
    return v


def read_xaf(path):
    data = open(path, "rb").read()
    if data[:4] != MAGIC:
        raise ValueError("bad magic in %s" % path)
    total_frames, channels, rate, block_bytes, block_count = struct.unpack_from("<IHIHI", data, 4)
    return {
        "data": data,
        "total": total_frames,
        "channels": channels,
        "rate": rate,
        "block_bytes": block_bytes,
        "blocks": block_count,
        "spc": total_frames // block_count,
    }


def read_wav(path):
    w = wave.open(path, "rb")
    ch = w.getnchannels()
    n = w.getnframes()
    raw = w.readframes(n)
    w.close()
    flat = struct.unpack("<%dh" % (n * ch), raw)
    out = [[] for _ in range(ch)]
    for i in range(n):
        for c in range(ch):
            out[c].append(flat[i * ch + c])
    return out


def block_nibbles(payload, nib_off, count):
    vals = []
    for b in payload[nib_off:]:
        vals.append(b & 0x0F)
        vals.append((b >> 4) & 0x0F)
    return vals[:count]


def signed_nib(n):
    return n - 16 if n >= 8 else n


def chan_header_size(info):
    nib_bytes = info["spc"] * info["channels"] // 2
    return (info["block_bytes"] - nib_bytes) // info["channels"]


def detect_packing(infos):
    for sel_fn, idx_fn in (
        (lambda p: p & 0x07, lambda p: (p >> 3) & 0x1F),
        (lambda p: (p >> 5) & 0x07, lambda p: p & 0x1F),
    ):
        sels = set()
        idxs = set()
        for info in infos:
            chs = chan_header_size(info)
            for b in range(info["blocks"]):
                off = HEADER_BYTES + b * info["block_bytes"]
                for c in range(info["channels"]):
                    p = info["data"][off + c * chs]
                    sels.add(sel_fn(p))
                    idxs.add(idx_fn(p))
        if len(sels) == 6 and max(idxs) < NUM_STEPS:
            return sel_fn, idx_fn
    raise RuntimeError("could not determine header packing")


def detect_state_offsets(infos, pcms, chs_map):
    cands = None
    for info, pcm in zip(infos, pcms):
        chs = chs_map[id(info)]
        spc = info["spc"]
        for b in range(1, min(info["blocks"], 12)):
            off = HEADER_BYTES + b * info["block_bytes"]
            for c in range(info["channels"]):
                base = off + c * chs
                s1 = pcm[c][b * spc - 1]
                s2 = pcm[c][b * spc - 2]
                found = set()
                for o1 in range(chs - 1):
                    for o2 in range(chs - 1):
                        if o1 == o2:
                            continue
                        v1 = struct.unpack_from("<h", info["data"], base + o1)[0]
                        v2 = struct.unpack_from("<h", info["data"], base + o2)[0]
                        if v1 == s1 and v2 == s2:
                            found.add((o1, o2))
                cands = found if cands is None else (cands & found)
    if not cands:
        raise RuntimeError("could not locate predictor state in block header")
    return sorted(cands)[0]


def gather_zero_blocks(infos, pcms, chs_map, sel_fn):
    per_sel = {}
    for info, pcm in zip(infos, pcms):
        chs = chs_map[id(info)]
        spc = info["spc"]
        nib_off = info["channels"] * chs
        for b in range(info["blocks"]):
            off = HEADER_BYTES + b * info["block_bytes"]
            payload = info["data"][off:off + info["block_bytes"]]
            if any(x != 0 for x in payload[nib_off:]):
                continue
            sel = sel_fn(payload[0])
            rows = per_sel.setdefault(sel, [])
            for c in range(info["channels"]):
                start = b * spc
                if start < 2:
                    continue
                p1 = pcm[c][start - 1]
                p2 = pcm[c][start - 2]
                for i in range(spc):
                    s = pcm[c][start + i]
                    rows.append((p1, p2, s))
                    p2 = p1
                    p1 = s
    return per_sel


def fit_coefficients(rows):
    n = len(rows)
    s11 = s12 = s22 = 0.0
    for p1, p2, _ in rows:
        s11 += p1 * p1
        s12 += p1 * p2
        s22 += p2 * p2
    for shift in range(6, 15):
        scale = 1 << shift
        rnd = 1 << (shift - 1)
        b1 = b2 = 0.0
        for p1, p2, s in rows:
            t = s * scale
            b1 += p1 * t
            b2 += p2 * t
        det = s11 * s22 - s12 * s12
        if abs(det) < 1e-6:
            guesses = [(0, 0)]
        else:
            a1 = (b1 * s22 - b2 * s12) / det
            a2 = (b2 * s11 - b1 * s12) / det
            guesses = [(int(round(a1)), int(round(a2)))]
        for base1, base2 in guesses:
            for d1 in range(-3, 4):
                for d2 in range(-3, 4):
                    c1 = base1 + d1
                    c2 = base2 + d2
                    ok = True
                    for p1, p2, s in rows:
                        if (c1 * p1 + c2 * p2 + rnd) >> shift != s:
                            ok = False
                            break
                    if ok:
                        return c1, c2, rnd, shift
    return None


def predict(coef, shift, rnd, sel, s1, s2):
    a1, a2 = coef[sel]
    return (a1 * s1 + a2 * s2 + rnd) >> shift


def constrain(rng, idx, nib, delta):
    if nib == 0:
        return
    if nib > 0:
        lo = ceil_div(8 * delta, nib)
        hi = (8 * delta + 7) // nib
    else:
        lo = ceil_div(8 * delta + 7, nib)
        hi = (8 * delta) // nib
    if lo < 1:
        lo = 1
    cur = rng[idx]
    if lo > cur[0]:
        cur[0] = lo
    if hi < cur[1]:
        cur[1] = hi


def collect_blocks(infos, pcms, chs_map, sel_fn, idx_fn, off1, off2):
    out = []
    for info, pcm in zip(infos, pcms):
        chs = chs_map[id(info)]
        spc = info["spc"]
        nib_off = info["channels"] * chs
        for b in range(info["blocks"]):
            off = HEADER_BYTES + b * info["block_bytes"]
            payload = info["data"][off:off + info["block_bytes"]]
            nibs = block_nibbles(payload, nib_off, spc * info["channels"])
            for c in range(info["channels"]):
                base = c * chs
                p = payload[base]
                out.append({
                    "sel": sel_fn(p),
                    "idx0": idx_fn(p),
                    "s1": struct.unpack_from("<h", payload, base + off1)[0],
                    "s2": struct.unpack_from("<h", payload, base + off2)[0],
                    "nibs": [signed_nib(nibs[i * info["channels"] + c]) for i in range(spc)],
                    "out": pcm[c][b * spc:(b + 1) * spc],
                })
    return out


def recover_quantiser(blocks, coef, shift, rnd):
    rng = [[1, 60000] for _ in range(NUM_STEPS)]
    for blk in blocks:
        s1, s2 = blk["s1"], blk["s2"]
        s = blk["out"][0]
        if abs(s) >= 32767:
            continue
        constrain(rng, blk["idx0"], blk["nibs"][0], s - predict(coef, shift, rnd, blk["sel"], s1, s2))

    adj = [None] * NUM_MAGS
    for m in range(NUM_MAGS):
        cands = set(range(-8, 9))
        for blk in blocks:
            if abs(blk["nibs"][0]) != m or len(blk["nibs"]) < 2:
                continue
            n1 = blk["nibs"][1]
            if n1 == 0:
                continue
            s0 = blk["out"][0]
            s1v = blk["out"][1]
            if abs(s0) >= 32767 or abs(s1v) >= 32767:
                continue
            delta1 = s1v - predict(coef, shift, rnd, blk["sel"], s0, blk["s1"])
            keep = set()
            for a in cands:
                j = blk["idx0"] + a
                if j < 0:
                    j = 0
                if j > NUM_STEPS - 1:
                    j = NUM_STEPS - 1
                lo, hi = rng[j]
                if n1 > 0:
                    l = ceil_div(8 * delta1, n1)
                    h = (8 * delta1 + 7) // n1
                else:
                    l = ceil_div(8 * delta1 + 7, n1)
                    h = (8 * delta1) // n1
                if max(l, lo) <= min(h, hi):
                    keep.add(a)
            cands = keep
            if len(cands) == 1:
                break
        if len(cands) != 1:
            raise RuntimeError("index adjust for magnitude %d not determined: %s" % (m, sorted(cands)))
        adj[m] = cands.pop()

    for _ in range(4):
        for blk in blocks:
            idx = blk["idx0"]
            s1, s2 = blk["s1"], blk["s2"]
            for i, nib in enumerate(blk["nibs"]):
                s = blk["out"][i]
                if abs(s) < 32767:
                    constrain(rng, idx, nib, s - predict(coef, shift, rnd, blk["sel"], s1, s2))
                s2 = s1
                s1 = s
                idx += adj[min(abs(nib), NUM_MAGS - 1)]
                if idx < 0:
                    idx = 0
                if idx > NUM_STEPS - 1:
                    idx = NUM_STEPS - 1
    return [r[0] for r in rng], adj


def decode(info, coef, shift, rnd, steps, adj, chs, off1, off2, sel_fn, idx_fn):
    spc = info["spc"]
    channels = info["channels"]
    nib_off = channels * chs
    pcm = [[] for _ in range(channels)]
    for b in range(info["blocks"]):
        off = HEADER_BYTES + b * info["block_bytes"]
        payload = info["data"][off:off + info["block_bytes"]]
        nibs = block_nibbles(payload, nib_off, spc * channels)
        for c in range(channels):
            base = c * chs
            p = payload[base]
            sel = sel_fn(p)
            idx = idx_fn(p)
            s1 = struct.unpack_from("<h", payload, base + off1)[0]
            s2 = struct.unpack_from("<h", payload, base + off2)[0]
            for i in range(spc):
                nib = signed_nib(nibs[i * channels + c])
                delta = (nib * steps[idx]) >> 3
                s = clamp16(predict(coef, shift, rnd, sel, s1, s2) + delta)
                pcm[c].append(s)
                s2 = s1
                s1 = s
                idx += adj[min(abs(nib), NUM_MAGS - 1)]
                if idx < 0:
                    idx = 0
                if idx > NUM_STEPS - 1:
                    idx = NUM_STEPS - 1
    return pcm


def write_wav(path, pcm, channels, rate, total):
    frames = bytearray()
    for i in range(total):
        for c in range(channels):
            frames += struct.pack("<h", pcm[c][i])
    w = wave.open(path, "wb")
    w.setnchannels(channels)
    w.setsampwidth(2)
    w.setframerate(rate)
    w.writeframes(bytes(frames))
    w.close()


def main():
    corpus_dir = "/app/data/corpus"
    holdout_dir = "/app/data/holdout"
    out_dir = "/app/out"
    os.makedirs(out_dir, exist_ok=True)

    names = sorted(f[:-4] for f in os.listdir(corpus_dir) if f.endswith(".xaf"))
    infos = [read_xaf(os.path.join(corpus_dir, n + ".xaf")) for n in names]
    pcms = [read_wav(os.path.join(corpus_dir, n + ".wav")) for n in names]
    chs_map = {id(i): chan_header_size(i) for i in infos}

    sel_fn, idx_fn = detect_packing(infos)
    off1, off2 = detect_state_offsets(infos, pcms, chs_map)

    zero_rows = gather_zero_blocks(infos, pcms, chs_map, sel_fn)
    coef = {}
    shift = rnd = None
    for sel in sorted(zero_rows):
        fit = fit_coefficients(zero_rows[sel])
        if fit is None:
            raise RuntimeError("coefficient fit failed for selector %d" % sel)
        a1, a2, r, s = fit
        coef[sel] = (a1, a2)
        shift, rnd = s, r

    blocks = collect_blocks(infos, pcms, chs_map, sel_fn, idx_fn, off1, off2)
    steps, adj = recover_quantiser(blocks, coef, shift, rnd)

    for info, pcm, name in zip(infos, pcms, names):
        got = decode(info, coef, shift, rnd, steps, adj, chs_map[id(info)], off1, off2, sel_fn, idx_fn)
        if got != pcm:
            raise RuntimeError("recovered decoder disagrees with %s" % name)

    for f in sorted(os.listdir(holdout_dir)):
        if not f.endswith(".xaf"):
            continue
        info = read_xaf(os.path.join(holdout_dir, f))
        chs = chan_header_size(info)
        pcm = decode(info, coef, shift, rnd, steps, adj, chs, off1, off2, sel_fn, idx_fn)
        write_wav(os.path.join(out_dir, f[:-4] + ".wav"), pcm, info["channels"], info["rate"], info["total"])

    return 0


if __name__ == "__main__":
    sys.exit(main())
