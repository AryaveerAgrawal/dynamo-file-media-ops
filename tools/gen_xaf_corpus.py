import math
import os
import random
import struct
import wave

MAGIC = b"XAF1"
HEADER_BYTES = 20
CHAN_HEADER_BYTES = 6
SAMPLE_RATE = 48000
PRED_SHIFT = 8
PRED_ROUND = 128

COEF = [
    (0, 0),
    (208, 0),
    (352, -112),
    (436, -180),
    (288, -96),
    (500, -252),
]

RARE_SELECTOR = 5

STEP_TABLE = [
    8, 10, 12, 15, 19, 24, 30, 37,
    46, 58, 72, 90, 112, 140, 175, 219,
    273, 341, 427, 533, 667, 833, 1041, 1301,
    1626, 2033, 2541, 3177, 3971, 4964, 6205, 7756,
]

INDEX_ADJ = [-1, -1, -1, 0, 1, 2, 4, 6]

STEP_MAX = len(STEP_TABLE) - 1


def clamp16(v):
    if v < -32768:
        return -32768
    if v > 32767:
        return 32767
    return v


def clamp_index(v):
    if v < 0:
        return 0
    if v > STEP_MAX:
        return STEP_MAX
    return v


def predict(selector, s1, s2):
    a1, a2 = COEF[selector]
    return (a1 * s1 + a2 * s2 + PRED_ROUND) >> PRED_SHIFT


def decode_block(payload, channels, samples_per_channel):
    state = []
    for ch in range(channels):
        off = ch * CHAN_HEADER_BYTES
        packed = payload[off]
        selector = packed & 0x07
        index = (packed >> 3) & 0x1F
        s1 = struct.unpack_from("<h", payload, off + 2)[0]
        s2 = struct.unpack_from("<h", payload, off + 4)[0]
        state.append([selector, index, s1, s2])

    nib_off = channels * CHAN_HEADER_BYTES
    nibbles = []
    for byte in payload[nib_off:]:
        nibbles.append(byte & 0x0F)
        nibbles.append((byte >> 4) & 0x0F)

    out = [[] for _ in range(channels)]
    pos = 0
    for _ in range(samples_per_channel):
        for ch in range(channels):
            nib = nibbles[pos]
            pos += 1
            signed = nib - 16 if nib >= 8 else nib
            st = state[ch]
            step = STEP_TABLE[st[1]]
            delta = (signed * step) >> 3
            sample = clamp16(predict(st[0], st[2], st[3]) + delta)
            out[ch].append(sample)
            st[3] = st[2]
            st[2] = sample
            st[1] = clamp_index(st[1] + INDEX_ADJ[min(abs(signed), 7)])
    return out


def encode_block(target, channels, selector_choice, start_state, force_zero):
    samples_per_channel = len(target[0])
    state = []
    for ch in range(channels):
        sel, index, s1, s2 = start_state[ch]
        state.append([sel if selector_choice is None else selector_choice, index, s1, s2])

    nibbles = []
    decoded = [[] for _ in range(channels)]
    for i in range(samples_per_channel):
        for ch in range(channels):
            st = state[ch]
            step = STEP_TABLE[st[1]]
            pred = predict(st[0], st[2], st[3])
            if force_zero:
                best = 0
            else:
                want = target[ch][i] - pred
                best = 0
                best_err = None
                for cand in range(-8, 8):
                    delta = (cand * step) >> 3
                    err = abs(clamp16(pred + delta) - target[ch][i])
                    if best_err is None or err < best_err:
                        best_err = err
                        best = cand
            delta = (best * step) >> 3
            sample = clamp16(pred + delta)
            decoded[ch].append(sample)
            nibbles.append(best & 0x0F)
            st[3] = st[2]
            st[2] = sample
            st[1] = clamp_index(st[1] + INDEX_ADJ[min(abs(best), 7)])

    end_state = [[st[0], st[1], st[2], st[3]] for st in state]
    return nibbles, decoded, end_state


def pack_block(channels, headers, nibbles, block_bytes):
    buf = bytearray(block_bytes)
    for ch in range(channels):
        off = ch * CHAN_HEADER_BYTES
        selector, index, s1, s2 = headers[ch]
        buf[off] = (index << 3) | selector
        buf[off + 1] = 0
        struct.pack_into("<h", buf, off + 2, s1)
        struct.pack_into("<h", buf, off + 4, s2)
    nib_off = channels * CHAN_HEADER_BYTES
    for i in range(0, len(nibbles), 2):
        lo = nibbles[i]
        hi = nibbles[i + 1] if i + 1 < len(nibbles) else 0
        buf[nib_off + i // 2] = (hi << 4) | lo
    return bytes(buf)


def make_signal(rng, frames, channels, kind):
    out = [[] for _ in range(channels)]
    for ch in range(channels):
        phase = rng.uniform(0, math.tau)
        if kind == "tone":
            f = rng.uniform(180, 900)
            amp = rng.uniform(6000, 20000)
            for n in range(frames):
                out[ch].append(int(amp * math.sin(math.tau * f * n / SAMPLE_RATE + phase)))
        elif kind == "sweep":
            f0 = rng.uniform(80, 200)
            f1 = rng.uniform(2000, 6000)
            amp = rng.uniform(8000, 22000)
            for n in range(frames):
                t = n / max(frames - 1, 1)
                f = f0 * (f1 / f0) ** t
                out[ch].append(int(amp * math.sin(math.tau * f * n / SAMPLE_RATE + phase)))
        elif kind == "noise":
            amp = rng.uniform(3000, 12000)
            for n in range(frames):
                out[ch].append(int(rng.uniform(-amp, amp)))
        elif kind == "transient":
            amp = rng.uniform(15000, 30000)
            for n in range(frames):
                env = math.exp(-((n % 2400) / 400.0))
                out[ch].append(int(amp * env * math.sin(math.tau * 320 * n / SAMPLE_RATE + phase)))
        else:
            for n in range(frames):
                out[ch].append(0)
    return out


def build_file(rng, channels, block_count, block_bytes, plan):
    nib_capacity = (block_bytes - channels * CHAN_HEADER_BYTES) * 2
    samples_per_channel = nib_capacity // channels
    total_frames = block_count * samples_per_channel

    kinds = ["tone", "sweep", "noise", "transient"]
    signal = [[] for _ in range(channels)]
    remaining = total_frames
    while remaining > 0:
        seg = min(remaining, rng.randint(samples_per_channel, samples_per_channel * 3))
        chunk = make_signal(rng, seg, channels, rng.choice(kinds))
        for ch in range(channels):
            signal[ch].extend(chunk[ch])
        remaining -= seg

    blocks = []
    state = [[0, 8, 0, 0] for _ in range(channels)]
    for b in range(block_count):
        lo = b * samples_per_channel
        hi = lo + samples_per_channel
        target = [signal[ch][lo:hi] for ch in range(channels)]

        entry = plan.get(b, {})
        force_zero = entry.get("zero", False)
        selector = entry.get("selector")

        if selector is None and not force_zero:
            best_sel = None
            best_err = None
            for cand in range(len(COEF)):
                if cand == RARE_SELECTOR:
                    continue
                _, dec, _ = encode_block(target, channels, cand, state, False)
                err = sum(
                    abs(dec[ch][i] - target[ch][i])
                    for ch in range(channels)
                    for i in range(samples_per_channel)
                )
                if best_err is None or err < best_err:
                    best_err = err
                    best_sel = cand
            selector = best_sel

        headers = []
        for ch in range(channels):
            headers.append([selector, state[ch][1], state[ch][2], state[ch][3]])

        nibbles, decoded, state = encode_block(target, channels, selector, headers, force_zero)
        blocks.append(pack_block(channels, headers, nibbles, block_bytes))

    payload = b"".join(blocks)
    header = MAGIC + struct.pack("<IHIHI", total_frames, channels, SAMPLE_RATE, block_bytes, block_count)
    xaf = header + payload

    pcm = [[] for _ in range(channels)]
    for b in range(block_count):
        off = b * block_bytes
        dec = decode_block(xaf[HEADER_BYTES + off:HEADER_BYTES + off + block_bytes], channels, samples_per_channel)
        for ch in range(channels):
            pcm[ch].extend(dec[ch])

    return xaf, pcm, total_frames


def write_wav(path, pcm, channels, total_frames):
    frames = bytearray()
    for i in range(total_frames):
        for ch in range(channels):
            frames += struct.pack("<h", pcm[ch][i])
    with wave.open(path, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(bytes(frames))


def zero_tail_plan(rng, block_count, count, selector):
    plan = {}
    picks = rng.sample(range(2, block_count - 1), count)
    for b in picks:
        plan[b] = {"zero": True, "selector": selector}
    return plan


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    corpus_dir = os.path.join(root, "task", "environment", "data", "corpus")
    holdout_dir = os.path.join(root, "task", "environment", "data", "holdout")
    expected_dir = os.path.join(root, "task", "tests", "expected")
    for d in (corpus_dir, holdout_dir, expected_dir):
        os.makedirs(d, exist_ok=True)
        for f in os.listdir(d):
            os.remove(os.path.join(d, f))

    specs = [
        (1, 90, 128),
        (2, 70, 160),
        (1, 110, 128),
        (2, 64, 160),
        (1, 96, 144),
        (2, 72, 160),
        (1, 100, 128),
        (2, 60, 176),
        (1, 88, 144),
        (2, 68, 160),
    ]

    rng = random.Random(20260808)

    for i, (channels, block_count, block_bytes) in enumerate(specs, start=1):
        plan = zero_tail_plan(rng, block_count, 3, None)
        for b in plan:
            plan[b]["selector"] = rng.randint(0, 4)
        if i == 4:
            rare_blocks = rng.sample(range(2, block_count - 1), 2)
            for b in rare_blocks:
                plan[b] = {"zero": False, "selector": RARE_SELECTOR}
            zb = rng.choice([b for b in range(2, block_count - 1) if b not in rare_blocks])
            plan[zb] = {"zero": True, "selector": RARE_SELECTOR}

        xaf, pcm, total = build_file(rng, channels, block_count, block_bytes, plan)
        name = "c%02d" % i
        with open(os.path.join(corpus_dir, name + ".xaf"), "wb") as f:
            f.write(xaf)
        write_wav(os.path.join(corpus_dir, name + ".wav"), pcm, channels, total)

    hold_specs = [
        (1, 84, 128),
        (2, 66, 160),
        (1, 92, 144),
        (2, 58, 176),
        (1, 78, 128),
    ]

    for i, (channels, block_count, block_bytes) in enumerate(hold_specs, start=1):
        plan = zero_tail_plan(rng, block_count, 2, None)
        for b in plan:
            plan[b]["selector"] = rng.randint(0, 4)
        heavy = rng.sample(range(2, block_count - 1), max(6, block_count // 5))
        for b in heavy:
            plan[b] = {"zero": False, "selector": RARE_SELECTOR}

        xaf, pcm, total = build_file(rng, channels, block_count, block_bytes, plan)
        name = "h%02d" % i
        with open(os.path.join(holdout_dir, name + ".xaf"), "wb") as f:
            f.write(xaf)
        write_wav(os.path.join(expected_dir, name + ".wav"), pcm, channels, total)

    print("corpus:", len(specs), "holdout:", len(hold_specs))


if __name__ == "__main__":
    main()
