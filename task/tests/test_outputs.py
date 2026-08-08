import hashlib
import json
import os
import struct
import wave

import pytest

OUT_DIR = "/app/out"
DATA_DIR = "/app/data"
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
EXPECTED_DIR = os.path.join(TESTS_DIR, "expected")

NAMES = ["h01", "h02", "h03", "h04", "h05"]


def graded_path(name):
    return os.path.join(OUT_DIR, name + ".wav")


def assert_real_file(path):
    assert not os.path.islink(path), "%s must be a real file, not a symlink" % path
    real = os.path.realpath(path)
    assert real.startswith(OUT_DIR + os.sep), "%s must resolve inside %s" % (path, OUT_DIR)
    assert os.path.isfile(real), "%s must be a regular file" % path
    return real


def read_wav(path):
    with wave.open(path, "rb") as w:
        params = (w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes())
        frames = w.readframes(w.getnframes())
    return params, frames


def source_header(name):
    with open(os.path.join(DATA_DIR, "holdout", name + ".xaf"), "rb") as f:
        head = f.read(20)
    total, channels, rate, _, _ = struct.unpack_from("<IHIHI", head, 4)
    return total, channels, rate


@pytest.mark.parametrize("name", NAMES)
def test_output_file_exists(name):
    """Each holdout file named in instruction.md has a corresponding WAV in /app/out."""
    path = graded_path(name)
    assert os.path.exists(path), "missing required output %s" % path
    assert_real_file(path)


@pytest.mark.parametrize("name", NAMES)
def test_output_is_not_aliased(name):
    """Graded outputs are real files under /app/out, not symlinks redirected at the answer key."""
    path = graded_path(name)
    assert os.path.exists(path), "missing required output %s" % path
    real = assert_real_file(path)
    assert os.path.commonpath([real, TESTS_DIR]) != TESTS_DIR, "%s resolves into the verifier's own tree" % path


@pytest.mark.parametrize("name", NAMES)
def test_output_is_16_bit_pcm(name):
    """Each output is a readable RIFF WAVE file carrying 16-bit signed PCM."""
    real = assert_real_file(graded_path(name))
    params, _ = read_wav(real)
    assert params[1] == 2, "%s must be 16-bit PCM, got sample width %d bytes" % (name, params[1])


@pytest.mark.parametrize("name", NAMES)
def test_output_matches_source_header(name):
    """Channel count, sample rate and frame count match the declared fields of the source .xaf header."""
    real = assert_real_file(graded_path(name))
    params, _ = read_wav(real)
    total, channels, rate = source_header(name)
    assert params[0] == channels, "%s: expected %d channels, got %d" % (name, channels, params[0])
    assert params[2] == rate, "%s: expected sample rate %d, got %d" % (name, rate, params[2])
    assert params[3] == total, "%s: expected %d frames, got %d" % (name, total, params[3])


@pytest.mark.parametrize("name", NAMES)
def test_decoded_pcm_is_bit_exact(name):
    """Every decoded sample equals the reference decode, which only a correct codec reproduces."""
    real = assert_real_file(graded_path(name))
    got_params, got_frames = read_wav(real)
    exp_params, exp_frames = read_wav(os.path.join(EXPECTED_DIR, name + ".wav"))
    assert got_params == exp_params, "%s: WAV parameters %s do not match reference %s" % (
        name, got_params, exp_params)
    assert len(got_frames) == len(exp_frames), "%s: %d PCM bytes, reference has %d" % (
        name, len(got_frames), len(exp_frames))
    if got_frames != exp_frames:
        count = sum(1 for a, b in zip(got_frames, exp_frames) if a != b)
        first = next(i for i, (a, b) in enumerate(zip(got_frames, exp_frames)) if a != b)
        raise AssertionError(
            "%s: %d of %d PCM bytes differ, first at byte %d" % (name, count, len(exp_frames), first))


def test_outputs_are_distinct():
    """The five decodes differ from one another, so a single file copied five times cannot pass."""
    digests = {}
    for name in NAMES:
        real = assert_real_file(graded_path(name))
        _, frames = read_wav(real)
        digests[name] = hashlib.sha256(frames).hexdigest()
    assert len(set(digests.values())) == len(NAMES), "outputs are not distinct: %s" % digests


def test_supplied_inputs_unmodified():
    """The corpus and holdout inputs are byte-identical to those shipped, as instruction.md requires."""
    with open(os.path.join(TESTS_DIR, "input_hashes.json")) as f:
        pinned = json.load(f)
    for rel, digest in sorted(pinned.items()):
        path = os.path.join(DATA_DIR, rel)
        assert os.path.isfile(path), "input %s is missing" % rel
        with open(path, "rb") as f:
            actual = hashlib.sha256(f.read()).hexdigest()
        assert actual == digest, "input %s was modified" % rel
