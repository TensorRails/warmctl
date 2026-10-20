"""Bounded JSON transport shared by the worker RPC and the files the host reads."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path

MAX_METADATA = 64 * 1024 * 1024


def plain(value, path='$'):
    """The JSON-plain copy of value: bool, int, finite float, str, list, dict with str keys."""
    if value is None or type(value) in (bool, int):
        return value
    if isinstance(value, str):
        return str(value)
    if type(value) is float and math.isfinite(value):
        return value
    if type(value) in (list, tuple):
        return [plain(item, '%s[%d]' % (path, index)) for index, item in enumerate(value)]
    if type(value) is dict and all(type(key) is str for key in value):
        return {key: plain(item, path + '.' + key) for key, item in value.items()}
    raise ValueError('unsupported or nonfinite value at %s: %s' % (path, type(value).__name__))


def encode(value):
    raw = (json.dumps(plain(value), sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
    if len(raw) > MAX_METADATA:
        raise ValueError('metadata exceeds bound')
    return raw


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError('duplicate JSON key: ' + key)
        result[key] = value
    return result


def _nonfinite(_value):
    raise ValueError('nonfinite JSON')


def decode(raw):
    if not raw or len(raw) > MAX_METADATA:
        raise ValueError('missing or oversized metadata')
    return plain(json.loads(raw, object_pairs_hook=_pairs, parse_constant=_nonfinite))


def absolute(value):
    canonical = isinstance(value, str) and value.startswith('/') and str(Path(value)) == value
    if not canonical or '..' in Path(value).parts:
        raise ValueError('noncanonical absolute path')
    return value


def publish(filename, value):
    """Write value as JSON at filename, create-only: a stale result cannot overwrite a fresh one."""
    absolute(filename)
    target = Path(filename)
    if not target.parent.is_dir():
        raise ValueError('parent directory missing: %s' % target.parent)
    raw = encode(value)
    fd, name = tempfile.mkstemp(prefix='.warmctl-', dir=target.parent)
    try:
        # Readable by the host like the weight parts, not mkstemp's 0600.
        os.fchmod(fd, 0o644)
        with os.fdopen(fd, 'wb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(name, target, follow_symlinks=False)
        directory = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        os.unlink(name)
    return {'path': str(target), 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}


def operation(value):
    if not isinstance(value, str) or not re.fullmatch('[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}', value):
        raise ValueError('invalid operation id')
    return value
