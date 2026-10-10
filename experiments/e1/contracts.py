"""Canonical frozen documents and strict numeric/time contracts."""
import hashlib
import json
import math
from pathlib import Path

import pandas as pd

STEP = pd.Timedelta(minutes=15)


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def content_hash(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True,
                                   ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def save_snapshot(path, kind, payload):
    write_json(path, {'kind': kind, 'payload': payload, 'sha256': content_hash(payload)})


def read_snapshot(path, kind):
    document = json.loads(Path(path).read_text(encoding='utf-8'))
    if document.get('kind') != kind:
        raise ValueError(f'Expected frozen document kind {kind}')
    if content_hash(document['payload']) != document.get('sha256'):
        raise ValueError('Frozen input hash mismatch')
    return document['payload']


def finite(value, name, positive=False):
    if isinstance(value, bool):
        raise ValueError(f'{name} must be numeric, not boolean')
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be numeric') from exc
    if not math.isfinite(result) or (result <= 0 if positive else result < 0):
        raise ValueError(f'{name} must be finite and {"positive" if positive else "nonnegative"}')
    return result


def utc_time(value, aligned=False):
    result = pd.Timestamp(value)
    if pd.isna(result) or result.tzinfo is None:
        raise ValueError('Timestamp must be explicitly UTC-aware')
    result = result.tz_convert('UTC')
    if aligned and result != result.floor('15min'):
        raise ValueError('Timestamp must lie on the 15-minute grid')
    return result


def window(start, end):
    start, end = utc_time(start, True), utc_time(end, True)
    if end <= start:
        raise ValueError('end must be after start')
    return start, end


def interval_times(start, end):
    start, end = window(start, end)
    return tuple(pd.date_range(start, end, freq=STEP, inclusive='left'))
