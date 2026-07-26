"""Regression datum chung (STEP3 mục 2, test 9).

Bước 3 cần hai recording sống trong cùng một ENU. Đường cũ (``datum=None``)
phải giữ nguyên từng bit, nếu không mọi số vòng 3–5 mất hiệu lực.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from data_tools.map_frame import first_quality4_datum
from data_tools.gps_sources import load_nmea_replay

ROOT = Path(".cache/data/4seasons")
QUERY = ROOT / "recording_2021-02-25_13-39-06"
MAPPING = ROOT / "recording_2021-05-10_19-15-19"

pytestmark = pytest.mark.skipif(
    not QUERY.exists() or not MAPPING.exists(),
    reason="Dataset 4Seasons chưa có trong .cache",
)


def positions(replay) -> np.ndarray:
    return np.array(
        [
            [measurement.x, measurement.y]
            for measurement in replay.measurements
            if measurement.has_position
        ]
    )


def test_default_datum_is_bit_identical_to_previous_behaviour():
    baseline = load_nmea_replay(QUERY)
    explicit = load_nmea_replay(
        QUERY,
        datum=(
            baseline.metadata["datum_latitude"],
            baseline.metadata["datum_longitude"],
        ),
    )

    assert np.array_equal(positions(baseline), positions(explicit))
    assert baseline.metadata["datum_source"] == "recording_first_quality4"
    assert explicit.metadata["datum_source"] == "shared"


def test_shared_datum_puts_both_recordings_in_one_frame():
    datum = first_quality4_datum(QUERY)
    own = load_nmea_replay(MAPPING)
    shared = load_nmea_replay(MAPPING, datum=datum.as_tuple())

    own_positions = positions(own)
    shared_positions = positions(shared)
    shift = shared_positions - own_positions

    # Datum khác nhau chỉ là tịnh tiến ENU: cùng hình dạng, khác gốc.
    assert np.ptp(shift, axis=0).max() < 0.05
    assert np.linalg.norm(shift.mean(axis=0)) > 1.0


def test_shared_datum_overlaps_two_garage_traversals():
    datum = first_quality4_datum(QUERY).as_tuple()
    query = positions(load_nmea_replay(QUERY, datum=datum))
    mapping = positions(load_nmea_replay(MAPPING, datum=datum))

    distances = np.linalg.norm(query[:, None, :] - mapping[None, :, :], axis=2)

    assert float(np.median(distances.min(axis=1))) < 5.0
