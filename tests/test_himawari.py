"""Tests: Himawari geometry and cloud features."""

import numpy as np

from hazards.himawari import NAMES, disc, segments_for, summarise, to_pixel


def nav():
    return {"sub_lon": 140.7, "cfac": 20466275, "lfac": 20466275, "coff": 2750.5, "loff": 2750.5}


def test_subsatellite_point_is_image_centre():
    c, l = to_pixel(np.array([0.0]), np.array([140.7]), nav())
    assert abs(c[0] - 2750.5) < 1e-6 and abs(l[0] - 2750.5) < 1e-6


def test_north_is_up_and_east_is_right():
    c, l = to_pixel(np.array([20.0]), np.array([150.0]), nav())
    assert l[0] < 2750.5 and c[0] > 2750.5


def test_segments_cover_position():
    assert (
        segments_for(20.0, 130.0) == [3] or segments_for(20.0, 130.0) == [2, 3] or segments_for(20.0, 130.0) == [3, 4]
    )
    assert all(1 <= s <= 10 for s in segments_for(-5.0, 120.0))


def test_disc_stays_inside_radius():
    _, _, r, _ = disc(20.0, 130.0)
    assert r.max() <= 200


def test_symmetric_storm_has_low_asymmetry_and_lopsided_has_high():
    _, _, r, b = disc(20.0, 130.0)
    even = 200 + r * 0.2
    lop = even + 40 * (np.cos(b) > 0)
    a, c = summarise(even, b), summarise(lop, b)
    assert a["tb_asym"] < 1 < c["tb_asym"]
    assert a["tb_min"] == even.min()
    assert set(a) == set(NAMES)


def test_too_much_missing_data_gives_no_features():
    _, _, r, b = disc(20.0, 130.0)
    tb = np.full(r.shape, np.nan)
    tb[:10] = 250.0
    assert np.isnan(summarise(tb, b)["cold_frac"])


def test_crop_is_north_up_and_centred():
    import hazards.himawari as hw

    part = {
        "sub_lon": 140.7,
        "cfac": 20466275,
        "lfac": 20466275,
        "coff": 2750.5,
        "loff": 2750.5,
        "first_line": 1,
        "lines": 5500,
        "cols": 5500,
        "img": np.tile(np.arange(5500, dtype=np.uint16)[:, None], (1, 5500)),
        "err": 65535,
        "out": 65534,
        "gain": 1.0,
        "const": 0.0,
        "wave": 10.4,
        "coef": (0.0, 1.0, 0.0),
        "c": 299792458.0,
        "h": 6.62606957e-34,
        "k": 1.3806488e-23,
    }
    orig = hw.brightness
    hw.brightness = lambda counts, s: counts.astype(float) / 5500 * 140 + 180
    try:
        c = hw.crop(20.0, 140.0, [part])
    finally:
        hw.brightness = orig
    assert c.shape == (hw.CROP, hw.CROP) and c.dtype == np.uint8
    assert c[0, 32] < c[-1, 32]  # lines count southward on the disc, so the top row is north
