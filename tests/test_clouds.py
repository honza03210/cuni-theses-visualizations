"""Word cloud layout caching."""

from clouds import CLOUD_W, DEFAULT, PREPARED, Settings, _key, default_views, layout, width_bucket


def test_width_buckets():
    assert width_bucket(None) == CLOUD_W
    assert width_bucket(2000) == CLOUD_W
    assert width_bucket(350) == 320 and width_bucket(359) == 320 and width_bucket(360) == 360
    assert width_bucket(100) == 300


def test_default_view_is_among_the_prepared_ones():
    assert Settings(**DEFAULT, width=CLOUD_W) in default_views()


def test_prepared_layout_matches_a_fresh_one():
    if not PREPARED:  # prepare.py has not been run
        return
    settings = Settings(**DEFAULT, width=CLOUD_W)
    from clouds import _compute
    assert PREPARED[_key(settings)] == _compute(settings)
    assert layout(settings) == _compute(settings)
