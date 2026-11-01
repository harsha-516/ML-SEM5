import pandas as pd

from hazards.typhoon import build_table, parse_bst

SAMPLE = """66666 2001   3      2001 0 6                                   X
01090100 002 3 100 1300  998     040
01090106 002 4 110 1300  985     055
01090112 002 5 120 1300  970     070
66666 2002   1      2002 0 6                                   Y
02090100 002 2 100 1300 1004     000
"""


def test_parser_reads_two_storms_and_zero_wind_is_missing():
    t = parse_bst(SAMPLE)
    assert t.storm.nunique() == 2 and t.wind.iloc[2] == 70 and pd.isna(t.wind.iloc[3])
    assert t.time.iloc[0] == pd.Timestamp("2001-09-01 00:00")


def test_label_and_exclusion_of_typhoon_rows():
    d = build_table(parse_bst(SAMPLE), first_year=1977)
    assert d[d.storm == "2001"].y.tolist() == [1, 1]
    assert d[d.storm == "2002"].y.tolist() == [0]


def test_past_only_trend():
    d = build_table(parse_bst(SAMPLE), first_year=1977)
    first, second = d[d.storm == "2001"].iloc[0], d[d.storm == "2001"].iloc[1]
    assert pd.isna(first.dwind_6) and second.dwind_6 == 15
