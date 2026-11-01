import pandas as pd
import pytest

from src import graph

F1, F2 = "0251111111", "0252222222"
W1, W2 = "8991111111", "8992222222"
M1 = "2101111111"
N1 = "8421111111"
R = [f"89900000{i:02d}" for i in range(1, 10)]


def call(a, b, imsi, ts):
    return {"a": a, "b": b, "imsi": imsi, "ts": pd.Timestamp(ts)}


CALLS = pd.DataFrame(
    [call(F1, b, "S1", "2026-10-01 10:00") for b in R[0:3]]
    + [call(F2, b, "S1", "2026-10-02 10:00") for b in R[3:6]]
    + [call(W1, b, "S5", "2026-10-03 12:00") for b in [R[0], *R[6:9]]]
    + [call(W2, b, "S6", "2026-10-03 13:00") for b in R[6:9]]
    + [call(M1, "8990000098", "S2", "2026-10-01 09:00"), call(M1, "8990000099", "S3", "2026-10-02 09:00")]
    + [call(N1, R[2], None, "2026-10-04 15:00")]
)

SCORES = pd.DataFrame({
    "level": ["высокий", "высокий", "высокий", "проверка", "высокий", "низкий"],
    "pattern": ["burst", "burst", "wangiri", "wangiri", "multi_sim", ""],
}, index=[F1, F2, W1, W2, M1, N1])


@pytest.fixture(scope="module")
def cand():
    return graph.candidates(SCORES)


def test_candidates_exclude_low_risk(cand):
    assert set(cand.index) == {F1, F2, W1, W2, M1}


def test_sim_links_keep_only_shared_or_multi_sim():
    links = graph.sim_links(CALLS)
    assert set(map(tuple, links[["a", "imsi"]].to_numpy())) == {
        (F1, "S1"), (F2, "S1"), (M1, "S2"), (M1, "S3")}


def test_recipient_overlap(cand):
    ov = graph.recipient_overlap(CALLS, cand.index).set_index(["a_1", "a_2"])
    assert ov.loc[(W1, W2), "shared"] == 3
    assert ov.loc[(W1, W2), "jaccard"] == pytest.approx(3 / 4)
    assert ov.loc[(F1, W1), "jaccard"] == pytest.approx(1 / 6)
    assert (F1, F2) not in ov.index


def test_pattern_overlap_shows_fan(cand):
    po = graph.pattern_overlap(CALLS, cand)
    assert po.loc["burst", "overlap_share"] == 0
    assert po.loc["wangiri", "overlap_share"] == pytest.approx(1 - 4 / 7)


def test_victims_hit_twice(cand):
    twice = graph.victims_hit_twice(CALLS, cand.index)
    assert twice.to_dict() == {R[0]: 2, R[6]: 2, R[7]: 2, R[8]: 2}


def test_schedule(cand):
    s = graph.schedule(CALLS, cand)
    assert s.loc["burst", "01.10"] == 1 and s.loc["burst", "02.10"] == 1
    assert s.loc["wangiri", "03.10"] == 2


def test_groups(cand):
    links = graph.sim_links(CALLS)
    groups = graph.build_groups(links, graph.recipient_overlap(CALLS, cand.index))
    assert groups.loc[F1, "group_id"] == groups.loc[F2, "group_id"]
    assert groups.loc[W1, "group_id"] == groups.loc[W2, "group_id"]
    assert groups.loc[F1, "group_id"] != groups.loc[W1, "group_id"]
    assert M1 not in groups.index and N1 not in groups.index
    desc = graph.describe_groups(groups, CALLS, links, SCORES)
    fan = desc.loc[groups.loc[F1, "group_id"]]
    assert fan["size"] == 2 and fan["shared_sims"] == 1 and fan["recipients"] == 6
    assert fan["composition"] == "burst×2"


def test_run_end_to_end(tmp_path):
    CALLS.to_parquet(tmp_path / "calls.parquet", index=False)
    SCORES.to_parquet(tmp_path / "scores.parquet")
    graph.run(tmp_path, tmp_path / "graph")
    assert (tmp_path / "groups.parquet").exists()
    assert (tmp_path / "graph" / "groups.csv").exists()
