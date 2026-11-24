import sqlite3

import pytest

from qros.registry import db
from qros.registry.core import Registry, RegistryError
from qros.registry.models import Periods
from helpers import make_returns


def test_registry_reopens_and_discovers(tmp_path, monkeypatch):
    reg = Registry.init(tmp_path / ".qros")
    rid = reg.registry_id
    reg.close()
    sub = tmp_path / "a" / "b"
    sub.mkdir(parents=True)
    monkeypatch.chdir(sub)
    monkeypatch.delenv("QROS_REGISTRY", raising=False)
    again = Registry.open()
    assert again.registry_id == rid
    again.close()


def test_missing_registry_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        Registry(tmp_path / "nope")


def test_hypothesis_lifecycle_and_versions(reg):
    h = reg.add_hypothesis("Momentum", "Winners keep winning", prediction={"direction": "positive"}, family="mom")
    assert h.status == "draft" and h.registered_at is None
    h = reg.register_hypothesis(h.id)
    assert h.status == "registered" and h.registered_at
    with pytest.raises(RegistryError):
        reg.register_hypothesis(h.id)
    h2 = reg.amend_hypothesis(h.id, reason="clarify", statement="Winners keep winning for a year")
    assert h2.current_version == 2
    assert [v.statement for v in h2.versions] == ["Winners keep winning", "Winners keep winning for a year"]
    assert h2.versions[0].text_sha256 != h2.versions[1].text_sha256


def test_append_only_tables_are_enforced_by_sqlite(reg):
    h = reg.add_hypothesis("H", "statement", register=True)
    with pytest.raises(sqlite3.IntegrityError):
        reg.conn.execute("UPDATE hypothesis_versions SET statement = 'rewritten' WHERE hypothesis_id = ?", (h.id,))
    with pytest.raises(sqlite3.IntegrityError):
        reg.conn.execute("DELETE FROM events")
    with pytest.raises(sqlite3.IntegrityError):
        reg.conn.execute("UPDATE events SET payload_json = '{}' WHERE seq = 1")


def test_audit_chain_detects_tampering(tmp_path):
    reg = Registry.init(tmp_path / ".qros")
    reg.add_hypothesis("H", "statement", register=True)
    reg.add_holdout("ho", "2020-01-01", "2020-12-31")
    assert reg.verify(deep=False)["audit_chain"]["ok"]
    # Bypass the trigger the way an attacker with file access would, then check the chain breaks.
    reg.conn.execute("DROP TRIGGER events_append_only_u")
    reg.conn.execute("UPDATE events SET payload_json = '{\"title\":\"edited\"}' WHERE seq = 2")
    res = reg.verify(deep=False)["audit_chain"]
    assert not res["ok"] and res["first_bad_seq"] == 2
    reg.close()


def test_finished_experiments_are_immutable(reg):
    exp = reg.create_experiment("e", params={"a": 1}, periods=Periods.of("2010:2014", "2015:2019"))
    reg.complete_experiment(exp.id, make_returns(n=2600))
    with pytest.raises(sqlite3.IntegrityError):
        reg.conn.execute("UPDATE experiments SET params_json = '{\"a\": 2}' WHERE id = ?", (exp.id,))
    with pytest.raises(sqlite3.IntegrityError):
        reg.conn.execute("UPDATE metrics SET value = 99 WHERE experiment_id = ?", (exp.id,))
    with pytest.raises(sqlite3.IntegrityError):
        reg.log_metrics(exp.id, {"late": 1.0})


def test_spec_hash_identifies_inputs_and_dedupes_trials(reg):
    a = reg.create_experiment("a", family="f", params={"x": 1}, seed=1)
    b = reg.create_experiment("b", family="f", params={"x": 1}, seed=1)
    c = reg.create_experiment("c", family="f", params={"x": 2}, seed=1)
    assert a.spec_hash == b.spec_hash != c.spec_hash
    for e in (a, b, c):
        reg.complete_experiment(e.id, make_returns(seed=hash(e.id) % 100), validate=False)
    counts = reg.family_trial_count("f")
    assert counts == {"registered": 3, "distinct": 2, "external": 0, "total": 2}
    reg.declare_external_trials("f", 10, "notebook exploration before qros")
    assert reg.family_trial_count("f")["total"] == 12


def test_tags_notes_and_listing(reg):
    e = reg.create_experiment("e", family="fam", tags=["x"])
    reg.tag(e.id, "y", "z")
    reg.untag(e.id, "z")
    reg.add_note("looked fine", experiment=e.id)
    assert reg.get_experiment(e.id).tags == ["x", "y"]
    assert [n["text"] for n in reg.notes(experiment=e.id)] == ["looked fine"]
    assert [x.id for x in reg.list_experiments(tag="y")] == [e.id]
    assert reg.list_experiments(family="other") == []


def test_holdout_ledger_records_every_look(reg):
    ho = reg.add_holdout("final", "2018-01-01", "2019-12-31", max_touches=1)
    e1 = reg.create_experiment("look 1", holdout=ho.id, open_holdout=True, holdout_purpose="final test")
    e2 = reg.create_experiment("look 2", holdout=ho.id, open_holdout=True, holdout_purpose="peek")
    acc = reg.holdout_accesses(ho.id)
    assert [a["experiment_id"] for a in acc] == [e1.id, e2.id]
    with pytest.raises(sqlite3.IntegrityError):
        reg.conn.execute("DELETE FROM holdout_access")


def test_counters_and_ids(reg):
    assert reg.add_hypothesis("a", "s").id == "H-001"
    assert reg.add_hypothesis("b", "s").id == "H-002"
    assert reg.create_experiment("e").id == "EXP-0001"


def test_schema_version_guard(tmp_path):
    reg = Registry.init(tmp_path / ".qros")
    reg.conn.execute(f"PRAGMA user_version = {db.SCHEMA_VERSION + 1}")
    reg.close()
    with pytest.raises(RuntimeError):
        Registry(tmp_path / ".qros")
