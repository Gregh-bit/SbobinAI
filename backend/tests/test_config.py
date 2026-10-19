from __future__ import annotations

import json
import os
import threading

import config


def test_save_settings_is_atomic_and_roundtrips(tmp_path, monkeypatch):
    target = tmp_path / "settings.json"
    monkeypatch.setattr(config, "SETTINGS_FILE", str(target))
    config.save_settings({"llm_model": "x", "vad_sensitivity": 3})
    assert not (tmp_path / "settings.json.tmp").exists()
    loaded = config.load_settings()
    assert loaded["llm_model"] == "x"
    assert loaded["vad_sensitivity"] == 3
    # i default mancanti vengono uniti
    assert "embedding_model" in loaded


def test_load_settings_with_corrupted_file_returns_defaults(tmp_path, monkeypatch):
    target = tmp_path / "settings.json"
    target.write_text("{ non json", encoding="utf-8")
    monkeypatch.setattr(config, "SETTINGS_FILE", str(target))
    assert config.load_settings() == dict(config.DEFAULT_SETTINGS)


def test_load_settings_non_object_returns_defaults(tmp_path, monkeypatch):
    target = tmp_path / "settings.json"
    target.write_text("[1, 2]", encoding="utf-8")
    monkeypatch.setattr(config, "SETTINGS_FILE", str(target))
    assert config.load_settings() == dict(config.DEFAULT_SETTINGS)


def test_update_settings_concurrent_writers_do_not_lose_keys(tmp_path, monkeypatch):
    target = tmp_path / "settings.json"
    monkeypatch.setattr(config, "SETTINGS_FILE", str(target))
    config.save_settings({})

    def writer(i):
        config.update_settings({f"k{i}": i})

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    data = json.loads(target.read_text(encoding="utf-8"))
    assert all(data.get(f"k{i}") == i for i in range(20))


def test_legacy_storage_path_is_migrated(tmp_path, monkeypatch):
    target = tmp_path / "settings.json"
    target.write_text(json.dumps({"storage_path": config.LEGACY_STORAGE_ROOT}), encoding="utf-8")
    monkeypatch.setattr(config, "SETTINGS_FILE", str(target))
    assert config._same_path(config.load_settings()["storage_path"], config.DEFAULT_STORAGE_ROOT)


def test_same_path_is_case_and_separator_insensitive_where_os_is():
    a = os.path.join("C:" + os.sep, "Users", "x") if os.name == "nt" else "/tmp/x"
    b = a.upper() if os.name == "nt" else "/tmp/./x"
    assert config._same_path(a, b)


def test_update_storage_path_updates_active_root(tmp_path):
    """Regressione: update_storage_path assegnava _active_storage_root senza `global`."""
    old_root = config.get_storage_path()
    new_root = str(tmp_path / "nuovo_storage")
    try:
        res = config.update_storage_path(new_root, migrate_data=True)
        assert res["status"] == "success"
        assert config._same_path(config.get_storage_path(), new_root)
        assert config._same_path(config.load_settings()["storage_path"], new_root)
        import database
        assert config._same_path(database.DB_PATH, os.path.join(new_root, "sqlite", "sbobinai.db"))
        assert os.path.exists(os.path.join(new_root, "sqlite", "sbobinai.db"))
    finally:
        config.update_storage_path(old_root, migrate_data=False)
    assert config._same_path(config.get_storage_path(), old_root)
