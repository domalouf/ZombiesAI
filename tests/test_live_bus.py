from zombiesai.viz.live_bus import Publisher, Subscriber, brain_bus, bus_path


def test_a_publisher_with_nobody_listening_drops_quietly(tmp_path):
    pub = Publisher(tmp_path / "nobody.sock")
    assert pub.send({"kind": "recorder"}) is False
    assert pub.dropped == 1
    pub.close()
    assert pub.send({"kind": "recorder"}) is False  # closed: still no exception


def test_the_subscriber_keeps_the_newest_message_of_each_kind(tmp_path):
    path = tmp_path / "record.sock"
    sub = Subscriber(path)
    pub = Publisher(path)
    for step in range(5):
        assert pub.send({"kind": "recorder", "step": step})
    pub.send({"kind": "pad", "lx": 0.5})
    assert sub.poll(now=10.0)
    assert sub.get("recorder")["step"] == 4
    assert sub.get("pad")["lx"] == 0.5
    assert sub.get("pad", max_age_s=1.0, now=10.5) is not None
    assert sub.get("pad", max_age_s=1.0, now=11.5) is None
    assert not sub.poll()
    sub.close()
    assert not path.exists()


def test_a_stale_socket_from_a_dead_reader_is_replaced(tmp_path):
    path = tmp_path / "record.sock"
    Subscriber(path)  # never closed, as if killed
    sub = Subscriber(path)
    Publisher(path).send({"kind": "recorder"})
    assert sub.poll()


def test_unserialisable_and_oversized_messages_are_dropped(tmp_path):
    path = tmp_path / "record.sock"
    sub = Subscriber(path)
    pub = Publisher(path)
    assert not pub.send({"kind": "x", "bad": object()})
    assert not pub.send({"kind": "x", "big": "a" * 70_000})
    assert not sub.poll()


def test_bus_paths_are_per_feed_and_per_display(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    assert bus_path("record") == tmp_path / "zombiesai" / "record.sock"
    assert brain_bus(":60") == tmp_path / "zombiesai" / "brain-60.sock"


def test_closing_an_old_reader_leaves_its_replacements_socket_alone(tmp_path):
    path = tmp_path / "brain-60.sock"
    old = Subscriber(path)
    new = Subscriber(path)  # a relaunched viewer binds before the old one has cleaned up
    old.close()
    assert path.exists()
    Publisher(path).send({"kind": "brain"})
    assert new.poll()
    new.close()
    assert not path.exists()
