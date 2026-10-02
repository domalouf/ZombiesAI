import queue

from zombiesai.rl import fleet
from zombiesai.rl.fleet import FleetClient, _offer_latest, response_verdict


def test_only_a_success_is_contact_with_the_learner():
    assert response_verdict(200, "heartbeat") == (True, None)
    for code in (204, 304):  # the weights' "nothing new": contact there, nowhere else
        assert response_verdict(code, "weights", success=(200, 204, 304)) == (True, None)
        assert response_verdict(code, "heartbeat") == (False, None)


def test_an_answer_that_is_no_success_leaves_the_watchdog_running():
    # Some other service on the port, a bad request, a learner that is behind: none of it proves the learner is
    # there for us, and none of it ends the session by itself -- `lost_s` does, if nothing better comes.
    for code in (400, 403, 404, 405, 413, 500, 502, 503):
        assert response_verdict(code, "heartbeat") == (False, None), code


def test_a_new_run_or_a_new_token_ends_the_session_at_once():
    contact, gone = response_verdict(409, "segment")
    assert not contact and "no longer knows this machine (segment)" in gone
    contact, gone = response_verdict(401, "weights", success=(200, 204, 304))
    assert not contact and "token" in gone and fleet.TOKEN_ENV in gone


def test_weights_from_something_that_is_not_the_learner_carry_no_version():
    client = FleetClient("127.0.0.1:1", "t", "rig2")
    client._request = lambda method, path, data=None, headers=None: (200, b"<html>", {"X-Version": "nope"})
    assert client.weights(-1) == (200, -1, b"<html>")
    client._request = lambda method, path, data=None, headers=None: (200, b"blob", {"X-Version": "7"})
    assert client.weights(-1) == (200, 7, b"blob")


def test_offering_a_segment_to_a_full_outbox_evicts_the_oldest_and_says_so():
    q = queue.Queue(maxsize=2)
    assert _offer_latest(q, "a") == 0 and _offer_latest(q, "b") == 0
    assert _offer_latest(q, "c") == 1
    assert [q.get_nowait(), q.get_nowait()] == ["b", "c"]


def test_an_outbox_emptied_by_the_sender_in_between_evicts_nothing():
    class Racy(queue.Queue):
        """Full on the first put, then the sender takes everything before we can evict."""

        def __init__(self):
            super().__init__(maxsize=1)
            self.raced = False

        def put_nowait(self, item):
            if not self.raced:
                self.raced = True
                raise queue.Full
            super().put_nowait(item)

    q = Racy()
    assert _offer_latest(q, "a") == 0 and q.get_nowait() == "a"
