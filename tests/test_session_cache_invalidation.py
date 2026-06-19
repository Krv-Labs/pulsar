"""Feature-evidence cache must not survive a data rebind.

The evidence index is precomputed once per session and reused across tool
calls, keyed by a fingerprint over cluster labels / threshold / run_id. That
fingerprint does not cover the underlying DataFrame, so rebinding session data
(e.g. re-ingesting a different dataset) must explicitly drop the cache;
otherwise a subsequent tool call could serve evidence computed against the old
data. See _bind_session_data in pulsar/mcp/session.py.
"""

import pandas as pd

from pulsar.mcp.session import _PulsarSession, _bind_session_data


def _populate_cache(session: _PulsarSession) -> None:
    session.feature_evidence_index = object()
    session.feature_evidence_fingerprint = "stale-fingerprint"
    session.feature_evidence_cluster_meta = {"n_clusters": 3}
    session.clusters = pd.Series([0, 1, 0])


def test_rebinding_data_invalidates_evidence_cache():
    session = _PulsarSession()
    _populate_cache(session)

    _bind_session_data(
        session, pd.DataFrame({"a": [1, 2, 3]}), data_path="/tmp/new.csv"
    )

    assert session.data is not None
    assert session.feature_evidence_index is None
    assert session.feature_evidence_fingerprint is None
    assert session.feature_evidence_cluster_meta is None
    assert session.clusters is None
