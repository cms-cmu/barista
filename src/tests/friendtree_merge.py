"""
Regression tests for friend-tree merging of jagged collections.

Guards against a bug in the NanoAOD zip transform
(:mod:`src.data_formats.awkward.zip`) where a jagged collection whose count
branch has a lowercase letter after the ``n`` -- e.g. ``nbJetCand`` for the
``bJetCand`` collection -- was not recognised as a count branch. The collection
was therefore left *unzipped*: its per-object columns (``bJetCand_pt``, ...)
stayed loose alongside an orphan ``nbJetCand`` count branch. With newer uproot,
writing that layout during the friend-tree merge aborts with::

    AttributeError: 'numpy.ndarray' object has no attribute 'layout'

These tests are self-contained (no remote files / grid proxy): they build a
small array with both standard (``nJet``) and lowercase-prefixed (``nbJetCand``)
collections, then exercise the transform directly and through a real
write -> read -> merge round trip.

Run with::

    ./run_container python -m src.tests.friendtree_merge
"""

import tempfile
import unittest

import awkward as ak
import numpy as np

from src.data_formats.awkward.zip import NanoAOD
from src.data_formats.root.chunk import Chunk
from src.data_formats.root.io import TreeReader, TreeWriter


def _sample_array(n=2000):
    """A friend-tree-like array mixing jagged collections (with count branches
    whose names start with both an uppercase and a lowercase letter), flat
    collections, and flat scalar branches."""
    rng = np.random.default_rng(0)
    counts = rng.integers(0, 4, size=n)

    def jagged(scale=1.0):
        return ak.unflatten(rng.random(int(counts.sum())) * scale, counts)

    return ak.Array(
        {
            # standard NanoAOD-style jagged collection: count is nJet
            "nJet": counts,
            "Jet_pt": jagged(100.0),
            "Jet_eta": jagged(5.0),
            # lowercase-after-n count branch (the regression case): nbJetCand
            "nbJetCand": counts,
            "bJetCand_pt": jagged(100.0),
            "bJetCand_btagScore": jagged(1.0),
            # another lowercase case
            "nnonbJetCand": counts,
            "nonbJetCand_pt": jagged(100.0),
            "nonbJetCand_eta": jagged(5.0),
            # flat (regular) collection, no count branch
            "MET_pt": rng.random(n) * 100.0,
            "MET_phi": rng.random(n) * 3.14,
            # flat scalar branches
            "weight": rng.random(n),
            "event": np.arange(n, dtype=np.int64),
        }
    )


def _orphan_count_branches(array):
    """Return ``n<prefix>`` fields that still have loose ``<prefix>_*`` members
    -- i.e. collections the transform failed to zip."""
    fields = set(array.fields)
    orphans = {}
    for field in array.fields:
        if field.startswith("n") and len(field) > 1:
            prefix = field[1:]
            loose = sorted(f for f in fields if f.startswith(f"{prefix}_"))
            if loose:
                orphans[field] = loose
    return orphans


class FriendTreeMerge(unittest.TestCase):
    # The friend-tree merge uses NanoAOD(regular=False, jagged=True)
    # (see runner.py process_friend_trees).
    transform = NanoAOD(regular=False, jagged=True)

    def test_lowercase_count_branch_is_zipped(self):
        """``nbJetCand`` must be recognised so ``bJetCand`` is zipped, leaving
        no orphan count branch."""
        out = self.transform(_sample_array())

        for coll in ("Jet", "bJetCand", "nonbJetCand"):
            self.assertIn(
                coll, out.fields,
                f"collection {coll!r} was not zipped by the transform",
            )
            self.assertTrue(
                ak.fields(out[coll]),
                f"{coll!r} is not a record (zip produced no subfields)",
            )

        orphans = _orphan_count_branches(out)
        self.assertEqual(
            orphans, {},
            f"unzipped collection(s) left orphan count branches: {orphans}",
        )

    def test_flat_collection_left_regular(self):
        """A collection with no count branch (``MET``) must stay flat when
        ``regular=False``, not be wrongly zipped."""
        out = self.transform(_sample_array())
        self.assertNotIn("MET", out.fields)
        self.assertIn("MET_pt", out.fields)

    def test_merge_round_trip(self):
        """Write -> read -> merge round trip must not raise (this is the path
        that crashed) and must preserve every event and jagged length."""
        data = _sample_array()

        with tempfile.TemporaryDirectory() as d:
            source = f"{d}/source.root"
            with TreeWriter()(source) as w:
                w.extend(self.transform(data))

            # Re-merge across multiple baskets, as the friend-tree merge does.
            merged = f"{d}/merged.root"
            chunk = Chunk(source=source, name="Events")
            with TreeWriter()(merged) as w:
                for arr in TreeReader().iterate(chunk, step=500):
                    w.extend(self.transform(arr))

            out = Chunk(source=merged, name="Events")
            self.assertEqual(out.num_entries, len(data))

            import uproot

            with uproot.open(merged) as f:
                tree = f["Events"]
                stored = ak.to_numpy(tree["nbJetCand"].array(library="ak"))
                actual = ak.to_numpy(
                    ak.num(tree["bJetCand_pt"].array(library="ak"))
                )
                self.assertTrue(
                    np.array_equal(stored, actual),
                    "bJetCand count branch does not match jagged lengths "
                    "after round trip",
                )


if __name__ == "__main__":
    unittest.main()
