import unittest
import tempfile
import numpy as np
import awkward as ak
import uproot
import os

from src.skimmer.picoaod import PicoAOD


class DummySingleStreamSkimmer(PicoAOD):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def select(self, events):
        mask = events.event % 2 == 0
        return mask, None, {}


class DummyMultiStreamSkimmer(PicoAOD):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def select(self, events):
        mask_even = events.event % 2 == 0
        mask_odd = events.event % 2 == 1
        return {
            "even": (mask_even, None, {}),
            "odd": (mask_odd, None, {}),
        }


class TestPicoAODSplitting(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_file = os.path.join(self.temp_dir.name, "input.root")
        
        # Create a mock ROOT file with Events tree
        data = {
            "event": np.arange(100, dtype=np.int64),
            "run": np.full(100, 1, dtype=np.int32),
            "luminosityBlock": np.full(100, 10, dtype=np.int32),
            "Jet_pt": ak.Array([[20.0, 30.0], [40.0]] * 50),
        }
        with uproot.recreate(self.root_file) as f:
            f["Events"] = data

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_single_stream_backward_compatibility(self):
        """Verify that single-stream skimmer produces standard dataset output."""
        output_dir = os.path.join(self.temp_dir.name, "output_single")
        skimmer = DummySingleStreamSkimmer(base_path=output_dir, step=50)
        
        from coffea.nanoevents import NanoEventsFactory, BaseSchema
        factory = NanoEventsFactory.from_root(
            {self.root_file: "Events"},
            schemaclass=BaseSchema,
            metadata={
                "dataset": "TestDataset",
                "filename": self.root_file,
                "fileuuid": "12345678-1234-5678-1234-567812345678",
                "entrystart": 0,
                "entrystop": 100,
                "treename": "Events",
            }
        )
        events = factory.events()
        res = skimmer.process(events)
            
        self.assertIn("TestDataset", res)
        self.assertEqual(res["TestDataset"]["saved_events"], 50)
        self.assertEqual(res["TestDataset"]["total_events"], 100)
        self.assertEqual(len(res["TestDataset"]["files"]), 1)

    def test_multi_stream_splitting(self):
        """Verify that multi-stream dictionary returns generate separate outputs."""
        output_dir = os.path.join(self.temp_dir.name, "output_multi")
        skimmer = DummyMultiStreamSkimmer(base_path=output_dir, step=50)
        
        from coffea.nanoevents import NanoEventsFactory, BaseSchema
        factory = NanoEventsFactory.from_root(
            {self.root_file: "Events"},
            schemaclass=BaseSchema,
            metadata={
                "dataset": "TestDataset",
                "filename": self.root_file,
                "fileuuid": "12345678-1234-5678-1234-567812345678",
                "entrystart": 0,
                "entrystop": 100,
                "treename": "Events",
            }
        )
        events = factory.events()
        res = skimmer.process(events)
            
        self.assertIn("TestDataset_even", res)
        self.assertIn("TestDataset_odd", res)
        self.assertEqual(res["TestDataset_even"]["saved_events"], 50)
        self.assertEqual(res["TestDataset_odd"]["saved_events"], 50)
        self.assertEqual(len(res["TestDataset_even"]["files"]), 1)
        self.assertEqual(len(res["TestDataset_odd"]["files"]), 1)
        
        # Verify the contents of the written ROOT files
        file_even = res["TestDataset_even"]["files"][0].path
        file_odd = res["TestDataset_odd"]["files"][0].path
        
        with uproot.open(file_even) as f_even:
            tree_even = f_even["Events"]
            self.assertEqual(tree_even.num_entries, 50)
            events_even = tree_even["event"].array()
            self.assertTrue(np.all(events_even % 2 == 0))
            
        with uproot.open(file_odd) as f_odd:
            tree_odd = f_odd["Events"]
            self.assertEqual(tree_odd.num_entries, 50)
            events_odd = tree_odd["event"].array()
            self.assertTrue(np.all(events_odd % 2 == 1))


if __name__ == "__main__":
    unittest.main()
