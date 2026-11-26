import tempfile
import unittest
from pathlib import Path

from src.plotting import make_gallery


class TestMakeGallery(unittest.TestCase):
    def _touch(self, path: Path, data: bytes = b"png") -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def test_scan_default_and_pdf_twin_folding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._touch(root / "RunII" / "region_SR" / "m4j.png")
            self._touch(root / "RunII" / "region_SR" / "m4j.pdf", b"%PDF")
            self._touch(root / "RunII" / "region_SB" / "lone.pdf", b"%PDF")
            items = make_gallery.scan(root, "index.html")
            self.assertEqual(len(items), 2)
            by_path = {it["path"]: it for it in items}
            self.assertEqual(by_path["RunII/region_SR/m4j.png"]["kind"], "img")
            self.assertEqual(by_path["RunII/region_SR/m4j.png"]["pdf"], "RunII/region_SR/m4j.pdf")
            self.assertEqual(by_path["RunII/region_SB/lone.pdf"]["kind"], "link")

    def test_scan_include_and_exclude_prefixes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._touch(root / "plots_v0" / "selJets_noJCM_n.png")
            self._touch(root / "plots_v1" / "tagJets_noJCM_n.png")
            self._touch(root / "test_v0_closure" / "plots" / "selJets_n.png")

            only_jcm = make_gallery.scan(root, "index.html", include_prefixes=["plots_v"])
            self.assertEqual(
                [it["path"] for it in only_jcm],
                ["plots_v0/selJets_noJCM_n.png", "plots_v1/tagJets_noJCM_n.png"],
            )

            no_test = make_gallery.scan(root, "index.html", exclude_prefixes=["test_"])
            self.assertEqual(
                [it["path"] for it in no_test],
                ["plots_v0/selJets_noJCM_n.png", "plots_v1/tagJets_noJCM_n.png"],
            )

    def test_mark_summary_and_main_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._touch(root / "plots_v0" / "selJets_noJCM_n.png")
            self._touch(root / "plots_v1" / "tagJets_noJCM_n.png")
            self._touch(root / "test_v0_closure" / "plots" / "canJet0_pt.png")

            rc = make_gallery.main([
                str(root),
                "--include-prefix", "plots_v",
                "--summary", "selJets_noJCM.n",
                "--title", "Subsample_JCM_Fits",
            ])
            self.assertEqual(rc, 0)
            html_out = (root / "index.html").read_text()
            self.assertIn("Subsample_JCM_Fits", html_out)
            self.assertIn("plots_v0/selJets_noJCM_n.png", html_out)
            self.assertIn("plots_v1/tagJets_noJCM_n.png", html_out)
            self.assertNotIn("test_v0_closure", html_out)

    def test_format_stage_title(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._touch(root / "m4j.png")
            rc = make_gallery.main([
                str(root),
                "--title", "Stage_F_1_Mixed-Data_Background_Systematics",
            ])
            self.assertEqual(rc, 0)
            html_out = (root / "index.html").read_text()
            self.assertIn("<title>Stage F.1 Mixed-Data Background Systematics</title>", html_out)
            self.assertIn("<h1>Stage F.1 Mixed-Data Background Systematics</h1>", html_out)


if __name__ == "__main__":
    unittest.main()
