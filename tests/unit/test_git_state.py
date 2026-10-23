from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
import unittest

from volleyball_vision.git_state import inspect_git_state


class GitStateTests(unittest.TestCase):
    def test_tracks_committed_untracked_and_ignored_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            (root / ".gitignore").write_text("ignored/\n", encoding="utf-8")
            (root / "tracked.txt").write_text("initial\n", encoding="utf-8")
            subprocess.run(
                ["git", "add", ".gitignore", "tracked.txt"],
                cwd=root,
                check=True,
            )
            subprocess.run(
                [
                    "git",
                    "-c",
                    "user.name=Volleyball Test",
                    "-c",
                    "user.email=volleyball-test@example.invalid",
                    "commit",
                    "-qm",
                    "initial",
                ],
                cwd=root,
                check=True,
            )

            clean = inspect_git_state(root)
            self.assertTrue(clean.clean)
            self.assertEqual(clean.revision_label, clean.revision)

            ignored = root / "ignored" / "artifact.bin"
            ignored.parent.mkdir()
            ignored.write_bytes(b"ignored")
            self.assertTrue(inspect_git_state(root).clean)

            (root / "untracked.txt").write_text("new\n", encoding="utf-8")
            dirty = inspect_git_state(root)
            self.assertFalse(dirty.clean)
            self.assertEqual(dirty.revision_label, f"{dirty.revision}-dirty")


if __name__ == "__main__":
    unittest.main()
