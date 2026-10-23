"""运行证据使用的 Git 状态检查。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess


class GitStateError(RuntimeError):
    """无法获得可审计 Git 状态。"""


@dataclass(frozen=True, slots=True)
class GitState:
    revision: str
    clean: bool

    @property
    def revision_label(self) -> str:
        return self.revision if self.clean else f"{self.revision}-dirty"


def inspect_git_state(root: str | Path) -> GitState:
    """返回 HEAD 与 tracked/untracked（忽略规则之外）工作树状态。"""

    directory = Path(root).resolve()
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=directory,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout.strip()
        status = subprocess.run(
            [
                "git",
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
                "--ignore-submodules=none",
            ],
            cwd=directory,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise GitStateError(f"无法读取 Git 状态: {directory}") from error
    if not revision:
        raise GitStateError(f"Git HEAD 为空: {directory}")
    return GitState(revision=revision, clean=not status.strip())
