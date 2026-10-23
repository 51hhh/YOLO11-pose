"""内容寻址资产使用的共享哈希工具。"""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    """流式计算文件 SHA-256，避免把模型或原始资产整体读入内存。"""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
