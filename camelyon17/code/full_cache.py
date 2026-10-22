"""固定revision全量PNG的私有、无压缩tar缓存；不修改实验或训练状态。"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import stat
import tarfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

PINNED_REVISION = "d784d5344ba6c967f83f9f3d9b2f1e2a4d6eb78f"
EXPECTED_IMAGES = 455954
TAR_NAME = "images.tar"
METADATA_NAME = "images.metadata.json"
PNG_HEADER = b"\x89PNG\r\n\x1a\n"
_VALIDATED_CACHES = {}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_relative(name, revision):
    if not isinstance(name, str) or not name or "\\" in name or any(ord(c) < 32 for c in name):
        raise ValueError(f"Unsafe image/member path: {name!r}")
    path = PurePosixPath(name)
    if (path.is_absolute() or ".." in path.parts or path.as_posix() != name or
            len(path.parts) < 4 or tuple(path.parts[:3]) != ("data", "images", revision) or
            path.suffix != ".png"):
        raise ValueError(f"Unsafe or unexpected PNG path: {name!r}")
    return name


def _manifest_identity(root, manifest_path, revision, expected_images):
    root = Path(root).resolve()
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("revision must be an immutable 40-character lowercase commit SHA")
    if int(expected_images) < 1:
        raise ValueError("expected_images must be positive")
    manifest_path = Path(manifest_path)
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    with manifest_path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None or "path" not in reader.fieldnames:
            raise ValueError("Manifest must contain path")
        paths = [_safe_relative(row["path"], revision) for row in reader]
    if len(paths) != int(expected_images) or len(set(paths)) != len(paths):
        raise ValueError(f"Manifest must contain exactly {expected_images} unique PNG paths")
    paths.sort()
    # 不含partition或runtime本地mtime；相同图像的三套病人划分共用缓存。
    identity = {"revision": revision, "expected_images": int(expected_images),
                "relative_paths_sha256": hashlib.sha256(("\n".join(paths)+"\n").encode()).hexdigest(),
                "path_digest_format": "UTF-8 sorted POSIX paths, one per line, final newline"}
    return root, paths, identity


def _checked_local_path(root, relative):
    candidate = root / relative
    if not candidate.resolve().is_relative_to(root):
        raise ValueError(f"Local image path escapes root: {relative}")
    # 即使符号链接指向root内部也不跟随，避免替换或复用意外的别名。
    current = root
    for component in PurePosixPath(relative).parts:
        current = current / component
        if current.is_symlink():
            raise ValueError(f"Symlink in image path: {relative}")
    return candidate


def _complete_metadata(cache, identity):
    metadata_path, archive_path = cache/METADATA_NAME, cache/TAR_NAME
    if not metadata_path.exists():
        # tar或*.partial独自存在不构成完整cache。
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError("Existing cache metadata is unreadable/corrupt") from error
    if (metadata.get("schema_version") != 1 or metadata.get("status") != "complete" or
            metadata.get("identity") != identity or metadata.get("tar_name") != TAR_NAME):
        raise ValueError("Existing cache identity/schema does not match this manifest")
    if (not archive_path.is_file() or archive_path.is_symlink() or
            not isinstance(metadata.get("tar_size"), int) or
            archive_path.stat().st_size != metadata["tar_size"] or
            not re.fullmatch(r"[0-9a-f]{64}", str(metadata.get("tar_sha256", "")))):
        raise ValueError("Existing complete cache tar is missing or its size/metadata is corrupt")
    return metadata


def _copy_file(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Path(source).open("rb") as incoming, destination.open("wb") as outgoing:
        shutil.copyfileobj(incoming, outgoing, length=8 << 20)
        outgoing.flush()
        os.fsync(outgoing.fileno())


def _validate_archive(path, paths):
    expected, seen = set(paths), set()
    end_of_data = 0
    try:
        with tarfile.open(path, mode="r:") as archive:
            for member in archive:
                if (not member.isfile() or member.issparse() or member.pax_headers or
                        member.name not in expected or member.name in seen):
                    raise ValueError(f"Unsafe, duplicate, or unlisted tar member: {member.name!r}")
                if member.size < len(PNG_HEADER):
                    raise ValueError(f"Invalid PNG member size: {member.name!r}")
                source = archive.extractfile(member)
                if source is None or source.read(len(PNG_HEADER)) != PNG_HEADER:
                    raise ValueError(f"Invalid PNG member bytes: {member.name!r}")
                source.close()
                seen.add(member.name)
                end_of_data = member.offset_data + ((member.size+511)//512)*512
                if len(seen) % 10000 == 0:
                    print(f"[image cache validate] {len(seen)}/{len(paths)}", flush=True)
        if seen != expected:
            raise ValueError(f"Tar member set differs from manifest: {len(seen)}/{len(expected)}")
        # tarfile本身容忍缺少EOF或追加垃圾；完整缓存额外要求标准零终止块。
        with Path(path).open("rb") as source:
            source.seek(end_of_data)
            if source.read(1024) != bytes(1024):
                raise ValueError("Tar is truncated or missing its complete end marker")
            for block in iter(lambda: source.read(8 << 20), b""):
                if block.strip(b"\0"):
                    raise ValueError("Unexpected nonzero data after tar end marker")
    except (tarfile.TarError, EOFError, OSError) as error:
        raise ValueError("Cache tar is corrupt or truncated") from error


def _write_metadata(path, metadata):
    partial = path.with_name(path.name+".partial")
    with partial.open("w", encoding="utf-8") as destination:
        json.dump(metadata, destination, indent=2, ensure_ascii=False)
        destination.write("\n")
        destination.flush()
        os.fsync(destination.fileno())
    partial.replace(path)


def _cache_token(cache):
    status=(cache/TAR_NAME).stat()
    return (str(cache.resolve()),status.st_size,status.st_mtime_ns,
            _sha256(cache/METADATA_NAME))


def _mark_validated(cache, local_tar):
    status=local_tar.stat()
    _VALIDATED_CACHES[_cache_token(cache)]=(str(local_tar),status.st_size,status.st_mtime_ns)


def _verified_local_archive(root, cache, metadata, paths):
    """远端只顺序读一次；本次已经验证且未变化的tar可供restore/export复用。"""
    token=_cache_token(cache)
    previous=_VALIDATED_CACHES.get(token)
    if previous is not None:
        local=Path(previous[0])
        if local.is_file() and not local.is_symlink():
            status=local.stat()
            if (status.st_size,status.st_mtime_ns)==previous[1:]:
                return local
    directory=root/".full_image_cache"/metadata["identity"]["revision"]
    directory.mkdir(parents=True,exist_ok=True)
    partial=directory/"images.tar.restore.partial"
    _copy_file(cache/TAR_NAME,partial)
    if partial.stat().st_size!=metadata["tar_size"] or _sha256(partial)!=metadata["tar_sha256"]:
        raise ValueError("Complete cache tar SHA256/size mismatch; no PNGs restored")
    _validate_archive(partial,paths)
    if _cache_token(cache)!=token:
        raise ValueError("Complete cache changed during validation")
    local=directory/TAR_NAME
    partial.replace(local)
    _mark_validated(cache,local)
    return local


def export_cache(root, manifest_path, cache_dir, *, revision=PINNED_REVISION,
                 expected_images=EXPECTED_IMAGES) -> Path:
    """打包调用者已经验证的全部PNG；返回完整tar路径，同identity完整cache核验后复用。

    cache_dir必须由调用者传入私有Drive目录；本函数没有Drive权限或分享操作。
    expected_images的非默认值仅供隔离的小软件fixture验证。
    """
    root, paths, identity = _manifest_identity(root, manifest_path, revision, expected_images)
    cache = Path(cache_dir)
    metadata = _complete_metadata(cache, identity)
    if metadata is not None:
        _verified_local_archive(root,cache,metadata,paths)
        print(f"[image cache] verified existing complete cache: {len(paths)} PNGs", flush=True)
        return cache/TAR_NAME
    image_directory = _checked_local_path(root, f"data/images/{revision}")
    available = {p.relative_to(root).as_posix() for p in image_directory.rglob("*.png")}
    if available != set(paths):
        raise ValueError("Local PNG tree does not exactly match the full manifest")
    local_directory = root/".full_image_cache"/revision
    local_directory.mkdir(parents=True, exist_ok=True)
    local_partial = local_directory/"images.tar.local.partial"
    with local_partial.open("wb") as destination:
        with tarfile.open(fileobj=destination, mode="w", format=tarfile.USTAR_FORMAT) as archive:
            for number, relative in enumerate(paths, 1):
                source_path = _checked_local_path(root, relative)
                before = source_path.stat()
                if not stat.S_ISREG(before.st_mode):
                    raise ValueError(f"Source PNG is not a regular file: {relative}")
                member = tarfile.TarInfo(relative)
                member.size, member.mode, member.mtime = before.st_size, 0o644, 0
                with source_path.open("rb") as source:
                    if source.read(len(PNG_HEADER)) != PNG_HEADER:
                        raise ValueError(f"Source is not PNG: {relative}")
                    source.seek(0)
                    archive.addfile(member, source)
                after = source_path.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise ValueError(f"Source PNG changed during export: {relative}")
                if number % 10000 == 0:
                    print(f"[image cache export] {number}/{len(paths)}", flush=True)
        destination.flush()
        os.fsync(destination.fileno())
    checksum, size = _sha256(local_partial), local_partial.stat().st_size
    local_tar=local_directory/TAR_NAME
    local_partial.replace(local_tar)
    cache.mkdir(parents=True, exist_ok=True)
    drive_partial = cache/(TAR_NAME+".partial")
    _copy_file(local_tar, drive_partial)
    if drive_partial.stat().st_size != size or _sha256(drive_partial) != checksum:
        raise ValueError("Drive partial tar copy checksum/size mismatch; cache not committed")
    drive_partial.replace(cache/TAR_NAME)
    metadata = {"schema_version": 1, "status": "complete", "identity": identity,
                "tar_name": TAR_NAME, "tar_sha256": checksum, "tar_size": size,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "format": "uncompressed USTAR, regular PNG files only",
                "source_validation": "caller-prevalidated PNGs; regular files, PNG signature, size/mtime stability checked on export"}
    _write_metadata(cache/METADATA_NAME, metadata)
    _mark_validated(cache,local_tar)
    print(f"[image cache] committed {len(paths)} PNGs, {size} bytes", flush=True)
    return cache/TAR_NAME


def restore_cache(root, manifest_path, cache_dir, *, revision=PINNED_REVISION,
                  expected_images=EXPECTED_IMAGES) -> bool:
    """完整cache复制到本地并核验后，逐图原子恢复；缺少完整cache返回False。

    完整cache损坏/identity不同必须报错；从不把partial当成成功、不删除旧实验。
    恢复完仍由既有prepare构建当前runtime的图片digest清单。
    """
    root, paths, identity = _manifest_identity(root, manifest_path, revision, expected_images)
    cache = Path(cache_dir)
    metadata = _complete_metadata(cache, identity)
    if metadata is None:
        return False
    local_tar=_verified_local_archive(root,cache,metadata,paths)
    with tarfile.open(local_tar, mode="r:") as archive:
        for number, member in enumerate(archive, 1):
            destination = _checked_local_path(root, member.name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            partial = destination.with_name(destination.name+".restore.partial")
            with archive.extractfile(member) as source, partial.open("wb") as output:
                shutil.copyfileobj(source, output, length=1 << 20)
            if partial.stat().st_size != member.size:
                raise ValueError(f"Incomplete restored PNG: {member.name}")
            partial.replace(destination)
            if number % 10000 == 0:
                print(f"[image cache restore] {number}/{len(paths)}", flush=True)
    print(f"[image cache] restored {len(paths)} PNGs; rebuild image digest manifest next", flush=True)
    return True
