#!/usr/bin/env python3
"""Build a loadable, single-layer Docker archive in storage-constrained hosts.

This helper is only a packaging fallback. It reconstructs the trusted upstream
base image from its distribution layers, installs the same files and packages
as Dockerfile, and writes the legacy docker-save archive format accepted by
``docker load`` and Grand Challenge.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


LAYERS = (
    "41140838cd967104f816a16f4132acefcc18315b8585b5918eaed7dbf9e164c1",
    "de89bbc9f3e10d19a5bb25ac4ce9d55cc016c65b5ee60d7dedbfb35a5af45a95",
    "e050793e54358013cb36fae58970540b26aee8603a4e179d460ba87d117f8f12",
    "2afd86f86046354253c055d4e0c6310f160cc217cbee4633b82740f33cf91e5f",
    "7188d322ae797c2bca265d48e78d5f1c81edd057f4cd3c52aad290c85f41e37c",
    "6fcd771462b712c223b407a2db21d385f629f9b3e68f43237f547409d5eadb99",
)
TAG = "isles26-strokefusion-final:latest"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_member_path(name: str) -> PurePosixPath:
    while name.startswith("./"):
        name = name[2:]
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe archive member: {name}")
    return path


def remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.is_dir():
        shutil.rmtree(path)


def extract_layers(blob_dir: Path, rootfs: Path) -> None:
    rootfs.mkdir(parents=True, exist_ok=True)
    for digest in LAYERS:
        blob = blob_dir / f"{digest}.tar.gz"
        if sha256(blob) != digest:
            raise RuntimeError(f"bad distribution layer digest: {blob}")
        print(f"Extracting {blob.name}", flush=True)
        with tarfile.open(blob, mode="r:gz") as archive:
            members = archive.getmembers()
            for member in members:
                relative = safe_member_path(member.name)
                basename = relative.name
                if not basename.startswith(".wh."):
                    continue
                parent = rootfs.joinpath(*relative.parent.parts)
                if basename == ".wh..wh..opq":
                    if parent.is_dir():
                        for child in parent.iterdir():
                            remove_path(child)
                else:
                    remove_path(parent / basename.removeprefix(".wh."))
            normal_members = [
                member
                for member in members
                if not safe_member_path(member.name).name.startswith(".wh.")
            ]
            archive.extractall(rootfs, members=normal_members)


def configure_rootfs(rootfs: Path, context: Path) -> None:
    app = rootfs / "opt" / "app"
    app.mkdir(parents=True, exist_ok=True)
    for name in ("app.py", "inference.py", "requirements.txt"):
        shutil.copy2(context / name, app / name)
    shutil.copytree(
        context / "deploy_trainers",
        app / "deploy_trainers",
        dirs_exist_ok=True,
    )
    resolv_conf = rootfs / "etc" / "resolv.conf"
    remove_path(resolv_conf)
    shutil.copy2(Path("/etc/resolv.conf"), resolv_conf)

    setup = r"""
set -euo pipefail
getent group user >/dev/null || groupadd -r user
id -u user >/dev/null 2>&1 || useradd -m --no-log-init -r -g user user
/usr/bin/python3 -m pip install --break-system-packages --no-cache-dir --no-color --requirement /opt/app/requirements.txt
/usr/bin/python3 /opt/app/deploy_trainers/install.py
chown -R user:user /opt/app /home/user
rm -rf /root/.cache /tmp/*
"""
    subprocess.run(
        ["chroot", str(rootfs), "/bin/bash", "-lc", setup],
        check=True,
    )
    smoke = (
        "import torch, nnunetv2, SimpleITK, fastapi, uvicorn; "
        "from nnunetv2.training.nnUNetTrainer.nnUNetTrainerV12RASS "
        "import nnUNetTrainerV12RASS; "
        "from nnunetv2.training.nnUNetTrainer.nnUNetTrainerV15NativeRefinement "
        "import nnUNetTrainerV15NativeRefinement; "
        "print(torch.__version__, torch.version.cuda, nnunetv2.__file__)"
    )
    env = (
        "PATH=/usr/local/sbin:/usr/local/bin:"
        "/usr/sbin:/usr/bin:/sbin:/bin "
        "HOME=/tmp nnUNet_compile=false "
    )
    subprocess.run(
        [
            "chroot",
            "--userspec=user:user",
            str(rootfs),
            "/bin/bash",
            "-lc",
            f"{env}/usr/bin/python3 -c {json.dumps(smoke)}",
        ],
        check=True,
    )


def package_layer(layer_tar: Path, base_config: Path, staging: Path) -> Path:
    staging.mkdir(parents=True, exist_ok=True)
    layer_id = sha256(layer_tar)
    layer_dir = staging / layer_id
    layer_dir.mkdir()
    layer_tar.replace(layer_dir / "layer.tar")
    (layer_dir / "VERSION").write_text("1.0", encoding="utf-8")

    created = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    legacy = {
        "id": layer_id,
        "created": created,
        "container_config": {"Cmd": None},
    }
    (layer_dir / "json").write_text(json.dumps(legacy), encoding="utf-8")

    config = json.loads(base_config.read_text(encoding="utf-8"))
    image_config = config.setdefault("config", {})
    environment = {
        item.split("=", 1)[0]: item.split("=", 1)[1]
        for item in image_config.get("Env", [])
        if "=" in item
    }
    environment.update(
        {
            "PYTHONUNBUFFERED": "1",
            "nnUNet_compile": "false",
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "HOME": "/tmp",
            "XDG_CACHE_HOME": "/tmp/cache",
            "MPLCONFIGDIR": "/tmp/matplotlib",
            "TORCH_HOME": "/tmp/torch",
            "CUDA_MODULE_LOADING": "LAZY",
            "PATH": environment.get("PATH", ""),
        }
    )
    image_config.update(
        {
            "User": "user",
            "Env": [f"{key}={value}" for key, value in environment.items()],
            "WorkingDir": "/opt/app",
            "Entrypoint": ["python3", "app.py"],
            "Cmd": None,
            "ExposedPorts": {"4743/tcp": {}},
            "Labels": {"org.grand-challenge.api-method": "invoke"},
        }
    )
    config["created"] = created
    config["architecture"] = "amd64"
    config["os"] = "linux"
    config["rootfs"] = {"type": "layers", "diff_ids": [f"sha256:{layer_id}"]}
    config["history"] = [
        {"created": created, "created_by": "ISLES26 final squashed packaging"}
    ]
    config.pop("container_config", None)
    config_bytes = json.dumps(config, separators=(",", ":")).encode()
    config_id = hashlib.sha256(config_bytes).hexdigest()
    (staging / f"{config_id}.json").write_bytes(config_bytes)

    manifest = [
        {
            "Config": f"{config_id}.json",
            "RepoTags": [TAG],
            "Layers": [f"{layer_id}/layer.tar"],
        }
    ]
    (staging / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    repository, tag = TAG.rsplit(":", 1)
    repositories = {repository: {tag: layer_id}}
    (staging / "repositories").write_text(
        json.dumps(repositories), encoding="utf-8"
    )
    print(f"Layer diff ID: sha256:{layer_id}", flush=True)
    return layer_dir / "layer.tar"


def package_rootfs(rootfs: Path, base_config: Path, staging: Path) -> Path:
    staging.mkdir(parents=True, exist_ok=True)
    layer_tar = staging / "layer.tar"
    subprocess.run(
        [
            "tar",
            "--numeric-owner",
            "--xattrs",
            "--acls",
            "-C",
            str(rootfs),
            "-cf",
            str(layer_tar),
            ".",
        ],
        check=True,
    )
    return package_layer(layer_tar, base_config, staging)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    extract = subparsers.add_parser("extract")
    extract.add_argument("blob_dir", type=Path)
    extract.add_argument("rootfs", type=Path)
    configure = subparsers.add_parser("configure")
    configure.add_argument("rootfs", type=Path)
    configure.add_argument("context", type=Path)
    package = subparsers.add_parser("package")
    package.add_argument("rootfs", type=Path)
    package.add_argument("base_config", type=Path)
    package.add_argument("staging", type=Path)
    package_layer_parser = subparsers.add_parser("package-layer")
    package_layer_parser.add_argument("layer_tar", type=Path)
    package_layer_parser.add_argument("base_config", type=Path)
    package_layer_parser.add_argument("staging", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "extract":
        extract_layers(args.blob_dir.resolve(), args.rootfs.resolve())
    elif args.command == "configure":
        configure_rootfs(args.rootfs.resolve(), args.context.resolve())
    elif args.command == "package":
        package_rootfs(
            args.rootfs.resolve(), args.base_config.resolve(), args.staging.resolve()
        )
    else:
        package_layer(
            args.layer_tar.resolve(),
            args.base_config.resolve(),
            args.staging.resolve(),
        )


if __name__ == "__main__":
    main()
