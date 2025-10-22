import typing
import tarfile
from collections import abc
from pathlib import Path, PurePath
from contextlib import contextmanager, ExitStack
import itertools

def is_tar(path: PurePath, suffixes={(".tar",), (".tar", ".gz"), (".tgz",)}) -> bool:
    """Does the path look like a TAR"""
    return tuple(path.suffixes) in suffixes

def list_directory(root_path: Path) -> typing.Iterator[tuple[str, ...]]:
    iter_archives = list[typing.Iterator[tuple[str, ...]]]()
    for file in root_path.iterdir():
        iter_archives.append(list_archive(file))
    yield itertools.chain(iter_archives)
# ---

def list_archive(root_path: Path) -> typing.Iterator[tuple[str, ...]]:
    """Recursive TAR walk"""
    path = str(root_path)

    if not is_tar(root_path):
        yield (path,)
        return

    with ExitStack() as fp_stack:
        stack: list[tuple[tarfile.TarFile, tuple[str, ...]]] = []

        tar = fp_stack.enter_context(tarfile.open(path, "r"))
        stack.append((tar, (path,)))

        while stack:
            tar, base_path = stack.pop()

            for member in tar.getmembers():
                if not member.isfile():
                    continue

                path = Path(member.name)
                full_path = (*base_path, str(path))

                if is_tar(path):
                    sub_file = fp_stack.enter_context(tar.extractfile(member))
                    sub_tar = fp_stack.enter_context(tarfile.open(fileobj=sub_file, mode="r"))
                    stack.append((sub_tar, full_path))
                else:
                    yield full_path


@contextmanager
def load_archive(*paths: str) -> abc.Generator[typing.IO[bytes]]:
    """Recursive TAR loader"""
    with ExitStack() as stack:
        # First: on disk
        file = stack.enter_context(open(paths[0], "rb"))

        if len(paths) <= 1:
            yield file
            return

        tar = stack.enter_context(tarfile.open(fileobj=file))

        # Intermediate: nested tars
        for fp in paths[1:-1]:
            file = stack.enter_context(tar.extractfile(fp))
            tar = stack.enter_context(tarfile.open(fileobj=file))

        # Last: return
        file = stack.enter_context(tar.extractfile(paths[-1]))
        yield file

