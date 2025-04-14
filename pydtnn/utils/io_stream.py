"""Streaming IO"""

# NOTE: Packed format:
# - Network byte order (big-endian)
# - Size only includes stream (not itself)
# - Negative size signifies ancillary stream
#
# +---------------+------------------+
# | Size (int64) | Stream (variable) |
# +---------------+------------------+

# TODO: Use intvar (VLQ) insted of int64 in packer

import io
import pickle
import struct
from collections import abc, deque


__all__ = (
    "Stream",
    "PackerStream",
    "AncillaryStream",
    "StreamSerializer"
)


def byteview(b: abc.Buffer) -> memoryview:
    """Return a byte view of a buffer"""
    with memoryview(b) as view:
        return view.cast("B")


# Fast path
try:
    from pydtnn.cython_modules import memoryview_index

# Slow path
except ImportError:
    def memoryview_index(view: memoryview, sub: bytes) -> int:
        """Find lowest index where substring is found"""
        if len(sub) != 1:
            raise TypeError("Only single byte substring are supported")
        for i, byte in enumerate(view):
            if byte == sub:
                return i
        else:
            raise ValueError("Substring not found")


class Stream(io.BufferedIOBase):
    """
    Zero-copy non-blocking pipe-like

    Interface mimics a non-blocking BufferedRWPair,
    but operations return memoryviews insted of bytes.

    Operations are not thread-safe.
    Reader is responsible of releasing chunks.
    """

    __slots__ = ("_chunks",)

    def __init__(self):
        """Initialize stream"""
        self._chunks = deque[memoryview]()

    # stream methods
    def empty(self) -> bool:
        """Is stream empty (would read block)"""
        return len(self._chunks) <= 0

    @property
    def nbytes(self) -> int:
        """Number of bytes held in stream"""
        return sum(map(len, self._chunks))

    def unreadchunk(self, chunk: memoryview) -> int:
        """Unread a chunk from the steam"""
        size = len(chunk)
        if size > 0:
            self._chunks.appendleft(chunk)
        else:
            chunk.release()
        return size

    def readchunk(self) -> memoryview:
        """Read a chunks from stream"""
        if self.empty():
            raise BlockingIOError()

        return self._chunks.popleft()

    def readchunks(self) -> abc.Iterable[memoryview]:
        """Read all chunks from steam"""
        while not self.empty():
            yield self.readchunk()

    def writechunk(self, chunk: memoryview) -> int:
        """Write a chunk into the steam"""
        size = len(chunk)
        if size > 0:
            self._chunks.append(chunk)
        else:
            chunk.release()
        return size

    def writechunks(self, chunks: abc.Iterable[memoryview], /) -> int:
        """Write many chunks into the steam"""
        size = 0
        for chunk in chunks:
            size += self.writechunk(chunk)
        return size

    def copy(self):
        """Shallow copy of stream"""
        other = self.__class__()
        for chunk in self._chunks:
            other.write(chunk)
        return other

    def __copy__(self):
        """Shallow copy of stream"""
        return self.copy()

    # io methods
    def write(self, b: abc.Buffer) -> int:
        """Inserts buffer into stream"""
        chunk = byteview(b)
        size = self.writechunk(chunk)
        return size

    def readline(self) -> memoryview:
        """Read a line and return a memoryview (may copy)"""
        with Stream() as stream:
            for chunk in self.readchunks():
                try:
                    i = memoryview_index(chunk, b"\n")
                except ValueError:
                    stream.write(chunk)
                    continue
                else:
                    self.unreadchunk(chunk)
                    stream.write(self.read1(i))
                    break
            return stream.read()

    def read1(self, size: int = -1, /) -> memoryview:
        """Reads, with at most one operation, and returns a memoryview"""
        if self.empty():
            raise BlockingIOError()

        chunk = self.readchunk()

        if size < 0 or size >= len(chunk):
            return chunk

        with chunk:
            read, keep = chunk[:size], chunk[size:]
        self.unreadchunk(keep)
        return read

    def readinto1(self, b: abc.Buffer, /) -> int:
        """Reads, with at most one operation, into a buffer"""
        if self.empty():
            raise BlockingIOError()

        with byteview(b) as view:
            chunk = self.read1(len(view))
            size = len(chunk)

            view[:size] = chunk

        return size

    def readinto(self, b: abc.Buffer, /) -> int:
        """Reads, until drained, into a buffer"""
        if self.empty():
            raise BlockingIOError()

        read = 0
        with byteview(b) as view:
            while not self.empty() and read < len(view):
                read += self.readinto1(view[read:])

        return read

    def read(self, size: int = -1, /) -> memoryview:
        """Reads, until drained, and returns a memoryview (may copy)"""
        if self.empty():
            raise BlockingIOError()

        if size >= 0:
            size = min(size, self.nbytes)
        else:
            size = self.nbytes

        # View path
        if len(self._chunks[0]) >= size:
            return self.read1(size)

        # Copy path
        buffer = bytearray(size)
        self.readinto(buffer)
        return byteview(buffer)

    def __del__(self) -> None:
        """Best effort finalizer"""
        try:
            self.close()
        except:  # noqa: E722
            pass

    def close(self) -> None:
        """Release stream chunks"""
        for chunk in self._chunks:
            chunk.release()
        self._chunks.clear()

    # io stubs
    def __iter__(self) -> abc.Iterable[memoryview]:
        return self.readlines()

    def fileno(self) -> int:
        raise OSError()

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return False

    def readable(self) -> bool:
        return True

    def readlines(self):
        while not self.empty():
            yield self.readline()

    def seek(self, offset, whence=io.SEEK_SET, /) -> int:
        raise OSError()

    def seekable(self) -> bool:
        return False

    def tell(self) -> int:
        raise OSError()

    def truncate(self, size=None, /) -> int:
        raise OSError()

    def writable(self) -> bool:
        return True

    def writelines(self, lines: abc.Iterable[abc.Buffer], /) -> None:
        for line in lines:
            self.write(line)

    def detach(self) -> abc.Buffer:
        raise io.UnsupportedOperation()


class AncillaryStream(BlockingIOError):
    """Blocked with ancillary stream (no normal stream available)"""

    __slots__ = ("stream",)

    def __init__(self, stream: Stream, *args: object) -> None:
        """Inizialize ancillary error"""
        self.stream = stream
        super().__init__(*args)


class PackerStream(Stream):
    """
    Packer stream

    Packs or unpacks multiple streams into one.
    Supports ancillary streams for control data.

    Operations are not thread-safe.
    Unpacks of ancillary raise AncillaryStream.
    """

    __slots__ = ()
    _format_size = "!q"
    _sizeof_size = struct.calcsize(_format_size)

    def unpack(self) -> Stream:
        """Extracts stream from packer (raises BlockingIOError if no stream)"""
        # Check if size available
        size = self._sizeof_size
        if self.nbytes < size:
            raise BlockingIOError()

        # Check if data available
        chunk = self.read(size)
        size = struct.unpack(self._format_size, chunk)[0]
        size, ancillary = abs(size), size < 0
        if self.nbytes < size:
            self.unreadchunk(chunk)
            raise BlockingIOError()
        else:
            chunk.release()

        # Ensure contained reads
        stream = Stream()
        while size > 0:
            chunk = self.read1(size)
            stream.writechunk(chunk)
            size -= len(chunk)

        # Return stream (or ancillary stream)
        if ancillary:
            raise AncillaryStream(stream)
        return stream

    def pack(self, stream: Stream, ancillary: bool = False) -> int:
        """Inserts stream into packer, returns bytes written"""
        # Ensure contained writes
        size = stream.nbytes
        pack = struct.pack(self._format_size, -size if ancillary else size)
        size += self.write(pack)
        self.writechunks(stream.readchunks())
        return size


class StreamSerializer:
    """Pickle-stream serializer"""

    __slots__ = ()

    def dump(self, obj) -> Stream:
        """Transform a object into a stream"""
        stream = Stream()
        pickle.dump(obj=obj, file=stream, protocol=5)
        return stream

    def load(self, stream: Stream):
        """Transform a stream into a object"""
        return pickle.load(stream)
