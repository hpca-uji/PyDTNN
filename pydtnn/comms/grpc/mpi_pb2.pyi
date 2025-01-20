from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Op(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    BCAST: _ClassVar[Op]
    ALLGATHER: _ClassVar[Op]
    ALLREDUCE: _ClassVar[Op]
BCAST: Op
ALLGATHER: Op
ALLREDUCE: Op

class SendRequest(_message.Message):
    __slots__ = ("rank", "data")
    RANK_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    rank: int
    data: bytes
    def __init__(self, rank: _Optional[int] = ..., data: _Optional[bytes] = ...) -> None: ...

class SendResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class RecvRequest(_message.Message):
    __slots__ = ("rank", "size", "op")
    RANK_FIELD_NUMBER: _ClassVar[int]
    SIZE_FIELD_NUMBER: _ClassVar[int]
    OP_FIELD_NUMBER: _ClassVar[int]
    rank: int
    size: int
    op: Op
    def __init__(self, rank: _Optional[int] = ..., size: _Optional[int] = ..., op: _Optional[_Union[Op, str]] = ...) -> None: ...

class RecvResponse(_message.Message):
    __slots__ = ("data",)
    DATA_FIELD_NUMBER: _ClassVar[int]
    data: bytes
    def __init__(self, data: _Optional[bytes] = ...) -> None: ...
