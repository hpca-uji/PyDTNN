"""C-based extension implementing fast integer bit sets."""

from collections import abc as _abc


class intbitset(set[int]):
    """Data object holding unordered sets of unsigned integers with ultra fast set operations"""

    def __init__(self, rhs: int | "intbitset" | bytes | _abc.Sequence = 0, preallocate: int = -1, trailing_bits: int = 0) -> None:
        ...

    def extract_finite_list(self) -> _abc.Sequence:
        """
        Return a finite list of elements sufficient to be passed to intbitset constructor
        toghether with the proper value of trailing_bits in order to reproduce this intbitset.
        At least up_to integer are looked for when they are inside the intbitset but not
        necessarily needed to build the intbitset
        """

    def fastdump(self) -> bytes:
        """
        Return a compressed string representation suitable to be saved somewhere.
        """

    def fastload(self, data: bytes, /) -> None:
        """
        Load a compressed string representation produced by a previous call to the
        fastdump method into the current intbitset. The previous content will be replaced.
        """

    def get_allocated(self) -> int:
        ...

    def get_size(self) -> int:
        ...

    def get_wordbitsize(self) -> int:
        ...

    def get_wordbytsize(self) -> int:
        ...

    def strbits(self) -> str:
        """
        Return a string of 0s and 1s representing the content in memory of the intbitset.
        """

    def update_with_signs(self, rhs: _abc.Mapping[int, int]) -> None:
        """
        Given a dictionary rhs whose keys are integers, remove all the integers whose value
        are less than 0 and add every integer whose value is 0 or more
        """
