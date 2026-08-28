"""Fancy terminal support"""
import sys

__all__ = (
    "FANCY",
    "BOLD", "RESET",
    "BOX_TL", "BOX_TR", "BOX_R", "BOX_L",
    "BOX_T", "BOX_R", "BOX_B", "BOX_L",
    "BOX_C", "BOX_H", "BOX_V"
)


FANCY = sys.stderr.isatty()  # pyright: ignore[reportConstantRedefinition]
try:
    from _colorize import can_colorize  # pyright: ignore[reportMissingTypeStubs]
    FANCY = can_colorize()  # pyright: ignore[reportConstantRedefinition]
    FANCY = can_colorize(file=sys.stderr)  # pyright: ignore[reportConstantRedefinition]
except Exception:
    pass


# Characters
if FANCY:
    RESET, BOLD, COLOR = "\x1b[0m", "\x1b[1m", "\x1b[34m"  # pyright: ignore[reportConstantRedefinition]
    BOX_TL, BOX_TR, BOX_BR, BOX_BL = "┌", "┐", "┘", "└"  # pyright: ignore[reportConstantRedefinition]
    BOX_T, BOX_R, BOX_B, BOX_L = "┬", "┤", "┴", "├"  # pyright: ignore[reportConstantRedefinition]
    BOX_C, BOX_H, BOX_V = "┼", "─", "│"  # pyright: ignore[reportConstantRedefinition]
else:
    RESET = BOLD = COLOR = ""  # pyright: ignore[reportConstantRedefinition]
    BOX_TL = BOX_TR = BOX_BR = BOX_BL = "+"  # pyright: ignore[reportConstantRedefinition]
    BOX_T = BOX_R = BOX_B = BOX_L = "+"  # pyright: ignore[reportConstantRedefinition]
    BOX_C, BOX_H, BOX_V = "+", "-", "|"  # pyright: ignore[reportConstantRedefinition]
