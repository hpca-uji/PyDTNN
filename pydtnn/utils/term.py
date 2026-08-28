"""Fancy terminal support"""
import sys

__all__ = (
    "FANCY",
    "BOLD", "RESET",
    "BOX_TL", "BOX_TR", "BOX_R", "BOX_L",
    "BOX_T", "BOX_R", "BOX_B", "BOX_L",
    "BOX_C", "BOX_H", "BOX_V"
)


try:
    from _colorize import can_colorize  # pyright: ignore[reportMissingTypeStubs]
except Exception:
    FANCY = sys.stderr.isatty()  # pyright: ignore[reportConstantRedefinition]
else:
    FANCY = can_colorize(file=sys.stderr)  # pyright: ignore[reportConstantRedefinition]


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
