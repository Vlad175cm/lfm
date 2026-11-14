"""Monochrome code highlighting for markdown fences.

Textual's markdown widget picks its own colourful Pygments theme. The whole UI
is black and white, so we swap in a theme built only from white, grey and
weight.
"""

from __future__ import annotations

from pygments.token import Token
from textual.highlight import HighlightTheme
from textual.widgets._markdown import MarkdownFence


class MonoHighlight(HighlightTheme):
    STYLES = {
        Token.Comment: "italic dim",
        Token.Error: "bold",
        Token.Generic.Heading: "bold underline",
        Token.Generic.Subheading: "bold",
        Token.Generic.Strong: "bold",
        Token.Generic.Emph: "italic",
        Token.Keyword: "",
        Token.Keyword.Type: "",
        Token.Name.Builtin: "",
        Token.Name.Class: "",
        Token.Name.Function: "",
        Token.Operator: "",
        Token.Operator.Word: "",
    }


def _mono_highlight(cls, code, language, ansi=False, dark=False):
    from textual.highlight import highlight

    return highlight(code, language=language or None, theme=MonoHighlight)


def install() -> None:
    MarkdownFence.highlight = classmethod(_mono_highlight)
