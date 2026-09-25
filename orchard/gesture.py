"""Gestures: the pre-linguistic channel that scaffolds the linguistic one.

Why this exists
---------------
Humans do not learn words from words alone. A parent points at the apple while
saying "apple"; a trader holds up three fingers while naming a quantity; a child
learning a new word assumes it names one *kind* of thing -- an object, a
property, a number -- and not an arbitrary conjunction of them. None of that is
language, and all of it is what makes language learnable: the gesture puts the
referent and the word in front of the listener at the same moment, and the
innate categories tell the listener what sort of thing the word is about.

The run this answers had passed fruit, colour and quality and stalled on
quantity for 850 updates at exactly chance (0.32-0.34 against 0.333). In the
quantity rounds the describer emitted one arbitrary atom -- `a15`, `a25`,
`a24` -- and the guesser's reading of the message was random, so the
straight-through gradient reaching the speaker had no consistent direction:
nine number-words had to break symmetry from nothing, through a listener whose
readout of the channel was itself untrained. That is the problem an iconic
gesture solves. Fingers for three mean three to any listener, so the listener's
*readout* of the dialogue -- "this slot says something about quantity, and it
says 3" -- gets trained on a signal that carries; and because the words a
speaker emits go through the very same readout, the gradient on the words now
has somewhere to point: towards whatever the listener already reads as "3".

What a gesture is
-----------------
A gesture is an extra symbol a speaker may put at the **start of its turn**,
occupying the turn's first dialogue slot. It is one of a fixed inventory whose
meanings are given by the world, never learned:

    fingers      one gesture per quantity 0..max_qty, and one per price bin
    pointing     one per fruit, per colour and per quality -- pointing at an
                 exemplar in the shared surroundings

So a gesture is ``(field, value)`` for one field of a lot, rendered as
``[3 fingers]`` or ``[points: APPLE]``. It is **truthful by construction**: the
value shown is read from the speaker's own observation
(:func:`gesture_columns`), never from the scenario, so a gesture can only ever
reveal something the speaker itself can see. A farmer at market can show its
floor price; it cannot show "how many of what you asked for", because that
would be the world telling the buyer what the farmer had not understood.

Who decides
-----------
Two parties. The **world** decides whether gesturing is possible in a given
round at all (:func:`gesture_share`): a share of rounds, high at the start of a
rung that is inventing words and withdrawn over the rung, so the words have to
carry the meaning on their own before the rung can be left -- and a small
standing share in every later rung, because traders do hold up fingers. The
**speaker** decides, in a round where it can, whether to gesture and which field
(:attr:`orchard.agents.CommNet.gesture_head`, sampled and trained by
REINFORCE like the decisions), and pays ``gesture.cost`` for doing so. Nothing
ever forces a gesture, and nothing forbids one where the world allows it.

What a gesture is not
---------------------
It is not a word, and nothing in the language is measured on it. Every
promotion gate, every probe and every ablation runs word-only
(:func:`orchard.rollout.run_episodes` never emits one), so a rung is left only
when the *words* carry what the gestures used to. Gestures are not parsed as
words, not costed as symbols, not counted in utterance length, and a newborn is
never taught to emit one (the slot is masked from its token lesson) -- though it
does learn to read them, like everyone else.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch

from .config import Config
from .world import LOT_FIELDS, LOT_KINDS, N_LOT_FIELDS, lot_spans, n_cells

GESTURE_NONE = 0                  # the gesture head's "say nothing with your hands"


def n_gesture_ids(cfg: Config) -> int:
    """One id per (field, value) of a lot: fingers for numbers, pointing for the rest."""
    return int(sum(lot_spans(cfg.world)))


def gesture_offsets(cfg: Config) -> list[int]:
    """Where each field's gesture ids start, relative to the first gesture id."""
    out, at = [], 0
    for span in lot_spans(cfg.world):
        out.append(at)
        at += span
    return out


def first_gesture_id(cfg: Config) -> int:
    """Gesture ids follow the symbol ids (atoms, HYPHEN, SPACE, END, PAD)."""
    return cfg.channel.n_symbol_ids


def n_token_ids(cfg: Config) -> int:
    """Size of the token embedding table: every symbol, plus every gesture.

    The table always has room for the gestures, whether or not ``gesture.enabled``
    -- so switching them off is a change of method, not of architecture, and a
    snapshot made with them on loads with them off.
    """
    return cfg.channel.n_symbol_ids + n_gesture_ids(cfg)


def gesture_id(cfg: Config, field: int, value: int) -> int:
    return first_gesture_id(cfg) + gesture_offsets(cfg)[int(field)] + int(value)


def is_gesture(cfg: Config, tok: int) -> bool:
    lo = first_gesture_id(cfg)
    return lo <= int(tok) < lo + n_gesture_ids(cfg)


def gesture_meaning(cfg: Config, tok: int) -> Optional[tuple[int, int]]:
    """``(field, value)`` of a gesture id, or None for anything else."""
    if not is_gesture(cfg, tok):
        return None
    rel = int(tok) - first_gesture_id(cfg)
    for field, (start, span) in enumerate(zip(gesture_offsets(cfg), lot_spans(cfg.world))):
        if start <= rel < start + span:
            return field, rel - start
    return None


def gesture_id_tensor(cfg: Config, field: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Vectorised :func:`gesture_id` for (B,) fields and values."""
    offs = torch.tensor(gesture_offsets(cfg), dtype=torch.long, device=field.device)
    return first_gesture_id(cfg) + offs[field] + value


def gesture_text(cfg: Config, tok: int) -> str:
    """How a transcript shows a gesture. Iconic, so it is named in plain words."""
    m = gesture_meaning(cfg, tok)
    if m is None:
        return ""
    field, v = m
    w = cfg.world
    if field == 3:
        return "[%d finger%s]" % (v, "" if v == 1 else "s")
    if field == 4:
        return "[%d finger%s for @%.2f]" % (v + 1, "" if v == 0 else "s", w.price_values[v])
    names = (w.variety_names, w.color_names, w.quality_names)[field]
    return "[points: %s]" % names[v]


def strip_gestures(cfg: Config, symbols: Sequence[int]) -> list[int]:
    """The spoken part of a turn: everything that is not a gesture."""
    lo = first_gesture_id(cfg)
    hi = lo + n_gesture_ids(cfg)
    return [int(s) for s in symbols if not (lo <= int(s) < hi)]


# --------------------------------------------------------------------------
# what a speaker may show, and from where
# --------------------------------------------------------------------------
def gesture_columns(cfg: Config, phase, role: int) -> list[tuple[int, int]]:
    """``[(field, observation column)]`` a speaker in this seat may gesture.

    The value shown is read off the speaker's *own observation* at that column
    and nowhere else, which is what keeps a gesture honest: it can reveal
    exactly what its maker can see. A describer in the naming rungs, either
    party in `mutual` and the buyer at market all look at a lot, so all five
    fields are available. The farmer at market looks at a barn, and the one
    fact of its own it can hold up fingers for is its floor price; the stock or
    quality of "the lot you asked for" depends on having understood the
    request, and a gesture the world computed for it would be the world doing
    the understanding.
    """
    from .env import FARMER
    from .world import K_PRICE
    if getattr(phase, "tuples", False) or role != FARMER:
        return [(j, j) for j in range(N_LOT_FIELDS)]
    # the barn: rows of (fruit, colour, quality, stock), then the floor price
    col = 4 * n_cells(cfg.world)
    return [(LOT_KINDS.index(K_PRICE), col)]


def gesture_option_mask(cfg: Config, phase, role: int, device=None) -> torch.Tensor:
    """(1 + N_LOT_FIELDS,) bool: which of the gesture head's options this seat has."""
    m = torch.zeros(1 + N_LOT_FIELDS, dtype=torch.bool)
    m[GESTURE_NONE] = True
    for field, _ in gesture_columns(cfg, phase, role):
        m[1 + field] = True
    return m if device is None else m.to(device)


def gesture_tokens_for(cfg: Config, phase, role: int, obs: torch.Tensor,
                       choice: torch.Tensor) -> torch.Tensor:
    """(B,) the gesture id each speaker's choice amounts to; PAD where none.

    ``choice`` is the gesture head's sample: 0 for none, ``1 + field`` otherwise.
    A field this seat cannot gesture (the mask should have prevented it) is
    treated as none.
    """
    B = obs.shape[0]
    out = torch.full((B,), cfg.channel.pad_id, dtype=torch.long, device=obs.device)
    for field, col in gesture_columns(cfg, phase, role):
        sel = choice == (1 + field)
        if bool(sel.any()):
            val = obs[:, col].clamp(0, lot_spans(cfg.world)[field] - 1)
            ids = gesture_id_tensor(cfg, torch.full_like(val, field), val)
            out = torch.where(sel, ids, out)
    return out


# --------------------------------------------------------------------------
# when the world allows it
# --------------------------------------------------------------------------
def gesture_share(cfg: Config, phase, updates_in_phase: int) -> float:
    """The share of rounds in which gesturing is possible, right now.

    High at the start of a rung that still has a word to invent and withdrawn
    over ``gesture.anneal_updates`` -- a parent stops pointing once the child
    has the word, and a rung is judged word-only in any case, so a word that
    never took over from the gesture fails the gate as before. In every rung
    that only reuses words a small standing share stays: fingers are part of a
    market, and whether the speakers still bother with them once the words
    work is something to measure.
    """
    g = cfg.gesture
    if not g.enabled:
        return 0.0
    if getattr(phase, "invents", False):
        from .rollout import anneal
        return float(anneal(g.share_start, g.share_end, updates_in_phase,
                            g.anneal_updates))
    return float(g.share_reuse)


def draw_availability(share: float, n: int, device, generator=None) -> torch.Tensor:
    """(n,) bool: in which rounds of this batch gesturing is possible."""
    if share <= 0:
        return torch.zeros(n, dtype=torch.bool, device=device)
    if share >= 1:
        return torch.ones(n, dtype=torch.bool, device=device)
    return torch.rand(n, device=device, generator=generator) < share


# --------------------------------------------------------------------------
# what the listener is told by a gesture
# --------------------------------------------------------------------------
def gestured_fields(cfg: Config, tokens: torch.Tensor, turn_starts: Sequence[int]
                    ) -> tuple[torch.Tensor, torch.Tensor]:
    """For each episode: the field the other party gestured (-1 if none) and its value.

    Read off the first slot of each of the other party's turns; if it gestured
    in more than one turn the last one wins (one lesson per round is enough).
    """
    B = tokens.shape[0]
    field = torch.full((B,), -1, dtype=torch.long, device=tokens.device)
    value = torch.zeros(B, dtype=torch.long, device=tokens.device)
    lo = first_gesture_id(cfg)
    offs = torch.tensor(gesture_offsets(cfg), dtype=torch.long, device=tokens.device)
    spans = torch.tensor(lot_spans(cfg.world), dtype=torch.long, device=tokens.device)
    bounds = offs + spans
    for p in turn_starts:
        tok = tokens[:, p]
        rel = tok - lo
        is_g = (rel >= 0) & (rel < int(bounds[-1]))
        if not bool(is_g.any()):
            continue
        # which field's range the id falls in
        f = (rel.unsqueeze(1) >= offs.unsqueeze(0)).sum(1) - 1
        f = f.clamp(0, N_LOT_FIELDS - 1)
        v = rel - offs[f]
        field = torch.where(is_g, f, field)
        value = torch.where(is_g, v, value)
    return field, value


def field_name(field: int) -> str:
    return LOT_FIELDS[int(field)]
