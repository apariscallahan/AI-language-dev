"""Agent brains: small, randomly-initialised causal transformers.

NO PRETRAINING ANYWHERE.  Every weight in this file is initialised by PyTorch's
default random init at the moment an agent is born.  There is no text corpus, no
tokenizer trained on language, no pretrained embedding table.  The "vocabulary"
is ``range(vocab_size)`` -- integer ids into a randomly-initialised lookup table
whose only source of meaning is the reinforcement signal from trading.

Sequence layout
---------------
Every agent, both roles, sees one fixed-length sequence::

    idx 0            BOS (+ role embedding)
    idx 1..N         its private observation, one slot per position (N is the
                     shared layout's width -- the barn needs the most room)
    idx N+1          SEP
    idx N+2..N+1+D   the dialogue, D = n_turns * max_msg_len, PAD where unspoken
    idx N+2+D        DECIDE

Every observation position carries a *field-kind* embedding, a *position*
embedding and a *value* embedding, so the network knows which number is which.
A thing to talk about is always a lot -- (fruit, colour, quality, quantity,
price) in that order, in slots 1..5 -- whether it is the lot a naming round
asks about or the request a buyer brings to market; a barn is a list of lot
rows. The dialogue positions carry a token embedding plus a self/other speaker
embedding -- an agent therefore always knows which words were its own.

The model is causal (masked self-attention), which makes two things identical:

* rollout, where we feed the prefix that exists so far and read the last
  position, and
* the update, where we feed the whole finished episode *once* and read the
  logits at every position the agent spoke from.

That equivalence is what makes training affordable: one forward per agent per
batch in the backward pass instead of one per emitted token.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config
from .env import BUYER, FARMER, speaker_of_turn
from .gesture import gesture_offsets, n_token_ids
from .world import (K_COLOR, K_EMPTY, K_FIELD, K_PRICE, K_QTY, K_QUALITY, K_VARIETY,
                    LOT_KINDS, N_LOT_FIELDS, QUERY_ALL, lot_spans, n_cells, n_obs_slots,
                    obs_schema)

# Innate concepts (`model.innate_concepts`): the kind of thing each field of a lot
# is. A fruit is an object kind, a colour or a quality a property, a quantity or
# a price a magnitude -- the semantic categories that nouns, adjectives and
# numerals are about. Keyed by field kind; a slot that holds no attribute of a
# lot (the query, padding) has no entry.
CONCEPT_OBJECT, CONCEPT_PROPERTY, CONCEPT_MAGNITUDE = 0, 1, 2
CONCEPT_OF_KIND = {K_VARIETY: CONCEPT_OBJECT, K_COLOR: CONCEPT_PROPERTY,
                   K_QUALITY: CONCEPT_PROPERTY, K_QTY: CONCEPT_MAGNITUDE,
                   K_PRICE: CONCEPT_MAGNITUDE}
# The word class that names each field of a lot, in lot order: the fruit is
# named by a noun, colour and quality by adjectives, quantity and price by
# numerals.
WORD_CLASSES = ("noun", "adjective", "numeral", "none")
CLASS_OF_FIELD = (0, 1, 1, 2, 2)


def thermometer(values: torch.Tensor, n_values: int) -> torch.Tensor:
    """(...,) ints -> (..., n_values - 1) floats: [v >= 1, v >= 2, ...].

    The innate number line's code: two magnitudes share one feature for every
    step they have in common, so neighbours are alike and extremes are not.
    """
    steps = torch.arange(1, n_values, device=values.device)
    return (values.unsqueeze(-1) >= steps).float()


class LexicalSpeaker(nn.Module):
    """Production through a mental lexicon: pick a part of the thing, say its word.

    The production half of the language faculty (`model.lexical_speaker`), and
    the reader's counterpart. When a speaker is describing a lot it can see,
    its token logits get a second term: the hidden state chooses which of the
    lot's five parts it is naming now -- an attention over the parts' concepts,
    each built from the field's value and kind alone -- and one shared output
    layer turns that concept into atoms. That layer never sees the context, so
    the word for red is the same word whether red is asked about on its own or
    is the second thing said about a red apple: a word learned alone is
    available in company, the production twin of the reader's out-of-context
    lookup (Levelt's lexical access).

    Why it exists: measured on 2026-09-30, speakers taught a one-word dialect
    for every field value (each field named alone at 0.99-1.00) and put into
    `name-all` grew descriptions to 2.5 words and then cut them back to one --
    the fruit and nothing else -- within 100 updates, reusing their own words
    for 11% of fields. A second word came from the transformer's token head,
    for which "the colour word, second, in a whole-lot round" was a new
    context: it was a random atom, usually some other value's name, charged by
    the composition term and misread by the listener, so continuing was
    punished and the speakers learned to stop. With the lexicon, whatever part
    the speaker attends to, the word that comes out is that part's word.

    What stays learned: which atoms name which value (the output layer), which
    part to name first and next (the attention), how many words to say and
    when to stop (the token head's END), and every structural choice. Where
    the observation is not a lot -- a barn, a lineup -- the term is absent.
    """

    # How hard a speaker describing a whole thing is pushed on past a word while
    # parts of it are still unnamed, in nats taken from ending and given to
    # going on. Learned from there; see `CommNet.describing`. Measured with two
    # founders drilled on "a word, then stop": at 3.0 one founder went on 99%
    # of the time and the other 10%, so its listener never learned its longer
    # descriptions and it stayed at one word for 100 updates; when both did go
    # on, their token heads had learned it so hard that descriptions ran to 8.5
    # words. At 5.0 both said exactly five words, one per part, by update 50.
    GO_ON_INIT = 5.0
    # How strongly a part already named in this turn is passed over when
    # choosing the part to name next (inhibition of return), in attention
    # logits. Learned from there.
    INHIBIT_INIT = 4.0

    def __init__(self, cfg: Config, d: int):
        super().__init__()
        self.atomic_vocab = cfg.channel.atomic_vocab
        self.n_emittable = cfg.channel.n_emittable
        self.query = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)
        self.say = nn.Linear(d, self.atomic_vocab)
        self.gain = nn.Parameter(torch.ones(()))
        self.go_on = nn.Parameter(torch.tensor(self.GO_ON_INIT))
        self.inhibit = nn.Parameter(torch.tensor(self.INHIBIT_INIT))
        push = torch.zeros(self.n_emittable)
        push[cfg.channel.end_id] = -1.0
        push[cfg.channel.space_id] = 1.0
        self.register_buffer("_push", push, persistent=False)

    @torch.no_grad()
    def part_words(self, concepts: torch.Tensor) -> torch.Tensor:
        """(B, 5): the atom this speaker says for each part of the lot -- what its
        lexicon produces when it attends to that part alone."""
        return self.say(self.norm(concepts)).argmax(-1)

    def forward(self, h: torch.Tensor, concepts: torch.Tensor,
                said: Optional[torch.Tensor] = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """``h`` (B, d) or (B, K, d) hidden states, ``concepts`` (B, 5, d);
        ``said`` ((B, 5) or (B, K, 5)) the parts already named in this turn.

        Returns token-logit contributions (h's shape with n_emittable last;
        zero on everything but atoms) and the attention over the five parts.
        A part already named is passed over when choosing the next (``inhibit``):
        made to go on without it, speakers named a second part and then said
        its word again until the turn's cap -- `a11 a28 a28 a28 a28`.
        """
        squeeze = h.dim() == 2
        if squeeze:
            h = h.unsqueeze(1)                                           # (B, 1, d)
        q = self.query(h)                                                # (B, K, d)
        scores = torch.einsum("bkd,bfd->bkf", q, concepts) / math.sqrt(q.shape[-1])
        if said is not None:
            said = said.to(scores.dtype)
            scores = scores - self.inhibit * (said.unsqueeze(1) if said.dim() == 2 else said)
        att = torch.softmax(scores, dim=-1)                              # (B, K, 5)
        mix = torch.einsum("bkf,bfd->bkd", att, concepts)                 # (B, K, d)
        atoms = self.gain * self.say(self.norm(mix))                      # (B, K, A)
        out = F.pad(atoms, (0, self.n_emittable - self.atomic_vocab))
        if squeeze:
            return out.squeeze(1), att.squeeze(1)
        return out, att


class LexicalReader(nn.Module):
    """Comprehension through a mental lexicon: words in, attributes out.

    The innate half of the language faculty on the listening side
    (`model.lexical_reader`). What is innate is the *architecture of
    understanding*; everything it maps is learned:

    1. **Words are units.** The other party's turn is segmented where the
       medium segments it -- a word is a run of atoms joined by HYPHENs --
       and each word is encoded from its own atoms alone, in order, with no
       context. So a word means the same thing in every utterance it appears
       in: learned alone ("apple") it is understood in company ("two red
       apple"). This is also where duality of patterning would live: the
       atoms are meaningless, the word built from them is not.
    2. **Words come in classes.** Each word is read as a noun, an adjective, a
       numeral or none of these, and within its class as naming one attribute
       -- a noun names the kind of fruit, an adjective the colour or the
       quality, a numeral the quantity or the price. Which words are nouns is
       learned; that there are nouns is not.
    3. **Numerals sit on a number line.** A numeral's meaning is a place on the
       line and a precision, plus whatever exceptions it learns, so "about
       four" is as easy to mean as "four", and near misses are near (a
       heavy-tailed kernel, so a far value is unlikely, never impossible).
    4. **A description is assembled from its words** (compositional semantics):
       each attribute is read from the word that names it -- an attention over
       the words by how strongly each is of that attribute's class, with "no
       word names it" as an option -- and a missing attribute is left
       uncertain rather than guessed from the others. No word's chance of
       naming an attribute falls below ``ATTRIBUTION_FLOOR``, so a word filed
       under the wrong class can still be re-filed.

    The output is five log-distributions, one per field of a lot, which the
    agent adds to its five belief heads (a product of experts with the
    context-reading transformer). Muted, the reader hears no word and returns
    uniform distributions, so the channel controls keep their meaning.
    """
    MAX_ATOMS = 8            # position-in-word embeddings; longer words share the last
    ATTRIBUTION_FLOOR = 0.05  # the least chance any word has of naming any attribute

    def __init__(self, cfg: Config, d: int):
        super().__init__()
        c = cfg.channel
        self.atomic_vocab = c.atomic_vocab
        self.hyphen_id = c.hyphen_id
        self.spans = lot_spans(cfg.world)
        self.atom_emb = nn.Embedding(n_token_ids(cfg), d)
        # position in the word, as a gain on the atom's vector: a7-a2 and
        # a2-a7 are different words
        self.slot_gain = nn.Parameter(torch.ones(self.MAX_ATOMS, d))
        # how many atoms the word has, added after pooling: the LayerNorm below
        # removes scale, and without this `a19` and `a19-a19` differed only in
        # scale and read as one word (cosine 0.997 at birth)
        self.word_len = nn.Embedding(self.MAX_ATOMS + 1, d)
        self.norm = nn.LayerNorm(d)
        self.encode_word = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.word_class = nn.Linear(d, len(WORD_CLASSES))
        self.adjective_of = nn.Linear(d, 2)      # colour or quality
        self.numeral_of = nn.Linear(d, 2)        # quantity or price
        self.meaning = nn.ModuleList([nn.Linear(d, s) for s in self.spans])
        self.line_place = nn.Linear(d, 2)        # where on the line (quantity, price)
        self.line_width = nn.Linear(d, 2)        # how precisely
        # "no word names this attribute": the score a word has to beat
        self.unnamed = nn.Parameter(torch.zeros(N_LOT_FIELDS))
        # how much the listener trusts its reading of the words, per field
        self.gain = nn.Parameter(torch.ones(N_LOT_FIELDS))

    def reset_innate(self) -> None:
        """Initial values that are part of the design, set after the generic init.

        A word has to be read as of an attribute's class with some confidence
        (log p(class) above -2) before it outscores "no word names it"; the
        number line starts broad (a numeral means "somewhere around here")
        and sharpens as the word does. The position gains start far apart --
        each position its own random direction per dimension -- so a word's
        atoms in a different order are a different word from birth (at
        1 +- 0.1 `a19-a20` and `a20-a19` had cosine 0.98-0.996); the word
        length embedding starts on the atoms' scale, so repeating an atom
        makes another word too.
        """
        with torch.no_grad():
            self.slot_gain.copy_(torch.randn_like(self.slot_gain))
            self.word_len.weight.normal_(0.0, 0.02)
            self.unnamed.fill_(-2.0)
            self.gain.fill_(1.0)
            self.line_width.bias.fill_(2.0)

    # ------------------------------------------------------------------
    def segment(self, ids: torch.Tensor, heard: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Where the words are. ``ids`` (B, n) symbol ids, ``heard`` (B, n) bool.

        Returns (atom mask, word index per position, position in word, number
        of word slots). A word starts at every heard atom that is not joined
        to the symbol before it by a HYPHEN -- the word grammar makes that
        exactly "the first atom of the turn, or the first after a SPACE".
        """
        atom = heard & (ids < self.atomic_vocab)
        # the symbol before, as heard: a slot the listener did not hear (its own
        # turn) cannot join its atom to the next word
        prev = torch.cat([torch.full_like(ids[:, :1], -1),
                          torch.where(heard[:, :-1], ids[:, :-1],
                                      torch.full_like(ids[:, :-1], -1))], dim=1)
        start = atom & (prev != self.hyphen_id)
        # Positions before a row's first word get -1, clamped to word 0; none
        # of them is an atom, so they contribute nothing wherever they point.
        word = (start.long().cumsum(1) - 1).clamp(min=0)
        n_words = int(word[atom].max()) + 1 if bool(atom.any()) else 0
        rank = atom.long().cumsum(1) - 1                     # atoms so far, per position
        # the rank of each word's first atom; everything that is not a word's
        # start writes into a spare last column that is never read
        B = ids.shape[0]
        spare = torch.full_like(word, n_words)
        first = torch.zeros((B, n_words + 1), dtype=torch.long, device=ids.device)
        first = first.scatter(1, torch.where(start, word, spare), rank)
        pos = (rank - first.gather(1, word)).clamp(0, self.MAX_ATOMS - 1)
        return atom, word, pos, n_words

    def forward(self, dialogue: torch.Tensor, ids: torch.Tensor,
                heard: torch.Tensor) -> Optional[tuple[torch.Tensor, ...]]:
        """Five (B, span) log-distributions, or None if nothing was said at all.

        ``dialogue`` (B, D) ids or (B, D, V) one-hot weights (the straight-
        through channel's, which is how a gradient reaches the speaker's
        atoms); ``ids`` (B, D) the same symbols as ids; ``heard`` (D,) bool,
        the dialogue slots the *other* party produced.
        """
        B, D = ids.shape
        heard = heard[:D].to(ids.device).unsqueeze(0).expand(B, D)
        if not bool((heard & (ids < self.atomic_vocab)).any()):
            return None
        # read no further than the last atom anyone heard in this batch
        used = int(((heard & (ids < self.atomic_vocab)).any(0)).nonzero().max()) + 1
        ids, heard = ids[:, :used], heard[:, :used]
        dialogue = dialogue[:, :used]
        atom, word, pos, W = self.segment(ids, heard)
        d = self.atom_emb.embedding_dim
        if dialogue.dtype == torch.long:
            vec = self.atom_emb(dialogue)
        else:
            vec = dialogue @ self.atom_emb.weight
        vec = vec * self.slot_gain[pos] * atom.unsqueeze(-1).float()
        words = torch.zeros((B, W, d), device=ids.device, dtype=vec.dtype)
        words = words.scatter_add(1, word.unsqueeze(-1).expand(-1, -1, d), vec)
        # atoms per word slot: which slots hold a word in this row, and how long
        n_atoms = torch.zeros((B, W), device=ids.device).scatter_add(1, word, atom.float())
        exists = n_atoms > 0
        words = words + self.word_len(n_atoms.long().clamp(max=self.MAX_ATOMS))
        v = self.encode_word(self.norm(words))                       # (B, W, d)

        # word class, then which attribute within the class
        cls = F.log_softmax(self.word_class(v), dim=-1)
        adj = F.log_softmax(self.adjective_of(v), dim=-1)
        num = F.log_softmax(self.numeral_of(v), dim=-1)
        of_field = torch.stack([cls[..., 0],
                                cls[..., 1] + adj[..., 0], cls[..., 1] + adj[..., 1],
                                cls[..., 2] + num[..., 0], cls[..., 2] + num[..., 1]],
                               dim=-1)                                # (B, W, 5)
        # Every word keeps a small chance of naming every attribute. Without it
        # a word filed under the wrong attribute early was never read as its
        # own: its weight went to 0, so its meaning for that attribute was never
        # trained, so there was nothing to gain by moving it back -- measured
        # with the agents' own initialisation, words learned one at a time were
        # read at 0.49-0.85 on some field in every seed of five.
        floor = self.ATTRIBUTION_FLOOR
        of_field = torch.logaddexp(of_field + math.log1p(-floor),
                                   torch.full_like(of_field, math.log(floor)))
        of_field = of_field.masked_fill(~exists.unsqueeze(-1), float("-inf"))
        none = self.unnamed.view(1, 1, N_LOT_FIELDS).expand(B, 1, N_LOT_FIELDS)
        attend = F.log_softmax(torch.cat([of_field, none], dim=1), dim=1)   # (B, W+1, 5)

        out = []
        line_at = torch.sigmoid(self.line_place(v))                  # (B, W, 2)
        line_w = F.softplus(self.line_width(v)) + 0.3
        for f, span in enumerate(self.spans):
            lg = self.meaning[f](v)                                  # (B, W, span)
            if CLASS_OF_FIELD[f] == 2:                               # a numeral
                j = f - 3
                grid = torch.arange(span, device=ids.device, dtype=lg.dtype)
                mu = line_at[..., j].unsqueeze(-1) * (span - 1)
                width = line_w[..., j].unsqueeze(-1)
                # Heavy-tailed: near misses are near, and a far value is
                # unlikely but not impossible. The Gaussian this replaces gave a
                # numeral whose place had drifted to 0 at the narrowest width a
                # penalty of ~270 nats on "seven", and nothing could move it
                # back: the value was read as never meant.
                lg = lg - torch.log1p(((grid - mu) / width) ** 2)
            lp = F.log_softmax(lg, dim=-1)
            flat = torch.full((B, 1, span), -math.log(span), device=ids.device, dtype=lp.dtype)
            lp = torch.cat([lp, flat], dim=1)                        # (B, W+1, span)
            out.append(torch.logsumexp(attend[..., f].unsqueeze(-1) + lp, dim=1))
        return tuple(out)

# Sequence layout.  The number of observation slots is whatever the world's
# schema needs (a farm with several varieties has more to look at than a buyer
# with one shopping list), and the shorter role is padded, so both roles share
# one layout and one set of position indices.
SLOT_BOS, SLOT_SEP, SLOT_DIALOGUE, SLOT_DECIDE = 0, 1, 2, 3
N_FIXED_SLOT_TYPES = 4
N_SLOT_TYPES = N_FIXED_SLOT_TYPES + 7        # + one per field kind


def dialogue_offset(cfg: Config) -> int:
    """Index of the first dialogue slot: BOS + the role-padded obs slots + SEP."""
    return 1 + n_obs_slots(cfg.world, cfg) + 1


def sequence_len(cfg: Config) -> int:
    return dialogue_offset(cfg) + cfg.channel.dialogue_len + 1


def own_dialogue_positions(cfg: Config, role: int) -> list[int]:
    """Dialogue-buffer indices (0..D-1) at which ``role`` speaks *in the trading task*.

    This is the buyer-opens schedule only. Anything that can run in a lineup
    rung must ask the phase instead (:meth:`orchard.curriculum.Phase.own_positions`):
    there the farmer describes first, and using this schedule there silently
    swaps whose words are whose.
    """
    L = cfg.channel.max_msg_len
    out: list[int] = []
    for turn in range(cfg.channel.n_turns):
        if speaker_of_turn(turn) == role:
            out.extend(range(turn * L, (turn + 1) * L))
    return out


def speaker_self_mask(cfg: Config, role: int) -> torch.Tensor:
    """(D,) bool: True where this agent is the speaker."""
    D = cfg.channel.dialogue_len
    m = torch.zeros(D, dtype=torch.bool)
    m[own_dialogue_positions(cfg, role)] = True
    return m


class CommNet(nn.Module):
    """Policy + value network for one agent.  Randomly initialised, always."""

    def __init__(self, cfg: Config, role: int):
        super().__init__()
        self.cfg = cfg
        self.role = role
        m, c, w = cfg.model, cfg.channel, cfg.world
        d = m.d_model
        self.d_model = d
        self.seq_len = sequence_len(cfg)

        self.schema = obs_schema(w, role, cfg)
        self.n_obs = len(self.schema)
        self.dialogue_offset = dialogue_offset(cfg)

        # Every symbol, plus every gesture (orchard/gesture.py): a gesture sits
        # in a dialogue slot like a symbol and is read through the same table,
        # which is what lets the words that share the slot inherit the readout
        # the gesture trains. The table always has the room, so switching
        # gestures off is a change of method, not of architecture.
        self.tok_emb = nn.Embedding(n_token_ids(cfg), d)
        self.variety_emb = nn.Embedding(w.n_varieties, d)
        self.qty_emb = nn.Embedding(w.max_qty + 1, d)
        self.quality_emb = nn.Embedding(w.n_quality, d)
        self.color_emb = nn.Embedding(w.n_colors, d)
        self.price_emb = nn.Embedding(w.n_price_bins, d)
        # "which field are you being asked about" -- the naming rungs put this in
        # the describer's observation, and it is what a word for a colour alone
        # has to be conditioned on. One value per field of a lot, plus "all of
        # it", which is also what a buyer's request carries into the market.
        self.field_emb = nn.Embedding(QUERY_ALL + 1, d)
        self.empty_emb = nn.Embedding(1, d)
        self.slot_emb = nn.Embedding(N_SLOT_TYPES, d)
        # One embedding per observation *position*, so "stock of GREEN" is a
        # different thing to look at from "stock of GOLD" even though both are
        # quantities.
        self.obs_pos_emb = nn.Embedding(self.n_obs, d)
        self.speaker_emb = nn.Embedding(2, d)     # 0 = me, 1 = the other party
        self.role_emb = nn.Embedding(2, d)
        self.pos_emb = nn.Embedding(self.seq_len, d)

        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=m.n_heads, dim_feedforward=m.d_ff,
            dropout=m.dropout, activation="gelu",
            batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=m.n_layers,
                                             enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)

        self.token_head = nn.Linear(d, c.n_emittable)
        self.accept_head = nn.Linear(d, 2)
        self.variety_head = nn.Linear(d, w.n_varieties)
        self.decide_qty_head = nn.Linear(d, w.max_qty + 1)
        self.decide_price_head = nn.Linear(d, w.n_price_bins)
        # What this agent thinks the OTHER party's private situation is.  Read at
        # the same position as the decision, from the same state, but scored
        # against the other's hidden facts rather than against the deal -- this is
        # what makes "did you understand me" a thing either side can be paid for.
        self.belief_variety_head = nn.Linear(d, w.n_varieties)
        self.belief_qty_head = nn.Linear(d, w.max_qty + 1)
        self.belief_quality_head = nn.Linear(d, w.n_quality)
        self.belief_price_head = nn.Linear(d, w.n_price_bins)
        # Appended after the choice head so every earlier head keeps its index.
        self.belief_color_head = nn.Linear(d, w.n_colors)
        # Which candidate in the lineup (referential phase only).  This is a
        # pointer rather than a flat classifier: it scores the hidden state *at
        # each candidate's own slots*, so "compare the message against this
        # candidate" is something the attention can express directly instead of a
        # relational trick the network has to discover from nothing.  Present in
        # every phase so the architecture -- and the carried weights -- never
        # change at a curriculum boundary.
        # With `model.factored_choice` the choice is read off the five belief
        # heads instead (see `choice_logits`), and the pointer below does not
        # exist; a parameter nothing reads would still be saved and compared.
        self.factored_choice = bool(m.factored_choice)
        if not self.factored_choice:
            self.choice_proj = nn.Linear(d, d)
            # Both sides of the match are normalised before the dot product.
            # Without this the candidate side is a sum of three freshly-initialised
            # embeddings (norm ~0.24) against a query of norm ~6.8, which put the
            # choice logits at std 0.03 where every other head sits near 1.0 -- a
            # policy so close to uniform that the gradient could not move it, and
            # the lineup game sat exactly at chance no matter how long it ran.
            self.choice_ln_cand = nn.LayerNorm(d)
            self.choice_ln_query = nn.LayerNorm(d)
        self.n_candidates = max(2, cfg.curriculum.n_candidates)
        # Whether to gesture at the start of a turn, and about which field of
        # the lot: none, or one of the five. Sampled from the same hidden state
        # that emits the turn's first symbol, trained by REINFORCE like the
        # decisions (orchard/gesture.py). Which options a seat actually has, and
        # whether the round allows any, is masked in by the rollout.
        self.gesture_head = nn.Linear(d, 1 + N_LOT_FIELDS)
        self.value_head = nn.Linear(d, 1)
        # The barn lookup (ModelConfig.barn_lookup): a query from each hidden
        # state against every barn row's identity, reading back the matching
        # row's contents. Only ever applied to a barn, so the naming rungs never
        # touch it; its output starts small so a farmer arriving at `order`
        # keeps the listening it has.
        self.n_cells = n_cells(w)
        self.lookup_q = nn.Linear(d, d)
        self.lookup_out = nn.Linear(d, d)
        # The innate reader (`model.lexical_reader`): the other party's words,
        # read one at a time out of context into the attributes they name, and
        # added to the five belief heads above.
        self.lexical = bool(m.lexical_reader)
        if self.lexical:
            self.reader = LexicalReader(cfg, d)
        # ...and its production twin (`model.lexical_speaker`): the word for the
        # part of the lot the speaker is naming, added to the token logits.
        self.speaks_lexically = bool(m.lexical_speaker)
        if self.speaks_lexically:
            self.speaker_lexicon = LexicalSpeaker(cfg, d)
            self._gesture_offsets = gesture_offsets(cfg)
        # Innate concepts (`model.innate_concepts`): the kind of thing each lot
        # field is, and a number line under quantities and prices.
        self.innate_concepts = bool(m.innate_concepts)
        if self.innate_concepts:
            self.concept_emb = nn.Embedding(3, d)
            self.qty_line = nn.Linear(w.max_qty, d, bias=False)
            self.price_line = nn.Linear(max(1, w.n_price_bins - 1), d, bias=False)

        self.register_buffer("_self_mask", speaker_self_mask(cfg, role), persistent=False)
        causal = torch.triu(torch.full((self.seq_len, self.seq_len), float("-inf")), diagonal=1)
        self.register_buffer("_causal", causal, persistent=False)

        self.apply(self._init)
        # Small, not zero: a zero-initialised output gives the query no gradient
        # at all until the output has moved, and the lookup then trails the plain
        # network for hundreds of steps. At a tenth of the usual gain it perturbs
        # a hidden state of norm ~sqrt(d) by a few percent and learns at once.
        nn.init.xavier_uniform_(self.lookup_out.weight, gain=0.1)
        if self.lexical:
            self.reader.reset_innate()
        if self.innate_concepts:
            # On the embeddings' scale, so the number line orders the values
            # without drowning which field the slot holds.
            nn.init.normal_(self.qty_line.weight, mean=0.0, std=0.02)
            nn.init.normal_(self.price_line.weight, mean=0.0, std=0.02)

    @staticmethod
    def _init(mod: nn.Module) -> None:
        if isinstance(mod, nn.Embedding):
            nn.init.normal_(mod.weight, mean=0.0, std=0.02)
        elif isinstance(mod, nn.Linear):
            nn.init.xavier_uniform_(mod.weight)
            if mod.bias is not None:
                nn.init.zeros_(mod.bias)

    def _table(self, kind: int) -> nn.Embedding:
        return {K_VARIETY: self.variety_emb, K_QTY: self.qty_emb,
                K_QUALITY: self.quality_emb, K_PRICE: self.price_emb,
                K_COLOR: self.color_emb, K_FIELD: self.field_emb}[kind]

    def field_embed(self, kind: int, values: torch.Tensor) -> torch.Tensor:
        """The embedding of a field value: its learned vector, plus -- for a
        magnitude, with `model.innate_concepts` -- its place on the number line."""
        table = self._table(kind)
        vec = table(values)
        if self.innate_concepts and kind == K_QTY:
            vec = vec + self.qty_line(thermometer(values, table.num_embeddings))
        elif self.innate_concepts and kind == K_PRICE:
            vec = vec + self.price_line(thermometer(values, table.num_embeddings))
        return vec

    def is_barn(self, schema: "list[int] | None") -> bool:
        """Does this observation layout hold the farmer's barn (lot rows)?"""
        s = schema if schema is not None else self.schema
        w = 4 * self.n_cells
        return (len(s) > w and s[w] == K_PRICE
                and s[:4] == [K_VARIETY, K_COLOR, K_QUALITY, K_QTY])

    def barn_lookup(self, h: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        """(B, L, d) -> (B, L, d): each state reads the barn row it asks for.

        Keys are each row's (fruit, colour) embeddings -- the same tables the
        words for fruit and colour are grounded in everywhere else -- and values
        its (quality, stock). A state that has decoded "green pears" from the
        buyer's words only has to reproduce those two embeddings as its query
        to read back how many green pears there are and how good they are.
        """
        B = obs.shape[0]
        rows = obs[:, :4 * self.n_cells].reshape(B, self.n_cells, 4)
        key = self.variety_emb(rows[:, :, 0]) + self.color_emb(rows[:, :, 1])   # (B,C,d)
        val = self.quality_emb(rows[:, :, 2]) + self.field_embed(K_QTY, rows[:, :, 3])
        q = self.lookup_q(h)                                                    # (B,L,d)
        att = torch.softmax(torch.bmm(q, key.transpose(1, 2)) / math.sqrt(self.d_model),
                            dim=-1)                                             # (B,L,C)
        return h + self.lookup_out(torch.bmm(att, val))

    # ------------------------------------------------------------------
    def embed(self, obs: torch.Tensor, tokens: torch.Tensor,
              schema: "list[int] | None" = None,
              self_mask: Optional[torch.Tensor] = None,
              upto: Optional[int] = None) -> torch.Tensor:
        """obs: (B,4) long -> (B, seq_len, d).

        ``tokens`` is either (B,D) integer ids, or (B,D,n_token_ids) of
        per-slot weights.  The float form is what the straight-through Gumbel
        channel passes: the forward values are still exact one-hots, so the
        message that crosses the channel is genuinely discrete, but the lookup
        becomes a differentiable matrix product and a gradient can reach the
        speaker that produced it.

        ``self_mask`` (D,) marks the dialogue slots this agent produced. It
        defaults to the trading schedule; a phase with a different speaking order
        passes its own, so an agent's own words are always embedded as its own.

        ``upto`` embeds only the first ``upto`` positions. During generation that
        is the conversation so far; embedding the whole dialogue buffer and then
        discarding most of it cost memory (the soft tokens are kept for the
        backward pass) in proportion to the buffer, at every symbol step.
        ``tokens`` may be just the dialogue prefix that ``upto`` needs.
        """
        B = obs.shape[0]
        n = self.seq_len if upto is None else upto
        n_dial = max(0, min(self.cfg.channel.dialogue_len, n - self.dialogue_offset))
        d = self.d_model
        dev = obs.device
        parts = []

        bos = self.slot_emb.weight[SLOT_BOS] + self.role_emb.weight[self.role]
        parts.append(bos.expand(B, 1, d))

        cols = []
        for i, kind in enumerate(schema if schema is not None else self.schema):
            if kind == K_EMPTY:
                vec = self.empty_emb.weight[0].expand(B, d)
            else:
                vec = self.field_embed(kind, obs[:, i])
                if self.innate_concepts and kind in CONCEPT_OF_KIND:
                    vec = vec + self.concept_emb.weight[CONCEPT_OF_KIND[kind]]
            cols.append(vec
                        + self.slot_emb.weight[N_FIXED_SLOT_TYPES + kind]
                        + self.obs_pos_emb.weight[i])
        parts.append(torch.stack(cols, dim=1))

        parts.append(self.slot_emb.weight[SLOT_SEP].expand(B, 1, d))

        mine = (self._self_mask if self_mask is None else self_mask)[:n_dial]
        spk = torch.where(mine.to(dev), 0, 1)                            # (n_dial,)
        tokens = tokens[:, :n_dial]
        tok_vec = (self.tok_emb(tokens) if tokens.dtype == torch.long
                   else tokens @ self.tok_emb.weight)
        dial = (tok_vec
                + self.speaker_emb(spk).unsqueeze(0)
                + self.slot_emb.weight[SLOT_DIALOGUE])
        parts.append(dial)

        if n > self.dialogue_offset + self.cfg.channel.dialogue_len:
            parts.append(self.slot_emb.weight[SLOT_DECIDE].expand(B, 1, d))

        x = torch.cat(parts, dim=1)[:, :n]
        return x + self.pos_emb.weight[:x.shape[1]].unsqueeze(0)

    def encode(self, obs: torch.Tensor, tokens: torch.Tensor,
               upto: Optional[int] = None,
               schema: "list[int] | None" = None,
               self_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Hidden states for the prefix of length ``upto`` (default: whole sequence)."""
        n = self.seq_len if upto is None else upto
        mask = self._causal[:n, :n]

        def run(tok):
            x = self.embed(obs, tok, schema, self_mask, upto=n)
            h = self.norm(self.encoder(x, mask=mask))
            if self.cfg.model.barn_lookup and self.is_barn(schema):
                h = self.barn_lookup(h, obs)
            return h
        # Gradient checkpointing covers embedding, layers and the final norm, so
        # all the backward pass keeps per call is the (soft) tokens that went in.
        # Generation re-encodes the conversation at every symbol step and the
        # Gumbel path backpropagates through all of them, so without this the
        # saved activations grow with steps x prefix length and ran a 24 GB card
        # out of memory in the first rung. Keyed on grad mode, not train mode:
        # newborns come out of their apprenticeship in eval mode.
        if self.cfg.train.grad_checkpoint and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint
            return checkpoint(run, tokens, use_reentrant=False)
        return run(tokens)

    # ------------------------------------------------------------------
    def lot_concepts(self, obs: torch.Tensor, schema=None) -> Optional[torch.Tensor]:
        """(B, 5, d): the speaker's own lot, field by field, as concepts -- each
        field's value and kind, with no context -- or None where the
        observation is not a lot to describe (a barn, a lineup)."""
        s = schema if schema is not None else self.schema
        n = N_LOT_FIELDS
        if len(s) <= n or tuple(s[:n]) != tuple(LOT_KINDS) or s[n] != K_FIELD:
            return None
        vecs = []
        for i, kind in enumerate(LOT_KINDS):
            v = self.field_embed(kind, obs[:, i]) + self.slot_emb.weight[N_FIXED_SLOT_TYPES + kind]
            if self.innate_concepts:
                v = v + self.concept_emb.weight[CONCEPT_OF_KIND[kind]]
            vecs.append(v)
        return torch.stack(vecs, dim=1)

    def speak(self, h: torch.Tensor, obs: torch.Tensor, schema=None,
              turn: Optional[tuple] = None) -> torch.Tensor:
        """Token logits from hidden state(s) ``h`` ((B, d) or (B, K, d)): the
        token head's, plus -- with `model.lexical_speaker`, describing a lot --
        the mental lexicon's word for the part being named (:class:`LexicalSpeaker`).
        ``turn`` is (push, said) from :meth:`turn_so_far`: which parts this
        turn has named (passed over when choosing the next), and whether a
        whole thing is being described with parts of it still unnamed (there
        ending loses ``go_on`` nats to going on). Every place that emits or
        scores a symbol reads this, so generation, evaluation and a newborn's
        lessons agree."""
        logits = self.token_head(h)
        if self.speaks_lexically:
            concepts = self.lot_concepts(obs, schema)
            if concepts is not None:
                lex = self.speaker_lexicon
                push, said = turn if turn is not None else (None, None)
                logits = logits + lex(h, concepts, said)[0]
                if push is not None:
                    logits = logits + (lex.go_on * push.to(logits.dtype)).unsqueeze(-1) * lex._push
        return logits

    def describing(self, obs: torch.Tensor, schema, tokens: torch.Tensor,
                   positions: Sequence[int]) -> Optional[torch.Tensor]:
        """(B, P) bool: the push part of :meth:`turn_so_far`."""
        got = self.turn_so_far(obs, schema, tokens, positions)
        return None if got is None else got[0]

    @torch.no_grad()
    def turn_so_far(self, obs: torch.Tensor, schema, tokens: torch.Tensor,
                    positions: Sequence[int]) -> Optional[tuple]:
        """(push (B, P) bool, said (B, P, 5) bool) before each dialogue position
        in ``positions``: which parts of the lot this turn has named, and is
        this speaker describing a whole lot with some of them still unnamed?

        The innate pragmatics of the production side -- say as much as the
        question asks (Grice's maxim of quantity). The speaker sees the question
        (the query slot: one field, or all of it) and monitors its own turn: a
        part counts as named once the turn holds that part's word (what the
        lexicon says for it, :meth:`LexicalSpeaker.part_words`) or a gesture at
        it. Asked for a whole lot, it is pushed on while any part is unnamed,
        and never past one word per part; asked for one field, never.

        Why it exists: measured on 2026-09-30, speakers who could name every
        field alone, and whose next word -- when made to go on -- named a
        different part of the lot 99-100% of the time, went on after their first
        word 0.02-0.03% of the time. One-field rounds teach "a word, then stop",
        nothing in a whole-lot round says there is more to say, and a
        continuation that is never tried cannot be learned. How hard the push
        is (`go_on`) is learned like everything else.

        None where it does not apply: no production lexicon, or not a lot.
        """
        if not self.speaks_lexically:
            return None
        concepts = self.lot_concepts(obs, schema)
        if concepts is None:
            return None
        c = self.cfg.channel
        L, A = c.max_msg_len, c.atomic_vocab
        B, D = tokens.shape
        whole = obs[:, N_LOT_FIELDS] >= QUERY_ALL                       # (B,)
        words = self.speaker_lexicon.part_words(concepts)               # (B, 5)
        is_atom = tokens < A
        names = is_atom.unsqueeze(-1) & (tokens.unsqueeze(-1) == words.unsqueeze(1))
        # a gesture names the part it points at
        g = tokens - c.n_symbol_ids
        offs = torch.tensor(self._gesture_offsets, device=tokens.device)
        in_g = (g >= 0) & (g < int(sum(lot_spans(self.cfg.world))))
        g_field = (torch.bucketize(g.clamp(min=0), offs, right=True) - 1).clamp(0, N_LOT_FIELDS - 1)
        names = names | (in_g.unsqueeze(-1)
                         & (g_field.unsqueeze(-1) == torch.arange(N_LOT_FIELDS, device=tokens.device)))
        prev = torch.cat([torch.full_like(tokens[:, :1], c.pad_id), tokens[:, :-1]], dim=1)
        at_turn_start = (torch.arange(D, device=tokens.device) % L == 0).unsqueeze(0)
        prev = torch.where(at_turn_start, torch.full_like(prev, c.pad_id), prev)
        starts = is_atom & (prev != c.hyphen_id)
        zero_n = torch.zeros((B, 1, N_LOT_FIELDS), dtype=torch.long, device=tokens.device)
        zero_s = torch.zeros((B, 1), dtype=torch.long, device=tokens.device)
        cs_n = torch.cat([zero_n, names.long().cumsum(1)], dim=1)       # (B, D+1, 5)
        cs_s = torch.cat([zero_s, starts.long().cumsum(1)], dim=1)      # (B, D+1)
        pos = torch.as_tensor(list(positions), dtype=torch.long, device=tokens.device).clamp(0, D)
        t0 = (pos // L) * L
        said = (cs_n[:, pos] - cs_n[:, t0]) > 0                           # (B, P, 5)
        n_words = cs_s[:, pos] - cs_s[:, t0]                              # (B, P)
        unnamed = N_LOT_FIELDS - said.sum(-1)
        push = whole.unsqueeze(1) & (unnamed > 0) & (n_words >= 1) & (n_words < N_LOT_FIELDS)
        return push, said

    def next_token_logits(self, obs: torch.Tensor, tokens: torch.Tensor,
                          seq_pos: int, schema=None, self_mask=None
                          ) -> tuple[torch.Tensor, torch.Tensor]:
        """Logits for the token that will occupy ``seq_pos``, plus that state's value."""
        h = self.encode(obs, tokens, upto=seq_pos, schema=schema,
                        self_mask=self_mask)[:, -1]
        turn = self.turn_so_far(obs, schema, tokens, [seq_pos - dialogue_offset(self.cfg)])
        return (self.speak(h, obs, schema, None if turn is None
                           else (turn[0][:, 0], turn[1][:, 0])),
                self.value_head(h).squeeze(-1))

    def read_words(self, dialogue: torch.Tensor, ids: Optional[torch.Tensor] = None,
                   self_mask: Optional[torch.Tensor] = None
                   ) -> Optional[tuple[torch.Tensor, ...]]:
        """What the other party's words say about each field of a lot, read by
        the innate reader -- five log-distributions -- or None (reader off, or
        nothing heard). ``dialogue`` is (B, D) ids or (B, D, V) soft one-hots,
        in which case ``ids`` gives the same symbols as ids."""
        if not self.lexical:
            return None
        if ids is None:
            ids = dialogue if dialogue.dtype == torch.long else dialogue.argmax(-1)
        mine = self._self_mask if self_mask is None else self_mask
        return self.reader(dialogue, ids, ~mine.to(ids.device))

    def decision_logits(self, obs: torch.Tensor, tokens: torch.Tensor, schema=None,
                        self_mask=None):
        """Every discrete head, then the value.  Order matches curriculum.py's
        head indices, so callers can slice the first N_HEADS and trust it."""
        h = self.encode(obs, tokens, schema=schema, self_mask=self_mask)[:, -1]
        lex = self.read_words(tokens, None, self_mask)
        return self.all_heads(h, obs, lex) + (self.value_head(h).squeeze(-1),)

    def decision_heads(self, h: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return (self.accept_head(h), self.variety_head(h),
                self.decide_qty_head(h), self.decide_price_head(h))

    def belief_heads(self, h: torch.Tensor, lex=None) -> tuple[torch.Tensor, ...]:
        """The H_BELIEF heads: (fruit, quantity, quality, price)."""
        r = self.report_logits(h, lex)
        return (r[0], r[3], r[2], r[4])

    def report_color(self, h: torch.Tensor, lex=None) -> torch.Tensor:
        return self.report_logits(h, lex)[1]

    def candidate_embeddings(self, obs: torch.Tensor) -> torch.Tensor:
        """(B, K, d) -- each lineup candidate embedded from its own five fields.

        Built from the raw observation rather than from hidden states, because the
        encoder is causal: a candidate sits early in the sequence and cannot
        attend forward to the message. Scoring it against a hidden state taken at
        a candidate slot would therefore be scoring it against something that has
        not heard anything, which is exactly how the first version of this head
        managed to be entirely independent of what was said.
        """
        K = self.n_candidates
        W = N_LOT_FIELDS
        kinds = (K_VARIETY, K_COLOR, K_QUALITY, K_QTY, K_PRICE)
        vecs = []
        for k in range(K):
            i = W * k
            if i + W - 1 >= obs.shape[1]:
                vecs.append(torch.zeros_like(vecs[0]) if vecs else
                            self.empty_emb.weight[0].expand(obs.shape[0], self.d_model))
                continue
            # Outside the lineup phase these slots hold other fields whose ranges
            # do not match these tables, and the head's output is unused.
            # Clamping keeps the lookup legal rather than making every call site
            # have to know which phase it is in.
            # a candidate is a lot: (fruit, colour, quality, quantity, price)
            vec = None
            for j, kind in enumerate(kinds):
                n = self._table(kind).num_embeddings
                e = self.field_embed(kind, obs[:, i + j].clamp(0, n - 1))
                vec = e if vec is None else vec + e
            vecs.append(vec)
        return torch.stack(vecs, dim=1)

    def report_logits(self, h: torch.Tensor, lex=None) -> tuple[torch.Tensor, ...]:
        """The five belief heads in lot order: fruit, colour, quality, quantity, price.

        ``lex`` is the innate reader's reading of the other party's words
        (:meth:`read_words`), added to each head: the transformer's reading of
        the whole context and the lexicon's reading of the words, as a product
        of experts. None leaves the heads as the transformer alone reads them.
        """
        out = (self.belief_variety_head(h), self.belief_color_head(h),
               self.belief_quality_head(h), self.belief_qty_head(h),
               self.belief_price_head(h))
        if lex is None:
            return out
        g = self.reader.gain
        return tuple(o + g[j] * lex[j] for j, o in enumerate(out))

    def choice_logits(self, h_last: torch.Tensor, obs: torch.Tensor, lex=None,
                      rep: Optional[tuple] = None) -> torch.Tensor:
        """(B, K) -- how well each candidate matches what was just heard.

        Two listeners, chosen by ``model.factored_choice``.

        **Factored** (the default): the message is first read into the five
        belief heads -- what fruit, colour, quality, quantity and price was that
        about -- and a candidate scores the sum over fields of the log-probability
        its value gets under the matching head. The listener innately parses a
        description into a kind of thing, its properties and a number, and
        matches attribute by attribute; a code with a word per field is read
        directly, a holistic label has to be squeezed through five independent
        readouts. On a round that varies one field the other four terms are the
        same for every candidate, so the choice *is* that field's head restricted
        to the three values on offer -- which is also the head the report and
        trading rungs will score, so naming trains reporting from the first rung.

        **Pointer** (off): a dot product between a projection of the final hidden
        state and each candidate's summed embedding -- the standard listener for
        a signalling game, kept as the control.
        """
        if not self.factored_choice:
            cand = self.choice_ln_cand(self.candidate_embeddings(obs))   # (B, K, d)
            q = self.choice_ln_query(self.choice_proj(h_last)).unsqueeze(-1)
            return torch.bmm(cand, q).squeeze(-1) / math.sqrt(self.d_model)
        K = self.n_candidates
        W = N_LOT_FIELDS
        B = h_last.shape[0]
        scores = h_last.new_zeros((B, K))
        heads = rep if rep is not None else self.report_logits(h_last, lex)
        for k in range(K):
            i = W * k
            if i + W - 1 >= obs.shape[1]:
                continue
            for j, lg in enumerate(heads):
                lp = F.log_softmax(lg, dim=-1)
                # Outside a lineup these slots hold other fields; clamping keeps
                # the gather legal, and the head's output is unused there.
                v = obs[:, i + j].clamp(0, lp.shape[-1] - 1)
                scores[:, k] = scores[:, k] + lp.gather(-1, v.unsqueeze(-1)).squeeze(-1)
        return scores

    def all_heads(self, h: torch.Tensor, obs: Optional[torch.Tensor] = None,
                  lex=None) -> tuple[torch.Tensor, ...]:
        """Every discrete output, in the fixed order curriculum.py indexes.

        ``lex`` (:meth:`read_words`) is the innate reader's reading of what the
        other party said; the five belief heads -- and the lineup choice, which
        is read through them -- include it.
        """
        rep = self.report_logits(h, lex)
        choice = (self.choice_logits(h, obs, rep=rep) if obs is not None
                  else h.new_zeros((h.shape[0], self.n_candidates)))
        return (self.decision_heads(h) + (rep[0], rep[3], rep[2], rep[4]) + (choice,)
                + (rep[1],))

    def full_pass(self, obs: torch.Tensor, tokens: torch.Tensor,
                  read_positions: torch.Tensor, schema=None, self_mask=None):
        """One causal forward over the finished episode.

        ``read_positions`` are sequence indices whose hidden state produced this
        agent's own tokens (i.e. ``DIALOGUE_OFFSET + p - 1`` for each own
        dialogue slot ``p``).  Returns token logits and values at those
        positions, plus the four decision logits and the value at DECIDE.
        """
        h = self.encode(obs, tokens, schema=schema, self_mask=self_mask)   # (B, L, d)
        hr = h[:, read_positions]                           # (B, K, d)
        # the dialogue position each of those states emits into
        turn = (self.turn_so_far(obs, schema, tokens,
                                 (read_positions - dialogue_offset(self.cfg) + 1).tolist())
                if read_positions.numel() else None)
        tok_logits = self.speak(hr, obs, schema, turn)      # (B, K, V+1)
        tok_values = self.value_head(hr).squeeze(-1)        # (B, K)
        hd = h[:, -1]
        dec = self.all_heads(hd, obs, self.read_words(tokens, None, self_mask))
        dec_value = self.value_head(hd).squeeze(-1)
        return tok_logits, tok_values, dec, dec_value


@dataclass
class Agent:
    """A living agent: its brain, its optimiser, and its lifecycle bookkeeping."""
    agent_id: int
    role: int
    slot: int                     # index into the population list ("lineage slot")
    generation: int
    net: CommNet
    opt: torch.optim.Optimizer
    birth_episode: int
    lifespan: int
    age: int = 0                  # episodes this agent has participated in
    updates: int = 0              # training updates it has taken part in
    days_alive: int = 0
    # running tallies, reported in per-generation summaries
    n_success: int = 0
    n_episodes: int = 0
    reward_sum: float = 0.0
    apples_traded: int = 0
    value_traded: float = 0.0
    profit: float = 0.0
    bottleneck_info: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return "%s%d/g%d" % ("F" if self.role == FARMER else "B", self.slot, self.generation)

    @property
    def success_rate(self) -> float:
        return self.n_success / self.n_episodes if self.n_episodes else 0.0

    def is_expired(self) -> bool:
        """Lifespans count training updates taken part in (see PopulationConfig)."""
        return self.updates >= self.lifespan


def make_agent(cfg: Config, *, agent_id: int, role: int, slot: int, generation: int,
               birth_episode: int, lifespan: int, device: str = "cpu") -> Agent:
    if str(device) == "auto":                      # resolve the config sentinel
        from .hardware import resolve_device
        device = resolve_device("auto")
    net = CommNet(cfg, role).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.train.lr)
    return Agent(agent_id=agent_id, role=role, slot=slot, generation=generation,
                 net=net, opt=opt, birth_episode=birth_episode, lifespan=lifespan)


def count_parameters(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters())
