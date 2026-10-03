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


def lexicon_offsets(cfg: Config) -> list[int]:
    """Where each field's meanings start among the rows of a speaker's lexicon
    -- every value of every field of a lot, in lot order
    (:meth:`CommNet.lexicon_table`)."""
    out, at = [], 0
    for span in lot_spans(cfg.world):
        out.append(at)
        at += span
    return out


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
    the observation is a lineup the term is absent; where it is a barn, the
    parts are those of the row the farmer is talking about
    (:meth:`CommNet.barn_concepts`).

    **Which part** is an attention over the five parts, scored against one
    learned key per part (``part_key``), not against the parts' concepts. The
    concepts are sums of embeddings that start at 0.02, so a score against
    them was a few tenths at most however the query was trained, the choice of
    part stayed close to uniform, and on the 2026-10-01 run it was the
    scaffold below that chose every part, never the speaker. The keys start at
    unit scale and the query small: the choice starts uniform and can be
    learned in tens of updates.

    **The scaffold.** Three biases put the pragmatics of description in from
    outside while the words are being learned: answer the question asked
    (``ask``), go on until every part is named and then stop (``go_on``), and
    do not name a part twice (``inhibit``). They are fixed strengths, all
    multiplied by ``scaffold``, which the trainer takes from 1 to 0 during
    `curriculum.scaffold_fade_rung` -- and while it is above 0 the speaker's
    own policy (the learned scores here and the token head) is trained to do
    unaided what the scaffold asks for (`train.scaffold_distil`), the way a
    supported skill becomes a habit. Everything after that rung -- all of
    trading -- runs with no scaffold at all. On the 2026-10-01 run
    the three were learnable parameters that barely moved (ask 4.0 -> 4.2), the
    token head had learned nothing in six rungs, and every description in
    `mutual` was the scaffold's.

    Two things keep the scaffold in charge for as long as it is on. The
    speaker's own scores are bounded (``BOUND``), and the two biases on the
    choice of part are well above the bound: whatever the speaker has learned,
    the asked part wins and a named part loses. And what the speaker's policy
    is taught is what the scaffold *asks for*, not what the scaffolded speaker
    did. Taught the latter, on the first CPU run with keys it could learn
    from, `name-fruit` -- where the asked part is always the fruit -- taught
    it "the fruit, whatever is asked", strongly enough to overrule the
    scaffold: at update 50 it answered a question about colour, quality,
    quantity or price with the fruit's name (4 distinct words for 27
    meanings), and a teacher that is the pupil plus a nudge agreed with it.
    """

    # How hard a speaker describing a whole thing is pushed on past a word while
    # parts of it are still unnamed, in nats taken from ending and given to
    # going on -- and, once it has said as much as it was asked for, taken from
    # starting another word. See `CommNet.turn_so_far`. Measured with two
    # founders drilled on "a word, then stop": at 3.0 one founder went on 99%
    # of the time and the other 10%, so its listener never learned its longer
    # descriptions and it stayed at one word for 100 updates; when both did go
    # on, their token heads had learned it so hard that descriptions ran to 8.5
    # words. At 5.0 both said exactly five words, one per part, by update 50.
    GO_ON = 5.0
    # Describing a whole lot, how strongly a part already named in this turn is
    # passed over when choosing the part the next *word* names (inhibition of
    # return), in attention logits.
    INHIBIT = 8.0
    # Asked about one field, how strongly the lexicon attends to that field, in
    # attention logits: answer the question you were asked.
    ASK = 8.0
    # The speaker's own score for a part lies within +-BOUND: enough to choose
    # a part 99% of the time, and too little to overrule the scaffold.
    BOUND = 3.0

    def __init__(self, cfg: Config, d: int):
        super().__init__()
        self.atomic_vocab = cfg.channel.atomic_vocab
        self.n_emittable = cfg.channel.n_emittable
        self.query = nn.Linear(d, d)
        self.part_key = nn.Parameter(torch.randn(N_LOT_FIELDS, d))
        self.norm = nn.LayerNorm(d)
        self.say = nn.Linear(d, self.atomic_vocab)
        self.gain = nn.Parameter(torch.ones(()))
        # the scaffold: fixed strengths, and how much of them is left (1 -> 0)
        self.register_buffer("go_on", torch.tensor(self.GO_ON))
        self.register_buffer("inhibit", torch.tensor(self.INHIBIT))
        self.register_buffer("ask", torch.tensor(self.ASK))
        self.register_buffer("scaffold", torch.ones(()))
        # The words this speaker has heard its elders use and understood: for
        # each meaning (every value of every field, in lot order), how often
        # lately each atom opened the word for it. What `reward.lexicon_imitate`
        # pulls its own lexicon towards. Decays by the update.
        self.register_buffer("heard", torch.zeros(sum(lot_spans(cfg.world)),
                                                  self.atomic_vocab))
        push = torch.zeros(self.n_emittable)
        push[cfg.channel.end_id] = -1.0
        push[cfg.channel.space_id] = 1.0
        self.register_buffer("_push", push, persistent=False)
        # Holding back: no new word -- and, where there are atoms enough to give
        # every meaning one of its own, no going on inside the word either, so
        # what is left is to stop. With fewer atoms than meanings (the
        # `duality` preset) a word needs more than one, and only the new word
        # is held back. Without the second half a speaker asked about one field
        # said its word and then, as often as not, a hyphen and more: 2.4-2.6
        # atoms a word over the first 24 updates of `name-fruit`, every one of
        # them a different word to a reader that reads words whole.
        hold = torch.zeros(self.n_emittable)
        hold[cfg.channel.space_id] = -1.0
        if self.atomic_vocab >= sum(lot_spans(cfg.world)):
            hold[cfg.channel.hyphen_id] = -1.0
        self.register_buffer("_hold", hold, persistent=False)
        # Whose atoms a word is made of. While this is 0 the token head adds
        # nothing to *which atom* is said -- only to whether a word goes on, a
        # new one starts or the turn ends -- so a word is the lexicon's and
        # nothing else's; at 1 the head's atoms are added, as they always were.
        # The trainer sets it (`curriculum.own_atoms_from_rung`): 0 through the
        # naming rungs wherever there are atoms enough for every meaning to
        # have one, 1 from the first trading rung on.
        #
        # Why: the head sees the context, and so it is a second place a word
        # can live. Measured on a CPU run of this design: a junior's lexicon
        # had taken its elder's word for a fruit (`a8`, 0.99) -- and it went on
        # saying its own old one, `a29`, which its token head had learned to
        # add 6.5 nats to in exactly that context, because that was the word
        # its listener could already read and the head was the one place the
        # game's gradient could still put it. One dialect cannot be reached
        # while every speaker has somewhere private to keep its own.
        self.register_buffer("own_atoms", torch.zeros(()))
        is_atom = torch.zeros(self.n_emittable)
        is_atom[:self.atomic_vocab] = 1.0
        self.register_buffer("_is_atom", is_atom, persistent=False)

    def gate(self, base: torch.Tensor) -> torch.Tensor:
        """The token head's logits with its say in *which atom* scaled by
        ``own_atoms`` (see there); everything else untouched."""
        return base * (1.0 - self._is_atom * (1.0 - self.own_atoms))

    def reset_innate(self) -> None:
        """Initial values that are part of the design, set after the generic
        init: unit-scale keys and a small query, so the choice of part starts
        uniform (within a tenth of a nat) and is quick to learn."""
        with torch.no_grad():
            self.part_key.normal_(0.0, 1.0)
            self.query.weight.mul_(0.1)
            self.query.bias.zero_()

    @torch.no_grad()
    def restore_strengths(self) -> bool:
        """Put the scaffold's three strengths back at their innate values; True
        if any had moved.

        They are constants, and in the state dict only so that a snapshot says
        what it ran with. A file from when they were learned (before
        2026-10-01) loads wherever they had drifted to into the same names --
        `ask` 4.2 on the run that prompted the change -- and at that strength
        the scaffold no longer outvotes a speaker's own scores (``BOUND``).
        """
        moved = False
        for name, value in (("go_on", self.GO_ON), ("inhibit", self.INHIBIT),
                            ("ask", self.ASK)):
            buf = getattr(self, name)
            if float(buf) != float(value):
                buf.fill_(float(value))
                moved = True
        return moved

    @torch.no_grad()
    def part_words(self, concepts: torch.Tensor) -> torch.Tensor:
        """(B, 5): the atom this speaker says for each part of the lot -- what its
        lexicon produces when it attends to that part alone."""
        return self.say(self.norm(concepts)).argmax(-1)

    def forward(self, h: torch.Tensor, concepts: torch.Tensor,
                avoid: Optional[torch.Tensor] = None,
                asked: Optional[torch.Tensor] = None, habit: bool = False
                ) -> tuple[torch.Tensor, ...]:
        """``h`` (B, d) or (B, K, d) hidden states; ``concepts`` (B, 5, d), or
        (B, K, 5, d) where the parts differ from state to state (a barn);
        ``avoid`` ((B, 5) or (B, K, 5)) parts to pass over -- describing a
        whole lot, the ones this turn has already named, when a new word
        starts (:meth:`CommNet.turn_so_far`); ``asked`` (B, 5) the field the
        question asks about, if it asks about one.

        Returns token-logit contributions (h's shape with n_emittable last;
        zero on everything but atoms) and the attention over the five parts.
        With ``habit``, two more: the speaker's own choice of part, with no
        scaffold at all, and the choice the scaffold asks for -- the asked
        part where one is asked about; otherwise the speaker's own choice among
        the parts not yet named -- which is what `train.scaffold_distil` trains
        the first towards. The lesson is about *which part*, never about which
        atom: taught the scaffolded speaker's atoms instead, the token head
        learned them, context and all -- measured on a CPU run of that design,
        a junior's lexicon had moved to its elder's word for a fruit (`a8`)
        while its token head went on saying its own old one (`a29`, 6.4 nats
        up), and the word a speaker says has to be its lexicon's.

        Asked about one field, the lexicon attends to it (``ask``): it used
        not to look at the question at all, so in `name-color` it went on
        naming the fruit it had learned to name in `name-fruit`, and the colour
        could only be reached further along -- words carried the fruit as well
        as the colour (fruit 0.66 against colour 0.51 of each field's
        information in colour rounds, on the 2026-10-01 GPU run), the same
        word served a fruit and a colour (4 distinct names for 8 meanings),
        and by `name-quantity` words were 11 atoms long.

        Describing a whole lot, a part already named is passed over when a
        new word starts (``inhibit``): pushed on without it, speakers named a
        second part and then said its word again until the turn's cap --
        `a11 a28 a28 a28 a28`. Only at a word's start, and only for a whole
        lot: applied inside words and to one-field questions, it pushed a
        word's second atom onto another part, so the words became chains of
        parts, and a speaker that pointed at the quantity was pushed to say
        anything but the quantity -- the point-and-say lesson the quantity
        words form on.
        """
        squeeze = h.dim() == 2
        if squeeze:
            h = h.unsqueeze(1)                                           # (B, 1, d)
            if concepts.dim() == 4:
                concepts = concepts.squeeze(1)
        mixing = "bkf,bfd->bkd" if concepts.dim() == 3 else "bkf,bkfd->bkd"
        q = self.query(h)                                                # (B, K, d)
        raw = torch.einsum("bkd,fd->bkf", q, self.part_key) / math.sqrt(q.shape[-1])
        learned = self.BOUND * torch.tanh(raw / self.BOUND)
        bias = torch.zeros_like(learned)
        a_ = None
        if asked is not None:
            a_ = asked.to(learned.dtype)
            a_ = a_.unsqueeze(1) if a_.dim() == 2 else a_
            bias = bias + self.ask * a_
        if avoid is not None:
            avoid = avoid.to(learned.dtype)
            bias = bias - self.inhibit * (avoid.unsqueeze(1) if avoid.dim() == 2 else avoid)
        att = torch.softmax(learned + self.scaffold * bias, dim=-1)       # (B, K, 5)
        mix = torch.einsum(mixing, att, concepts)                         # (B, K, d)
        atoms = self.gain * self.say(self.norm(mix))                      # (B, K, A)
        out = F.pad(atoms, (0, self.n_emittable - self.atomic_vocab))
        own = teach = None
        if habit:
            own = torch.softmax(learned, dim=-1)
            # a question decides the part outright; with none, which of the
            # parts not yet named comes next is the speaker's own to say
            free = (1.0 - a_.sum(-1, keepdim=True).clamp(max=1.0)) if a_ is not None else 1.0
            teach = torch.softmax(learned.detach() * free + bias, dim=-1)
        if squeeze:
            out, att = out.squeeze(1), att.squeeze(1)
            if habit:
                own, teach = own.squeeze(1), teach.squeeze(1)
        return (out, att, own, teach) if habit else (out, att)


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
        got = self.read(dialogue, ids, heard)
        return None if got is None else got[0]

    def read(self, dialogue: torch.Tensor, ids: torch.Tensor, heard: torch.Tensor,
             concepts: Optional[Sequence[torch.Tensor]] = None
             ) -> Optional[tuple[tuple[torch.Tensor, ...], Optional[torch.Tensor],
                                 torch.Tensor]]:
        """:meth:`forward`'s five log-distributions; -- given ``concepts``,
        one (span, d) table per field of what each value *is* to this listener
        -- each heard word's meaning, as a (B, D, d) vector at every dialogue
        position holding one of its atoms (zero elsewhere); for each field,
        the first atom of the word the listener took to name it ((B, 5), -1
        where it took no word to); and what the *first* heard word is read as
        naming ((B, 5) log-probabilities over the five fields, before the
        floor; meaningless in a row with no word, which the last value, (B,)
        bool, marks).

        A word's meaning is what it is read as, in the listener's own terms:
        over the five attributes it may name, the expected concept of the
        value it names. Looked up out of context like everything here, so it
        is the same vector wherever the word is heard, and exactly causal --
        nothing said later changes it. How sure the reading is comes from the
        reader as it stands (no gradient into it from here): the reading is
        trained where it is scored, on the five belief heads.
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
        first_names = of_field[:, 0]                                  # (B, 5)
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
        word_means = (torch.zeros((B, W, d), device=ids.device, dtype=v.dtype)
                      if concepts is not None else None)
        names = of_field.exp().detach()                              # (B, W, 5); 0 where no word
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
            if word_means is not None:
                word_means = word_means + names[..., f:f + 1] * (
                    lp.exp().detach() @ concepts[f].to(v.dtype))
            flat = torch.full((B, 1, span), -math.log(span), device=ids.device, dtype=lp.dtype)
            lp = torch.cat([lp, flat], dim=1)                        # (B, W+1, span)
            out.append(torch.logsumexp(attend[..., f].unsqueeze(-1) + lp, dim=1))
        meaning = None
        if word_means is not None:
            at = word_means.gather(1, word.unsqueeze(-1).expand(-1, -1, d))
            meaning = F.pad(at * atom.unsqueeze(-1).to(at.dtype), (0, 0, 0, D - used))
        # which word was taken to name each field: the one the field's reading
        # attends to most, if any beats "no word names it"
        with torch.no_grad():
            words_lp = attend[:, :W]
            best = words_lp.argmax(dim=1)                             # (B, 5)
            named = words_lp.max(dim=1).values > attend[:, W]
            opens = torch.zeros((B, W + 1), dtype=torch.long, device=ids.device).scatter(
                1, torch.where(atom & (pos == 0), word, torch.full_like(word, W)), ids)
            said = torch.where(named, opens.gather(1, best), torch.full_like(best, -1))
        return tuple(out), meaning, said, first_names, exists[:, 0]

# Sequence layout.  The number of observation slots is whatever the world's
# schema needs (a farm with several varieties has more to look at than a buyer
# with one shopping list), and the shorter role is padded, so both roles share
# one layout and one set of position indices.
SLOT_BOS, SLOT_SEP, SLOT_DIALOGUE, SLOT_DECIDE = 0, 1, 2, 3
N_FIXED_SLOT_TYPES = 4
N_SLOT_TYPES = N_FIXED_SLOT_TYPES + 7        # + one per field kind


@dataclass
class Heard:
    """What an agent has made of the other party's words (:meth:`CommNet.listen`)."""
    fields: tuple                           # five (B, span) log-distributions, lot order
    meaning: Optional[torch.Tensor] = None  # (B, n, d): each word's meaning, at its atoms
    said: Optional[torch.Tensor] = None     # (B, 5): the word taken to name each field
                                            # (its first atom; -1: none was)
    first: Optional[torch.Tensor] = None    # (B, 5): log P(the first word names each field)
    spoke: Optional[torch.Tensor] = None    # (B,) bool: there was a first word


# `CommNet.encode(heard=AUTO)`: listen here, rather than be told what was heard.
AUTO = object()


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
        # In the market (`model.lexical_barn`): the farmer finds the lot it was
        # asked about through its reader, and talks about that row with its
        # lexicon. (`model.heard_meaning`): what a listener has understood of
        # each word reaches its own state, not only its report heads.
        self.barn_lexicon = bool(m.lexical_barn) and self.speaks_lexically
        self.reads_rows = bool(m.lexical_barn) and self.lexical
        self.hears_meaning = bool(m.heard_meaning) and self.lexical
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
        if self.speaks_lexically:
            self.speaker_lexicon.reset_innate()
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

    def barn_rows(self, obs: torch.Tensor) -> torch.Tensor:
        """(B, C, 4): the barn's rows -- (fruit, colour, quality, stock) each."""
        return obs[:, :4 * self.n_cells].reshape(obs.shape[0], self.n_cells, 4)

    def row_attention(self, h: torch.Tensor, obs: torch.Tensor,
                      heard: Optional[tuple] = None) -> torch.Tensor:
        """(B, C) or (B, K, C): which barn row each state ((B, d) or (B, K, d))
        is about.

        Two readings, added. The state's own: a query against each row's
        (fruit, colour) embeddings. And, with `model.lexical_barn`, the innate
        reader's: ``heard`` is what the other party's words say about each
        field (:meth:`read_words`), and a row scores the log-probability of
        its own fruit and colour under it -- the factored lineup choice
        (:meth:`choice_logits`) with the farmer's own lots as the candidates.
        A buyer that says "green pear" in words the farmer can read has
        pointed at a row; nothing has to be learned in the market for that to
        work, where the query alone had to be learned there from nothing.
        Nothing heard, or nothing said about fruit or colour, leaves the rows
        level.
        """
        rows = self.barn_rows(obs)
        key = self.variety_emb(rows[:, :, 0]) + self.color_emb(rows[:, :, 1])   # (B,C,d)
        q = self.lookup_q(h)
        flat = h.dim() == 2
        scores = (torch.einsum("bd,bcd->bc", q, key) if flat
                  else torch.bmm(q, key.transpose(1, 2))) / math.sqrt(self.d_model)
        if heard is not None and self.reads_rows:
            said = (heard[0].gather(1, rows[:, :, 0].clamp(0, heard[0].shape[1] - 1))
                    + heard[1].gather(1, rows[:, :, 1].clamp(0, heard[1].shape[1] - 1)))
            scores = scores + (said if flat else said.unsqueeze(1)).to(scores.dtype)
        return torch.softmax(scores, dim=-1)

    def barn_lookup(self, h: torch.Tensor, obs: torch.Tensor,
                    heard: Optional[tuple] = None) -> torch.Tensor:
        """(B, L, d) -> (B, L, d) (or (B, d) -> (B, d)): each state reads the
        barn row it asks for.

        Keys are each row's (fruit, colour) embeddings -- the same tables the
        words for fruit and colour are grounded in everywhere else -- and values
        its (quality, stock). A state that has decoded "green pears" from the
        buyer's words only has to reproduce those two embeddings as its query
        to read back how many green pears there are and how good they are; and
        the reader's own decoding of them picks the row directly
        (:meth:`row_attention`).
        """
        rows = self.barn_rows(obs)
        val = self.quality_emb(rows[:, :, 2]) + self.field_embed(K_QTY, rows[:, :, 3])
        att = self.row_attention(h, obs, heard)                                 # (B,[L,]C)
        got = (torch.einsum("bc,bcd->bd", att, val) if h.dim() == 2
               else torch.bmm(att, val))
        return h + self.lookup_out(got)

    def barn_concepts(self, h: torch.Tensor, obs: torch.Tensor,
                      heard: Optional[tuple] = None) -> torch.Tensor:
        """(B, 5, d) or (B, K, 5, d): the lot a farmer is talking about, as the
        five concepts its lexicon names -- the fruit, colour, quality and stock
        of the barn row it is attending to (:meth:`row_attention`), and its
        floor price.

        A lot in a barn is the same kind of thing as a lot held in the hand,
        so the same words name it. Without this the lexicon was absent
        wherever the observation was a barn, and a farmer answering a request
        had only its token head to speak with -- which had said nothing in six
        naming rungs (measured on the 2026-10-01 run: the token head alone
        told the values of a field apart no better than chance), so `offer`
        would have had to invent the numbers again.
        """
        rows = self.barn_rows(obs)
        att = self.row_attention(h, obs, heard)
        parts = torch.stack([self.concept(kind, rows[:, :, j].clamp(
            0, self._table(kind).num_embeddings - 1))
            for j, kind in enumerate((K_VARIETY, K_COLOR, K_QUALITY, K_QTY))], dim=2)
        price = self.concept(K_PRICE, obs[:, 4 * self.n_cells].clamp(
            0, self.price_emb.num_embeddings - 1))                              # (B, d)
        if h.dim() == 2:
            mixed = torch.einsum("bc,bcfd->bfd", att, parts)                    # (B, 4, d)
            return torch.cat([mixed, price.unsqueeze(1)], dim=1)
        mixed = torch.einsum("bkc,bcfd->bkfd", att, parts)                      # (B, K, 4, d)
        return torch.cat([mixed, price[:, None, None, :].expand(-1, mixed.shape[1], 1, -1)],
                         dim=2)

    # ------------------------------------------------------------------
    def embed(self, obs: torch.Tensor, tokens: torch.Tensor,
              schema: "list[int] | None" = None,
              self_mask: Optional[torch.Tensor] = None,
              upto: Optional[int] = None,
              meaning: Optional[torch.Tensor] = None) -> torch.Tensor:
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

        ``meaning`` (B, n, d) is what this agent has understood of the other
        party's words (:meth:`listen`): each heard word's meaning, added at
        the slots its atoms sit in, so that what was understood is something
        the state can use and not only something the report heads can say.
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
        if meaning is not None and n_dial > 0:
            m = meaning[:, :n_dial]
            if m.shape[1] < n_dial:
                m = F.pad(m, (0, 0, 0, n_dial - m.shape[1]))
            dial = dial + m.to(dial.dtype)
        parts.append(dial)

        if n > self.dialogue_offset + self.cfg.channel.dialogue_len:
            parts.append(self.slot_emb.weight[SLOT_DECIDE].expand(B, 1, d))

        x = torch.cat(parts, dim=1)[:, :n]
        return x + self.pos_emb.weight[:x.shape[1]].unsqueeze(0)

    def encode(self, obs: torch.Tensor, tokens: torch.Tensor,
               upto: Optional[int] = None,
               schema: "list[int] | None" = None,
               self_mask: Optional[torch.Tensor] = None,
               heard: Any = AUTO, lookup: bool = True) -> torch.Tensor:
        """Hidden states for the prefix of length ``upto`` (default: whole sequence).

        ``heard`` is what this agent has made of the other party's words in
        that prefix (:meth:`listen`): by default it listens here; a caller that
        has already listened passes the result (or None for "nothing heard")
        so the words are read once. The heard words' meanings go in with the
        dialogue (:meth:`embed`) and, looking at a barn, pick the row
        (:meth:`row_attention`). That last reading is of the whole prefix, so
        it is right for the *last* state -- the one generation and every
        decision read; :meth:`full_pass`, which reads states inside earlier
        turns too, passes ``lookup=False`` and looks the rows up turn by turn.
        """
        n = self.seq_len if upto is None else upto
        mask = self._causal[:n, :n]
        barn = bool(self.cfg.model.barn_lookup) and lookup and self.is_barn(schema)
        if heard is AUTO:
            n_dial = max(0, min(self.cfg.channel.dialogue_len, n - self.dialogue_offset))
            want = self.hears_meaning or (barn and self.reads_rows)
            heard = (self.listen(tokens[:, :n_dial], None, self_mask)
                     if want and n_dial > 0 else None)
        meaning = heard.meaning if heard is not None else None
        fields = tuple(heard.fields) if heard is not None and barn else ()

        def run(tok, meaning, *fields):
            x = self.embed(obs, tok, schema, self_mask, upto=n, meaning=meaning)
            h = self.norm(self.encoder(x, mask=mask))
            if barn:
                h = self.barn_lookup(h, obs, fields or None)
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
            return checkpoint(run, tokens, meaning, *fields, use_reentrant=False)
        return run(tokens, meaning, *fields)

    # ------------------------------------------------------------------
    def concept(self, kind: int, values: torch.Tensor) -> torch.Tensor:
        """A field value as a concept, with no context: its embedding, the kind
        of field it is and -- with `model.innate_concepts` -- the kind of thing."""
        v = self.field_embed(kind, values) + self.slot_emb.weight[N_FIXED_SLOT_TYPES + kind]
        if self.innate_concepts:
            v = v + self.concept_emb.weight[CONCEPT_OF_KIND[kind]]
        return v

    def lot_concepts(self, obs: torch.Tensor, schema=None) -> Optional[torch.Tensor]:
        """(B, 5, d): the speaker's own lot, field by field, as concepts -- each
        field's value and kind, with no context -- or None where the
        observation is not a lot to describe (a barn, a lineup)."""
        s = schema if schema is not None else self.schema
        n = N_LOT_FIELDS
        if len(s) <= n or tuple(s[:n]) != tuple(LOT_KINDS) or s[n] != K_FIELD:
            return None
        return torch.stack([self.concept(kind, obs[:, i])
                            for i, kind in enumerate(LOT_KINDS)], dim=1)

    def value_concepts(self) -> list[torch.Tensor]:
        """One (span, d) table per field of a lot: every value it can take, as
        a concept. What a heard word is understood *as* (:meth:`listen`), and
        the rows of the speaker's own lexicon (:meth:`lexicon_table`)."""
        dev = self.slot_emb.weight.device
        return [self.concept(kind, torch.arange(span, device=dev))
                for kind, span in zip(LOT_KINDS, lot_spans(self.cfg.world))]

    def lexicon_table(self) -> torch.Tensor:
        """(M, A) logits: this speaker's whole lexicon -- for every value of
        every field of a lot (M of them, in lot order), the atoms its word
        starts with. What `reward.lexicon_exclusive` keeps one-to-one and
        `reward.lexicon_imitate` moves towards a word heard and understood."""
        lex = self.speaker_lexicon
        return lex.gain * lex.say(lex.norm(torch.cat(self.value_concepts(), dim=0)))

    def set_scaffold(self, value: float) -> None:
        """How much of the description scaffold is left, 1 to 0 (the trainer's
        schedule: `curriculum.scaffold_fade_rung`)."""
        if self.speaks_lexically:
            self.speaker_lexicon.scaffold.fill_(float(min(1.0, max(0.0, value))))

    def set_own_atoms(self, on: bool) -> None:
        """May the token head add atoms of its own to what the lexicon says
        (:attr:`LexicalSpeaker.own_atoms`; the trainer's schedule)?"""
        if self.speaks_lexically:
            self.speaker_lexicon.own_atoms.fill_(1.0 if on else 0.0)

    def speak(self, h: torch.Tensor, obs: torch.Tensor, schema=None,
              turn: Optional[tuple] = None, heard: Optional[tuple] = None,
              habit: bool = False):
        """Token logits from hidden state(s) ``h`` ((B, d) or (B, K, d)): the
        token head's, plus -- with `model.lexical_speaker` -- the mental
        lexicon's word for the part being named (:class:`LexicalSpeaker`).

        Describing a lot, the question (the observation's query slot) is read
        here: asked about one field, the lexicon attends to it. ``turn`` is
        (push, avoid) from :meth:`turn_so_far`: the parts to pass over as a
        new word starts, and whether to go on (parts still unnamed: ending
        loses ``go_on`` nats to a new word) or hold (everything asked for has
        been said: a new word loses them). All of that is the scaffold, and
        fades with it.

        Looking at a barn (`model.lexical_barn`), the parts are those of the
        row being talked about, which ``heard`` -- the reader's reading of the
        other party's words so far -- helps find (:meth:`barn_concepts`). No
        scaffold there: by the market it is gone.

        With ``habit`` the return is (logits, lesson): ``lesson`` is what
        `train.scaffold_distil` needs -- (the token head's logits alone, its
        own choice of part, the choice the scaffold asks for, the next symbol
        the scaffold asks for as logits (zero where it asks for nothing)) --
        or None where there is no scaffold to do without. Every place that
        emits or scores a symbol reads this method, so generation, evaluation
        and a newborn's lessons agree."""
        base = self.token_head(h)
        logits, own = base, None
        if self.speaks_lexically:
            lex = self.speaker_lexicon
            concepts = self.lot_concepts(obs, schema)
            if concepts is not None:
                base = lex.gate(base)
                push, avoid = turn if turn is not None else (None, None)
                q = obs[:, N_LOT_FIELDS]
                asked = (F.one_hot(q.clamp(0, N_LOT_FIELDS - 1), N_LOT_FIELDS)
                         * (q < QUERY_ALL).unsqueeze(-1))
                out = lex(h, concepts, avoid, asked, habit=habit)
                logits = base + out[0]
                plan = torch.zeros_like(base)
                if push is not None:
                    p = push.to(logits.dtype).unsqueeze(-1)
                    plan = lex.go_on * (p.clamp(min=0) * lex._push
                                        + (-p).clamp(min=0) * lex._hold)
                    logits = logits + lex.scaffold * plan
                if habit:
                    own = (base, out[2], out[3], plan)
            elif self.barn_lexicon and self.is_barn(schema):
                base = lex.gate(base)
                logits = base + lex(h, self.barn_concepts(h, obs, heard))[0]
        return (logits, own) if habit else logits

    def describing(self, obs: torch.Tensor, schema, tokens: torch.Tensor,
                   positions: Sequence[int]) -> Optional[torch.Tensor]:
        """(B, P) bool: is the speaker being pushed on (:meth:`turn_so_far`)?"""
        got = self.turn_so_far(obs, schema, tokens, positions)
        return None if got is None else got[0] > 0

    @torch.no_grad()
    def turn_so_far(self, obs: torch.Tensor, schema, tokens: torch.Tensor,
                    positions: Sequence[int]) -> Optional[tuple]:
        """(push (B, P) in {+1, 0, -1}, avoid (B, P, 5) bool) before each
        dialogue position in ``positions``: should this speaker go on (+1: it
        is describing a whole lot and parts are still unnamed), hold (-1: it
        has said as much as it was asked for), or neither; and -- if a new
        word starts there -- which parts has the turn already named? Nothing
        is avoided on a question about one field: there the speaker names the
        field it was asked about, however many atoms its word has, and
        whatever it pointed at.

        The innate pragmatics of the production side -- say as much as the
        question asks, and no more (Grice's maxim of quantity). The speaker
        sees the question (the query slot: one field, or all of it) and
        monitors its own turn: a part counts as named once the turn holds that
        part's word (what the lexicon says for it,
        :meth:`LexicalSpeaker.part_words`) or a gesture at it. Asked for a
        whole lot, it is pushed on while any part is unnamed, never past one
        word per part, and held once every part is named; asked for one field,
        it is held after one word.

        Why it exists: measured on 2026-09-30, speakers who could name every
        field alone, and whose next word -- when made to go on -- named a
        different part of the lot 99-100% of the time, went on after their first
        word 0.02-0.03% of the time. One-field rounds teach "a word, then stop",
        nothing in a whole-lot round says there is more to say, and a
        continuation that is never tried cannot be learned. And the hold:
        nothing said when to stop either, so on the 2026-10-01 run descriptions
        in `mutual` ran to 8.3 words for five parts, 5.4 of them distinct.
        Both are the scaffold's (:class:`LexicalSpeaker`): they fade, and the
        token head is taught to do the same unaided while they do.

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
        w = whole.unsqueeze(1)
        go = w & (unnamed > 0) & (n_words >= 1) & (n_words < N_LOT_FIELDS)
        # as much as was asked for: every part of a whole lot (or a word per
        # part, whatever those words named), or one word for one field
        done = (n_words >= 1) & torch.where(
            w, (unnamed == 0) | (n_words >= N_LOT_FIELDS), torch.ones_like(w))
        push = go.to(torch.int8) - done.to(torch.int8)
        # a new word starts where the symbol before is not a hyphen (the turn's
        # start, a space, a gesture); inside a word the part stays the same
        prev = torch.where(pos > t0, tokens[:, (pos - 1).clamp(min=0)],
                           torch.full_like(tokens[:, :1], c.pad_id).expand(B, pos.numel()))
        new_word = prev != c.hyphen_id
        avoid = said & (whole.unsqueeze(1) & new_word).unsqueeze(-1)
        return push, avoid

    def next_token_logits(self, obs: torch.Tensor, tokens: torch.Tensor,
                          seq_pos: int, schema=None, self_mask=None
                          ) -> tuple[torch.Tensor, torch.Tensor]:
        """Logits for the token that will occupy ``seq_pos``, plus that state's value."""
        p = seq_pos - dialogue_offset(self.cfg)
        hd = self.listen(tokens[:, :p], None, self_mask) if p > 0 else None
        h = self.encode(obs, tokens, upto=seq_pos, schema=schema,
                        self_mask=self_mask, heard=hd)[:, -1]
        turn = self.turn_so_far(obs, schema, tokens, [p])
        return (self.speak(h, obs, schema, None if turn is None
                           else (turn[0][:, 0], turn[1][:, 0]),
                           heard=None if hd is None else hd.fields),
                self.value_head(h).squeeze(-1))

    def listen(self, dialogue: torch.Tensor, ids: Optional[torch.Tensor] = None,
               self_mask: Optional[torch.Tensor] = None) -> Optional[Heard]:
        """What this agent makes of the other party's words in ``dialogue``:
        the innate reader's five log-distributions over what was said about
        each field of a lot, and -- with `model.heard_meaning` -- each heard
        word's meaning at the slots it was heard in (:meth:`LexicalReader.read`).
        None with the reader off, or nothing heard. ``dialogue`` is (B, n) ids
        or (B, n, V) soft one-hots, in which case ``ids`` gives the same
        symbols as ids; n may be any prefix of the dialogue."""
        if not self.lexical or dialogue.shape[1] == 0:
            return None
        if ids is None:
            ids = dialogue if dialogue.dtype == torch.long else dialogue.argmax(-1)
        mine = self._self_mask if self_mask is None else self_mask
        got = self.reader.read(dialogue, ids, ~mine.to(ids.device),
                               self.value_concepts() if self.hears_meaning else None)
        return None if got is None else Heard(fields=got[0], meaning=got[1], said=got[2],
                                              first=got[3], spoke=got[4])

    def read_words(self, dialogue: torch.Tensor, ids: Optional[torch.Tensor] = None,
                   self_mask: Optional[torch.Tensor] = None
                   ) -> Optional[tuple[torch.Tensor, ...]]:
        """What the other party's words say about each field of a lot, read by
        the innate reader -- five log-distributions -- or None (reader off, or
        nothing heard). The reading half of :meth:`listen`."""
        if not self.lexical or dialogue.shape[1] == 0:
            return None
        if ids is None:
            ids = dialogue if dialogue.dtype == torch.long else dialogue.argmax(-1)
        mine = self._self_mask if self_mask is None else self_mask
        return self.reader(dialogue, ids, ~mine.to(ids.device))

    def decision_logits(self, obs: torch.Tensor, tokens: torch.Tensor, schema=None,
                        self_mask=None):
        """Every discrete head, then the value.  Order matches curriculum.py's
        head indices, so callers can slice the first N_HEADS and trust it."""
        hd = self.listen(tokens, None, self_mask)
        h = self.encode(obs, tokens, schema=schema, self_mask=self_mask, heard=hd)[:, -1]
        lex = hd.fields if hd is not None else None
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
        heard = self.listen(tokens, None, self_mask)
        lex = heard.fields if heard is not None else None
        barn = bool(self.cfg.model.barn_lookup) and self.is_barn(schema)
        # looking at a barn, the rows are looked up below, turn by turn: a
        # state found its row by what had been said *when it spoke*, which for
        # every turn but the last is less than the whole conversation
        h = self.encode(obs, tokens, schema=schema, self_mask=self_mask,
                        heard=heard, lookup=not barn)                      # (B, L, d)
        # the dialogue position each of those states emits into
        pos = read_positions - dialogue_offset(self.cfg) + 1
        if barn and read_positions.numel():
            L = self.cfg.channel.max_msg_len
            turn_of = pos // L
            states, logits = [], []
            for t in sorted(set(turn_of.tolist())):
                at = read_positions[turn_of == t]
                said = self.read_words(tokens[:, :t * L], None, self_mask) if t > 0 else None
                hr_t = self.barn_lookup(h[:, at], obs, said)
                states.append(hr_t)
                logits.append(self.speak(hr_t, obs, schema, None, heard=said))
            # read positions come in dialogue order, so the turns are in order
            hr, tok_logits = torch.cat(states, dim=1), torch.cat(logits, dim=1)
        else:
            hr = h[:, read_positions]                       # (B, K, d)
            turn = (self.turn_so_far(obs, schema, tokens, pos.tolist())
                    if read_positions.numel() else None)
            tok_logits = self.speak(hr, obs, schema, turn)  # (B, K, V+1)
        tok_values = self.value_head(hr).squeeze(-1)        # (B, K)
        hd = self.barn_lookup(h[:, -1], obs, lex) if barn else h[:, -1]
        dec = self.all_heads(hd, obs, lex)
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
