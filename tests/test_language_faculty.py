"""Tests for the innate language faculty: words, word classes, composition.

What each class protects:

* a name is a *word*: repeating it does not make it another name, mutual
  exclusivity runs across fields, and a name is paid in full only when said
  once;
* describing a whole lot pays for the speaker's own words for its parts, field
  by field, charges a wrong value's word, and does not care about order -- the
  word-order term does, pair by pair, and adding a field never breaks it;
* which rounds are descriptions: a whole-lot round's describer, each side in
  `mutual`, a buyer's request -- never a one-field round, the guesser or a barn;
* the innate reader segments words as the word grammar makes them, ignores the
  listener's own words and gestures, reads a muted turn as uniform, passes a
  gradient to the speaker's atoms -- and, built as a brain builds it, on three
  seeds, understands a five-word description made of words it only ever
  learned one at a time;
* the production lexicon says a part's word in a context it never trained in;
  asked about a whole lot a speaker is pushed on while parts are unnamed (never
  past one word per part, never on a one-field question), and a part once named
  is passed over;
* names and word order keep their own clocks, a split twin speaks its
  original's words, and a speaker with none describes with the community's;
* innate concepts: the number line orders magnitudes, and switching the faculty
  off restores the plain brain;
* the productivity test is decided by the reserved fields alone, and is
  compared with the same round on trained combinations;
* `name-all` has to cover each field, and a rehearsal nobody measured is unmet;
* in the naming rungs the convention bonus agrees with the community's words.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from orchard.agents import CommNet, LexicalReader, count_parameters, thermometer
from orchard.batched import TensorWorld
from orchard.config import Config
from orchard.conventions import (POPULATION, PopulationUsage, SpeakerLexicon, lot_keys,
                                 naming_keys)
from orchard.curriculum import (ASK_ALL, ReferentialWorld, evaluate_rung, phase_named,
                                rung_budget)
from orchard.env import BUYER, FARMER
from orchard.gesture import gesture_id
from orchard.gumbel import run_and_update_gumbel
from orchard.world import N_LOT_FIELDS, lot_spans

from test_curriculum import agents, cfg_small
from test_rungs import _swap_evidence


def utter(cfg, *words) -> tuple:
    """An utterance's symbols; each word is a tuple of atoms."""
    c = cfg.channel
    out: list[int] = []
    for i, w in enumerate(words):
        if i:
            out.append(c.space_id)
        for j, a in enumerate(w):
            if j:
                out.append(c.hyphen_id)
            out.append(a)
    return tuple(out)


def teach(lex, agent, key, utterance, times=10):
    lex.observe([agent] * times, [key] * times, [tuple(utterance)] * times)


def roomy() -> Config:
    """The small world with an atom for every value of every field (28 values)."""
    cfg = cfg_small()
    cfg.channel.atomic_vocab = 32
    cfg.channel.max_symbols = 24
    return cfg


def offsets(cfg) -> list[int]:
    spans = lot_spans(cfg.world)
    return [sum(spans[:f]) for f in range(N_LOT_FIELDS)]


def code(cfg, f, v) -> tuple:
    """A fixed one-atom word for (field, value): a code with a word per value."""
    return (offsets(cfg)[f] + int(v),)


def turn(cfg, symbols, D=None) -> torch.Tensor:
    c = cfg.channel
    t = torch.full((D or c.dialogue_len,), c.pad_id, dtype=torch.long)
    seq = list(symbols) + [c.end_id]
    t[:len(seq)] = torch.tensor(seq)
    return t


# ==========================================================================
class TestNamesAreWords(unittest.TestCase):
    def setUp(self):
        self.cfg = cfg_small()
        self.cfg.channel.max_symbols = 12
        self.cfg.reward.lexicon = 1.0
        self.cfg.reward.lexicon_min_support = 3

    def test_repeating_a_word_does_not_make_it_another_name(self):
        """Measured on 2026-09-29: banana `a16` and red `a16 a16 a16 ...` -- one
        word naming two things, which utterance-level edit distance read as two
        forms 96% apart."""
        cfg = self.cfg
        lex = SpeakerLexicon(cfg)
        banana, red = (0, 1), (1, 2)
        teach(lex, 1, banana, utter(cfg, (15,)))
        teach(lex, 1, red, utter(cfg, (15,), (15,), (15,)))
        self.assertEqual(lex.names(1)[banana], lex.names(1)[red])
        self.assertEqual(lex.shared(1), {(15,)})
        bonus, used = lex.terms([1, 1], [banana, red],
                                [utter(cfg, (15,)), utter(cfg, (15,), (15,), (15,))])
        self.assertEqual(used, [False, False], "a shared word is nobody's name")
        self.assertLessEqual(max(bonus), 1e-9)

    def test_mutual_exclusivity_runs_across_fields(self):
        cfg = self.cfg
        lex = SpeakerLexicon(cfg)
        apple, green = (0, 0), (1, 3)
        teach(lex, 1, apple, utter(cfg, (3,)))
        teach(lex, 1, green, utter(cfg, (5,)))
        bonus, _ = lex.terms([1], [green], [utter(cfg, (3,))])
        self.assertAlmostEqual(bonus[0], -1.0, places=6,
                               msg="the apple word said of a colour is charged in full")

    def test_a_name_is_said_once(self):
        cfg = self.cfg
        lex = SpeakerLexicon(cfg)
        apple, banana = (0, 0), (0, 1)
        teach(lex, 1, apple, utter(cfg, (3,)))
        teach(lex, 1, banana, utter(cfg, (5,)))
        said = [utter(cfg, (3,)), utter(cfg, (3,), (3,)), utter(cfg, (3,), (9,), (9,)),
                utter(cfg, (5,), (9,))]
        bonus, used = lex.terms([1] * 4, [apple] * 4, said)
        self.assertAlmostEqual(bonus[0], 1.0, places=6)
        self.assertAlmostEqual(bonus[1], 0.5, places=6)
        self.assertAlmostEqual(bonus[2], 1.0 / 3, places=6)
        self.assertAlmostEqual(bonus[3], -1.0, places=6,
                               msg="saying more never softens a wrong word")
        self.assertEqual(used, [True, True, True, False])

    def test_a_name_is_short(self):
        """Twelve hyphenated atoms fill a 24-symbol turn: a name that long can
        never sit beside another word in a description."""
        cfg = self.cfg
        lex = SpeakerLexicon(cfg)
        plum, red, blue = (0, 3), (1, 2), (1, 3)
        teach(lex, 1, plum, utter(cfg, (9,)))
        teach(lex, 1, red, utter(cfg, (1, 2, 3, 4, 5, 6)))
        teach(lex, 1, blue, utter(cfg, (7, 8)))
        bonus, _ = lex.terms([1, 1], [red, blue],
                             [utter(cfg, (1, 2, 3, 4, 5, 6)), utter(cfg, (7, 8))])
        self.assertAlmostEqual(bonus[1], 1.0, places=6, msg="two atoms is short enough")
        self.assertAlmostEqual(bonus[0], 1.0 / 3, places=6, msg="six atoms share it three ways")
        # and a word said of the wrong thing is still charged in full
        bonus, _ = lex.terms([1], [plum], [utter(cfg, (7, 8))])
        self.assertAlmostEqual(bonus[0], -1.0, places=6)

    def test_a_repetition_is_a_stepping_stone_not_a_free_word(self):
        """Measured locally on 2026-09-30: `a19` twelve times over for red beside
        `a19` for plum. A long repetition earns little (short-name rule plus its
        resemblance to the word it repeats); a two-atom one is half a new word;
        a genuinely different word earns in full."""
        cfg = self.cfg
        lex = SpeakerLexicon(cfg)
        plum, red, blue, green = (0, 3), (1, 2), (1, 3), (1, 1)
        teach(lex, 1, plum, utter(cfg, (9,)))
        teach(lex, 1, red, utter(cfg, (9,) * 6))
        teach(lex, 1, blue, utter(cfg, (9, 9)))
        teach(lex, 1, green, utter(cfg, (4,)))
        bonus, _ = lex.terms([1, 1, 1], [red, blue, green],
                             [utter(cfg, (9,) * 6), utter(cfg, (9, 9)), utter(cfg, (4,))])
        self.assertLess(bonus[0], bonus[1], "the long repetition earns least")
        self.assertLess(bonus[1], bonus[2])
        self.assertAlmostEqual(bonus[2], 1.0, places=6)

    def test_a_turn_opened_with_a_gesture_teaches_no_name(self):
        """Measured locally on 2026-09-30: learned from every round, both
        speakers' colour "names" were one of their fruit words -- what they said
        by habit while the gesture did the naming."""
        cfg = self.cfg
        usage = PopulationUsage(cfg)
        rw = ReferentialWorld(cfg, generator=torch.Generator().manual_seed(6))
        ph = phase_named(cfg, "name-fruit").with_informer(FARMER)
        rb = rw.sample(8, informer=FARMER, query=0)
        g = gesture_id(cfg, 0, 0)
        toks = torch.stack([turn(cfg, [g, 3]) if i % 2 == 0 else turn(cfg, [5])
                            for i in range(8)])
        obs = {FARMER: rb.obs(cfg, FARMER), BUYER: rb.obs(cfg, BUYER)}
        terms = usage.speaker_terms(ph, toks, obs, rarity=False, convention=False,
                                    agent_ids={FARMER: [7] * 8})
        usage.observe(terms, 8)
        mine = {w for d in usage.lexicon.forms.get(7, {}).values() for w in d}
        ours = {w for d in usage.pop_lexicon.forms.get(POPULATION, {}).values() for w in d}
        self.assertEqual(mine, {(5,)}, "only the words said without pointing")
        self.assertEqual(ours, {(5,)})

    def test_the_first_word_is_the_name(self):
        cfg = self.cfg
        lex = SpeakerLexicon(cfg)
        teach(lex, 2, (3, 4), utter(cfg, (7, 1), (9,)))       # "a7-a1 a9"
        self.assertEqual(lex.names(2), {(3, 4): (7, 1)})
        self.assertEqual(lex.summary()["2"]["names"], {"quantity=4": "a7-a1"})

    def test_state_round_trip_and_utterance_states_are_refused(self):
        cfg = self.cfg
        lex = SpeakerLexicon(cfg)
        teach(lex, 4, (2, 1), utter(cfg, (1, 2)), 5)
        lex.observe_orders([4] * 4, [[0, 1, 3]] * 4)
        lex._decay(3)
        other = SpeakerLexicon(cfg)
        self.assertTrue(other.load_state(lex.state()))
        self.assertEqual(other.names(4), lex.names(4))
        self.assertAlmostEqual(other.order_count(4, 0, 3), lex.order_count(4, 0, 3), places=6)
        old = {"scale": 1.0, "forms": {1: {(0, 0): {(3, cfg.channel.space_id, 3): 5.0}}},
               "total": {1: {(0, 0): 5.0}}}
        fresh = SpeakerLexicon(cfg)
        self.assertFalse(fresh.load_state(old), "a lexicon of whole utterances is not restored")
        self.assertEqual(fresh.names(1), {})


# ==========================================================================
class TestDescribingAThing(unittest.TestCase):
    def setUp(self):
        self.cfg = roomy()
        self.cfg.reward.lexicon_min_support = 3
        self.lex = SpeakerLexicon(self.cfg)
        for f, span in enumerate(lot_spans(self.cfg.world)):
            for v in range(span):
                teach(self.lex, 1, (f, v), utter(self.cfg, code(self.cfg, f, v)))
        self.lot = (1, 2, 3, 4, 5)

    def words(self, *fields, lot=None):
        lot = lot or self.lot
        return utter(self.cfg, *[code(self.cfg, f, lot[f]) for f in fields])

    def test_all_five_right_words_earn_it_whole(self):
        ct = self.lex.compose_terms([1], [self.lot], [self.words(0, 1, 2, 3, 4)], coef=1.0)
        self.assertAlmostEqual(ct["bonus"][0], 1.0, places=6)
        self.assertEqual(ct["stats"]["fields_reused"], 5)
        self.assertEqual(ct["stats"]["descriptions"], 1)

    def test_each_field_named_is_a_fifth_and_order_does_not_matter(self):
        ct = self.lex.compose_terms([1, 1], [self.lot, self.lot],
                                    [self.words(3, 0, 4), self.words(4, 3, 0)], coef=1.0)
        self.assertAlmostEqual(ct["bonus"][0], 3 / 5, places=6)
        self.assertAlmostEqual(ct["bonus"][1], 3 / 5, places=6)

    def test_a_wrong_value_is_charged_and_a_word_for_nothing_is_free(self):
        cfg = self.cfg
        wrong_colour = (self.lot[1] + 1) % lot_spans(cfg.world)[1]
        u = utter(cfg, code(cfg, 0, self.lot[0]), code(cfg, 1, wrong_colour))
        ct = self.lex.compose_terms([1], [self.lot], [u], coef=1.0)
        self.assertAlmostEqual(ct["bonus"][0], 0.0, places=6, msg="+1 fruit, -1 colour")
        junk = utter(cfg, code(cfg, 0, self.lot[0]), (31,))
        ct = self.lex.compose_terms([1], [self.lot], [junk], coef=1.0)
        self.assertAlmostEqual(ct["bonus"][0], 1 / 5, places=6)

    def test_nothing_named_yet_is_nothing_to_compose_from(self):
        ct = self.lex.compose_terms([99], [self.lot], [self.words(0, 1)], coef=1.0)
        self.assertEqual(ct["bonus"], [0.0])
        self.assertEqual(ct["stats"]["descriptions"], 1)
        ct = self.lex.compose_terms([1], [None], [self.words(0, 1)], coef=1.0)
        self.assertEqual((ct["bonus"], ct["stats"]["descriptions"]), ([0.0], 0),
                         "a one-field round is not a description")

    def test_word_order_is_learned_pair_by_pair(self):
        for _ in range(10):
            self.lex.observe_orders([1], [[3, 1, 0]])        # number, colour, fruit
        said = [self.words(3, 1, 0), self.words(0, 1, 3), self.words(3, 2, 1, 0)]
        ct = self.lex.compose_terms([1] * 3, [self.lot] * 3, said, coef=0.0, order_coef=1.0)
        self.assertEqual(ct["seqs"][0], [3, 1, 0])
        self.assertAlmostEqual(ct["order"][0], 0.5, places=6)
        self.assertAlmostEqual(ct["order"][1], -0.5, places=6)
        self.assertGreater(ct["order"][2], 0.0, "adding quality breaks no pair already said")
        self.assertEqual(self.lex.usual_order(1), [3, 1, 0])
        self.assertEqual(self.lex.summary()["1"]["order"], ["quantity", "colour", "fruit"])


# ==========================================================================
class TestWhatIsADescription(unittest.TestCase):
    def test_which_rounds_describe_a_whole_lot(self):
        cfg = cfg_small()
        rw = ReferentialWorld(cfg, generator=torch.Generator().manual_seed(0))
        ph = phase_named(cfg, "name-all").with_informer(FARMER)
        rb = rw.sample(96, informer=FARMER, mix=ph.mix)
        lots = lot_keys(cfg, ph, FARMER, rb.obs(cfg, FARMER))
        keys = naming_keys(cfg, ph, FARMER, rb.obs(cfg, FARMER))
        for i in range(96):
            if int(rb.query[i]) == ASK_ALL:
                self.assertEqual(lots[i], tuple(int(x) for x in rb.true_meaning[i]))
                self.assertIsNone(keys[i])
            else:
                self.assertIsNone(lots[i])
                self.assertIsNotNone(keys[i])
        self.assertTrue(all(x is None for x in lot_keys(cfg, ph, BUYER, rb.obs(cfg, BUYER))),
                        "the guesser describes nothing")
        mutual = phase_named(cfg, "mutual")
        mb = rw.sample_mutual(8)
        for role in (FARMER, BUYER):
            got = lot_keys(cfg, mutual, role, mb.obs(cfg, role))
            want = [tuple(int(x) for x in r) for r in mb.meaning_of(role)]
            self.assertEqual(got, want)

    def test_a_request_is_a_description_and_a_barn_is_not(self):
        cfg = cfg_small()
        tw = TensorWorld(cfg, device="cpu", generator=torch.Generator().manual_seed(0))
        sb = tw.sample(16)
        order = phase_named(cfg, "order")
        self.assertTrue(all(x is None for x in lot_keys(cfg, order, FARMER, sb.obs(cfg, FARMER))))
        req = lot_keys(cfg, order, BUYER, sb.obs(cfg, BUYER))
        self.assertTrue(all(x is not None and len(x) == N_LOT_FIELDS for x in req))


# ==========================================================================
class TestTheInnateReader(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()
        torch.manual_seed(0)
        self.net = CommNet(self.cfg, BUYER)
        self.nobody = torch.zeros(self.cfg.channel.dialogue_len, dtype=torch.bool)

    def test_words_are_segmented_as_the_grammar_makes_them(self):
        c = self.cfg.channel
        H, S = c.hyphen_id, c.space_id
        toks = torch.stack([turn(self.cfg, [3, H, 7, S, 5]), turn(self.cfg, [9]),
                            turn(self.cfg, [1, S, 2, S, 3, S, 4, S, 6])])
        atom, word, pos, W = self.net.reader.segment(toks, torch.ones_like(toks, dtype=torch.bool))
        self.assertEqual(W, 5)
        got = [[(int(toks[b, i]), int(word[b, i]), int(pos[b, i]))
                for i in atom[b].nonzero().flatten().tolist()] for b in range(3)]
        self.assertEqual(got[0], [(3, 0, 0), (7, 0, 1), (5, 1, 0)], "a3-a7 is one word, a5 another")
        self.assertEqual(got[1], [(9, 0, 0)])
        self.assertEqual([w for _, w, _ in got[2]], [0, 1, 2, 3, 4])

    def test_a_word_means_the_same_wherever_it_is_said(self):
        """Out of context: after a gesture, later in the turn, or with the
        listener's own words about -- the reading of the same words is the same."""
        cfg, c = self.cfg, self.cfg.channel
        S = c.space_id
        plain = turn(cfg, [3, S, 5])
        pointed = turn(cfg, [gesture_id(cfg, 3, 2), 3, S, 5])
        a = self.net.read_words(plain.unsqueeze(0), None, self.nobody)
        b = self.net.read_words(pointed.unsqueeze(0), None, self.nobody)
        for x, y in zip(a, b):
            self.assertTrue(torch.allclose(x, y, atol=1e-6))
        # the listener's own turn is not heard
        L = c.max_msg_len
        mine = torch.zeros(c.dialogue_len, dtype=torch.bool)
        mine[L:2 * L] = True
        chatty = plain.clone()
        chatty[L:L + 3] = torch.tensor([9, S, 11])
        a = self.net.read_words(plain.unsqueeze(0), None, mine)
        b = self.net.read_words(chatty.unsqueeze(0), None, mine)
        for x, y in zip(a, b):
            self.assertTrue(torch.allclose(x, y, atol=1e-6))

    def test_silence_reads_as_uniform_and_nothing_heard_as_nothing(self):
        cfg = self.cfg
        toks = torch.stack([turn(cfg, [3]), turn(cfg, [])])
        lex = self.net.read_words(toks, None, self.nobody)
        for x, span in zip(lex, lot_spans(cfg.world)):
            self.assertTrue(torch.allclose(x[1], torch.full((span,), -torch.log(torch.tensor(float(span))))))
            self.assertAlmostEqual(float(x[0].detach().exp().sum()), 1.0, places=5)
        self.assertIsNone(self.net.read_words(turn(cfg, []).unsqueeze(0), None, self.nobody))

    def test_the_gradient_reaches_the_speakers_atoms(self):
        cfg = self.cfg
        toks = turn(cfg, [3, cfg.channel.space_id, 5]).unsqueeze(0)
        soft = F.one_hot(toks, self.net.tok_emb.num_embeddings).float().requires_grad_(True)
        lex = self.net.read_words(soft, toks, self.nobody)
        sum(x[:, 0].sum() for x in lex).backward()
        self.assertGreater(float(soft.grad[0, 0].abs().sum()), 0.0)
        self.assertGreater(float(soft.grad[0, 2].abs().sum()), 0.0)
        self.assertEqual(float(soft.grad[0, 10].abs().sum()), 0.0, "PAD carries nothing")

    def test_words_learned_alone_are_understood_together(self):
        """The point of the faculty. Trained only on one-word utterances -- one
        field at a time, as the naming rungs teach -- the reader decodes
        five-word descriptions it has never heard, in any word order, and says
        "don't know" about a field no word names.

        The agents' own reader, initialised as a brain is (a bare
        `LexicalReader` with PyTorch's defaults passed this while the real one
        read some field at 0.49-0.85 in every seed: words filed under the
        wrong attribute early were never re-filed). Three seeds."""
        for seed in range(3):
            self._composes(seed)

    def _composes(self, seed: int) -> None:
        cfg = Config()
        c = cfg.channel
        spans = lot_spans(cfg.world)
        torch.manual_seed(seed)
        reader = CommNet(cfg, BUYER).reader
        opt = torch.optim.Adam(reader.parameters(), lr=3e-3)
        D = c.max_msg_len
        heard = torch.ones(D, dtype=torch.bool)

        def batch_of(words_per_row):
            rows = []
            for ws in words_per_row:
                rows.append(turn(cfg, utter(cfg, *ws), D=D))
            return torch.stack(rows)
        g = torch.Generator().manual_seed(1 + seed)
        for _ in range(400):
            f = torch.randint(0, N_LOT_FIELDS, (128,), generator=g)
            v = torch.stack([torch.randint(0, spans[int(x)], (1,), generator=g)[0] for x in f])
            ids = batch_of([[code(cfg, int(a), int(b))] for a, b in zip(f, v)])
            lp = reader(ids, ids, heard)
            loss = 0.0
            for j in range(N_LOT_FIELDS):
                sel = f == j
                if bool(sel.any()):
                    loss = loss - lp[j][sel].gather(1, v[sel].unsqueeze(1)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        # five-word descriptions of random lots, in random word orders
        n = 256
        lots = torch.stack([torch.randint(0, s, (n,), generator=g) for s in spans], dim=1)
        rows = []
        for i in range(n):
            order = torch.randperm(N_LOT_FIELDS, generator=g).tolist()
            rows.append([code(cfg, f, lots[i, f]) for f in order])
        ids = batch_of(rows)
        with torch.no_grad():
            lp = reader(ids, ids, heard)
        for f in range(N_LOT_FIELDS):
            acc = float((lp[f].argmax(-1) == lots[:, f]).float().mean())
            self.assertGreater(acc, 0.95, "seed %d: field %d read at %.2f from a five-word "
                               "description" % (seed, f, acc))
        # a description that leaves the price out: price is left uncertain
        rows = [[code(cfg, f, lots[i, f]) for f in range(4)] for i in range(n)]
        ids = batch_of(rows)
        with torch.no_grad():
            lp = reader(ids, ids, heard)
        self.assertLess(float(lp[4].exp().max(-1).values.mean()), 0.5,
                        "a field no word names should not be read as if it were")
        self.assertGreater(float((lp[0].argmax(-1) == lots[:, 0]).float().mean()), 0.95)


# ==========================================================================
class TestTheSpeakersLexicon(unittest.TestCase):
    """The production half: a word learned alone is available in company."""

    def test_a_word_learned_alone_is_said_in_company(self):
        """Taught only one-field rounds -- the naming rungs -- a speaker asked
        for a whole lot, a context it never trained in, still says the word
        for one of that lot's parts. Measured without the lexicon on two
        seeds: 0.22 and 0.75; with it, 1.00 and 1.00."""
        from orchard.agents import dialogue_offset
        from orchard.curriculum import phase_schema
        cfg = Config()
        spans = lot_spans(cfg.world)
        off = offsets(cfg)
        c = cfg.channel
        torch.manual_seed(0)
        net = CommNet(cfg, FARMER)
        self.assertTrue(net.speaks_lexically)
        rw = ReferentialWorld(cfg, generator=torch.Generator().manual_seed(0))
        ph = phase_named(cfg, "name-all").with_informer(FARMER)
        schema, mask, pos = phase_schema(cfg, FARMER, ph), ph.self_mask(cfg, FARMER), dialogue_offset(cfg)
        opt = torch.optim.Adam(net.parameters(), lr=3e-3)
        silent = torch.full((64, c.dialogue_len), c.pad_id, dtype=torch.long)
        for _ in range(120):
            rb = rw.sample(64, informer=FARMER, mix=(0.2,) * N_LOT_FIELDS + (0.0,))
            lg, _ = net.next_token_logits(rb.obs(cfg, FARMER), silent, pos, schema=schema,
                                          self_mask=mask)
            f = rb.query.clamp(max=N_LOT_FIELDS - 1)
            tgt = torch.tensor([off[int(x)] for x in f]) + rb.true_meaning[torch.arange(64), f]
            opt.zero_grad()
            F.cross_entropy(lg, tgt).backward()
            opt.step()
        with torch.no_grad():
            rb = rw.sample(256, informer=FARMER, query=ASK_ALL)
            obs = rb.obs(cfg, FARMER)
            h = net.encode(obs, silent[:1].expand(256, -1), upto=pos, schema=schema,
                           self_mask=mask)[:, -1]
            words = torch.stack([off[f] + rb.true_meaning[:, f] for f in range(N_LOT_FIELDS)], 1)
            said = net.speak(h, obs, schema)[:, :c.atomic_vocab].argmax(-1)
            lex, att = net.speaker_lexicon(h, net.lot_concepts(obs, schema))
            alone = lex[:, :c.atomic_vocab].argmax(-1)
        self.assertGreater(float((said.unsqueeze(1) == words).any(1).float().mean()), 0.9,
                           "the first word of a whole-lot description names none of its parts")
        self.assertGreater(float((alone.unsqueeze(1) == words).any(1).float().mean()), 0.9)
        self.assertTrue(torch.allclose(att.sum(-1), torch.ones(256), atol=1e-5))

    def test_the_word_depends_on_the_part_not_the_context(self):
        """Whatever the hidden state, attending to one part says that part's
        word: the output layer never sees the context."""
        from orchard.agents import LexicalSpeaker
        cfg = Config()
        torch.manual_seed(1)
        spk = LexicalSpeaker(cfg, 16)
        concepts = torch.randn(2, N_LOT_FIELDS, 16)
        concepts[1, 2] = concepts[0, 2]                 # the same colour, other parts differ
        with torch.no_grad():
            spk.query.weight.zero_()
            spk.query.bias.zero_()
            # a query that picks out part 2 in both rows
            spk.query.bias.copy_(concepts[0, 2] * 50.0)
            out, att = spk(torch.randn(2, 16), concepts)
        self.assertGreater(float(att[0, 2]), 0.99)
        self.assertGreater(float(att[1, 2]), 0.99)
        self.assertTrue(torch.allclose(out[0], out[1], atol=1e-3))
        self.assertTrue(bool((out[:, cfg.channel.atomic_vocab:] == 0).all()),
                        "only atoms are ever pushed")

    def test_asked_for_the_whole_lot_it_goes_on_until_every_part_is_named(self):
        """Say as much as the question asks. Measured before this existed:
        speakers whose next word, when made to go on, named a different part
        99-100% of the time went on after their first word 0.02-0.03% of the
        time, so combining was never tried and could not be learned."""
        from orchard.curriculum import phase_schema
        cfg = Config()
        c = cfg.channel
        S = c.space_id
        torch.manual_seed(0)
        net = CommNet(cfg, FARMER)
        rw = ReferentialWorld(cfg, generator=torch.Generator().manual_seed(0))
        ph = phase_named(cfg, "name-all").with_informer(FARMER)
        schema = phase_schema(cfg, FARMER, ph)
        obs = rw.sample(3, informer=FARMER, query=ASK_ALL).obs(cfg, FARMER)
        w = net.speaker_lexicon.part_words(net.lot_concepts(obs, schema))
        toks = torch.full((3, c.dialogue_len), c.pad_id, dtype=torch.long)

        def put(row, seq):
            toks[row, :len(seq)] = torch.tensor(seq)
        # one part named; every part named; a gesture for the fruit and four words
        put(0, [int(w[0, 0])])
        put(1, sum([[int(w[1, f]), S] for f in range(N_LOT_FIELDS)], [])[:-1])
        put(2, [gesture_id(cfg, 0, int(obs[2, 0]))]
            + sum([[int(w[2, f]), S] for f in range(1, N_LOT_FIELDS)], [])[:-1])
        named_all = {1: 2 * N_LOT_FIELDS - 1, 2: 2 * (N_LOT_FIELDS - 1)}
        push = net.describing(obs, schema, toks, range(12))
        self.assertFalse(bool(push[:, 0].any()), "nothing to push before the first word")
        self.assertTrue(bool(push[0, 1:].all()), "one part named of five: go on")
        for row, end in named_all.items():
            names = len(set(int(x) for x in w[row].tolist()))
            if names == N_LOT_FIELDS:            # distinct words: all named at the end
                self.assertFalse(bool(push[row, end]), "every part named: free to stop")
        # never past one word per part, even saying one word over and over
        toks[0] = c.pad_id
        put(0, sum([[int(w[0, 0]), S] for _ in range(N_LOT_FIELDS)], [])[:-1])
        push = net.describing(obs, schema, toks, [2 * N_LOT_FIELDS - 1])
        self.assertFalse(bool(push[0, 0]))
        # asked about one field, never
        one = rw.sample(3, informer=FARMER, query=1).obs(cfg, FARMER)
        self.assertFalse(bool(net.describing(one, schema, toks, range(12)).any()))
        # the push moves ending's weight to a new word, nothing else, and is learned
        h = torch.randn(3, cfg.model.d_model)
        with torch.no_grad():
            d = (net.speak(h, obs, schema, (torch.ones(3, dtype=torch.bool), None))
                 - net.speak(h, obs, schema))
        g = float(net.speaker_lexicon.go_on.detach())
        self.assertAlmostEqual(float(d[0, c.end_id]), -g, places=5)
        self.assertAlmostEqual(float(d[0, c.space_id]), g, places=5)
        self.assertAlmostEqual(float(d[0].abs().sum()), 2 * g, places=4)
        self.assertTrue(net.speaker_lexicon.go_on.requires_grad)

    def test_a_part_already_named_is_passed_over(self):
        """Inhibition of return. Pushed on without it, speakers named a second
        part and then said its word until the cap: `a11 a28 a28 a28 a28`."""
        from orchard.curriculum import phase_schema
        cfg = Config()
        c = cfg.channel
        torch.manual_seed(2)
        net = CommNet(cfg, FARMER)
        rw = ReferentialWorld(cfg, generator=torch.Generator().manual_seed(2))
        ph = phase_named(cfg, "name-all").with_informer(FARMER)
        schema = phase_schema(cfg, FARMER, ph)
        obs = rw.sample(64, informer=FARMER, query=ASK_ALL).obs(cfg, FARMER)
        concepts = net.lot_concepts(obs, schema)
        h = torch.randn(64, cfg.model.d_model)
        lex = net.speaker_lexicon
        with torch.no_grad():
            _, free = lex(h, concepts)
            top = free.argmax(-1)                                    # the part it would name
            said = F.one_hot(top, N_LOT_FIELDS).bool()
            _, after = lex(h, concepts, said)
        rows = torch.arange(64)
        self.assertTrue(bool((after[rows, top] < free[rows, top]).all()))
        self.assertLess(float(after[rows, top].mean()), 0.2,
                        "a named part still draws the attention")
        # and it runs through speak(): the push and the parts said, together
        toks = torch.full((64, c.dialogue_len), c.pad_id, dtype=torch.long)
        turn = net.turn_so_far(obs, schema, toks, [0])
        self.assertEqual(tuple(turn[1].shape), (64, 1, N_LOT_FIELDS))
        self.assertTrue(lex.inhibit.requires_grad)

    def test_asked_about_one_field_it_names_that_field(self):
        """Answer the question asked. The lexicon used not to look at the
        question, so in `name-color` it went on naming the fruit it learned in
        `name-fruit`; the 2026-10-01 GPU run's colour answers carried the fruit
        (0.66) better than the colour (0.51), and by `name-quantity` words were
        11 atoms long, walking through the parts to reach the one asked."""
        from orchard.curriculum import phase_schema
        cfg = Config()
        c = cfg.channel
        torch.manual_seed(3)
        net = CommNet(cfg, FARMER)
        rw = ReferentialWorld(cfg, generator=torch.Generator().manual_seed(3))
        ph = phase_named(cfg, "name-color").with_informer(FARMER)
        schema = phase_schema(cfg, FARMER, ph)
        lex = net.speaker_lexicon
        silent = torch.full((128, c.dialogue_len), c.pad_id, dtype=torch.long)
        for q in range(N_LOT_FIELDS):
            obs = rw.sample(128, informer=FARMER, query=q).obs(cfg, FARMER)
            asked = F.one_hot(obs[:, N_LOT_FIELDS].clamp(max=N_LOT_FIELDS - 1), N_LOT_FIELDS)
            with torch.no_grad():
                h = torch.randn(128, cfg.model.d_model)
                _, att = lex(h, net.lot_concepts(obs, schema), None, asked)
            self.assertGreater(float(att[:, q].mean()), 0.8,
                               "asked about field %d, the lexicon looks elsewhere" % q)
        self.assertTrue(lex.ask.requires_grad)
        # and speak() reads the question off the observation itself: its
        # lexical term is the asked field's word
        obs = rw.sample(128, informer=FARMER, query=3).obs(cfg, FARMER)
        with torch.no_grad():
            h = torch.randn(128, cfg.model.d_model)
            term = net.speak(h, obs, schema) - net.token_head(h)
            words = lex.part_words(net.lot_concepts(obs, schema))
        hit = term[:, :c.atomic_vocab].argmax(-1) == words[:, 3]
        self.assertGreater(float(hit.float().mean()), 0.8)

    def test_a_one_field_answer_is_never_pushed_or_steered_away(self):
        """Inhibition of return is for choosing the next *word* of a whole-lot
        description. Inside a word it pushed the second atom onto another part,
        and on a one-field question it steered a speaker that had pointed at
        the asked field away from naming it."""
        from orchard.curriculum import phase_schema
        cfg = Config()
        c = cfg.channel
        torch.manual_seed(4)
        net = CommNet(cfg, FARMER)
        rw = ReferentialWorld(cfg, generator=torch.Generator().manual_seed(4))
        ph = phase_named(cfg, "name-quantity").with_informer(FARMER)
        schema = phase_schema(cfg, FARMER, ph)
        obs = rw.sample(4, informer=FARMER, query=3).obs(cfg, FARMER)
        w = net.speaker_lexicon.part_words(net.lot_concepts(obs, schema))
        toks = torch.full((4, c.dialogue_len), c.pad_id, dtype=torch.long)
        toks[:, 0] = torch.tensor([gesture_id(cfg, 3, int(v)) for v in obs[:, 3]])
        toks[:, 1] = w[:, 3]
        toks[:, 2] = c.hyphen_id
        push, avoid = net.turn_so_far(obs, schema, toks, [1, 3])
        self.assertFalse(bool(push.any()))
        self.assertFalse(bool(avoid.any()), "pointing at the quantity steered the words off it")
        # describing a whole lot: inside a word nothing is avoided, a new word
        # passes over what the turn has named
        whole = rw.sample(2, informer=FARMER, query=ASK_ALL).obs(cfg, FARMER)
        w = net.speaker_lexicon.part_words(net.lot_concepts(whole, schema))
        toks = torch.full((2, c.dialogue_len), c.pad_id, dtype=torch.long)
        toks[0, :2] = torch.tensor([int(w[0, 0]), c.hyphen_id])
        toks[1, :2] = torch.tensor([int(w[1, 0]), c.space_id])
        _, avoid = net.turn_so_far(whole, schema, toks, [2])
        self.assertFalse(bool(avoid[0].any()), "inside a word the part stays the same")
        self.assertTrue(bool(avoid[1, 0, 0]), "a new word passes over the fruit just named")

    def test_a_barn_or_a_lineup_gets_no_lexical_term(self):
        from orchard.curriculum import phase_schema
        cfg = Config()
        net = CommNet(cfg, FARMER)
        order = phase_named(cfg, "order")
        barn = phase_schema(cfg, FARMER, order)
        guess = phase_schema(cfg, BUYER, phase_named(cfg, "name-all").with_informer(FARMER))
        obs = torch.zeros((2, len(barn)), dtype=torch.long)
        self.assertIsNone(net.lot_concepts(obs, barn))
        self.assertIsNone(net.lot_concepts(torch.zeros((2, len(guess)), dtype=torch.long), guess))


# ==========================================================================
class TestTheLexiconKeepsItsWords(unittest.TestCase):
    """Names and word order run on their own clocks, travel with an agent, and a
    speaker with none describes with the community's."""

    def setUp(self):
        self.cfg = roomy()
        self.cfg.reward.lexicon_min_support = 3
        self.cfg.reward.usage_half_life_updates = 10
        self.lex = SpeakerLexicon(self.cfg)
        for f, span in enumerate(lot_spans(self.cfg.world)):
            for v in range(span):
                teach(self.lex, 1, (f, v), utter(self.cfg, code(self.cfg, f, v)))
        self.lot = (1, 2, 3, 4, 5)

    def test_names_do_not_expire_where_nothing_can_teach_them(self):
        """A single clock expired every name ~750 updates after `name-all`, and
        composition stopped paying half-way through `mutual`."""
        n = len(self.lex.names(1))
        self.lex._decay(500, names=False, orders=True)
        self.assertEqual(len(self.lex.names(1)), n)
        self.lex._decay(500, names=True, orders=False)
        self.assertEqual(self.lex.names(1), {}, "and they do age where they can be taught")

    def test_word_order_keeps_its_own_clock(self):
        for _ in range(10):
            self.lex.observe_orders([1], [[3, 1, 0]])
        before = self.lex.order_count(1, 3, 1)
        self.lex._decay(50, names=True, orders=False)
        self.assertAlmostEqual(self.lex.order_count(1, 3, 1), before, places=6)
        self.lex._decay(10, names=False, orders=True)
        self.assertAlmostEqual(self.lex.order_count(1, 3, 1), before / 2, places=5)

    def test_a_twin_speaks_its_originals_words(self):
        self.lex.observe_orders([1] * 5, [[0, 1]] * 5)
        self.lex.copy_speaker(1, 7)
        self.assertEqual(self.lex.names(7), self.lex.names(1))
        self.assertAlmostEqual(self.lex.order_count(7, 0, 1), self.lex.order_count(1, 0, 1))
        words = utter(self.cfg, *[code(self.cfg, f, self.lot[f]) for f in range(N_LOT_FIELDS)])
        ct = self.lex.compose_terms([7], [self.lot], [words], coef=1.0)
        self.assertAlmostEqual(ct["bonus"][0], 1.0, places=6)

    def test_a_speaker_with_no_names_describes_with_the_communitys(self):
        community = SpeakerLexicon(self.cfg)
        for f, span in enumerate(lot_spans(self.cfg.world)):
            for v in range(span):
                teach(community, POPULATION, (f, v), utter(self.cfg, code(self.cfg, f, v)))
        words = utter(self.cfg, *[code(self.cfg, f, self.lot[f]) for f in range(N_LOT_FIELDS)])
        alone = self.lex.compose_terms([42], [self.lot], [words], coef=1.0)
        self.assertEqual(alone["bonus"], [0.0])
        helped = self.lex.compose_terms([42], [self.lot], [words], coef=1.0, fallback=community)
        self.assertAlmostEqual(helped["bonus"][0], 1.0, places=6)
        self.assertEqual(helped["stats"]["fields_reused"], N_LOT_FIELDS)

    def test_a_new_field_in_a_description_is_not_taxed(self):
        """Pairs with no history are left out of the average, not counted as 0."""
        for _ in range(10):
            self.lex.observe_orders([1], [[3, 1, 0]])
        said = [utter(self.cfg, *[code(self.cfg, f, self.lot[f]) for f in fs])
                for fs in ([3, 1, 0], [3, 1, 0, 2], [3, 1, 0, 2, 4])]
        ct = self.lex.compose_terms([1] * 3, [self.lot] * 3, said, coef=0.0, order_coef=1.0)
        self.assertAlmostEqual(ct["order"][0], 0.5, places=6)
        self.assertAlmostEqual(ct["order"][1], 0.5, places=6)
        self.assertAlmostEqual(ct["order"][2], 0.5, places=6)

    def test_the_trainer_copies_lexicons_to_the_twins_at_the_split(self):
        from orchard.train import Trainer
        cfg = Config()
        cfg.model.d_model, cfg.model.n_layers, cfg.model.d_ff = 32, 1, 64
        cfg.train.batch_size, cfg.train.device, cfg.log.plot = 32, "cpu", False
        with tempfile.TemporaryDirectory() as tmp:
            tr = Trainer(cfg, tmp, quiet=True)
            try:
                a = tr.pop.farmers[0]
                teach(tr.usage.lexicon, a.agent_id, (0, 1), utter(cfg, (3,)))
                tr.maybe_split_roles(phase_named(cfg, "haggle"), log=lambda *_: None)
                twin = tr.pop.buyers[0]
                self.assertNotEqual(twin.agent_id, a.agent_id)
                self.assertEqual(tr.usage.lexicon.names(twin.agent_id), {(0, 1): (3,)})
            finally:
                tr.close()


# ==========================================================================
class TestInnateConcepts(unittest.TestCase):
    def test_the_thermometer_code(self):
        t = thermometer(torch.tensor([0, 2, 4]), 5)
        self.assertEqual(t.tolist(), [[0, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 1]])

    def test_the_number_line_orders_magnitudes(self):
        torch.manual_seed(0)
        cfg = Config()
        net = CommNet(cfg, FARMER)
        v = torch.arange(cfg.world.max_qty + 1)
        with torch.no_grad():
            line = net.qty_line(thermometer(v, cfg.world.max_qty + 1))
        cos = F.cosine_similarity(line[4:5], line, dim=-1)
        self.assertGreater(float(cos[3]), float(cos[1]), "4 is nearer 3 than 1")
        self.assertGreater(float(cos[5]), float(cos[8]), "4 is nearer 5 than 8")

    def test_a_snapshot_from_before_the_faculty_resumes_without_it(self):
        """The file decides the architecture: a brain saved before the reader
        existed has no reader's weights, so it resumes as the brain it was."""
        from orchard.train import Trainer
        cfg = Config()
        cfg.model.d_model, cfg.model.n_layers, cfg.model.d_ff = 32, 1, 64
        cfg.train.batch_size, cfg.train.device, cfg.log.plot = 32, "cpu", False
        with tempfile.TemporaryDirectory() as tmp:
            tr = Trainer(cfg, tmp, quiet=True)
            try:
                old = cfg.to_dict()
                del old["model"]["lexical_reader"], old["model"]["innate_concepts"]
                tr._adopt_architecture({"config": old})
                self.assertFalse(tr.cfg.model.lexical_reader)
                self.assertFalse(tr.cfg.model.innate_concepts)
                plain = CommNet(tr.cfg, FARMER)
                self.assertFalse(hasattr(plain, "reader"))
            finally:
                tr.close()

    def test_switching_the_faculty_off_restores_the_plain_brain(self):
        cfg = Config()
        cfg.model.lexical_reader = False
        cfg.model.innate_concepts = False
        plain = CommNet(cfg, FARMER)
        self.assertFalse(hasattr(plain, "reader"))
        self.assertFalse(hasattr(plain, "concept_emb"))
        toks = turn(cfg, [3]).unsqueeze(0)
        self.assertIsNone(plain.read_words(toks, None, None))
        self.assertGreater(count_parameters(CommNet(Config(), FARMER)), count_parameters(plain))


# ==========================================================================
def _score(rb, fields):
    """A flawless describer naming only ``fields``, read by a listener that
    picks uniformly among the candidates consistent with what it heard."""
    m, t = rb.meanings, rb.target
    tgt = m[torch.arange(len(t)), t]
    if not fields:
        return 1.0 / m.shape[1]
    idx = list(fields)
    match = (m[:, :, idx] == tgt[:, None, idx]).all(-1).float()
    return float((1.0 / match.sum(1)).mean())


class TestTheProductivityTest(unittest.TestCase):
    def test_only_the_reserved_fields_decide(self):
        cfg = Config()
        rw = ReferentialWorld(cfg, generator=torch.Generator().manual_seed(3))
        for held in (True, False):
            rb = rw.sample(400, informer=FARMER, held_out=held, query=ASK_ALL, combo_only=True)
            m = rb.meanings
            K = m.shape[1]
            self.assertTrue(bool((m[:, :, 3] == m[:, :1, 3]).all()), "one quantity per round")
            self.assertTrue(bool((m[:, :, 4] == m[:, :1, 4]).all()), "one price per round")
            for a in range(K):
                for b in range(a + 1, K):
                    self.assertFalse(bool((m[:, a, :3] == m[:, b, :3]).all(-1).any()),
                                     "two candidates with one combination")
            res = rw.is_held_out(m.reshape(-1, N_LOT_FIELDS)).view(400, K)
            self.assertTrue(bool(res.all()) if held else bool((~res).all()))
            # a code without the combination scores chance; one with it, in full
            self.assertAlmostEqual(_score(rb, (3, 4)), 1.0 / K, places=6)
            self.assertAlmostEqual(_score(rb, (0, 1, 2)), 1.0, places=6)

    def test_the_trainer_compares_like_with_like(self):
        from orchard.train import Trainer
        cfg = Config()
        cfg.model.d_model, cfg.model.n_layers, cfg.model.d_ff = 32, 1, 64
        cfg.train.batch_size, cfg.train.device, cfg.log.plot = 32, "cpu", False
        with tempfile.TemporaryDirectory() as tmp:
            tr = Trainer(cfg, tmp, quiet=True)
            try:
                ph = phase_named(cfg, "name-all").with_informer(FARMER)
                held, seen = tr.holdout_sampler(ph)(64), tr.seen_sampler(ph)(64)
                for rb in (held, seen):
                    m = rb.meanings
                    self.assertTrue(bool((m[:, :, 3:] == m[:, :1, 3:]).all()))
                self.assertTrue(bool(held.held_out.all()))
                self.assertFalse(bool(seen.held_out.any()))
                # a one-field rung keeps its own rounds
                one = phase_named(cfg, "name-color").with_informer(FARMER)
                rb = tr.seen_sampler(one)(32)
                self.assertTrue(bool((rb.query == 1).all()))
            finally:
                tr.close()

    def test_the_evidence_measures_it_and_the_rehearsal_on_every_check(self):
        """Through the real evidence path, light check included: the held-out
        comparison is computed (a sampler the path could not call was once
        swallowed by its `except`, and the whole block silently went missing)
        and every rehearsed kind is scored."""
        from orchard.train import Trainer
        cfg = Config()
        cfg.model.d_model, cfg.model.n_layers, cfg.model.d_ff = 32, 1, 64
        cfg.train.batch_size, cfg.train.device, cfg.log.plot = 32, "cpu", False
        cfg.log.ablation_episodes, cfg.log.topsim_samples = 256, 30
        with tempfile.TemporaryDirectory() as tmp:
            tr = Trainer(cfg, tmp, quiet=True)
            try:
                names = [p.name for p in tr.curriculum.phases]
                tr.curriculum.index = names.index("name-all")
                ev = tr.gather_evidence(tr.curriculum.phase, light=True)
                self.assertEqual(ev["holdout_success"], ev["holdout_success"], "held-out not measured")
                self.assertEqual(ev["seen_success"], ev["seen_success"], "comparison not measured")
                self.assertEqual(sorted(int(k) for k in ev["by_kind"]), list(range(N_LOT_FIELDS + 1)))
                self.assertTrue(all(len(d.get("each", [])) == 2 for d in ev["by_kind"].values()),
                                "each describer is scored on its own")
                # field by field, per guesser: the whole round cannot tell reuse
                # of all three words from reuse of one
                self.assertEqual(sorted(ev["holdout_field_names"]),
                                 sorted("%s %s" % (r, n) for r in ("farmer", "buyer")
                                        for n in ("fruit", "colour", "quality")))
                self.assertEqual(set(ev["holdout_role_ratios"]), {"farmer", "buyer"})
                self.assertNotIn("holdout_error", ev)
                # a one-field rung is not asked, and prints no number
                tr.curriculum.index = names.index("name-color")
                ev = tr.gather_evidence(tr.curriculum.phase, light=True)
                self.assertNotEqual(ev["holdout_success"], ev["holdout_success"])
                self.assertNotIn("holdout_field_names", ev)
            finally:
                tr.close()


# ==========================================================================
class TestNameAllGates(unittest.TestCase):
    def _ev(self, per_field, by_kind=True):
        ev = _swap_evidence(True, True)
        for spk in ev["speakers"].values():
            spk["per_field_coverage"] = list(per_field)
        if by_kind:
            ev["by_kind"] = {k: {"success": 0.95} for k in range(N_LOT_FIELDS)}
        else:
            ev.pop("by_kind", None)
        return ev

    def test_every_field_has_to_be_named(self):
        cfg = cfg_small()
        rung = phase_named(cfg, "name-all")
        lo, _ = rung_budget(cfg, rung)
        ok, checks = evaluate_rung(cfg, rung, self._ev([0.6] * 5), lo)
        self.assertTrue(ok, checks)
        ok, checks = evaluate_rung(cfg, rung, self._ev([0.9, 0.9, 0.9, 0.05, 0.9]), lo)
        self.assertFalse(ok)
        c = checks["farmer describes: names each field"]
        self.assertFalse(c["met"])
        self.assertIn("quantity 0.05", c["detail"])

    def test_an_unmeasured_rehearsal_is_not_a_pass(self):
        cfg = cfg_small()
        rung = phase_named(cfg, "name-all")
        ok, checks = evaluate_rung(cfg, rung, self._ev([0.6] * 5, by_kind=False),
                                   rung_budget(cfg, rung)[0])
        self.assertFalse(ok)
        self.assertEqual(checks["still names fruit"],
                         {"met": False, "detail": "not measured"})


# ==========================================================================
class TestConventionsAreWords(unittest.TestCase):
    def test_in_the_naming_rungs_the_convention_is_the_communitys_word(self):
        cfg = cfg_small()
        cfg.channel.max_symbols = 12
        usage = PopulationUsage(cfg)
        n = 3 * cfg.reward.convention_min_support
        usage.pop_lexicon.observe([POPULATION] * n, [(0, 0)] * n, [utter(cfg, (3,))] * n)
        usage.pop_lexicon.observe([POPULATION] * n, [(0, 1)] * n, [utter(cfg, (5,))] * n)
        rw = ReferentialWorld(cfg, generator=torch.Generator().manual_seed(2))
        ph = phase_named(cfg, "name-fruit").with_informer(FARMER)
        rb = rw.sample(48, informer=FARMER, query=0)
        toks = torch.stack([turn(cfg, [3]) for _ in range(48)])
        obs = {FARMER: rb.obs(cfg, FARMER), BUYER: rb.obs(cfg, BUYER)}
        terms = usage.speaker_terms(ph, toks, obs, rarity=False, convention=True)
        conv = terms[FARMER]["convention"]
        w = cfg.reward.convention
        for i in range(48):
            fruit = int(rb.true_meaning[i, 0])
            want = w if fruit == 0 else -w
            self.assertAlmostEqual(float(conv[i]), want, places=5)


# ==========================================================================
class TestTheTrainingStep(unittest.TestCase):
    def test_descriptions_are_scored_and_one_field_rounds_are_not(self):
        cfg = roomy()
        cfg.channel.max_symbols = 12
        torch.manual_seed(4)
        f, b = agents(cfg)
        usage = PopulationUsage(cfg)
        # every atom a name, so random utterances meet the lexicon
        for a in f:
            for fld, span in enumerate(lot_spans(cfg.world)):
                for v in range(span):
                    teach(usage.lexicon, a.agent_id, (fld, v), utter(cfg, code(cfg, fld, v)))
        rw = ReferentialWorld(cfg, generator=torch.Generator().manual_seed(4))
        ph = phase_named(cfg, "name-all")
        rb = rw.sample(64, informer=FARMER, mix=ph.mix)
        i = torch.arange(64)
        batch, st = run_and_update_gumbel(cfg, rb, f, b, i % 2, (i // 2) % 2,
                                          phase=ph.with_informer(FARMER), usage=usage,
                                          gesture_share=0.0)
        whole = rb.query == ASK_ALL
        comp = batch.res["farmer_compose"]
        self.assertEqual(st.descriptions, int(whole.sum()))
        self.assertTrue(bool((comp[~whole] == 0).all()))
        self.assertTrue(bool((comp[whole] != 0).any()))
        self.assertTrue(0.0 <= st.names_reused <= 1.0)


if __name__ == "__main__":
    unittest.main()
