"""What the population has recently been saying, and the two terms built on it.

Two pressures live here, both paid to (or charged to) the *speaker*, both reward
terms rather than restrictions -- nothing here ever stops an agent from saying
anything:

* **Coining costs.** A word costs more the rarer it is in the population's recent
  usage: an established form is cheap, a form nobody has used is dear. This is
  the pressure the previous run lacked -- 913 distinct words, nearly all one-offs,
  because a brand-new four-atom word cost exactly what an established one did.

  The cost is *centred* on the batch: each word is charged its rarity minus the
  average rarity of the words spoken in that batch. So it moves speakers from
  rare forms towards established ones and never makes silence the cheapest thing
  to say. Uncentred, it did exactly that: at the start every word is novel, so
  the only way to avoid the charge was not to talk, and the lineup's describer
  went silent within 10k episodes.

* **Convention.** A speaker is paid for saying what the population currently says
  *for the meaning it is expressing*. Success with one partner can be had with a
  private code the two of you happen to share; this term is the only thing that
  rewards the *population* sharing one. Coherence was 0.000.

  It is contrastive: similarity (1 - normalised edit distance over the emitted
  symbols -- the measure coherence itself uses) to this meaning's modal form,
  minus the similarity to the *closest* of a sample of other meanings' modal
  forms. Without a second half the cheapest way to "agree" is one form for
  everything -- which is what happened (four spaces, every meaning, within 20k
  episodes). Silence and bare punctuation are not conventions: an utterance
  with no atom earns nothing, and modal forms are taken over utterances with at
  least one.

  The contrast subtracts the closest rather than the *average* other form,
  because the average punishes exactly what this project is trying to build. A
  compositional code's forms resemble each other -- that is what sharing a
  morpheme means -- so against the average it reads as undistinctive and is
  taxed for it. Scored over a 48-meaning space, the average form paid an
  arbitrary short code 0.281 and a compositional one 0.133, and paid a collapsed
  fruit-only code 0.204: more than the compositional code it replaces. A GPU run
  at `mutual`, where the task signal starts at zero and nothing else shapes what
  is said, duly collapsed onto it -- 7 words of 1.0 atoms, one word per
  utterance, coherence 0.92, field coverage [0.83, 0.13, 0.05].

  Against the closest other form the question becomes "is this meaning's
  convention the one my form is nearest to", which a collapsed code fails by
  construction: every meaning's modal is the same form, so the closest other is
  identical to its own and the bonus is exactly zero. Compositional 0.062,
  arbitrary 0.158, both collapses 0.000. Choosing between compositional and
  arbitrary is not this term's job -- `min_holdout_ratio` and
  `min_field_coverage` do that -- but paying for the collapse was.

  It is measured on symbols, not whole words, because partial agreement has to
  count for a convention to form at all: with ~11k word types in circulation
  early on, almost no utterance shares a whole word with a meaning's modal form,
  so a word-level bonus was zero nearly everywhere and six farmers stayed on six
  private codes (coherence 0.05).

"Recent" is an exponentially-decayed count with a half-life in training updates
(``reward.usage_half_life_updates``: the same span of learning at any batch
size), so a
convention can still change -- it just has to win over the population to do it.

Conventions are pooled by *meaning kind*, not by role: in the lineup rungs a
farmer and a buyer describing the same lot (fruit, colour, quality, quantity,
price) feed and are measured against the same convention -- and so does a buyer
placing that request in the market. That is what keeps the two roles speaking
one language rather than two.
"""
from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Optional, Sequence

import torch

from .config import Config
from .env import BUYER, FARMER, parse_words
from .world import K_EMPTY, K_FIELD, N_LOT_FIELDS


def _edit(a: Sequence[int], b: Sequence[int]) -> int:
    """Levenshtein distance, written for the call count rather than for looks.

    This is 97% of what the convention bonus costs, and the bonus is charged per
    episode against a sample of other meanings' forms, so it runs tens of
    thousands of times per training update. The row is carried in a rolling
    variable and written in place instead of being rebuilt, and the three-way
    minimum is unrolled: `min(x, y, z)` is a function call per cell.
    """
    if len(a) < len(b):
        a, b = b, a
    nb = len(b)
    if nb == 0:
        return len(a)
    prev = list(range(nb + 1))
    for i, ca in enumerate(a, 1):
        left = i                       # cur[j-1], carried rather than indexed
        diag = prev[0]                 # prev[j-1]
        prev[0] = i
        for j in range(1, nb + 1):
            up = prev[j]                           # delete
            v = diag if ca == b[j - 1] else diag + 1   # substitute
            if left + 1 < v:                       # insert
                v = left + 1
            if up + 1 < v:
                v = up + 1
            diag = up
            prev[j] = left = v
    return prev[nb]


def similarity(a: Sequence[int], b: Sequence[int]) -> float:
    """1 - normalised edit distance: the same measure coherence reports."""
    m = max(len(a), len(b))
    return 1.0 - (_edit(a, b) / m if m else 0.0)


def n_real_fields(cfg: Config, role: int, phase) -> int:
    """How many leading observation slots hold the speaker's actual meaning."""
    from .curriculum import phase_schema
    return sum(1 for k in phase_schema(cfg, role, phase) if k not in (K_EMPTY, K_FIELD))


def query_slots(cfg: Config, role: int, phase) -> list[int]:
    """Observation slots holding *what was asked*, not what is being described.

    A convention is a form for a meaning, and on a rung that asks different
    questions about the same thing the question is part of the meaning: in
    ``name-all`` the same (fruit, colour, quality) is asked about as a whole in
    70% of rounds and one field at a time in the rest, and the right answers are
    different utterances. Keyed on the tuple alone, one meaning's "convention"
    was the modal of a blend of answers to different questions, so matching it
    could not be right more than most of the time.
    """
    from .curriculum import phase_schema
    return [i for i, k in enumerate(phase_schema(cfg, role, phase)) if k == K_FIELD]


def naming_keys(cfg: Config, phase, role: int, obs: torch.Tensor) -> list:
    """(field, value) per episode where one field of a lot is asked about; None otherwise.

    The meaning a name is *for*: in a fruit round, the fruit; in a quantity
    round, the number. A round that asks for a whole lot has no single meaning
    and gets None.
    """
    B = obs.shape[0]
    q = query_slots(cfg, role, phase)
    if not q:
        return [None] * B
    cols = obs[:, [q[0]] + list(range(N_LOT_FIELDS))].tolist()
    out = []
    for row in cols:
        f = int(row[0])
        out.append((f, int(row[1 + f])) if 0 <= f < N_LOT_FIELDS else None)
    return out


def naming_mutual_information(lp: torch.Tensor, have: torch.Tensor, keys: Sequence
                              ) -> Optional[torch.Tensor]:
    """I(value; first spoken symbol) *within the asked-about field*, under the
    speaker's policy, in nats, plus the separation of the values' distributions.

    ``lp`` (B, E) is the log-distribution over the first symbol each speaker
    spoke, ``have`` (B,) which rows have one, ``keys`` the (field, value) per row
    (None for rows with no single meaning). The rows are split by field and the
    objective taken within each -- the distribution of each value is the mean of
    its rows' distributions, the marginal their weighted mean, the information
    H(marginal) - sum_v w_v H(value v), plus the mean pairwise separation --
    then averaged over fields, weighted by rows.

    Within the field, not across all meanings. Measured across all of them on
    `name-quantity` (four fields asked about, twenty values), both speakers
    settled on **one name per field** -- every fruit one form, every quantity
    another -- and the objective read 1.5-2.1 while every value sat at chance:
    with 139 of 190 meaning pairs lying across fields, distinguishing *what
    was asked* (which is in the speaker's observation) satisfied most of it and
    the values, which are the point, hardly counted. Telling fields apart is
    the mutual-exclusivity charge's job (:class:`SpeakerLexicon`), not this
    term's. Differentiable through ``lp``. None if no field has two values.
    """
    by_field: dict[int, list[int]] = {}
    for i, k in enumerate(keys):
        if k is not None and bool(have[i]):
            by_field.setdefault(int(k[0]), []).append(i)
    total = torch.zeros((), device=lp.device)
    n_rows = 0
    for rows in by_field.values():
        part = _naming_objective(lp, [keys[i][1] for i in rows], rows)
        if part is not None:
            total = total + part * len(rows)
            n_rows += len(rows)
    return total / n_rows if n_rows else None


def _naming_objective(lp: torch.Tensor, values: Sequence, rows: Sequence[int]
                      ) -> Optional[torch.Tensor]:
    """Information plus separation among the ``values`` named on these ``rows``."""
    ids: dict = {}
    gidx = [ids.setdefault(v, len(ids)) for v in values]
    if len(ids) < 2:
        return None
    dev = lp.device
    rows_t = torch.tensor(rows, dtype=torch.long, device=dev)
    g = torch.tensor(gidx, dtype=torch.long, device=dev)
    probs = lp[rows_t].exp()
    counts = torch.bincount(g, minlength=len(ids)).float()
    p_m = torch.zeros((len(ids), probs.shape[-1]), device=dev).index_add_(0, g, probs)
    p_m = p_m / counts.unsqueeze(-1)
    w = counts / counts.sum()
    p_bar = (w.unsqueeze(-1) * p_m).sum(0)

    def H(p):
        return -(p * (p + 1e-9).log()).sum(-1)
    mi = H(p_bar) - (w * H(p_m)).sum()
    # Mutual information is zero *with zero gradient* where every meaning's
    # distribution is the same -- which is where an untrained speaker starts, and
    # where it stayed for 90 updates at 0.00 bits. The mean pairwise L1 distance
    # between the meanings' distributions has a gradient of full size at any
    # asymmetry however small, in the direction of that asymmetry, so it is what
    # breaks the symmetry; the information term then sharpens what it started.
    G = p_m.shape[0]
    sep = (p_m.unsqueeze(0) - p_m.unsqueeze(1)).abs().sum(-1) / 2.0        # (G, G) in [0, 1]
    sep = sep.sum() / (G * (G - 1))
    return mi + sep


class SpeakerLexicon:
    """Each speaker's own names for the meanings it has been asked to name.

    The innate assumption behind it -- the one a child brings to a new word --
    is that a name is *for* one meaning and a meaning *has* one name. Nothing
    here says which sounds name which things; it says that whatever a speaker
    has been calling a thing is what it should go on calling it, and that a
    name it already uses for something else is not this thing's name. The
    population-level convention bonus (:class:`PopulationUsage`) asks the same
    of a community and waits for `name-all`; this is per speaker and on from
    the first round, because it is a fact about speakers, not about
    communities: a speaker that has not settled its own names has nothing to
    agree with anyone about.

    Why it matters for learning: the only other pressure on the words in the
    naming rungs is the gradient through the listener, and a listener whose
    reading of the channel is still random gives that gradient no consistent
    direction. This term does not go through the listener at all. A speaker
    that says a random thing each round earns nothing; one that repeats its
    own established name earns `reward.lexicon`; one whose name for this
    meaning is also its name for another is charged for the resemblance. So the
    speaker settles on distinct, consistent names on its own, and the listener
    is left with a stationary code to learn -- which it can, quickly, from the
    gestured rounds (:attr:`orchard.config.GestureConfig.ostensive_coef`).

    A meaning is the value of the field a round asks about -- (fruit, APPLE),
    (quantity, 3) -- so the prior applies wherever a single thing is being
    named: the query rounds of every naming rung, including the rehearsed
    ones. A round that asks for a whole lot has no single meaning to name and
    is left to the convention bonus.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.scale = 1.0
        # agent id -> meaning key -> utterance -> decayed count
        self.forms: dict[int, dict[tuple, dict[tuple, float]]] = {}
        self.total: dict[int, dict[tuple, float]] = {}

    # ---- bookkeeping ---------------------------------------------------
    def _decay(self, n_updates: int = 1) -> None:
        hl = max(1, self.cfg.reward.usage_half_life_updates)
        self.scale *= 0.5 ** (n_updates / hl)
        if self.scale < 1e-6:
            self._renormalise()

    def _renormalise(self) -> None:
        s = self.scale
        forms: dict[int, dict[tuple, dict[tuple, float]]] = {}
        totals: dict[int, dict[tuple, float]] = {}
        for a, per in self.forms.items():
            for key, d in per.items():
                kept = {u: c * s for u, c in d.items() if c * s > 1e-3}
                if kept:
                    forms.setdefault(a, {})[key] = kept
                    totals.setdefault(a, {})[key] = self.total[a][key] * s
        self.forms, self.total = forms, totals
        self.scale = 1.0

    def _has_atom(self, u: Sequence[int]) -> bool:
        A = self.cfg.channel.atomic_vocab
        return any(x < A for x in u)

    def support(self, agent: int, key: tuple) -> float:
        return self.total.get(agent, {}).get(key, 0.0) * self.scale

    def modal(self, agent: int, key: tuple) -> Optional[tuple[int, ...]]:
        d = self.forms.get(agent, {}).get(key)
        if not d:
            return None
        live = [(c, u) for u, c in d.items() if self._has_atom(u)]
        return max(live)[1] if live else None

    def names(self, agent: int) -> dict[tuple, tuple[int, ...]]:
        """This speaker's established names: meaning -> the form it uses for it."""
        need = self.cfg.reward.lexicon_min_support
        out = {}
        for key in self.forms.get(agent, {}):
            if self.support(agent, key) >= need:
                m = self.modal(agent, key)
                if m:
                    out[key] = m
        return out

    def distributions(self, agent: int) -> dict[tuple, list[tuple[tuple[int, ...], float]]]:
        """Per established meaning: its top recent forms and their conditional weights."""
        need = self.cfg.reward.lexicon_min_support
        top = max(1, int(self.cfg.reward.lexicon_top_forms))
        out = {}
        for key, d in self.forms.get(agent, {}).items():
            if self.support(agent, key) < need:
                continue
            live = sorted(((c, u) for u, c in d.items() if self._has_atom(u)), reverse=True)[:top]
            tot = sum(c for c, _ in live)
            if tot > 0:
                out[key] = [(u, c / tot) for c, u in live]
        return out

    # ---- what a round is about ------------------------------------------
    def keys(self, phase, role: int, obs: torch.Tensor) -> list:
        """(field, value) per episode where one field is asked about; None otherwise."""
        return naming_keys(self.cfg, phase, role, obs)

    # ---- the term --------------------------------------------------------
    def terms(self, agents: Sequence[int], keys: Sequence, firsts: Sequence[tuple]
              ) -> tuple[list[float], list[bool]]:
        """Per episode: the lexicon bonus, and whether the utterance *was* the
        speaker's established name for the meaning (the ostensive gate).

        own      expected similarity of the utterance to my recent forms for this
                 meaning (its top forms, weighted by recent use);
        closest  the same against the closest *other* meaning I have named: the
                 mutual-exclusivity charge.
        bonus = reward.lexicon x (own - closest).

        A speaker saying one thing for everything scores own = closest and earns
        nothing; a form that has co-occurred more with this meaning than with
        any other earns when said for it, which is what pulls names apart; a
        settled, distinct name earns the full amount every time it is used.

        ``used`` is true where the utterance is the speaker's modal form for the
        meaning and that form is not also its modal form for another meaning --
        the utterance was, by the speaker's own usage, a word -- and it gates
        the ostensive lesson (`gesture.ostensive_coef`).
        """
        coef = float(self.cfg.reward.lexicon)
        B = len(firsts)
        bonus = [0.0] * B
        used = [False] * B
        if coef <= 0:
            return bonus, used
        dist_cache: dict[int, dict] = {}
        name_cache: dict[int, dict] = {}
        shared_cache: dict[int, set] = {}
        sim_cache: dict[tuple, float] = {}
        exp_cache: dict[tuple, float] = {}

        def sim(a, b):
            k = (a, b)
            v = sim_cache.get(k)
            if v is None:
                v = sim_cache[k] = similarity(a, b)
            return v

        def expected(a, key, u):
            k = (a, key, u)
            v = exp_cache.get(k)
            if v is None:
                v = exp_cache[k] = sum(w * sim(u, f) for f, w in dist_cache[a][key])
            return v
        for i, (a, key, u) in enumerate(zip(agents, keys, firsts)):
            if key is None or not u or not self._has_atom(u):
                continue
            u = tuple(u)
            if a not in dist_cache:
                dist_cache[a] = self.distributions(a)
                nm = name_cache[a] = self.names(a)
                seen: dict[tuple, int] = {}
                for f in nm.values():
                    seen[f] = seen.get(f, 0) + 1
                shared_cache[a] = {f for f, n in seen.items() if n > 1}
            dists = dist_cache[a]
            own = expected(a, key, u) if key in dists else 0.0
            closest = max((expected(a, k, u) for k in dists if k != key), default=0.0)
            bonus[i] = coef * (own - closest)
            mine = name_cache[a].get(key)
            used[i] = mine is not None and mine not in shared_cache[a] and u == mine
        return bonus, used

    def observe(self, agents: Sequence[int], keys: Sequence, firsts: Sequence[tuple]
                ) -> None:
        inc = 1.0 / self.scale
        for a, key, u in zip(agents, keys, firsts):
            if key is None or not u:
                continue
            per = self.forms.setdefault(a, {})
            d = per.setdefault(key, {})
            d[tuple(u)] = d.get(tuple(u), 0.0) + inc
            t = self.total.setdefault(a, {})
            t[key] = t.get(key, 0.0) + inc

    # ---- reporting -------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        """Per speaker: how many meanings it has a name for, how many distinct
        names those are, and how many meanings share a name with another."""
        out: dict[str, Any] = {}
        for a in sorted(self.forms):
            nm = self.names(a)
            forms = list(nm.values())
            distinct = len(set(forms))
            out[str(a)] = {"meanings_named": len(nm), "distinct_names": distinct,
                           "shared_names": len(nm) - distinct,
                           "names": {"%s=%d" % (("fruit", "colour", "quality", "quantity",
                                                  "price")[k[0]], k[1]):
                                     "-".join(str(x) for x in f) for k, f in sorted(nm.items())}}
        return out

    def state(self) -> dict[str, Any]:
        return {"scale": self.scale,
                "forms": {a: {k: dict(d) for k, d in per.items()} for a, per in self.forms.items()},
                "total": {a: dict(t) for a, t in self.total.items()}}

    def load_state(self, st: dict[str, Any]) -> None:
        self.scale = float(st.get("scale", 1.0))
        self.forms = {int(a): {tuple(k): dict(d) for k, d in per.items()}
                      for a, per in (st.get("forms") or {}).items()}
        self.total = {int(a): {tuple(k): float(v) for k, v in t.items()}
                      for a, t in (st.get("total") or {}).items()}


class PopulationUsage:
    """Decayed counts of recent words, and of recent utterances per meaning.

    Counts are stored against a global scale factor so that decaying everything
    is one multiplication, not a pass over every entry.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.scale = 1.0
        # each speaker's own names for the things it has named -- the innate
        # one-name-per-meaning prior, on from the first round
        self.lexicon = SpeakerLexicon(cfg)
        self.words: dict[tuple[int, ...], float] = defaultdict(float)
        self.word_total = 0.0
        self.forms: dict[tuple, dict[tuple[int, ...], float]] = defaultdict(
            lambda: defaultdict(float))
        self.form_total: dict[tuple, float] = defaultdict(float)
        self.episodes = 0
        self._rng = random.Random(cfg.train.seed + 31)

    # ---- bookkeeping -----------------------------------------------------
    def _decay(self, n_updates: int = 1) -> None:
        hl = max(1, self.cfg.reward.usage_half_life_updates)
        self.scale *= 0.5 ** (n_updates / hl)
        if self.scale < 1e-6:
            self._renormalise()

    def _renormalise(self) -> None:
        s = self.scale
        self.words = defaultdict(float, {w: c * s for w, c in self.words.items()
                                         if c * s > 1e-3})
        self.word_total *= s
        forms = defaultdict(lambda: defaultdict(float))
        totals = defaultdict(float)
        for key, d in self.forms.items():
            kept = {u: c * s for u, c in d.items() if c * s > 1e-3}
            if kept:
                forms[key] = defaultdict(float, kept)
                totals[key] = self.form_total[key] * s
        self.forms, self.form_total = forms, totals
        self.scale = 1.0

    def word_share(self, word: tuple[int, ...]) -> float:
        if self.word_total <= 0:
            return 0.0
        return self.words.get(word, 0.0) / self.word_total

    def support(self, key: tuple) -> float:
        return self.form_total.get(key, 0.0) * self.scale

    def _has_atom(self, u: Sequence[int]) -> bool:
        A = self.cfg.channel.atomic_vocab
        return any(x < A for x in u)

    def modal(self, key: tuple) -> Optional[tuple[int, ...]]:
        """The population's commonest utterance (with at least one atom) for this meaning."""
        d = self.forms.get(key)
        if not d:
            return None
        live = [(c, u) for u, c in d.items() if self._has_atom(u)]
        if not live:
            return None
        return max(live)[1]

    # ---- reading a batch -----------------------------------------------
    def _utterances(self, phase, role: int, toks: list[list[int]]
                    ) -> tuple[list[tuple], list[list[tuple[int, ...]]]]:
        """(first utterance's symbols, all words) per episode for one speaking role.

        A form is the speaker's first turn as emitted -- atoms, hyphens and
        spaces, without the end mark. Words (for the coining cost) are parsed
        from every turn it took.
        """
        c = self.cfg.channel
        L = c.max_msg_len
        turns = phase.turns_of(self.cfg, role)
        firsts: list[tuple] = []
        words: list[list[tuple[int, ...]]] = []
        # A living language repeats itself, so parse each distinct turn once.
        seen: dict[tuple, tuple[tuple, list]] = {}
        for row in toks:
            first: tuple = ()
            ws: list[tuple[int, ...]] = []
            for j, t in enumerate(turns):
                seg = tuple(row[t * L:(t + 1) * L])
                hit = seen.get(seg)
                if hit is None:
                    clean = [x for x in seg if x != c.pad_id]
                    hit = (tuple(x for x in clean if x < c.end_id),
                           parse_words(self.cfg, clean))
                    seen[seg] = hit
                if j == 0:
                    first = hit[0]
                ws.extend(hit[1])
            firsts.append(first)
            words.append(ws)
        return firsts, words

    def _keys(self, phase, role: int, obs: torch.Tensor) -> list[tuple]:
        """(meaning kind, what was asked, what it is about) per episode.

        A lot asked about for its colour alone is not the same meaning as the
        whole lot: the population's form for the first is one word and for the
        second is several, so the two must not be folded into one modal form.
        A buyer's request in the market carries "the whole lot" in the same
        slot, so it shares a convention with the `name-all` describer's lot.

        One gather and one transfer: ``obs`` is on the training device, and
        bringing the query slots across separately would be a second
        synchronisation per role per update.
        """
        kind = phase.meaning_kind(role)
        n = n_real_fields(self.cfg, role, phase)
        q = query_slots(self.cfg, role, phase)
        # A slice is a view; gathering columns copies. The farmer's barn has no
        # query slot, so it keeps the slice.
        sel = obs[:, :n] if not q else obs[:, q + list(range(n))]
        return [(kind,) + tuple(r) for r in sel.tolist()]

    def speaker_terms(self, phase, tokens: torch.Tensor,
                      obs_of: dict[int, torch.Tensor], *, rarity: bool = True,
                      convention: bool = True,
                      agent_ids: "dict[int, Sequence[int]] | None" = None
                      ) -> dict[int, dict[str, torch.Tensor]]:
        """Per role: coining cost and convention bonus for each episode, (B,) each.

        ``rarity`` / ``convention`` False skip a term whose weight is currently
        zero (the cost gate), which early on -- when every utterance is new --
        is most of the work. Also returns the parsed batch so :meth:`observe`
        does not parse it twice.

        With ``agent_ids`` (per role, the speaking agent of each episode) the
        speaker's own lexicon term is computed too (:class:`SpeakerLexicon`):
        ``"lexicon"`` (B,) and ``"word_used"`` (B,) bool, the latter marking
        the episodes in which the utterance was the speaker's established name
        for what it was asked about.
        """
        R = self.cfg.reward
        B = tokens.shape[0]
        dev = tokens.device
        toks = tokens.tolist()
        out: dict[int, dict[str, Any]] = {}
        lo = math.log(max(R.rarity_common_share, 1e-12))
        hi = math.log(max(R.rarity_novel_share, 1e-12))
        span = lo - hi
        ready = self.word_total * self.scale >= 50.0
        cache: dict[tuple[int, ...], float] = {}

        def rarity_of(w: tuple[int, ...]) -> float:
            if w not in cache:
                p = self.word_share(w)
                if p <= 0:
                    cache[w] = 1.0
                else:
                    x = (lo - math.log(p)) / span if span > 0 else 1.0
                    cache[w] = min(1.0, max(0.0, x))
            return cache[w]

        parsed = {}
        for role in (FARMER, BUYER):
            if phase.speaks(self.cfg, role):
                parsed[role] = self._utterances(phase, role, toks)
        # the batch's average word rarity, over every word any speaker said
        do_rarity = rarity and ready and bool(R.rarity_cost)
        all_r = ([rarity_of(w) for _, words in parsed.values() for ws in words for w in ws]
                 if do_rarity else [])
        mean_r = sum(all_r) / len(all_r) if all_r else 0.0

        for role, (firsts, words) in parsed.items():
            keys = self._keys(phase, role, obs_of[role])
            rarity = [0.0] * B
            conv = [0.0] * B
            if do_rarity:
                for i, ws in enumerate(words):
                    rarity[i] = R.rarity_cost * sum(rarity_of(w) - mean_r for w in ws)
            if convention and R.convention:
                # a fixed sample of other meanings' conventions to contrast with
                kind = phase.meaning_kind(role)
                need = R.convention_min_support / max(self.scale, 1e-12)
                est = [k for k, v in self.form_total.items() if k[0] == kind and v >= need]
                n_contrast = max(1, int(R.convention_contrast_samples))
                others = self._rng.sample(est, min(n_contrast, len(est)))
                other_modal = [(k, self.modal(k)) for k in others]
                other_modal = [(k, m) for k, m in other_modal if m]
                other_keys = {ko: mo for ko, mo in other_modal}
                modal_cache: dict[tuple, Optional[tuple]] = {}
                sim_cache: dict[tuple, float] = {}
                base_cache: dict[tuple, list[tuple[tuple, float]]] = {}

                def sim(a, b):
                    key = (a, b)
                    v = sim_cache.get(key)
                    if v is None:
                        v = sim_cache[key] = similarity(a, b)
                    return v
                for i, (k, u) in enumerate(zip(keys, firsts)):
                    if not self._has_atom(u):
                        continue                   # no atom: not a convention
                    if k not in modal_cache:
                        modal_cache[k] = (self.modal(k) if self.support(k)
                                          >= R.convention_min_support else None)
                    m = modal_cache[k]
                    if m is None:
                        continue
                    if u not in base_cache:
                        base_cache[u] = [(ko, sim(u, mo))
                                         for ko, mo in other_keys.items()]
                    # The *closest* other convention, not the average one.
                    base = max((v for ko, v in base_cache[u] if ko != k), default=0.0)
                    conv[i] = R.convention * (sim(u, m) - base)
            lex = [0.0] * B
            used = [False] * B
            lkeys: list = [None] * B
            ids: list = []
            if agent_ids is not None and role in agent_ids and R.lexicon > 0:
                ids = [int(x) for x in agent_ids[role]]
                lkeys = self.lexicon.keys(phase, role, obs_of[role])
                lex, used = self.lexicon.terms(ids, lkeys, firsts)
            out[role] = {
                "rarity": torch.tensor(rarity, device=dev),
                "convention": torch.tensor(conv, device=dev),
                "lexicon": torch.tensor(lex, device=dev),
                "word_used": torch.tensor(used, dtype=torch.bool, device=dev),
                "_firsts": firsts, "_words": words, "_keys": keys,
                "_lexicon_keys": lkeys, "_agents": ids,
            }
        return out

    def observe(self, terms: dict[int, dict[str, Any]], n_episodes: int) -> None:
        """Fold a batch that was just played -- one training update -- into recent usage."""
        self._decay(1)
        self.lexicon._decay(1)
        inc = 1.0 / self.scale
        for role, d in terms.items():
            if d.get("_agents"):
                self.lexicon.observe(d["_agents"], d["_lexicon_keys"], d["_firsts"])
            for ws in d["_words"]:
                for w in ws:
                    self.words[w] += inc
                    self.word_total += inc
            for k, u in zip(d["_keys"], d["_firsts"]):
                # A farmer's market utterance is keyed on its whole barn, which
                # never repeats: no convention can form on it, and recording one
                # key per episode would grow without bound. Words still count.
                if k[0] == "barn":
                    continue
                self.forms[k][tuple(u)] += inc
                self.form_total[k] += inc
        self.episodes += n_episodes

    def key_arities(self) -> dict[str, int]:
        """How long a key is, per meaning kind, under the current code.

        Constant within a kind -- the schema decides it -- and it changed when
        the key started carrying what was asked.
        """
        from .curriculum import ladder
        out: dict[str, int] = {}
        for ph in ladder(self.cfg):
            for v in ph.views():
                for role in (FARMER, BUYER):
                    if not v.speaks(self.cfg, role):
                        continue
                    out[v.meaning_kind(role)] = (
                        1 + len(query_slots(self.cfg, role, v))
                        + n_real_fields(self.cfg, role, v))
        return out

    def drop_stale_forms(self) -> int:
        """Forget conventions recorded under a key format this code no longer writes.

        A snapshot from before the key carried *what was asked* stores keys of a
        different arity. Nothing crashes if they are kept -- the new keys simply
        start without support -- but the stale ones stay in the contrast set for
        a couple of half-lives, so a speaker is scored against the modal forms
        of meanings that are not what those keys now denote. They rebuild within
        an update at any real batch size, so dropping them is cheap and honest.
        The word counts are untouched: a word is keyed by its atoms.
        """
        want = self.key_arities()
        stale = [k for k in self.forms
                 if k and k[0] in want and len(k) != want[k[0]]]
        for k in stale:
            self.forms.pop(k, None)
            self.form_total.pop(k, None)
        return len(stale)

    def summary(self) -> dict[str, Any]:
        s = self.scale
        live = sorted(((c * s, w) for w, c in self.words.items()), reverse=True)
        total = self.word_total * s
        return {
            "recent_word_tokens": round(total, 1),
            "recent_word_types": sum(1 for c, _ in live if c >= 0.5),
            "established_types": sum(1 for c, _ in live
                                     if total and c / total >= self.cfg.reward.rarity_common_share),
            "meanings_with_convention": sum(
                1 for k in self.form_total
                if self.form_total[k] * s >= self.cfg.reward.convention_min_support),
        }
