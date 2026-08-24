"""Table setup: who sits where, and with which knobs.

Everything is **per seat**, because the point of a table is that five
differently configured opponents meet at it. All of it serialises to JSON, so a
UI can edit it and send it back.

The defaults are deliberately conservative. At a table of six, every cost
multiplies: a knob that is merely expensive for one seat becomes six times that
for a full table, and the slowest seat sets the pace for everyone waiting.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace

from .prompt_catalog import catalog, live_variant, recommended_variant

# Who may sit at the table.
#
#   human     — waits for a person to click
#   heuristic — the built-in bot: price against a rough read, no model, no key
#   llm       — a seat driven by a language model, on the user's own key
#
# This is not a closed list. Anything registered through `seat_registry` is just
# as valid a seat kind — that is how outside bots join without the arena having
# to know they exist.
SEAT_KINDS = ("human", "heuristic", "llm", "solver")
SEAT_LABELS = {
    "human": "human",
    "heuristic": "built-in bot",
    "llm": "language model",
    "solver": "solver",
}


def known_seat_kinds() -> frozenset[str]:
    """The built-in seat kinds, plus anything anyone has registered."""
    from .seat_registry import registered_kinds
    return frozenset(SEAT_KINDS) | frozenset(registered_kinds())


# Reasoning levels across providers. What a given model actually supports is a
# question for the provider layer; an unsupported level is translated down to
# the nearest one it does support, so no API is handed a value it will reject.
#
THINKING_LEVELS = ("off", "minimal", "low", "medium", "high", "xhigh")

# Suggestions for the model menu — NOT a whitelist. The client is LiteLLM, so
# any name it understands works; the UI lets you type one in. Validation only
# insists on a provider prefix, because a name without one fails at the first
# decision rather than while the table is being set up.
FLASH = "deepseek/deepseek-v4-flash"
PRO = "deepseek/deepseek-v4-pro"
GPT = "openai/gpt-5.2"
CLAUDE = "anthropic/claude-sonnet-5"
GEMINI = "gemini/gemini-2.5-flash-lite"
LOCAL = "ollama/llama4"          # needs ARENA_LLM_API_BASE; no key, nothing leaves the machine

AVAILABLE_MODELS = [FLASH, PRO, GPT, CLAUDE, GEMINI, LOCAL]


# How many CHIPS make one displayed betting unit. PokerKit counts in whole
# numbers, so a half-blind raise step must never land on half a chip — hence
# everything is held at double and divided only in the UI.
#
# The practical consequence: `big_blind` in chips is ALWAYS twice what the table
# shows. A 2/5 structure is not 2/5 in chips, it is 4/10.
CHIP_SCALE = 2

# The blind menu: label -> (small, big) in chips. The label is what a player
# sees and says out loud; chips are the internal unit.
#
# The structures differ in more than scale: 5/10 puts the small blind at half
# the big blind, while 2/5 puts it at 0.4 — a different preflop game, not the
# same table with new labels.
BLIND_STRUCTURES: dict[str, tuple[int, int]] = {
    "0,5/1": (1, 2),
    "2/5": (4, 10),
    "5/10": (10, 20),
}


class TableConfigError(ValueError):
    """The table setup cannot be used safely."""


@dataclass(frozen=True)
class SeatConfig:
    name: str
    kind: str = "heuristic"
    avatar: str = ""                 # icon filename, e.g. bot_gto.svg
    # --- knobs for model-driven seats ---
    range_model: str = FLASH         # first pass: estimating the range
    decision_model: str = FLASH      # second pass: the decision
    # A fallback model for when the primary one reports a capacity error: an
    # exhausted quota, a rate limit, an overload. "" disables it. This is not
    # cosmetic — a table and anything else on the same account run out together,
    # and here it costs an interrupted evening rather than money.
    fallback_model: str = ""
    # The prompt variant this seat plays; the catalogue decides what is on offer.
    prompt_variant: str = field(default_factory=recommended_variant)
    hedge: int = 1
    # On: preflop comes from precomputed matrices, and the model decides when a
    # spot does not map. Off: preflop comes from base charts. Postflop is the
    # model's either way.
    gto_preflop: bool = True
    solver_enabled: bool = False
    # The solver's budget per postflop decision. Six seconds is deliberately
    # generous: a solver seat pays in time rather than money, and a short budget
    # means it plays shallowly-solved spots.
    solver_budget_s: float = 6.0
    # Model reasoning: "off", or a level. What each provider supports differs —
    # some cannot be turned off at all, and "off" means their lowest level.
    #
    # The default is off, because reasoning multiplies the length of an answer
    # and with it the chance of a timeout.
    thinking: str = "off"
    # Sampling temperature; ``None`` uses the default of 0.2.
    temperature: float | None = None
    # A critic pass, for catalogue variants that support one: a second call that
    # looks for mistakes on the spots that matter — a big raise, an overbet.
    # "" disables it.
    critic_model: str = ""
    # This seat's own budget; ``None`` uses the table's. Models differ widely in
    # speed, so one shared ceiling either strangles the slow ones or leaves the
    # fast ones waiting. A slow model can ask for more here.
    timeout_s: float | None = None
    # How long the first pass may take; ``None`` (an empty field in the UI)
    # means the default: a share of the budget, capped at
    # `RANGE_CALL_CAP_S` (12 s).
    #
    # This is where most timeouts in the archive came from: 51 of 214 decisions
    # from one CLI model and 10 of 70 from another failed on "did not finish in
    # 12.0s". Those models write a range in 8-14s, so a 12s cap sat below their
    # normal time rather than above it.
    range_call_cap_s: float | None = None
    # The decision pass is read as a stream and stops as soon as the action is
    # complete; the rest of the answer is collected in the background for the
    # record. It needs a variant with the `field_order` rule, `hedge == 1`, and a
    # provider that streams.
    stream_early_stop: bool = False

    def validate(self) -> None:
        known = known_seat_kinds()
        if self.kind not in known:
            raise TableConfigError(
                f"{self.name}: kind must be one of {sorted(known)}, not {self.kind!r}")
        if self.kind == "solver" and self.solver_budget_s <= 0:
            raise TableConfigError(
                f"{self.name}: a solver seat needs a positive budget")
        if self.kind != "llm":
            return
        variants = catalog().variants
        if self.prompt_variant not in variants:
            raise TableConfigError(
                f"{self.name}: unknown prompt variant {self.prompt_variant!r}; "
                f"the choices are {', '.join(variants)}")
        if self.fallback_model and "/" not in self.fallback_model:
            raise TableConfigError(
                f"{self.name}: fallback_model={self.fallback_model!r} has no "
                f"provider prefix; something like {FLASH!r} is expected")
        if self.hedge < 1:
            raise TableConfigError(f"{self.name}: hedge must be at least 1")
        if self.stream_early_stop:
            # Three ways this knob can look enabled and do nothing: without
            # `field_order` the model may send its reasoning before the action,
            # with hedge > 1 the streaming branch is never used, and CLI bridges
            # return the whole answer at once.
            cat = catalog()
            if "field_order" not in cat.rule_ids(self.prompt_variant):
                raise TableConfigError(
                    f"{self.name}: stream_early_stop needs a variant with the "
                    f"'field_order' rule; {self.prompt_variant!r} has none")
            if self.hedge > 1:
                raise TableConfigError(
                    f"{self.name}: stream_early_stop and hedge > 1 exclude each "
                    f"other — the streaming branch is unused when hedging")
            if not cat.supports_streaming(self.decision_model):
                raise TableConfigError(
                    f"{self.name}: stream_early_stop: provider "
                    f"{self.decision_model!r} does not stream (CLI bridges return "
                    f"the whole answer at once)")
        # A provider is recognised only by the prefix. Without one, the call
        # fails at the first decision and the table quietly plays the fallback —
        # several hands gone, and nothing in the log but "provider not provided".
        for role, model in (("range_model", self.range_model),
                            ("decision_model", self.decision_model)):
            if "/" not in (model or ""):
                raise TableConfigError(
                    f"{self.name}: {role}={model!r} has no provider prefix; "
                    f"something like {FLASH!r} is expected")
        if self.critic_model:
            if "/" not in self.critic_model:
                raise TableConfigError(
                    f"{self.name}: critic_model={self.critic_model!r} has no "
                    f"provider prefix; something like {FLASH!r} is expected")
            # Outside those variants there is no critic pass, and ignoring the
            # setting silently would mean playing something other than what was
            # configured.
            bayes = catalog().bayes_chain
            if self.prompt_variant not in bayes:
                raise TableConfigError(
                    f"{self.name}: critic_model only applies to critic-capable "
                    f"variants ({', '.join(sorted(bayes))}), "
                    f"not to {self.prompt_variant!r}")
        if self.solver_enabled and self.solver_budget_s <= 0:
            raise TableConfigError(
                f"{self.name}: an enabled solver needs a positive budget")
        if self.thinking not in THINKING_LEVELS:
            raise TableConfigError(
                f"{self.name}: thinking must be one of {THINKING_LEVELS}, "
                f"not {self.thinking!r}")
        if self.temperature is not None and not 0.0 <= self.temperature <= 2.0:
            raise TableConfigError(
                f"{self.name}: temperature must be between 0 and 2, not {self.temperature}")
        if self.range_call_cap_s is not None:
            if self.range_call_cap_s <= 0:
                raise TableConfigError(
                    f"{self.name}: the first-pass cap must be positive "
                    f"(empty means the default), not {self.range_call_cap_s}")
            # A cap longer than the whole budget would leave the decision pass
            # nothing and the spot would end in the fallback.
            budget = self.timeout_s
            if budget and self.range_call_cap_s >= budget:
                raise TableConfigError(
                    f"{self.name}: the first-pass cap "
                    f"({self.range_call_cap_s:g}s) must be shorter than the "
                    f"seat's budget ({budget:g}s)")

    @property
    def uses_llm(self) -> bool:
        return self.kind == "llm"

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class TableConfig:
    seats: tuple[SeatConfig, ...]
    big_blind: int = 2               # in chips; a stack of 200 is 100bb
    # The small blind in chips. 0 means half the big blind. The field exists
    # because in some structures the small blind is NOT half — 2/5, at 0.4bb,
    # being the common one.
    #
    # Chips are the internal unit, not what anyone sees: the table and the hand
    # history both count in big blinds, so "2/5" means the ratio 0.4/1.0 and is
    # stored as 4/10 chips. An even big blind is a requirement, not a
    # preference: the UI steps raises by half a blind, and an odd one would put
    # them on half chips.
    small_blind: int = 0
    start_stack: int = 200
    seed: int = 1
    # The budget for one decision. Models measured between 11 and 22 seconds,
    # so a 20-second cap made the slowest one **time out on every postflop** and
    # the table played a safe check-call for it. Nobody is in a hurry here, so
    # the ceiling is generous; a seat can shorten it for itself.
    timeout_s: float = 45.0
    rebuy: bool = True
    # Move the dealer button after each hand. Off means fixed positions, with
    # everyone in the same seat all game. A group playing for fun often wants
    # that; for measuring a strategy it is a bias,
    # proto je default zapnuto.
    rotate_button: bool = True
    # How low a stack may fall before topping back up to `start_stack`. Never
    # more than a full stack — nobody sits deeper than they arrived.
    rebuy_below_bb: float = 70.0
    # Should seats build reads on their opponents from the table's history?
    # On means the usual statistics from earlier games of the same group, plus a
    # running update from the hands just played. Identity is the SEAT NAME, so a
    # friend has to sit under the name they used last time.
    #
    # Off is useful for measuring a strategy: a read is an input, so two games
    # with different histories are not the same experiment.
    exploits: bool = True

    def validate(self) -> None:
        if not 2 <= len(self.seats) <= 6:
            raise TableConfigError(
                f"a table seats 2 to 6, not {len(self.seats)}")
        names = [s.name for s in self.seats]
        if len(set(names)) != len(names):
            raise TableConfigError(f"duplicate seat names: {names}")
        if self.big_blind < 2:
            raise TableConfigError("the big blind is at least 2 chips "
                                   "(the small blind is half of it)")
        if self.big_blind % 2:
            raise TableConfigError(
                f"the big blind must be an even number of chips, not "
                f"{self.big_blind} — raises step by half a blind")
        if not 0 < self.sb < self.big_blind:
            raise TableConfigError(
                f"the small blind must be between 1 and {self.big_blind - 1} chips, "
                f"ne {self.sb}")
        if self.start_stack < 2 * self.big_blind:
            raise TableConfigError("the starting stack does not cover two blinds")
        if self.timeout_s <= 0:
            raise TableConfigError("the decision budget must be positive")
        stack_bb = self.start_stack / self.big_blind
        if not 0 < self.rebuy_below_bb <= stack_bb:
            raise TableConfigError(
                f"the top-up threshold must be between 0 and a full stack ({stack_bb:g}bb), "
                f"ne {self.rebuy_below_bb:g} bb")
        for seat in self.seats:
            seat.validate()

    @property
    def sb(self) -> int:
        """The small blind in chips; 0 in the config means half the big blind."""
        return self.small_blind or self.big_blind // 2

    @property
    def blinds_label(self) -> str:
        """The structure the way people say it out loud, in units rather than chips."""
        fmt = lambda v: f"{v:g}".replace(".", ",")   # noqa: E731
        return (f"{fmt(self.sb / CHIP_SCALE)}/"
                f"{fmt(self.big_blind / CHIP_SCALE)}")

    @property
    def humans(self) -> tuple[int, ...]:
        return tuple(i for i, s in enumerate(self.seats) if s.kind == "human")

    def estimated_cost_per_hand(self) -> float:
        """A rough cost per hand in USD, so the UI is not silent about spending.

        Based on measured medians for one decision chain and on 0.37 postflop
        decisions per hand per player from real play. It is an order-of-magnitude
        estimate, not accounting.
        """
        # Measured medians for the suggested models. Anything else falls back
        # to a middling guess — the point is an order of magnitude, so that a
        # table full of expensive seats cannot look free.
        per_chain = {FLASH: 0.00034, PRO: 0.00041, GPT: 0.0074,
                     CLAUDE: 0.0102, GEMINI: 0.0003, LOCAL: 0.0}
        total = 0.0
        for seat in self.seats:
            if not seat.uses_llm:
                continue
            price = (per_chain.get(seat.range_model, 0.001) / 2
                     + per_chain.get(seat.decision_model, 0.001) / 2)
            if seat.critic_model:
                # The critic is one extra call on triggered spots only; half a
                # chain is close enough for an order-of-magnitude estimate.
                price += per_chain.get(seat.critic_model, 0.001) / 2
            total += price * seat.hedge * 0.37
        return round(total, 5)

    def as_dict(self) -> dict:
        return {"seats": [s.as_dict() for s in self.seats],
                "big_blind": self.big_blind, "small_blind": self.sb,
                "blinds_label": self.blinds_label, "chip_scale": CHIP_SCALE,
                "start_stack": self.start_stack,
                "start_stack_bb": round(self.start_stack / self.big_blind, 2),
                "seed": self.seed, "timeout_s": self.timeout_s,
                "rebuy": self.rebuy, "rebuy_below_bb": self.rebuy_below_bb,
                "rotate_button": self.rotate_button,
                "exploits": self.exploits,
                "estimated_cost_per_hand": self.estimated_cost_per_hand()}

    @classmethod
    def from_dict(cls, payload: dict) -> "TableConfig":
        rows = payload.get("seats")
        if not isinstance(rows, list) or not rows:
            raise TableConfigError("the table setup needs a 'seats' list")
        known = {f for f in SeatConfig.__dataclass_fields__}
        seats = []
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise TableConfigError(f"seat #{index} is not an object")
            unknown = set(row) - known
            if unknown:
                raise TableConfigError(
                    f"seat #{index}: unknown keys {sorted(unknown)}")
            seats.append(SeatConfig(**{**{"name": f"Seat {index + 1}"}, **row}))
        big_blind = int(payload.get("big_blind", 2))
        # A stack can be given in big blinds, which is natural, or in chips.
        if payload.get("start_stack_bb") is not None:
            start_stack = int(round(float(payload["start_stack_bb"]) * big_blind))
        else:
            start_stack = int(payload.get("start_stack", 100 * big_blind))
        config = cls(
            seats=tuple(seats),
            big_blind=big_blind,
            # Missing or 0 means half the big blind, so older saved tables load.
            small_blind=int(payload.get("small_blind") or 0),
            start_stack=start_stack,
            seed=int(payload.get("seed", 1)),
            # The default comes from the dataclass, so it lives in one place.
            timeout_s=float(payload.get("timeout_s")
                            or cls.__dataclass_fields__["timeout_s"].default),
            rebuy=bool(payload.get("rebuy", True)),
            rebuy_below_bb=float(payload.get("rebuy_below_bb", 70.0)),
            rotate_button=bool(payload.get("rotate_button", True)),
            exploits=bool(payload.get("exploits", True)),
        )
        config.validate()
        return config


def _llm(name: str, range_model: str, decision_model: str, **kwargs) -> SeatConfig:
    return SeatConfig(name=name, kind="llm", range_model=range_model,
                      decision_model=decision_model, **kwargs)


def preset_mixed() -> TableConfig:
    """Mixed models — the tournament version.

    Mind the cost: these models run about a cent per decision chain, so this
    line-up comes to a few dollars per hundred hands.
    """
    return TableConfig(seats=(
        _llm("flash-1", FLASH, FLASH),
        _llm("flash-2", FLASH, FLASH),
        _llm("pro", PRO, PRO),
        _llm("gpt", GPT, GPT),
        _llm("claude", CLAUDE, CLAUDE),
        _llm("split", FLASH, PRO),      # cheap on the range, dearer on the decision
    ))


def preset_cheap() -> TableConfig:
    """One model against built-in bots — nearly free over a long run."""
    return TableConfig(seats=(
        _llm("model", FLASH, FLASH),
        *(SeatConfig(f"house-{i}", kind="heuristic") for i in range(1, 6)),
    ))


def preset_solver() -> TableConfig:
    """You against solver opponents: matrices preflop, a solver postflop.

    The strongest table available without paid calls. The cost is time — the
    solver takes seconds per decision.
    """
    return TableConfig(seats=(
        SeatConfig("You", kind="human"),
        *(SeatConfig(f"solver-{i}", kind="solver")
          for i in range(1, 4)),
        *(SeatConfig(f"house-{i}", kind="heuristic") for i in range(1, 3)),
    ))


def live_prompt_variant() -> str:
    """The prompt variant being played for real elsewhere.

    This preset is meant to show
    that strategy, so a hardcoded variant here would quietly show something
    other than what is actually being played.
    """
    return live_variant()


def live_stream_early_stop() -> bool:
    """May a seat mirroring the live strategy stream with an early stop?

    This cannot be a constant on the seat: an early stop needs a schema with the
    `field_order` rule — the decision before the reasoning — and not every
    variant has one. Hardcoding it on meant that switching the live variant
    failed validation of the whole preset, before the table even started.
    """
    return "field_order" in catalog().rule_ids(live_prompt_variant())


def live_seat_label(suffix: str = "") -> str:
    """A seat name derived from the live variant, e.g. `Zaira · live`.

    A name reads as an opponent's identity, so a seat labelled after one variant
    while playing another would mislead the people at the table and any bot
    building reads from names.
    """
    family = live_prompt_variant().split("_")[0].capitalize()
    return f"{family}{suffix} · live"


def preset_live_copies() -> TableConfig:
    """Six copies of the same strategy, to see how it fares against itself.

    The models are deliberately cheap: the question is how the prompt behaves,
    not which model is better. ``mixed`` is the preset for that.
    """
    variant = live_prompt_variant()
    return TableConfig(seats=tuple(
        _llm(f"live-{i + 1}", FLASH, FLASH, prompt_variant=variant)
        for i in range(6)))


def preset_human_vs_bots() -> TableConfig:
    """You and five bots. Cheap enough to play without watching the bill."""
    seats = [SeatConfig("You", kind="human"),
             _llm("flash", FLASH, FLASH),
             _llm("pro", PRO, PRO),
             SeatConfig("solver", kind="solver"),
             SeatConfig("house-1", kind="heuristic"),
             SeatConfig("house-2", kind="heuristic")]
    return TableConfig(seats=tuple(seats))


def preset_friends_vs_bots() -> TableConfig:
    """Three people against three bots — the usual evening.

    The names are placeholders. Seat names are how the table identifies a
    player, so put in the ones your group actually uses.
    """
    seats = [
        SeatConfig("Player 1", kind="human"),
        SeatConfig("Player 2", kind="human"),
        SeatConfig("Player 3", kind="human"),
        SeatConfig("Bot 1", kind="heuristic", avatar="bot_default.svg"),
        SeatConfig("Bot 2", kind="heuristic", avatar="bot_cyber.svg"),
        SeatConfig("Bot 3", kind="heuristic", avatar="bot_gto.svg"),
    ]
    return TableConfig(seats=tuple(seats))


def preset_friends4_vs_bots() -> TableConfig:
    """Four people against two bots.

    A table seats six, so a fourth person pushes a bot out rather than
    squeezing everyone in.
    """
    seats = [
        SeatConfig(f"Player {i}", kind="human") for i in range(1, 5)
    ] + [
        SeatConfig("Bot 1", kind="heuristic", avatar="bot_default.svg"),
        SeatConfig("Bot 2", kind="heuristic", avatar="bot_cyber.svg"),
    ]
    return TableConfig(seats=tuple(seats))


def preset_friends5_vs_bots() -> TableConfig:
    """Five people against one bot — nearly a full home game."""
    seats = [
        SeatConfig(f"Player {i}", kind="human") for i in range(1, 6)
    ] + [SeatConfig("Bot", kind="heuristic", avatar="bot_default.svg")]
    return TableConfig(seats=tuple(seats))


PRESETS = {
    "friends": preset_friends_vs_bots,
    "friends4": preset_friends4_vs_bots,
    "friends5": preset_friends5_vs_bots,
    "solver": preset_solver,
    "human": preset_human_vs_bots,
    "cheap": preset_cheap,
    "live_copies": preset_live_copies,
    "mixed": preset_mixed,
}


__all__ = ["SeatConfig", "TableConfig", "TableConfigError", "PRESETS",
           "BLIND_STRUCTURES", "CHIP_SCALE",
           "SEAT_KINDS", "SEAT_LABELS", "AVAILABLE_MODELS",
           "THINKING_LEVELS", "known_seat_kinds",
           "FLASH", "PRO", "GPT", "CLAUDE", "GEMINI", "LOCAL"]
