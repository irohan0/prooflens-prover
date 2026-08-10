"""The tactic prompt must be byte-identical to REAL-Prover's.

REAL-Prover-v1 is fine-tuned on exactly one string shape. A prompt that is *nearly* right yields a
model that is quietly worse at proving, and because retrieval changes the prompt, that degradation
would present as a retrieval result. So these tests assert whole expected strings, not properties:
a property test ("contains STATE:") passes on a prompt with the premises in the wrong place.

The reference values below are derived by hand from
`Realprover/manager/manage/prompt_manage.py` — `chat_template_to_prompt` and
`build_local_prompt_str` — not from our own implementation.

Hermetic: no model, no tokenizer, no network.
"""

from __future__ import annotations

import pytest

from prooflens_prover.prover.prompt import (
    DEFAULT_TEMPLATE,
    EOS_DEEPSEEK,
    QWEN_GENERIC_SYSTEM,
    QWEN_MATH_SYSTEM,
    build_tactic_content,
    build_tactic_prompt,
    render_chat,
)
from prooflens_prover.retrieval.base import PROMPT_PREMISE_LIMIT, Premise, format_premises

STATE = "G : Type u_1\ninst✝ : Group G\na b : G\n⊢ a * b = b * a"


def premise(n: int) -> Premise:
    return Premise(formal_name=f"lemma_{n}", formal_statement=f"∀ x, f{n} x = x", score=1.0 / n)


class TestDeepseekTemplate:
    """The template REAL-Prover's code hard-codes, and which their weights were **not** trained on.

    Every test here names `"deepseek"` explicitly. They used to rely on it being the default, which
    was true until the released checkpoint's own `tokenizer_config.json` settled the question the
    other way — and a test of a specific template should say which template it means regardless.
    Kept in full: `--template deepseek` reproduces their code exactly and is now an ablation.
    """

    def test_single_user_turn(self):
        # 'User: ' + content + '\n\n', then 'Assistant:' because the last message is from the user.
        assert render_chat([{"role": "user", "content": "hi"}], "deepseek") == (
            "User: hi\n\nAssistant:"
        )

    def test_no_bos_token(self):
        # Their `<｜begin▁of▁sentence｜>` line is commented out; the tokenizer supplies one.
        assert not render_chat(
            [{"role": "user", "content": "hi"}], "deepseek"
        ).startswith("<｜begin")

    def test_assistant_turn_is_terminated_with_the_eos_marker(self):
        out = render_chat([
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ], "deepseek")
        assert out == f"User: q\n\nAssistant:a{EOS_DEEPSEEK}"

    def test_no_trailing_assistant_cue_when_the_last_turn_is_the_assistant(self):
        out = render_chat([
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ], "deepseek")
        assert not out.endswith("Assistant:")

    def test_system_message_is_bare(self):
        out = render_chat([
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q"},
        ], "deepseek")
        assert out == "sys\n\nUser: q\n\nAssistant:"

    def test_eos_uses_fullwidth_bars_not_ascii(self):
        # `<|end_of_sentence|>` tokenizes as unrelated pieces and the model would never stop.
        assert EOS_DEEPSEEK == "<｜end▁of▁sentence｜>"
        assert "|" not in EOS_DEEPSEEK
        assert "_" not in EOS_DEEPSEEK


class TestOtherTemplates:
    def test_qwen_prepends_a_system_turn_and_opens_the_assistant(self):
        assert render_chat([{"role": "user", "content": "hi"}], "qwen") == (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\nhi<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def test_internlm(self):
        assert render_chat([{"role": "user", "content": "hi"}], "internlm") == (
            "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
        )

    def test_deepseek3(self):
        assert render_chat([{"role": "user", "content": "hi"}], "deepseek3") == (
            "<｜User｜>hi<｜Assistant｜>"
        )

    def test_unknown_template_is_rejected(self):
        # Their code raises NotImplementedError from inside the loop; failing early is equivalent
        # and means a typo in a config cannot silently produce an empty prompt.
        with pytest.raises(ValueError, match="unknown template"):
            render_chat([{"role": "user", "content": "hi"}], "llama")


class TestPremiseBlock:
    """`format_premises` already existed; these pin it against the transcribed source."""

    def test_matches_the_reference_rendering(self):
        assert format_premises([premise(1), premise(2)]) == (
            "ID:0\nFormal name: lemma_1\nInformal name: \nFormal statement: ∀ x, f1 x = x"
            "\n\n"
            "ID:1\nFormal name: lemma_2\nInformal name: \nFormal statement: ∀ x, f2 x = x"
        )

    def test_ids_start_at_zero(self):
        assert format_premises([premise(1)]).startswith("ID:0\n")

    def test_hard_limit_of_six(self):
        # Their `related_theorems[:6]` is not configurable. Retrieval returns top_k=10, so four
        # premises never reach the model, and a top-k ablation that ignores this measures nothing.
        assert PROMPT_PREMISE_LIMIT == 6
        out = format_premises([premise(i) for i in range(1, 11)])
        assert out.count("ID:") == 6
        assert "lemma_6" in out and "lemma_7" not in out

    def test_rank_order_is_preserved_best_first(self):
        # Consequence: the BEST premise is furthest from `STATE:` and the 6th is adjacent to it.
        out = format_premises([premise(1), premise(2), premise(3)])
        assert out.index("lemma_1") < out.index("lemma_2") < out.index("lemma_3")

    def test_empty_premise_list_renders_as_the_empty_string(self):
        assert format_premises([]) == ""


#: The user-message content, which no template change affects. Asserted whole rather than by
#: property, because the predecessor project documented three separate ways a reasonable-looking
#: reimplementation of a prompt formatter invalidates a study.
EXPECTED_CONTENT = (
    "Please generate a tactic in lean4 to solve the state.\n"
    "Here're some theorems that may be helpful:\n"
    "ID:0\nFormal name: lemma_1\nInformal name: \nFormal statement: ∀ x, f1 x = x\n"
    "STATE:\n"
    f"{STATE}\n"
    "TACTIC:\n"
)


class TestTacticPrompt:
    def test_content_is_byte_exact(self):
        assert build_tactic_content(STATE, [premise(1)]) == EXPECTED_CONTENT

    def test_full_prompt_is_byte_exact(self):
        """Default template: ChatML, exactly as the released `tokenizer_config.json` renders it."""
        expected = (
            "<|im_start|>system\n"
            "Please reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n"
            "<|im_start|>user\n"
            f"{EXPECTED_CONTENT}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        assert build_tactic_prompt(STATE, [premise(1)]) == expected

    def test_the_deepseek_rendering_is_still_available_as_an_ablation(self):
        expected = (
            f"User: {EXPECTED_CONTENT}\n\nAssistant:"
        )
        assert build_tactic_prompt(STATE, [premise(1)], "deepseek") == expected

    def test_premises_sit_above_the_state(self):
        out = build_tactic_prompt(STATE, [premise(1)])
        assert out.index("lemma_1") < out.index("STATE:")

    def test_ends_with_the_assistant_cue(self):
        # What makes the model continue rather than open a new turn. ChatML's cue, not deepseek's.
        assert build_tactic_prompt(STATE, [premise(1)]).endswith("<|im_start|>assistant\n")

    @pytest.mark.parametrize("premises", [None, []])
    def test_no_retrieval_keeps_the_header_and_the_prompt_shape(self, premises):
        """The `none` arm must send the same prompt shape as the retrieval arms.

        Dropping the header when there are no premises would make `none` differ from `li` in prompt
        *structure* as well as content, and no comparison between them would isolate retrieval.
        """
        out = build_tactic_prompt(STATE, premises)
        assert "Here're some theorems that may be helpful:\n\nSTATE:" in out
        assert "ID:" not in out

    def test_the_instruction_contraction_is_not_tidied(self):
        # "Here're" is theirs. Correcting it to "Here are" changes the token sequence.
        assert "Here're" in build_tactic_prompt(STATE, [])

    def test_state_is_inserted_verbatim_including_unicode(self):
        out = build_tactic_prompt(STATE, [])
        assert "inst✝ : Group G" in out
        assert "⊢ a * b = b * a" in out

    def test_template_defaults_to_the_one_the_checkpoint_ships(self):
        """The correction that cost a FATE-M run.

        `build_local_prompt_str` hard-codes deepseek even though REAL-Prover-v1 is Qwen-based, and
        following that produced multilingual token salad on real problems: one of ~55 tactics made
        any progress, against 290 progress steps from a 19-tactic repertoire on one of the same
        problems. The checkpoint's own `tokenizer_config.json` ships ChatML, and a checkpoint's
        declaration outranks a transcribed prompt builder.
        """
        assert DEFAULT_TEMPLATE == "qwen_chatml"
        assert build_tactic_prompt(STATE, []) == build_tactic_prompt(STATE, [], "qwen_chatml")
        assert build_tactic_prompt(STATE, []).startswith("<|im_start|>system")

    def test_the_two_chatml_variants_differ_only_in_the_system_message(self):
        """REAL-Prover's own `qwen` branch says "You are a helpful assistant."; the shipped template
        says Qwen2.5-Math's "reason step by step" line. Same turn structure, different prompt — so
        both are kept and neither is allowed to masquerade as the other."""
        shipped = build_tactic_prompt(STATE, [], "qwen_chatml")
        theirs = build_tactic_prompt(STATE, [], "qwen")
        assert QWEN_MATH_SYSTEM in shipped and QWEN_GENERIC_SYSTEM not in shipped
        assert QWEN_GENERIC_SYSTEM in theirs and QWEN_MATH_SYSTEM not in theirs
        # Everything after the system turn is identical.
        assert shipped.split("<|im_end|>\n", 1)[1] == theirs.split("<|im_end|>\n", 1)[1]

    def test_limit_is_overridable_for_the_top_k_ablation(self):
        out = build_tactic_prompt(STATE, [premise(i) for i in range(1, 5)], limit=2)
        assert out.count("ID:") == 2
