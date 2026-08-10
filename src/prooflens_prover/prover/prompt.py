"""REAL-Prover's tactic prompt, transcribed rather than reimplemented.

Source: `Realprover/manager/manage/prompt_manage.py` in `github.com/frenzymath/REAL-Prover`
(`PromptManage.chat_template_to_prompt` and `PromptManage.build_local_prompt_str`).

## Why transcribed

REAL-Prover-v1 is fine-tuned on exactly this string. A prompt that is *nearly* right produces a
model that is quietly worse at proving, and the failure looks like a retrieval result — the arms
would differ in premise quality *and* in how well the model can read them. The predecessor project
documented three separate ways a reasonable-looking reimplementation of a prompt formatter
invalidates a study (`prooflens/src/prooflens/generation/format.py`), which is why the tests below
assert whole expected strings rather than properties of them.

## Three details that are easy to get backwards

1. **Best-ranked premise goes FIRST, i.e. furthest from the goal.** `build_theorems_str` enumerates
   in rank order and the block is placed *above* `STATE:`, so the 6th-best premise is the one
   adjacent to the proof state. The predecessor project deliberately did the opposite — it prepended
   so the best-ranked premise sat next to the state. We follow REAL-Prover, because the model was
   trained that way; the ordering is a knob worth ablating later, not one to change silently.

2. **`build_local_prompt_str` hard-codes the `deepseek` template**, even though `best_first.py`
   threads a `template` argument through everything else and REAL-Prover-v1 is built on
   `Qwen2.5-Math-7B`. It reads like a bug in their code — and following it anyway was a mistake that
   cost a cluster run. **Measured**, on five FATE-M problems: the model produced multilingual token
   salad (`'俾 Evangel Daniel dialogueCSV refriger.diag stmt'`), tactic *fragments* ending in `]` as
   though continuing a `rw [` it had never opened, and bare terms where a tactic belongs. One of
   ~55 tactics made any progress; the 19-tactic model-free repertoire made 290 progress steps on one
   of the same problems. That is the signature of a prompt format a model has never seen, not of a
   weak model.

   The released checkpoint's `tokenizer_config.json` ships a **ChatML** `chat_template`, and
   `config.json` declares `Qwen2ForCausalLM`. A checkpoint's own declaration outranks a transcribed
   prompt builder, so `qwen_chatml` is the default. `deepseek` is kept, because reproducing their
   code exactly is now an interesting ablation rather than the main path.

3. **No BOS token.** Their `deepseek` branch has the `<｜begin▁of▁sentence｜>` line commented out,
   and the released checkpoint agrees: `tokenizer_config.json` has `bos_token: None`. The template
   must not add one.

4. **The turn-end token is not the EOS token.** `eos_token` is `<|endoftext|>` (151643), while a
   ChatML model ends its turn with `<|im_end|>` (151645). vLLM stops on EOS automatically and would
   run straight past `<|im_end|>`, appending whatever came next to the tactic — so `VLLMGenerator`
   resolves `<|im_end|>` to a token id and passes it as `stop_token_ids`. A stop *string* cannot do
   this job: `skip_special_tokens` removes special tokens from the text before any string match.

## The informal-name gap

Their premise blocks carry an `Informal name` — a natural-language gloss served from their own
answer dataset. Our corpus is extracted from Lean's elaborated environment and has no such field, so
`Premise.informal_name` is `""` and the line renders as `Informal name: `.

This is a real distribution shift against a model trained with those glosses present, and it biases
the **calibration gate** downward. It does not bias the arm comparison: the field is identically
empty for `none`, `bm25`, `sv` and `li`. If a `formal name -> informal name` mapping can be joined
onto the corpus from REAL-Prover's released data, every arm should get it at once.
"""

from __future__ import annotations

from prooflens_prover.retrieval.base import (
    PROMPT_PREMISE_LIMIT,
    Premise,
    format_premises,
)

#: DeepSeek's end-of-sequence marker, with the full-width vertical bars (U+FF5C) and the
#: lower-block separator (U+2581) their template uses. Copy-paste, never retype: the ASCII
#: lookalikes `|` and `_` tokenize differently and the model would never see a stop.
EOS_DEEPSEEK = "<｜end▁of▁sentence｜>"

#: Qwen2.5-Math's system message, and so REAL-Prover-v1's: the `chat_template` in the released
#: `tokenizer_config.json` inserts exactly this whenever the caller supplies no system message.
#: `\\boxed{}` reads oddly for tactic generation and is not ours to second-guess — it is what the
#: weights shipped with, and `--template qwen` selects the alternative if it ever proves harmful.
QWEN_MATH_SYSTEM = "Please reason step by step, and put your final answer within \\boxed{}."

#: What REAL-Prover's *own* `qwen` branch sends instead. Same ChatML turn structure, different
#: system message — so the two are different prompts, and only one came with the checkpoint.
QWEN_GENERIC_SYSTEM = "You are a helpful assistant."

_CHATML_SYSTEM = {"qwen": QWEN_GENERIC_SYSTEM, "qwen_chatml": QWEN_MATH_SYSTEM}

#: `chat_template_to_prompt` supports four; `qwen_chatml` is the fifth and is not theirs — it is the
#: `chat_template` the released checkpoint ships, which differs from their `qwen` branch only in the
#: system message.
TEMPLATES = ("deepseek", "qwen", "qwen_chatml", "internlm", "deepseek3")

#: The checkpoint's own declared format. See trap 2: `deepseek` is what their code hard-codes, and
#: sending it to these weights produced token salad on real problems.
DEFAULT_TEMPLATE = "qwen_chatml"

#: The instruction line, verbatim. Note "Here're" — their contraction, not a typo to fix.
TACTIC_INSTRUCTION = "Please generate a tactic in lean4 to solve the state."


def render_chat(messages: list[dict[str, str]], template: str = DEFAULT_TEMPLATE) -> str:
    """Render chat messages to a raw prompt string, as `chat_template_to_prompt` does.

    Line-for-line equivalent to theirs, including the trailing `Assistant:` appended when the last
    message is from the user — that is what makes the model continue rather than start a new turn.
    """
    if template not in TEMPLATES:
        raise ValueError(f"unknown template {template!r}; expected one of {TEMPLATES}")

    result = ""
    if template in _CHATML_SYSTEM:
        result += f"<|im_start|>system\n{_CHATML_SYSTEM[template]}<|im_end|>\n"

    total = len(messages)
    for i, message in enumerate(messages):
        role, content = message["role"], message["content"]
        last_user = (i + 1 == total) and role == "user"

        if template == "deepseek":
            if role == "user":
                result += "User: " + content + "\n\n"
            elif role == "assistant":
                result += "Assistant:" + content + EOS_DEEPSEEK
            elif role == "system":
                result += content + "\n\n"
            if last_user:
                result += "Assistant:"

        elif template in _CHATML_SYSTEM:
            # ChatML. `qwen` and `qwen_chatml` share this loop exactly; they differ only in the
            # system line emitted above, which is the whole point of keeping both.
            if role == "user":
                result += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "assistant":
                result += f"<|im_start|>assistant\n{content}<|im_end|>\n"
            if last_user:
                result += "<|im_start|>assistant\n"

        elif template == "internlm":
            result += "<|im_start|>" + role + "\n" + content
            if i + 1 != total:
                result += "<|im_end|>\n"
            elif role == "user":
                result += "<|im_end|>\n<|im_start|>assistant\n"

        elif template == "deepseek3":
            if role == "user":
                result += f"<｜User｜>{content}"
            elif role == "assistant":
                result += f"<｜Assistant｜>{content}{EOS_DEEPSEEK}"
            if last_user:
                result += "<｜Assistant｜>"

    return result


def build_tactic_content(
    state: str,
    premises: list[Premise] | None = None,
    limit: int = PROMPT_PREMISE_LIMIT,
) -> str:
    """The user-message content, before any chat template wraps it.

    Separated from `build_tactic_prompt` so the content and the wrapping can be checked
    independently: `VLLMGenerator.check_prompt_format` re-renders exactly this through the
    tokenizer's own `chat_template` and compares, which is only possible if the two layers are
    distinguishable. They were not, and the wrapping was wrong for a whole run.

    `premises=None` and `premises=[]` both render the premise block as empty — the header line
    "Here're some theorems that may be helpful:" stays, followed by a blank line. That is what their
    code does when retrieval returns nothing, and it is what the `none` arm must send: an arm that
    also dropped the header would differ from the retrieval arms in *prompt shape* as well as in
    premises, and no comparison between them would isolate retrieval.
    """
    theorems_str = format_premises(premises or [], limit=limit)
    return (
        f"{TACTIC_INSTRUCTION}\n"
        f"Here're some theorems that may be helpful:\n"
        f"{theorems_str}\n"
        f"STATE:\n{state}\nTACTIC:\n"
    )


def build_tactic_prompt(
    state: str,
    premises: list[Premise] | None = None,
    template: str = DEFAULT_TEMPLATE,
    limit: int = PROMPT_PREMISE_LIMIT,
) -> str:
    """The next-tactic prompt for a proof state, equivalent to `build_local_prompt_str`."""
    return render_chat(
        [{"role": "user", "content": build_tactic_content(state, premises, limit)}], template
    )
