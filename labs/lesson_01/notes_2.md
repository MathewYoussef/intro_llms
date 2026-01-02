EOS token:

Now we will take a look at `eos_token_id`.

- This is the end-of-sequence token, like a stop codon in DNA.
- If the model generates an EOS token ID, Transformers treats it as a stop signal and stops generation for that sequence (unless overridden by `min_new_tokens`).

We can inspect what EOS is currently set to by default:

```python
tokenizer.eos_token, tokenizer.eos_token_id
model.generation_config.eos_token_id
```

Example output:

```python
[2, 11]
```

Notes:

- `tokenizer.eos_token` is the special end token as text (a string).
- `tokenizer.eos_token_id` is the integer ID the tokenizer reserves for that token.
- The EOS string is treated as an atomic/special token and maps to one ID (different from normal text which may map to multiple IDs).
- `model.generation_config.eos_token_id` can be a list of IDs (any of these tokens can end generation).
- We can modify/add/override special tokens after loading the model, and we can also override them per `generate()` call.

In our case, we do not have a single EOS token, but a list of valid stop tokens.

We can see what they actually are:

```
tokenizer.eos_token      = <|im_end|>
tokenizer.eos_token_id   = 11
generation eos_token_id  = [2, 11]
2  '</s>'
11 '<|im_end|>'
```

Notes on EOS behavior:

- Our tokenizer’s official EOS token is `<|im_end|>` with an ID of `11`.
- The model’s generation defaults say “stop” if either ID `2` (`</s>`) or ID `11` (`<|im_end|>`) is generated.
- If we call `model.generate(..., eos_token_id=tokenizer.eos_token_id)`, we set it such that it will only stop on ID `11`.
- If we instead call `model.generate(..., eos_token_id=model.generation_config.eos_token_id)`, it will stop on either `2` or `11`.

Pad token:

Now we take a look at `pad_token_id`.

- `pad_token_id` is a token ID used for padding and filling shorter sequences so that a batch has the same length.
- Many operations expect rectangular tensors (same length along the sequence dimension).
- This is useful in beam search where some sequences may end earlier than others:
  - If one sequence hits EOS early and the others keep generating, Transformers pads the finished ones out to the same length to keep everything in one tensor.
- `pad_token_id` tells Transformers which token ID to use for that padding.

We can inspect what it is with:

```python
print("tokenizer.pad_token =", tokenizer.pad_token)
print("tokenizer.pad_token_id =", tokenizer.pad_token_id)

print("model.config.pad_token_id =", getattr(model.config, "pad_token_id", None))
print("model.generation_config.pad_token_id =", getattr(model.generation_config, "pad_token_id", None))
```

Example output:

```text
tokenizer.pad_token = None
tokenizer.pad_token_id = None
model.config.pad_token_id = 0
model.generation_config.pad_token_id = 0
```

- There is no defined padding token, so `tokenizer.pad_token_id` is `None`.
- However, the model itself uses token ID `0` as the pad.

We can inspect what ID `0` is:

```python
print(tokenizer.convert_ids_to_tokens(0))
print(repr(tokenizer.decode([0], skip_special_tokens=False)))
print(0 in tokenizer.all_special_ids, tokenizer.all_special_ids)
```

Example output:

```text
<unk>
'<unk>'
True [1, 11, 0]
```

So padding with ID `0` means padding with `<unk>`.

Just as a reminder, we can decode the special tokens:

```python
for tid in [0, 1, 2, 11]:
    print(tid, tokenizer.convert_ids_to_tokens(tid), repr(tokenizer.decode([tid], skip_special_tokens=False)))
```

Example output:

```text
0 <unk> '<unk>'
1 <s> '<s>'
2 </s> '</s>'
11 <|im_end|> '<|im_end|>'
```

This maps to:

- `0` = `<unk>` (unknown)
- `1` = `<s>` (start-of-sequence)
- `2` = `</s>` (end-of-sequence)
- `11` = `<|im_end|>` (end-of-message / tokenizer EOS)

`<unk>` isn’t necessarily a safe token to use for padding because it can semantically interfere with the model’s attention unless we pass an explicit `attention_mask`.

A safe fallback is to use `pad_token_id=tokenizer.eos_token_id`. It is usually safe to use the EOS token ID as padding because models are typically trained seeing EOS near the end.

Later, when we discuss `attention_mask`, we can pad with `<unk>` with no worries. This becomes more relevant when we discuss batching.

Beam search:

We now turn to beam search. Like greedy decoding (no sampling), this is a deterministic method in which we keep the top `num_beams` partial continuations at every generation step. These partial continuations are referred to as “beams” and are ranked by their cumulative log-probability.

At a new generation step:

- Logits are produced.
- Logits are converted to log-probabilities (log of the probability).
- The new token’s log-prob is added to the total log-prob for that beam.

Compute cost scales with the number of beams, because each beam is decoded at each generation step (typically batched). With `num_beams=3`, we can think of it as ~3× greedy per step.

Example intuition (`num_beams=3`):

- Step 1: we select the top 3 most probable tokens as the start of 3 beams (Prompt+A, Prompt+B, Prompt+C).
- Step 2: we do a forward pass for each beam (Prompt+A, Prompt+B, Prompt+C), producing a next-token distribution for each.
- We compute candidate continuations by extending each beam with its candidate tokens, updating cumulative scores.
- We then keep the cumulative top 3 beams overall (not “top 3 per beam”). Technically, all 3 could come from the same original beam if the other beams score poorly.

If a beam generates EOS, it is finished. Beam search typically ends when enough beams are finished (depending on `early_stopping` and max-length rules).

Early stopping:

Early stopping is a phenomenon in which beam search ends as a consequence of defined criteria (before `max_new_tokens`). The goal is to answer: “Can any unfinished beam beat the finished beam(s)?”

- Finished beams are those which have ended with EOS and are stored as complete beam hypotheses.
- To stop early, the algorithm requires an upper bound on how good an unfinished beam could possibly become if it were extended further.
- Hugging Face uses current scores and length normalization (`length_penalty`) to determine this.
- If `early_stopping=True`, then as soon as the number of finished candidates reaches `num_beams`, the search ends (active beams still searching are stopped).

Example with `early_stopping=True`:

```python
messages = [{"role": "user", "content": "Describe cyanobacteria photoprotective mechanisms"}]

input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    enable_thinking=False,  # on by default in Nemotron
    add_generation_prompt=True,
    return_tensors="pt",
).to(model.device)

with torch.inference_mode():
    out = model.generate(
        input_ids,
        max_new_tokens=50,
        min_new_tokens=20,  # determines minimum tokens before EOS
        do_sample=False,
        num_beams=5,
        num_return_sequences=5,  # so we can see the top beams produced
        early_stopping=True,  # once num_beams finished, stop
        length_penalty=1.0,  # default
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,  # safe as we are not batching yet
    )

prompt_len = input_ids.shape[-1]
for i in range(out.shape[0]):
    gen_ids = out[i, prompt_len:].detach().cpu().tolist()
    print(f"\n--- sequence {i+1} ---")
    print(tokenizer.decode(gen_ids, skip_special_tokens=True))
```

Example output (snippets):

```
--- sequence 1 ---
Cyanobacteria, as oxygenic photosynthetic prokaryotes, face intense light stress in aquatic and terrestrial environments. To protect themselves from photodamage while maintaining efficient photosynthesis, they employ a sophisticated array of **photoprotective mechanisms**. These mechanisms primarily

--- sequence 2 ---
Cyanobacteria, as oxygenic photosynthetic prokaryotes, face intense light stress in aquatic and terrestrial environments. To protect themselves from photodamage while maintaining efficient photosynthesis, they employ a sophisticated array of **photoprotective mechanisms**. These can be

--- sequence 3 ---
Cyanobacteria, as oxygenic photosynthetic prokaryotes, face intense light stress in aquatic and terrestrial environments. To protect themselves from photodamage while maintaining efficient photosynthesis, they employ a sophisticated array of **photoprotective mechanisms**, primarily centered on

--- sequence 4 ---
Cyanobacteria, as photosynthetic prokaryotes, employ a sophisticated array of **photoprotective mechanisms** to shield their photosynthetic apparatus from oxidative damage caused by excess light energy, particularly under high-light conditions or environmental stressors (e.g., intense sunlight

--- sequence 5 ---
Cyanobacteria, as oxygenic photosynthetic prokaryotes, face intense light stress in aquatic and terrestrial environments. To protect themselves from photodamage while maintaining efficient photosynthesis, they employ a sophisticated array of **photoprotective mechanisms**. These mechanisms work
```

If `early_stopping=False`, the search does not end as soon as we collect `num_beams` completed sequences. Instead, a heuristic check asks: “Can any unfinished beam still win?”

- Once `num_beams` finished beams are collected, Hugging Face computes an upper bound on how good the best active beam could possibly get under the scoring scheme (including `length_penalty`).
- If the best-case active score cannot beat the worst of the finished beams (after normalization), then it stops. Otherwise, it continues.
- This is heuristic: the decoder can’t know if a future token will make an active beam beat a finished beam, so it uses score normalization to decide when further search is unlikely to help.

Example with `early_stopping=False`:

```python
messages = [{"role": "user", "content": "Describe cyanobacteria photoprotective mechanisms in 1 sentences"}]

input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    enable_thinking=False,  # on by default in Nemotron
    add_generation_prompt=True,
    return_tensors="pt",
).to(model.device)

with torch.inference_mode():
    out = model.generate(
        input_ids,
        max_new_tokens=100,  # higher means more opportunity for EOS
        do_sample=False,
        num_beams=5,
        num_return_sequences=5,
        early_stopping=False,  # heuristic: can active beams beat finished beams?
        length_penalty=1.0,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

prompt_len = input_ids.shape[-1]
for i in range(out.shape[0]):
    gen_ids = out[i, prompt_len:].detach().cpu().tolist()
    print(f"\n--- sequence {i+1} ---")
    print(tokenizer.decode(gen_ids, skip_special_tokens=True))
```

Example output (snippets):

```
--- sequence 1 ---
Cyanobacteria protect themselves from excess light by employing photoprotective mechanisms such as non‑photochemical quenching (NPQ), where excess energy is safely dissipated as heat via carotenoid‑dependent pathways and the xanthophyll cycle, alongside the production of photoprotective pigments (e.g., scytonemin, mycosporine-like amino acids) and robust antioxidant systems that scavenge reactive oxygen species.

--- sequence 2 ---
Cyanobacteria protect themselves from excess light by employing photoprotective mechanisms such as non‑photochemical quenching (NPQ), where excess energy is safely dissipated as heat via carotenoid‑dependent pathways and the xanthophyll cycle, alongside the production of photoprotective pigments (e.g., scytonemin and mycosporine‑like amino acids) and robust antioxidant systems that scavenge reactive oxygen species.
```

Early stopping = `'never'`:

`early_stopping="never"` is the most conservative beam search stopping rule.

- While `early_stopping=False` uses the *current* length of the beam in its heuristic, `early_stopping="never"` uses the *maximum* length (`max_length`) for normalization.
- After we have `num_beams` finished beams, it does not stop until it determines there are no active beams that can ever beat the worst finished beam under the scoring function.
- It effectively assumes the log-prob of all remaining tokens until `max_length` is 0 (probability 1.0) to form an optimistic upper bound.
- Quick aside: `max_length` and `max_new_tokens` are different knobs, but we can often set them to be equivalent for experiments.

Example:

```python
messages = [{"role": "user", "content": "Describe cyanobacteria photoprotective mechanisms in 1 sentence"}]

input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    enable_thinking=False,  # on by default in Nemotron
    add_generation_prompt=True,
    return_tensors="pt",
).to(model.device)

with torch.inference_mode():
    out = model.generate(
        input_ids,
        max_new_tokens=100,
        do_sample=False,
        num_beams=5,
        num_return_sequences=5,
        early_stopping="never",
        length_penalty=1.0,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,  # safe as we are not batching yet
    )

prompt_len = input_ids.shape[-1]
for i in range(out.shape[0]):
    gen_ids = out[i, prompt_len:].detach().cpu().tolist()
    print(f"\n--- sequence {i+1} ---")
    print(tokenizer.decode(gen_ids, skip_special_tokens=True))
```

Example output:

```
--- sequence 1 ---
Cyanobacteria protect themselves from excess light by employing photoprotective mechanisms such as non‑photochemical quenching (NPQ) and carotenoid‑mediated energy dissipation, which safely dissipate surplus excitation energy as heat to prevent oxidative damage.

--- sequence 2 ---
Cyanobacteria protect themselves from excess light by employing photoprotective mechanisms such as non‑photochemical quenching (NPQ) and carotenoid‑mediated energy dissipation, which safely dissipate surplus excitation energy as heat.

--- sequence 3 ---
Cyanobacteria protect themselves from excess light by employing photoprotective mechanisms such as non‑photochemical quenching (NPQ) and carotenoid‑mediated energy dissipation, which safely dissipate surplus excitation energy as heat and prevent oxidative damage.

--- sequence 4 ---
Cyanobacteria protect themselves from excess light by employing photoprotective mechanisms such as non‑photochemical quenching (NPQ) and carotenoid‑mediated energy dissipation, which safely dissipate surplus excitation energy as heat and prevent oxidative damage to the photosynthetic apparatus.

--- sequence 5 ---
Cyanobacteria protect themselves from excess light by employing photoprotective mechanisms such as non‑photochemical quenching (NPQ), where excess excitation energy is safely dissipated as heat via carotenoid‑mediated pathways and the xanthophyll cycle, alongside the production of photoprotective pigments (e.g., scytonemin and mycosporine-like amino acids) that absorb and shield against harmful UV and high‑intensity radiation.
```

Beam search tracing / internal bookkeeping:

We now return to internal bookkeeping to get a feel for what is happening under the hood for these calls. We implement a custom `StoppingCriteria` that records the tail end of each active beam at each step (without stopping generation).

```python
import torch
from transformers import StoppingCriteria, StoppingCriteriaList

class BeamPrefixTracer(StoppingCriteria):
    def __init__(self, tokenizer, num_beams, max_steps: int = 25, tail_tokens: int = 12):
        self.tokenizer = tokenizer
        self.num_beams = num_beams
        self.max_steps = max_steps
        self.tail_tokens = tail_tokens
        self.step = 0
        self.records = []  # list of beam tail strings per step

    def __call__(self, input_ids, scores, **kwargs):
        if self.step == 0:
            print("beam rows (input_ids.shape[0]) =", input_ids.shape[0])
        if self.step < self.max_steps:
            rows = input_ids[: self.num_beams]  # keep only active beams
            tails = rows[:, -self.tail_tokens:].detach().cpu().tolist()
            tail_texts = [self.tokenizer.decode(t, skip_special_tokens=False) for t in tails]
            self.records.append(tail_texts)
        self.step += 1
        return False  # never stop early ourselves

messages = [{"role": "user", "content": "Describe cyanobacteria photoprotective mechanisms in 1 sentence"}]

input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    enable_thinking=False,
    add_generation_prompt=True,
    return_tensors="pt",
).to(model.device)

NUM_BEAMS = 5
prompt_len = input_ids.shape[-1]
tracer = BeamPrefixTracer(tokenizer, NUM_BEAMS, max_steps=25, tail_tokens=14)

with torch.inference_mode():
    gen = model.generate(
        input_ids,
        max_new_tokens=100,
        do_sample=False,
        num_beams=NUM_BEAMS,
        num_return_sequences=NUM_BEAMS,
        early_stopping="never",
        length_penalty=1.0,
        return_dict_in_generate=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        stopping_criteria=StoppingCriteriaList([tracer]),
    )

print("sequences shape:", tuple(gen.sequences.shape))
print("has beam_indices:", getattr(gen, "beam_indices", None) is not None)

print("\n=== Beam evolution (tail of each active beam per step) ===")
for step, tails in enumerate(tracer.records):
    print(f"\nstep {step}")
    for b, t in enumerate(tails):
        print(f"beam_row {b}: {t!r}")

for i in range(gen.sequences.shape[0]):
    gen_ids = gen.sequences[i, prompt_len:].detach().cpu().tolist()
    print(f"\n--- sequence {i+1} ---")
    print(tokenizer.decode(gen_ids, skip_special_tokens=True))
```

Observed output (selected):

```
beam rows (input_ids.shape[0]) = 10
sequences shape: (5, 117)
has beam_indices: True
```

We can see the beam evolution in the recorded tails, and the final decoded sequences match the earlier `early_stopping="never"` run.

When beam search shows its value, it is typically:

- longer outputs (more room for early choices to matter),
- tasks with a clearer “best” under model likelihood (translation, summarization, structured/constraint outputs),
- when you return multiple candidates (`num_return_sequences > 1`) and compare them.

For open-ended chat style, sampling (`top_p`/`top_k`/`min_p` + temperature) usually produces more obviously different outputs than beam search.

Repetition controls: n-grams and repetition penalty

Now we will look at n-gram blocking, repetition penalties, and n-gram sizes.

- An n-gram is a sequence of N tokens (like a codon).
  - 1-gram: one token
  - 2-gram: two tokens
  - etc.
- We can tune Transformers to restrict repetition on the n-gram scale by setting `no_repeat_ngram_size=N`, which prevents repeating any exact N-token sequence that has already appeared.

Example (`no_repeat_ngram_size=2`):

```python
messages = [{"role": "user", "content": "Describe cyanobacteria photoprotective mechanisms"}]

input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    enable_thinking=False,  # on by default in Nemotron
    add_generation_prompt=True,
    return_tensors="pt",
).to(model.device)

with torch.inference_mode():
    out = model.generate(
        input_ids,
        max_new_tokens=250,
        do_sample=True,
        temperature=0.8,
        top_k=15,
        top_p=0.9,
        no_repeat_ngram_size=2,  # tune n-gram size here
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

prompt_len = input_ids.shape[-1]
gen_ids = out[0, prompt_len:].detach().to("cpu").tolist()
print(tokenizer.decode(gen_ids, skip_special_tokens=True))
```

This effect can be hard to notice unless we generate very long outputs, but it is a critical guardrail to keep on for longer generations (to avoid frustrating repetition).

However, an alternative exists which is not as strict: `repetition_penalty`.

Repetition penalty:

`repetition_penalty` is a “soft” restriction compared to banning repetition via `no_repeat_ngram_size`. It reduces the probability of repeating tokens by modifying logits.

- In practice, Transformers adjusts logits for tokens that have appeared before so they are less attractive.
- `1.0` means no penalty (default).
- `> 1.0` penalizes repeats more.
- `< 1.0` encourages repetition.

Rule of thumb (as implemented in Transformers): logits are divided by the penalty if they are positive (and the token has appeared before), and multiplied by the penalty if they are negative. This can encourage/penalize tokens depending on the sign of the logit.

Experiment: run the same prompt and settings but set `repetition_penalty < 1` to encourage repetition.

Something interesting: to get the low-penalty value to “work”, I needed to set `min_new_tokens`. Without a minimum token count, I simply received the initial prompt.

```python
messages = [{"role": "user", "content": "Describe cyanobacteria photoprotective mechanisms"}]

input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    enable_thinking=False,  # on by default in Nemotron
    add_generation_prompt=True,
    return_tensors="pt",
).to(model.device)

with torch.inference_mode():
    out = model.generate(
        input_ids,
        max_new_tokens=250,
        min_new_tokens=40,
        do_sample=True,
        temperature=0.8,
        top_k=15,
        top_p=0.9,
        repetition_penalty=0.3,  # encourages lots of repeats
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

prompt_len = input_ids.shape[-1]
gen_ids = out[0, prompt_len:].detach().to("cpu").tolist()
print(tokenizer.decode(gen_ids, skip_special_tokens=True))
```

Output:

```
Describe cyanobacteria photoprotective mechanisms
</think>
Describe cyanobacteria photoprotective mechanisms
</think>
Describe cyanobacteria photoprotective mechanisms
</think>
Describe cyanobacteria photoprotective mechanisms
</think>
Describe cyanobacteria photoprotective mechanisms
</think>
Describe cyanobacteria photoprotective mechanisms
</think>
Describe cyanobacteria photoprotective mechanisms
</think>
Describe cyanobacteria photoprotective mechanisms
</think>
Describe cyanobacteria photoprotective mechanisms
</think>
Describe cyanobacteria photoprotective mechanisms
</think>
Describe cyanobacteria photoprotective mechanisms
</think>
Describe cyanobacteria photoprotective mechanisms
</think>
Describe cyanobacteria photoprotective mechanisms
</think>
Describe cyanobacteria photoprotective mechanisms
</think>
Describe cyanobacteria photoprotective mechanisms
</think>
Describe cyanobacteria photoprotective mechanisms
</think>
Describe cyanobacteria photoprotective mechanisms
```

Conclusion: the model became biased to repeat the same tokens over and over again, forcing a degenerate loop.

Another degenerate-loop example with a milder repetition encouragement:

Even with a higher (less extreme) penalty, the output can spiral quickly into nonsense as tokens get reinforced more and more.

Setting:

```text
repetition_penalty=0.7
```

Example output:

```text
Cyanobacteria, as phototrophic cyanobacteria, employ a sophisticated, multi-layered photoprotective mechanisms to cope with intense, rapidly fluctuating, and potentially photodamaging intense sunlight, especially to prevent photodamage to the photosystems and the photosystems, especially photosystems, especially photosystems, especially photosystems, especially photosystems, especially photosystems, especially photosystems, especially photosystems, especially photosystems, especially photosystems, especially photosystems, especially photosystems, especially photosystems, especially photosystems, especially photosystems, especially photosystems, especially photosystems, especially photosystems, especially photosystems, especially photosystems, especially photosystems, especially photosystems, especially photosystems, especially photosystems, especially photosystems, especially photosystems, especially photosystems, especially photosystems, especially photosystems, especially photosystems, especially photosystems, especially photosystems, especially photosystems, especially photosystems, especially photosystems, especially photosystems, especially photosystems, especially photosystems, especially photosystems, especially photosystems, especially photosystems, especially photosystems, especially photosystems, especially photosystems, especially photosystems, especially photosystems, especially photosystems, especially photosystems, especially photosystems, especially photosystems,
```

Part 3 — Tokens: see what the model actually consumes

We are making the `tokenization_peek` script.

Checkpoint: Why does “the model predicts the next *token*” matter more than “the model predicts the next *word*”?

An LLM cannot predict the next *word* as a stable unit. Logits (and therefore token probabilities) are the fundamental unit by which these models operate. Tokens are numbers that can appear in many combinations and can make up part of a word, multiple words, words and punctuation, etc. It is through generation steps over token probabilities that a decoder translates tokens into words.

One experiment made this concrete: I set EOS to be “cell” (token ID `1987`). But as we can see from the output, “cell” did not trigger end-of-sequence. This indicates a crucial point: when a word is embedded in context, its tokenization may be different compared to when it is isolated. Instead of being just the word, the token might be a single letter, the preceding word + the word “cell” combined, or a space+word variant. This speaks to the relationship between tokens and words and how tokens are translated into words. Tokens are the fundamental unit and have specific interpretability when decoded into text. However, a word might be associated with multiple tokens depending on how the tokenizer encodes text in context. Therefore it makes sense that “cell” itself need not have triggered EOS.

Example context (excerpt):

“Cyanobacteria are photoautotrophic bacteria that live at the interface of light and water, where they must harvest photons for photosynthesis while avoiding the damaging effects of excess excitation energy (EEE). Unlike eukaryotes whose photosynthetic apparatus is housed inside chloroplasts bounded by double membranes, cyanobacteria have their photosystems embedded directly into **a single, continuous plasma membrane** that also contains all other essential bioenergetic complexes (respiratory chains, transporters, etc.). This “prokaryotic” organization imposes unique constraints on how protective strategies can be assembled, deployed, and coordinated.

Below we walk through the main photoprotective pathways that cyanobacteria employ, linking each mechanism back to its structural context within the cell envelope. The focus is always on *how* the architecture—a lipid bilayer dotted with pigment–protein complexes—enables or limits those defenses.”

Therefore we deal in tokens, not in words. A token can relate to specific word variants (leading space, trailing space, word plus an extra letter after a space, etc.). A specific word on its own might correlate to a specific subset of tokens, but only in the context where that word is identified/tokenized. For example, `"cat"`, `" cat"`, `"cat "`, and `"cat a"` can be associated with different tokens.

Fundamentally, a model is trained to predict the next token; a word is not a stable unit. Tokenization is deterministic, and the model assigns probabilities to token IDs based on context. Decoding operates at the token level.
