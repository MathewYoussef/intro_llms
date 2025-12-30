2025-12-29 — Beginning Lesson 1.

I am beginning with a local-first approach as opposed to the cloud API. Specifically, I will be setting up vLLM for serving the model and running easy, web-UI-based tests, while also setting up a local download and lab via containerization and `docker-compose` on my Jetson Thor. This will be the primary box running the AI lessons unless we require a different, non-unified architecture for experiments, in which case we will transition to RTX GPUs. However, as it stands, I do not currently have the knowledge to differentiate between the pros and cons of either right now. We will begin by setting up containers on the Thor and a vLLM mode so we can work between the two.

We have added a `docker-compose.yml` to the repository to reliably and repeatably create environments for serving via vLLM and for running experiments in a lab-style environment.

In this workspace, any edits we make inside the container are saved back into our local files (via bind mounts).

To access JupyterLab for course work while hosting the model on Thor (and using Thor as the primary machine):

1) Start the lab container on Thor:

```bash
docker compose -f infra/compose.yml --env-file .env up lab
```

```
2) This builds/starts the Jupyter server. From the other machine (e.g., the Omni Sim box), create an SSH tunnel:

```bash
ssh -L 8888:127.0.0.1:8888 <ssh_user>@<thor_host>
```

3) Open JupyterLab locally in a browser:

```
http://127.0.0.1:8888/lab?token=<JUPYTER_TOKEN>
```

The shared cache is visible inside the container, which is good news because it means we can load the model weights and experiment as needed. For example:

```bash
echo $HF_HOME && ls -la /data/hf | head
```

Nemotron has finished downloading. I will now begin working on the lesson: Part 1, First Contact: "Hello, Nemotron".

We started Jupyter cell development and hit the first interesting behavior: when we attempted to run, we were prompted that Nemotron needed custom code to run and asked whether we wanted to install it. We proceeded.

We ran into dependency issues, though, as Nemotron requires `mamba-ssm` and some other packages. We will rebuild the image to include these custom dependencies.

Initial run notes:

Below is the script we initially received an output from:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
revision = "2e43387afd60157064e5bef4e9a583f887c6dfdd"  # cached snapshot
cache_dir = "/data/hf/hub"

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True,
    use_fast=True,
    revision=revision,
    cache_dir=cache_dir,
    local_files_only=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    revision=revision,
    cache_dir=cache_dir,
    local_files_only=True,
)

prompt = "Hello Nemotron. In one sentence, who are you?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.inference_mode():
    out = model.generate(**inputs, max_new_tokens=60, do_sample=False)

print(tokenizer.decode(out[0], skip_special_tokens=True))
```

- We loaded and defined a specific tokenizer and its parameters: what it is allowed to use and not allowed to use. We forced `local_files_only` and specified a `revision`, because previous runs without these constraints caused Transformers to fetch tensors/shards again during execution.
- We also defined a model and its parameters, including specifying `torch_dtype=bfloat16` (interesting).

First successful run produced logs like:

```
.../site-packages/transformers/utils/hub.py:110: FutureWarning: Using `TRANSFORMERS_CACHE` is deprecated and will be removed in v5 of Transformers. Use `HF_HOME` instead.
`torch_dtype` is deprecated! Use `dtype` instead!
Loading checkpoint shards: 100% 13/13
Some parameters are on the meta device because they were offloaded to the cpu.
NemotronH requires an initialized `NemotronHHybridDynamicCache` to return a cache. None was provided, so no cache will be returned.
```

Then (after Thor made some very loud noises) we got an output.

Prompt:

> "Hello Nemotron. In one sentence, who are you? Please answer in English."

Model output (1 sentence):

> "I am Nemotron, an AI language model created by NVIDIA."

Notes / next steps:

- Accelerate decided to offload some weights onto the CPU; next run we will force everything onto the GPU. This is related to `device_map="auto"`.
- We will switch from `torch_dtype` to `dtype` to avoid the deprecation warning.
- We will use `tegrastats` to monitor memory, since there is a weird phenomenon where weights stay cached.

Mini-checkpoint (write 2-3 sentences): What parts of the pipeline are visible here (messages in, text out), and what parts are hidden (tokenization, logits, decoding)?

Currently visible are some errors and warnings pertaining to deprecation in how I am using Torch; I will correct that in a future run. Next, I see that all 13 shards are loaded. But I read that some parameters are on the meta device because they were offloaded to the CPU. That could be because of standard tuning that allocates based on size; we will turn off `device_map="auto"` (or switch to something else) to force the model to host everything on Thor’s 128GB of unified memory.

Next, I see the question we asked Nemotron; however, it has a slight amendment at the end: “Please answer in English.” This is peculiar, as I am not quite sure what causes this to be appended by default. Perhaps it is part of the additional scripts we had to run from NVIDIA for using Nemotron. I will investigate this further.

In terms of the reply, I see some “thinking” from Nemotron: it appears to reason about the question and how to respond. It focuses first on English and that it must produce a response, then notices that it must “describe who I am”. This suggests some level of association or contextual awareness surrounding the “who are you?” part of the question. It then reasons more and says “as per identity”, which is interesting; it could mean it was trained on some data about itself. “Identity” sounds like a tag or label in a machine-readable form somewhere. It then reasons that it is Nemotron and created by NVIDIA and prompts itself to answer with the words “so answer”, which is again peculiar. Finally, it does write a one-sentence reply in quotations, but after the final quotation there is a random “that” which is cut off. This suggests the next-token prediction was about to continue. It also makes sense given we set a strict cutoff of 60 new tokens for the run, so it could be a consequence of that.

We switched our code to the chat-template path so we can reliably toggle Nemotron “thinking” on and off.

Device mapping note: `device_map` was set to a string (no brackets like `{cuda:0}`); using `"cuda:0"` worked.

Side note: Jetson Thor has odd issues holding memory even after a model has been shut down. To free RAM, we can run:

```bash
sudo sh -c "sync; echo 3 > /proc/sys/vm/drop_caches"
```

Second test (chat-template path, thinking toggled off):

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
revision = "2e43387afd60157064e5bef4e9a583f887c6dfdd"  # cached snapshot
cache_dir = "/data/hf/hub"

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True,
    use_fast=True,
    revision=revision,
    cache_dir=cache_dir,
    local_files_only=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    dtype=torch.bfloat16,
    device_map="cuda:0",
    revision=revision,
    cache_dir=cache_dir,
    local_files_only=True,
)

messages = [{"role": "user", "content": "Hello Nemotron. In one sentence, who are you?"}]
input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    enable_thinking=False,
    add_generation_prompt=True,
    return_tensors="pt",
).to(model.device)

out = model.generate(input_ids, max_new_tokens=60, do_sample=False)
print(tokenizer.decode(out[0], skip_special_tokens=True))
```

Output:

```
Loading checkpoint shards: 100% 13/13 [00:24<00:00, 1.84s/it]

NemotronH requires an initialized `NemotronHHybridDynamicCache` to return a cache. None was provided, so no cache will be returned.

system

user
Hello Nemotron. In one sentence, who are you?
assistant
<think></think>I am Nemotron, a large language model created by NVIDIA to assist with reasoning, creativity, and problem-solving across a wide range of tasks.
```

The `<think></think>` output was expected: per Nemotron’s instructions, when we toggle thinking display off, it emits an empty think block instead of a full reasoning trace. What is interesting, though, is that this result seems better/more verbose than the previous reply. I will need to investigate why that is.

I wonder if it has to do with the `max_new_tokens` setting. Given that it is still 60, is it possible that previously there were tokens allocated to the thinking trace, and now that thinking is off, all 60 tokens are available for the reply?

This seems very plausible: with thinking on, the model will generate a reasoning trace (the `<think>...</think>` content) before producing the final answer. If it uses too much of its token budget there, it is restricted in how much it can give to its final output. That would explain why toggling thinking off (while keeping the same token count) elicited a better response.

We then ran a new `model.generate()` call using the same `input_ids` defined in the previous run (everything was still alive in the notebook kernel). Effectively, we reused the same prompt. We captured timing, token throughput, peak CUDA memory usage, and the top-5 alternatives for the first generated token.

Code (reusing existing `model`, `tokenizer`, and `input_ids`):

```python
import time, torch

eos = tokenizer.eos_token_id

torch.cuda.reset_peak_memory_stats()
t0 = time.perf_counter()

with torch.inference_mode():
    gen = model.generate(
        input_ids,
        max_new_tokens=60,
        do_sample=False,
        eos_token_id=eos,
        pad_token_id=eos,
        return_dict_in_generate=True,
        output_scores=True,
    )

torch.cuda.synchronize()
t1 = time.perf_counter()

seq = gen.sequences[0]
prompt_tokens = input_ids.shape[-1]
generated = seq[prompt_tokens:]

# Trim at first EOS
if eos is not None:
    eos_pos = (generated == eos).nonzero(as_tuple=True)[0]
    if eos_pos.numel() > 0:
        generated = generated[: eos_pos[0]]

text = tokenizer.decode(
    generated.detach().to("cpu").tolist(),
    skip_special_tokens=True,
)

new_tokens = int(generated.numel())
dt = t1 - t0
peak_gb = torch.cuda.max_memory_allocated() / (1024**3)

print(text)
print(
    f"prompt_tokens={prompt_tokens} new_tokens={new_tokens} time_s={dt:.3f} "
    f"tok_per_s={(new_tokens/dt if dt>0 else float('inf')):.2f} "
    f"peak_cuda_mem_gb={peak_gb:.2f}"
)

# Top-5 alternatives for the first generated token
if gen.scores:
    logits0 = gen.scores[0][0].float()
    probs0 = torch.softmax(logits0, dim=-1)
    top = torch.topk(probs0, k=5)
    ids = top.indices.tolist()
    ps = top.values.tolist()
    toks = [tokenizer.decode([i], skip_special_tokens=False) for i in ids]
    print("top5_first_token:", list(zip(ids, ps, toks)))
```

Output:

```
The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.

I am Nemotron, a large language model created by NVIDIA to assist with reasoning, creativity, and problem-solving across a wide range of tasks.
prompt_tokens=28 new_tokens=31 time_s=28.099 tok_per_s=1.10 peak_cuda_mem_gb=58.94
top5_first_token: [(1073, 0.9805247783660889, 'I'), (122973, 0.007486398331820965, 'Nem'), (22177, 0.00660672364756465, 'Hello'), (37133, 0.0016704415902495384, 'Hi'), (4568, 0.0010131739545613527, 'You')]
```

The warning comes from `pad_token_id` being the same as `eos_token_id`, so Transformers cannot infer which tokens are padding vs real content. We likely need to pass an explicit `attention_mask` for reliable results.

We are also getting very slow tokens/sec (~1.10), which is surprising for this hardware. This is especially interesting because peak CUDA memory is listed at ~58.94 GB, but in system monitoring memory seemed to sit around ~66 GB for the whole session (no clear increase/decrease). We also observed CPU throttling, so it is possible some computation is moving to the CPU for some reason (worth troubleshooting).

I tested whether the slow performance was because something was offloading to the CPU instead of staying on the GPU. However, testing showed that the forward pass was indeed on the GPU (not the CPU). Additionally, when I ran the script again with verbosity off, CPU usage stayed low (~5%) at first; then I tried again and it spiked to ~80%.

I am making a note to figure out the bad tokens/sec, but for now we need to move on.

Follow-up: I wonder if there is a depth we can get into regarding optimizing tokens/sec, specifically how the GPU processes and memory behavior are occurring (inference performance engineering).

2025-12-30 — Day 2.

We are now getting into 6.3, where we will run the same prompt multiple times under different decoding settings and save the results. I need to adapt the sample code for local Transformers and the specific formatting that Nemotron expects.

First, I bring up the container via Docker Compose, then display the tail end of its logs so I can see the token for logging in.

Next, I tunnel in from another terminal on my desktop:

```bash
ssh -vv -N -C -o ExitOnForwardFailure=yes -L 8893:127.0.0.1:8888 <ssh_user>@<thor_host>
```

Then I go to JupyterLab, log in with the token, and we are in. I am making a new notebook at `labs/lesson_01/sampling_lab.ipynb`.

Decoding settings notes (Generation 101):

Decoding settings influence which tokens are allowed (filtering). This can include `top_k`, `top_p` (nucleus sampling), `min_p`, and other cutoffs that remove unlikely tokens.

We can also influence how random the model is (sampling “shape”) via `temperature` (which flattens or sharpens the distribution).

Other decoding/generation controls include:

- Greedy vs stochastic decoding (`do_sample=False` vs `do_sample=True`)
- `max_new_tokens`
- Stopping criteria such as EOS
- Repetition controls (repetition penalty, n-gram blocking)
- Search strategy (non-sampling): beam search settings

All of these settings are passed to Transformers via `model.generate(...)` (or stored in the model’s `generation_config.json`).

`top_k` keeps only the `k` highest-probability tokens at a given step.

`top_p` (nucleus sampling) keeps the smallest set of tokens whose cumulative probability is >= `p`. The set size can change per step.

`min_p` keeps tokens whose probability is at least `min_p` multiplied by the probability of the top token. This prevents sampling ultra-low-probability tokens even if `top_p` would include them.

The “distribution” refers to the model’s output logits (unnormalized scores) at each generation step over the vocabulary. Applying `softmax(logits)` yields a probability distribution over the next token.

Temperature rescales logits before softmax:

- `temperature > 1`: flattens (probabilities become more even), leading to more randomness
- `temperature < 1`: sharpens (top tokens become more dominant), leading to more predictable/deterministic outcomes

This flattening/sharpening applies to the next-token probability distribution.

When we talk about sampling vs picking the best token, “best” means the token with the highest probability at that step (the argmax). If we decode greedily (`do_sample=False`), we always pick the argmax token each step. If we sample (`do_sample=True`), we randomly draw a token from the (possibly filtered / temperature-scaled) distribution.

While greedy decoding is the most deterministic approach and lowering temperature also makes decoding more deterministic, they are not the same. Greedy is a discrete choice rule (pick top-1 token). Temperature modifies the probability landscape before a token is chosen. Temperature matters most when sampling; with greedy decoding, temperature usually does not change the result unless it changes which token is ranked #1. It would be interesting to test this explicitly.

An n-gram is a sequence of N tokens in a row:

- 1-gram: one token
- 2-gram (bigram): two consecutive tokens
- 3-gram (trigram): three consecutive tokens

We can reduce repetition at the n-gram scale via `no_repeat_ngram_size=N`, which prevents repeating any *exact* N-token sequence that has already occurred. For example, if N=3 and “the cat sat” appears, that exact trigram cannot appear again later; however, shorter sequences like “the” or “cat sat” can repeat as long as they do not recreate the banned trigram.

EOS is an end-of-sequence token: a special token ID that signals the output is complete (similar to a “stop” marker).

Repetition can be penalized/tuned: we can downweight tokens that have already appeared and/or prevent repeating n-grams.

A “beam” is a candidate partial output (a token sequence in progress). In greedy decoding we maintain one candidate sequence (one beam). In beam search, we maintain B candidate sequences (beams); each beam has its own token history and score. At each step, each beam proposes next-token continuations, and we keep the best-scoring continuations overall.

Overall decoding pipeline:

1) Start with the prompt (input tokens)
2) Model computes logits for the next token (scores over the vocabulary)
3) Convert logits to probabilities (softmax), optionally applying temperature
4) Apply constraints/filters (`top_k`, `top_p`, repetition controls, n-gram blocking, token bans, etc.)
5) Select next token (greedy argmax or sampling)
6) Stop if EOS is produced, `max_new_tokens` is reached, or other stopping criteria fire
7) Repeat until stopping

Next: set up the notebook to explore the effects of different decoding “knobs” and how they influence model output:

- `do_sample`
  - `temperature`
  - `top_k`
  - `top_p`
  - `min_p`
  - `max_new_tokens`
  - `min_new_tokens`
  - `eos_token_id`
  - `pad_token_id`
  - `early_stopping`
  - `repetition_penalty`
  - `no_repeat_ngram_size`
  - `num_beams`
  - `num_return_sequences`
  - `length_penalty`

Sanity checks (ensure the model will load locally):

```python
import os
print("HF_HOME =", os.environ.get("HF_HOME"))
```

```bash
pwd
ls -la /data | head
ls -la /data/hf | head
ls -la $HF_HOME | head
```

```python
import os
from pathlib import Path
from huggingface_hub import snapshot_download

model_id = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"

print("HF_HOME =", os.environ.get("HF_HOME"))
print("HF_HUB_CACHE =", os.environ.get("HF_HUB_CACHE"))

# This forces an offline check: it will fail if the files aren’t already cached.
snap_dir = snapshot_download(
    repo_id=model_id,
    local_files_only=True,
)

snap_path = Path(snap_dir)
print("Snapshot dir:", snap_path)

# Prove the 13 shards are present in this snapshot.
shards = sorted(snap_path.glob("model-*-of-00013.safetensors"))
print("Shard count:", len(shards))
print("First shard:", shards[0] if shards else None)

# Optional: show whether these are symlinks into blobs (typical HF cache layout)
if shards:
    print("First shard is_symlink:", shards[0].is_symlink())
    if shards[0].is_symlink():
        print("first shard ->", os.readlink(shards[0]))
```

Then we load the model:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
revision = "2e43387afd60157064e5bef4e9a583f887c6dfdd"  # cached snapshot
cache_dir = "/data/hf/hub"

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True,
    use_fast=True,
    revision=revision,
    cache_dir=cache_dir,
    local_files_only=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    dtype=torch.bfloat16,
    device_map="cuda:0",
    revision=revision,
    cache_dir=cache_dir,
    local_files_only=True,
)

print("Loaded tokenizer + model on", model.device)
```

First test: toggle `do_sample` and see how it influences the output.

```python
messages = [{"role": "user", "content": "Describe cyanobacteria photoprotective mechanisms"}]

input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    enable_thinking=False,  # on by default in Nemotron
    add_generation_prompt=True,
    return_tensors="pt",
).to(model.device)

with torch.inference_mode():  # disables autograd (faster)
    out = model.generate(
        input_ids,
        max_new_tokens=250,
        do_sample=True,  # for this experiment we toggle this on/off
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

prompt_len = input_ids.shape[-1]
gen_ids = out[0, prompt_len:].detach().to("cpu").tolist()  # decode answer only
print(tokenizer.decode(gen_ids, skip_special_tokens=True))
```

Result (with sampling on):

```
The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
NemotronH requires an initialized `NemotronHHybridDynamicCache` to return a cache. None was provided, so no cache will be returned.

Cyanobacteria, as photosynthetic prokaryotes, face intense light exposure in aquatic environments, which can lead to oxidative stress and photodamage. To mitigate these risks, they employ a sophisticated suite of **photoprotective mechanisms** ...
```

Then I flipped sampling to off (`do_sample=False`). The result began:

```
Cyanobacteria, as oxygenic photosynthetic prokaryotes, face intense light stress in aquatic and terrestrial environments. To protect themselves from photodamage while maintaining efficient photosynthesis, they employ a sophisticated array of **photoprotective mechanisms** ...
```

Interesting to note is how different they are; however, neither is obviously factually incorrect. They both focus differently.

Next experiment: use a larger sample size to see if a pattern emerges in the difference between the two. We run multiple trials, save the text, and compute simple similarity/diversity statistics.

- Within-set similarity:
  - greedy-vs-greedy (should be ~1.0 / identical)
  - sample-vs-sample (should be lower / variable)
- Between-set similarity:
  - greedy-vs-sample (how far sampling drifts from the greedy “mode”)

```python
import re
import torch

N_RUNS = 5
BASE_SEED = 1234  # control for randomness

def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())

def jaccard(a: str, b: str) -> float:
    wa = set(re.findall(r"[a-zA-Z]+", a.lower()))
    wb = set(re.findall(r"[a-zA-Z]+", b.lower()))
    return len(wa & wb) / max(1, len(wa | wb))

def generate_once(do_sample: bool, seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    with torch.inference_mode():
        out = model.generate(
            input_ids,
            max_new_tokens=250,
            do_sample=do_sample,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    prompt_len = input_ids.shape[-1]
    gen_ids = out[0, prompt_len:].detach().to("cpu").tolist()
    return normalize(tokenizer.decode(gen_ids, skip_special_tokens=True))

def pairwise_stats(texts_a, texts_b):
    sims = []
    for a in texts_a:
        for b in texts_b:
            sims.append(jaccard(a, b))
    return {"min": min(sims), "mean": sum(sims)/len(sims), "max": max(sims)}

# Run
greedy = [generate_once(False, BASE_SEED) for _ in range(N_RUNS)]
sample = [generate_once(True, BASE_SEED + i) for i in range(N_RUNS)]

print("Unique outputs:")
print("greedy:", len(set(greedy)), "of", N_RUNS)
print("sample:", len(set(sample)), "of", N_RUNS)

print("\nWithin-set similarity (Jaccard):")
print("greedy×greedy:", pairwise_stats(greedy, greedy))
print("sample×sample:", pairwise_stats(sample, sample))

print("\nBetween-set similarity (Jaccard):")
print("greedy×sample:", pairwise_stats(greedy, sample))

print("\n--- Outputs (greedy) ---")
for i, t in enumerate(greedy, 1):
    print(f"\n[greedy {i}]\n{t}")

print("\n--- Outputs (sample) ---")
for i, t in enumerate(sample, 1):
    print(f"\n[sample {i} seed={BASE_SEED + i - 1}]\n{t}")
```

Results:

```
Unique outputs:
greedy: 1 of 5
sample: 5 of 5

Within-set similarity (Jaccard):
greedy×greedy: {'min': 1.0, 'mean': 1.0, 'max': 1.0}
sample×sample: {'min': 0.143646408839779, 'mean': 0.35636687966727054, 'max': 1.0}

Between-set similarity (Jaccard):
greedy×sample: {'min': 0.19767441860465115, 'mean': 0.2175417088889349, 'max': 0.25153374233128833}
```

This showcases the effect of selecting the best token vs selecting a random token from the distribution. When sampling is off, we repeatedly get the exact same tokens each time; when sampling is on, we see a wide variety of tokens chosen.

To be extra careful, we redid the experiment one more time, but had greedy decoding also use the same increasing seed structure as sampling. This limits variables and lets us compare “apples to apples” and prove that (for greedy decoding) the seed does not matter: it still returns the exact same output.

```python
import re
import torch

N_RUNS = 5
BASE_SEED = 1234  # control for randomness

def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())

def jaccard(a: str, b: str) -> float:
    wa = set(re.findall(r"[a-zA-Z]+", a.lower()))
    wb = set(re.findall(r"[a-zA-Z]+", b.lower()))
    return len(wa & wb) / max(1, len(wa | wb))

def generate_once(do_sample: bool, seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    with torch.inference_mode():
        out = model.generate(
            input_ids,
            max_new_tokens=250,
            do_sample=do_sample,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    prompt_len = input_ids.shape[-1]
    gen_ids = out[0, prompt_len:].detach().to("cpu").tolist()
    return normalize(tokenizer.decode(gen_ids, skip_special_tokens=True))

def pairwise_stats(texts_a, texts_b):
    sims = []
    for a in texts_a:
        for b in texts_b:
            sims.append(jaccard(a, b))
    return {"min": min(sims), "mean": sum(sims)/len(sims), "max": max(sims)}

# Run
seeds = [BASE_SEED + i for i in range(N_RUNS)]
greedy = [generate_once(False, s) for s in seeds]
sample = [generate_once(True, s) for s in seeds]

print("Unique outputs:")
print("greedy:", len(set(greedy)), "of", N_RUNS)
print("sample:", len(set(sample)), "of", N_RUNS)

print("\nWithin-set similarity (Jaccard):")
print("greedy×greedy:", pairwise_stats(greedy, greedy))
print("sample×sample:", pairwise_stats(sample, sample))

print("\nBetween-set similarity (Jaccard):")
print("greedy×sample:", pairwise_stats(greedy, sample))

print("\n--- Outputs (greedy) ---")
for i, (s, t) in enumerate(zip(seeds, greedy), 1):
    print(f"\n[greedy {i} seed={s}]\n{t}")

print("\n--- Outputs (sample) ---")
for i, (s, t) in enumerate(zip(seeds, sample), 1):
    print(f"\n[sample {i} seed={s}]\n{t}")
```

Results:

```
Unique outputs:
greedy: 1 of 5
sample: 5 of 5

Within-set similarity (Jaccard):
greedy×greedy: {'min': 1.0, 'mean': 1.0, 'max': 1.0}
sample×sample: {'min': 0.143646408839779, 'mean': 0.35636687966727054, 'max': 1.0}

Between-set similarity (Jaccard):
greedy×sample: {'min': 0.19767441860465115, 'mean': 0.2175417088889349, 'max': 0.25153374233128833}
```

We now see clearly that even with changing RNG seeds, greedy decoding returns the same exact output each time because the most probable token is selected at every step.

Next: sampling is usually paired with temperature plus a constraint like `top_p`, `top_k`, or `min_p` to deal with the large tail of improbable tokens that can be selected under sampling.

Greedy decoding (`do_sample=False`) will always result in the exact same outcome. `top_k`, `top_p`, and `min_p` won’t change that output because they are sampling knobs. Later we will explore how greedy can change when we twist non-sampling knobs such as constraints or penalties.

Here we will compare two variables:

1) With `do_sample=True`, run the same prompt with varying `temperature`.
2) Then run the same prompt with the same `temperature` but different `top_k`, `top_p`, and `min_p` to explore how those influence things.

Temperature controls the variety available to pick from by changing how spread out the probabilities are. If the probabilities are very close to each other, the distribution is flatter and there is less bias toward a specific token (lower-ranked tokens become less “impossible”). If temperature is very low (toward 0), it makes the distribution very peaky, so the higher-ranked logits have much higher likelihood of being selected (behaving closer to greedy argmax).

So temperature influences the distribution of logits before the selection mechanism (sampling vs greedy) takes place.

It is important to remember: sampling is still probability-based. Greedy always picks the top token. Sampling still selects highly probable tokens more often than not. If something is 50% probable and it is the highest probability token, greedy takes it 100% of the time; sampling takes it ~50% of the time. Temperature influences that distribution, making token probabilities either more separated or more evenly likely.

Temperature experiment:

- Keep `do_sample=True`
- Test temperatures: `0.2`, `0.7`, `1.0`, `1.3`, `1.8`
- For each temperature, rerun 3 times (3 seeds)
- Measure within-temperature variability and between-temperature shifts

Observations:

- Within-temperature similarity decreased as temperature increased. This makes sense: as the distribution gets flatter (probabilities more evenly spread), it becomes more likely to select different tokens across runs, decreasing similarity.
- Between-temperature similarity also decreased as temperature increased. As decoding becomes less deterministic and more random, outputs at higher temperatures drift further away from outputs at lower temperatures.
- Coherence was impacted severely at the highest temperature setting: sentence structure barely made sense. At lower temperatures, responses were more coherent and complete while still varying in structure.

Unique outputs per temperature:

```
T=0.2: 3 of 3
T=0.7: 3 of 3
T=1.0: 3 of 3
T=1.3: 3 of 3
T=1.8: 3 of 3
```

Within-temp similarity (Jaccard):

```
T=0.2: {'min': 0.2289156626506024, 'mean': 0.6099530396719152, 'max': 1.0}
T=0.7: {'min': 0.17045454545454544, 'mean': 0.49382319095647464, 'max': 1.0}
T=1.0: {'min': 0.15384615384615385, 'mean': 0.47799160704712257, 'max': 1.0}
T=1.3: {'min': 0.1744186046511628, 'mean': 0.4583675330597355, 'max': 1.0}
T=1.8: {'min': 0.11956521739130435, 'mean': 0.41641829530028285, 'max': 1.0}
```

Between-temp similarity (Jaccard), temp i vs i+1:

```
0.2 vs 0.7: {'min': 0.11458333333333333, 'mean': 0.29572568683514566, 'max': 0.5625}
0.7 vs 1.0: {'min': 0.15294117647058825, 'mean': 0.24569283080079599, 'max': 0.37681159420289856}
1.0 vs 1.3: {'min': 0.11363636363636363, 'mean': 0.18418958611470856, 'max': 0.39705882352941174}
1.3 vs 1.8: {'min': 0.09375, 'mean': 0.15536525735440446, 'max': 0.21839080459770116}
```

Outputs (snippets):

```
===== temperature=0.2 =====

[run 1 seed=1234]
**Cyanobacteria – a quick reminder** Cyanobacteria are oxygenic photosynthetic prokaryotes ...

[run 2 seed=1235]
Cyanobacteria, as oxygenic photosynthetic prokaryotes, face intense light exposure ...

[run 3 seed=1236]
Cyanobacteria, as photosynthetic prokaryotes, face intense light stress ...

===== temperature=0.7 =====

[run 1 seed=1237]
Cyanobacteria, as oxygenic photosynthetic prokaryotes, face intense light stress ...

[run 2 seed=1238]
Cyanobacteria, like plants and algae, employ a sophisticated suite ...

[run 3 seed=1239]
Cyanobacteria employ a suite of sophisticated, interconnected photoprotective mechanisms ...

===== temperature=1.0 =====

[run 1 seed=1240]
## Cyanobacterial Photoprotection: A Multilayered Defense Against Solar Radiation ...

[run 2 seed=1241]
Cyanobacteria, as photosynthetic microorganisms, face intense oxidative stress ...

[run 3 seed=1242]
Cyanobacteria, as oxygenic photosynthetic prokaryotes, face intense light exposure ...

===== temperature=1.3 =====

[run 1 seed=1243]
**Cyanobacteria Photoprotective Mechanisms** Cyanobacteria, like plants, are photosynthetic microorganisms ...

[run 2 seed=1244]
## Light as a Double‑Edged Sword for Cyanobacteria ...

[run 3 seed=1245]
Cyanobacteria, like all photosynthetic organisms, face intense light stress ...

===== temperature=1.8 =====

[run 1 seed=1246]
**Cyanobacteria – a quick reminder** Cyanobacteria are ancient, oxygen‑producing, photosynthetic prokaryotes ...

[run 2 seed=1247]
Cyanobacteria, like other photosynthetic organisms, face intense solar radiation that can cause photodamage, primarily through ...

[run 3 seed=1248]
Cyanobacteria face intense variable (sometimes super-light) irradiance where photosynthesis can become photointoxicated ...
```

Sampling filters:

Now we will see how influencing which tokens are allowed to be selected influences the outputs.

Top-k:

- We pick the k most likely tokens at each step, renormalize probabilities to sum to one for the remaining k tokens, and sample from that reduced set.
- `top_k=1` should technically be the same as greedy decoding (`do_sample=False`) because we only allow the single best token to proceed into selection.

Top-k experiment:

- 5 levels of `top_k`: 1, 2, 3, 4, 5
  - `top_k=1` acts like a control and should be equivalent to greedy.
- We also run one explicit greedy control to confirm.
- Temperature is held constant at `0.5`.

```python
import re
import torch

TOP_KS = [1, 2, 3, 4, 5]
RUNS_PER_K = 3
TEMPERATURE = 0.5
BASE_SEED = 1234

def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())

def jaccard(a: str, b: str) -> float:
    wa = set(re.findall(r"[a-zA-Z]+", a.lower()))
    wb = set(re.findall(r"[a-zA-Z]+", b.lower()))
    return len(wa & wb) / max(1, len(wa | wb))

def generate_once(
    seed: int,
    do_sample: bool,
    temperature: float | None = None,
    top_k: int | None = None,
):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    gen_kwargs = dict(
        max_new_tokens=80,
        do_sample=do_sample,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    if temperature is not None:
        gen_kwargs["temperature"] = temperature
    if top_k is not None:
        gen_kwargs["top_k"] = top_k

    with torch.inference_mode():
        out = model.generate(input_ids, **gen_kwargs)

    prompt_len = input_ids.shape[-1]
    gen_ids = out[0, prompt_len:].detach().to("cpu").tolist()
    return normalize(tokenizer.decode(gen_ids, skip_special_tokens=True))

def pairwise_stats(texts_a, texts_b):
    sims = []
    for a in texts_a:
        for b in texts_b:
            sims.append(jaccard(a, b))
    return {"min": min(sims), "mean": sum(sims) / len(sims), "max": max(sims)}

results = {}  # label -> list[str]
meta = {}     # label -> list[int]

seed_counter = 0

# Control: greedy (do_sample=False)
label = "control_greedy"
seeds = [BASE_SEED + seed_counter]  # one run is enough; it should be deterministic
seed_counter += 1
results[label] = [generate_once(seed=seeds[0], do_sample=False)]
meta[label] = seeds

# top_k experiments (do_sample=True)
for k in TOP_KS:
    label = f"top_k={k}"
    seeds = []
    texts = []
    for _ in range(RUNS_PER_K):
        seed = BASE_SEED + seed_counter
        seed_counter += 1
        seeds.append(seed)
        texts.append(
            generate_once(
                seed=seed,
                do_sample=True,
                temperature=TEMPERATURE,
                top_k=k,
            )
        )
    results[label] = texts
    meta[label] = seeds

print("Unique outputs per setting:")
for label, texts in results.items():
    print(f"{label}: {len(set(texts))} of {len(texts)}")

print("\nWithin-setting similarity (Jaccard):")
for label, texts in results.items():
    print(f"{label}:", pairwise_stats(texts, texts))

print("\nBetween-setting similarity (Jaccard), adjacent top_k:")
for k1, k2 in zip(TOP_KS, TOP_KS[1:]):
    a = results[f"top_k={k1}"]
    b = results[f"top_k={k2}"]
    print(f"top_k={k1} vs top_k={k2}:", pairwise_stats(a, b))

print("\n--- Outputs ---")
for label, texts in results.items():
    print(f"\n===== {label} =====")
    for i, text in enumerate(texts, 1):
        seed = meta[label][i - 1]
        print(f"\n[run {i} seed={seed}]\n{text}")
```

Top-k experiment results and anomaly:

We predicted that at `top_k=1` we would see deterministic behavior, since only the top token (irrespective of temperature) is allowed to remain. However, we see some non-determinism: a single word/token seems to slip through, and this tiny variation changes the course of the output.

Temperature cannot create randomness when only one option remains to be picked from. Therefore, something else is influencing which options are available to be picked. Hypotheses:

- The model’s logits/probabilities are not perfectly deterministic run-to-run (upstream non-determinism), and the “top token” itself can change slightly.
- There is some behavior when `do_sample=True` (even with `top_k=1`) that influences the probability distribution.

To disprove/validate this, we can rerun the same experiment but cycle through 3 greedy runs with different seeds as well. If greedy is always identical, then the non-determinism is likely tied to sampling mode; if greedy is not identical, then the “top token” itself is changing (either due to probability weighting differences or upstream non-determinism in logits).

Results from this experiment:

```
Unique outputs per setting:
control_greedy: 1 of 1
top_k=1: 3 of 3
top_k=2: 3 of 3
top_k=3: 3 of 3
top_k=4: 3 of 3
top_k=5: 3 of 3

Within-setting similarity (Jaccard):
control_greedy: {'min': 1.0, 'mean': 1.0, 'max': 1.0}
top_k=1: {'min': 0.5454545454545454, 'mean': 0.7834534066151713, 'max': 1.0}
top_k=2: {'min': 0.17647058823529413, 'mean': 0.4765152643335815, 'max': 1.0}
top_k=3: {'min': 0.1518987341772152, 'mean': 0.5054729908992984, 'max': 1.0}
top_k=4: {'min': 0.125, 'mean': 0.4977115355757069, 'max': 1.0}
top_k=5: {'min': 0.2597402597402597, 'mean': 0.5247228158620563, 'max': 1.0}

Between-setting similarity (Jaccard), adjacent top_k:
top_k=1 vs top_k=2: {'min': 0.24358974358974358, 'mean': 0.3454948582551978, 'max': 0.576271186440678}
top_k=2 vs top_k=3: {'min': 0.17647058823529413, 'mean': 0.2963473401609485, 'max': 0.5901639344262295}
top_k=3 vs top_k=4: {'min': 0.14102564102564102, 'mean': 0.2893887693178497, 'max': 0.6031746031746031}
top_k=4 vs top_k=5: {'min': 0.13924050632911392, 'mean': 0.285800620152406, 'max': 0.4722222222222222}
```

Selected outputs showing the `top_k=1` phenomenon:

```
===== control_greedy =====

[run 1 seed=1234]
Cyanobacteria, as oxygenic photosynthetic prokaryotes, face intense light stress in aquatic and terrestrial environments. To protect themselves from photodamage while maintaining efficient photosynthesis, they employ a sophisticated array of **photoprotective mechanisms**...

===== top_k=1 =====

[run 1 seed=1235]
Cyanobacteria, as oxygenic photosynthetic prokaryotes, face intense light exposure in aquatic and terrestrial environments, where excess light can cause oxidative damage...

[run 2 seed=1236]
Cyanobacteria, as oxygenic photosynthetic prokaryotes, face intense light exposure in aquatic and terrestrial environments, where excess light can cause oxidative damage... repair photodamage.

[run 3 seed=1237]
Cyanobacteria, as oxygenic photosynthetic prokaryotes, face intense light stress in aquatic and terrestrial environments. To protect themselves from photodamage while maintaining efficient photosynthesis...
```

Follow-up experiment plan (pin down the source of non-determinism):

1) Greedy determinism check:
   - Run 3 rounds of greedy decoding (`do_sample=False`) with different seeds.
   - Expected: all three outputs are identical (seed should be irrelevant for greedy).
   - If the greedy rounds differ, then non-determinism is upstream in the model forward / GPU kernels (not due to sampling machinery).

2) `top_k=1` determinism check (same seed):
   - Run 3 rounds with `do_sample=True` and `top_k=1`, repeated with the same seed.
   - If outputs differ even with the same seed, then it is not RNG; there is non-determinism elsewhere.

3) Round out the matrix:
   - 3 rounds greedy with the same seed.
   - 3 rounds `do_sample=True, top_k=1` with different seeds.

Follow-up experiment run (results):

We ran the new experiment with this code:

```python
import torch

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def gen_text(**kwargs):
    with torch.inference_mode():
        out = model.generate(
            input_ids,
            max_new_tokens=80,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            **kwargs,
        )
    gen_ids = out[0, input_ids.shape[-1]:].detach().cpu().tolist()
    return tokenizer.decode(gen_ids, skip_special_tokens=True)

SEEDS = [111, 222, 333]

print("1) Greedy, SAME seed repeated (should match exactly):")
for i in range(3):
    set_seed(999)
    print(f"\nrepeat={i+1} seed=999\n{gen_text(do_sample=False)}")

print("\n2) Greedy, DIFFERENT seeds (should still match exactly):")
for s in SEEDS:
    set_seed(s)
    print(f"\nseed={s}\n{gen_text(do_sample=False)}")

print("\n3) top_k=1, SAME seed repeated (should match; if not, nondeterminism in sampling path):")
for i in range(3):
    set_seed(999)
    print(
        f"\nrepeat={i+1} seed=999\n"
        f"{gen_text(do_sample=True, temperature=0.5, top_k=1)}"
    )

print("\n4) top_k=1, DIFFERENT seeds (should still match; seed shouldn't matter with k=1):")
for s in SEEDS:
    set_seed(s)
    print(f"\nseed={s}\n{gen_text(do_sample=True, temperature=0.5, top_k=1)}")
```

Summary:

- Greedy with the same seed repeated and greedy with different seeds both produced the exact same output (deterministic forward + argmax path; seed irrelevant).
- `top_k=1` with the same seed repeated also produced the exact same output.
- However, `top_k=1` output (for the same seed) is not the same as the greedy outputs. This suggests a difference occurring when we enable sampling that changes how the token probabilities are distributed (even though only 1 token remains after filtering).
- More evidence: `top_k=1` with different seeds produced different outputs. Interestingly, seed=111 produced the same output as greedy.

Full output:

```
1) Greedy, SAME seed repeated (should match exactly):

repeat=1 seed=999
Cyanobacteria, as oxygenic photosynthetic prokaryotes, face intense light stress in aquatic and terrestrial environments. To protect themselves from photodamage while maintaining efficient photosynthesis, they employ a sophisticated array of **photoprotective mechanisms**. These mechanisms primarily focus on **dissipating excess excitation energy as heat** to prevent the formation of harmful reactive oxygen species (ROS) that can damage cellular components

repeat=2 seed=999
Cyanobacteria, as oxygenic photosynthetic prokaryotes, face intense light stress in aquatic and terrestrial environments. To protect themselves from photodamage while maintaining efficient photosynthesis, they employ a sophisticated array of **photoprotective mechanisms**. These mechanisms primarily focus on **dissipating excess excitation energy as heat** to prevent the formation of harmful reactive oxygen species (ROS) that can damage cellular components

repeat=3 seed=999
Cyanobacteria, as oxygenic photosynthetic prokaryotes, face intense light stress in aquatic and terrestrial environments. To protect themselves from photodamage while maintaining efficient photosynthesis, they employ a sophisticated array of **photoprotective mechanisms**. These mechanisms primarily focus on **dissipating excess excitation energy as heat** to prevent the formation of harmful reactive oxygen species (ROS) that can damage cellular components

2) Greedy, DIFFERENT seeds (should still match exactly):

seed=111
Cyanobacteria, as oxygenic photosynthetic prokaryotes, face intense light stress in aquatic and terrestrial environments. To protect themselves from photodamage while maintaining efficient photosynthesis, they employ a sophisticated array of **photoprotective mechanisms**. These mechanisms primarily focus on **dissipating excess excitation energy as heat** to prevent the formation of harmful reactive oxygen species (ROS) that can damage cellular components

seed=222
Cyanobacteria, as oxygenic photosynthetic prokaryotes, face intense light stress in aquatic and terrestrial environments. To protect themselves from photodamage while maintaining efficient photosynthesis, they employ a sophisticated array of **photoprotective mechanisms**. These mechanisms primarily focus on **dissipating excess excitation energy as heat** to prevent the formation of harmful reactive oxygen species (ROS) that can damage cellular components

seed=333
Cyanobacteria, as oxygenic photosynthetic prokaryotes, face intense light stress in aquatic and terrestrial environments. To protect themselves from photodamage while maintaining efficient photosynthesis, they employ a sophisticated array of **photoprotective mechanisms**. These mechanisms primarily focus on **dissipating excess excitation energy as heat** to prevent the formation of harmful reactive oxygen species (ROS) that can damage cellular components

3) top_k=1, SAME seed repeated (should match; if not, nondeterminism in sampling path):

repeat=1 seed=999
Cyanobacteria, as oxygenic photosynthetic prokaryotes, face intense light exposure in aquatic and terrestrial environments, where excess light can cause oxidative damage to photosystem II (PSII) and other cellular components. To mitigate this, they employ a sophisticated array of **photoprotective mechanisms** that work synergistically to dissipate excess energy as heat, prevent reactive oxygen species (ROS) formation,

repeat=2 seed=999
Cyanobacteria, as oxygenic photosynthetic prokaryotes, face intense light exposure in aquatic and terrestrial environments, where excess light can cause oxidative damage to photosystem II (PSII) and other cellular components. To mitigate this, they employ a sophisticated array of **photoprotective mechanisms** that work synergistically to dissipate excess energy as heat, prevent reactive oxygen species (ROS) formation,

repeat=3 seed=999
Cyanobacteria, as oxygenic photosynthetic prokaryotes, face intense light exposure in aquatic and terrestrial environments, where excess light can cause oxidative damage to photosystem II (PSII) and other cellular components. To mitigate this, they employ a sophisticated array of **photoprotective mechanisms** that work synergistically to dissipate excess energy as heat, prevent reactive oxygen species (ROS) formation,

4) top_k=1, DIFFERENT seeds (should still match; seed shouldn't matter with k=1):

seed=111
Cyanobacteria, as oxygenic photosynthetic prokaryotes, face intense light stress in aquatic and terrestrial environments. To protect themselves from photodamage while maintaining efficient photosynthesis, they employ a sophisticated array of **photoprotective mechanisms**. These mechanisms primarily focus on **dissipating excess excitation energy as heat** to prevent the formation of harmful reactive oxygen species (ROS) that can damage cellular components

seed=222
Cyanobacteria, as oxygenic photosynthetic prokaryotes, face intense light exposure in aquatic and terrestrial environments, where excess light can cause oxidative damage to photosynthetic machinery. To mitigate this, they employ a sophisticated array of **photoprotective mechanisms** that work synergistically to safely dissipate excess energy as heat, prevent reactive oxygen species (ROS) formation, and protect the photosynthetic apparatus. Below is

seed=333
Cyanobacteria, as oxygenic photosynthetic prokaryotes, face intense light stress in aquatic and terrestrial environments. To protect themselves from photodamage while maintaining efficient photosynthesis, they employ a sophisticated array of **photoprotective mechanisms**. These mechanisms primarily focus on **dissipating excess excitation energy as heat** to prevent the formation of harmful reactive oxygen species (ROS) that can damage cellular components
```

Hypothesis: ties during top-k filtering

It is possible that the apparent non-deterministic behavior is due to ties in token probabilities during filtering. With `top_k=1` we expect only one token to survive, but if two (or more) tokens share the same max logit (possible with lower precision like BF16), then it would make sense that with different seeds we get different results (because one of the tied tokens gets selected).

Next, we will test this tie hypothesis as the factor contributing to the apparent non-deterministic behavior. Specifically, we will look at how many tokens survive filtering under `top_k=1` at a given token position (the first position at which different runs diverge).

Tie hypothesis results:

As we can see, two tokens were allowed through even at `top_k=1`:

- `first_diff_generated_step = 15`
- `seed=111` token: `7500` (`" stress"`)
- `seed=222` token: `11915` (`" exposure"`)
- `allowed_tokens_after_top_k=1_at_that_step = 2`

So at least two tokens shared the same top score after temperature processing and Transformers kept both of them. Fundamentally, `top_k=1` is not always equivalent to greedy decoding (`do_sample=False`) in practice because of ties (which are made more likely due to kernel precision and BF16). Sampling breaks ties differently than greedy does.

Top-p (nucleus sampling):

Now we investigate the filtering technique of `top_p` (nucleus sampling).

- After temperature is applied, we sort tokens by probability (highest to lowest).
- We walk down that list and add up probabilities (`p1 + p2 + ...`) to form a cumulative probability mass.
- We then take the smallest set of tokens whose cumulative probability is >= the `top_p` we set.
- That leaves us with a smaller set of allowed tokens; we renormalize those probabilities to sum back to 1.
- We then sample from that renormalized set.

Temperature strongly affects how peaked the distribution is and therefore has a strong impact on how quickly cumulative mass accumulates (and how many tokens are required to surpass a given `top_p` threshold).

Top-p experiment:

We tested 5 levels of `top_p`: `0.01`, `0.1`, `0.3`, `0.5`, `0.8` (temperature held constant at `0.5`).

```python
import re
import torch

TOP_PS = [0.01, 0.1, 0.3, 0.5, 0.8]
RUNS_PER_P = 3
TEMPERATURE = 0.5
BASE_SEED = 1234

def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())

def jaccard(a: str, b: str) -> float:
    wa = set(re.findall(r"[a-zA-Z]+", a.lower()))
    wb = set(re.findall(r"[a-zA-Z]+", b.lower()))
    return len(wa & wb) / max(1, len(wa | wb))

def generate_once(
    seed: int,
    do_sample: bool,
    temperature: float | None = None,
    top_p: float | None = None,
):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    gen_kwargs = dict(
        max_new_tokens=80,
        do_sample=do_sample,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    if temperature is not None:
        gen_kwargs["temperature"] = temperature
    if top_p is not None:
        gen_kwargs["top_p"] = top_p

    with torch.inference_mode():
        out = model.generate(input_ids, **gen_kwargs)

    prompt_len = input_ids.shape[-1]
    gen_ids = out[0, prompt_len:].detach().to("cpu").tolist()
    return normalize(tokenizer.decode(gen_ids, skip_special_tokens=True))

def pairwise_stats(texts_a, texts_b):
    sims = []
    for a in texts_a:
        for b in texts_b:
            sims.append(jaccard(a, b))
    return {"min": min(sims), "mean": sum(sims) / len(sims), "max": max(sims)}

results = {}  # label -> list[str]
meta = {}     # label -> list[int]

seed_counter = 0

# Control: greedy (do_sample=False)
label = "control_greedy"
seeds = [BASE_SEED + seed_counter]  # one run is enough; it should be deterministic
seed_counter += 1
results[label] = [generate_once(seed=seeds[0], do_sample=False)]
meta[label] = seeds

# top_p experiments (do_sample=True)
for p in TOP_PS:
    label = f"top_p={p}"
    seeds = []
    texts = []
    for _ in range(RUNS_PER_P):
        seed = BASE_SEED + seed_counter
        seed_counter += 1
        seeds.append(seed)
        texts.append(
            generate_once(
                seed=seed,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=p,
            )
        )
    results[label] = texts
    meta[label] = seeds

print("Unique outputs per setting:")
for label, texts in results.items():
    print(f"{label}: {len(set(texts))} of {len(texts)}")

print("\nWithin-setting similarity (Jaccard):")
for label, texts in results.items():
    print(f"{label}:", pairwise_stats(texts, texts))

print("\nBetween-setting similarity (Jaccard), adjacent top_p:")
for p1, p2 in zip(TOP_PS, TOP_PS[1:]):
    a = results[f"top_p={p1}"]
    b = results[f"top_p={p2}"]
    print(f"top_p={p1} vs top_p={p2}:", pairwise_stats(a, b))

print("\n--- Outputs ---")
for label, texts in results.items():
    print(f"\n===== {label} =====")
    for i, text in enumerate(texts, 1):
        seed = meta[label][i - 1]
        print(f"\n[run {i} seed={seed}]\n{text}")
```

Results:

```
Unique outputs per setting:
control_greedy: 1 of 1
top_p=0.01: 1 of 3
top_p=0.1: 1 of 3
top_p=0.3: 1 of 3
top_p=0.5: 2 of 3
top_p=0.8: 3 of 3

Within-setting similarity (Jaccard):
control_greedy: {'min': 1.0, 'mean': 1.0, 'max': 1.0}
top_p=0.01: {'min': 1.0, 'mean': 1.0, 'max': 1.0}
top_p=0.1: {'min': 1.0, 'mean': 1.0, 'max': 1.0}
top_p=0.3: {'min': 1.0, 'mean': 1.0, 'max': 1.0}
top_p=0.5: {'min': 0.578125, 'mean': 0.8125, 'max': 1.0}
top_p=0.8: {'min': 0.37142857142857144, 'mean': 0.6853589196872778, 'max': 1.0}

Between-setting similarity (Jaccard), adjacent top_p:
top_p=0.01 vs top_p=0.1: {'min': 1.0, 'mean': 1.0, 'max': 1.0}
top_p=0.1 vs top_p=0.3: {'min': 1.0, 'mean': 1.0, 'max': 1.0}
top_p=0.3 vs top_p=0.5: {'min': 0.5454545454545454, 'mean': 0.6403659233847914, 'max': 0.8301886792452831}
top_p=0.5 vs top_p=0.8: {'min': 0.23170731707317074, 'mean': 0.43212590369069054, 'max': 0.6896551724137931}
```

Interpretation:

- As `top_p` increases, variation increases.
- We see identical results for `top_p=0.01`, `0.1`, and `0.3`, suggesting that (at `temperature=0.5`) the cumulative probability mass often included only ~1 token before reaching those thresholds. That would mean the nucleus “set” was effectively size 1 for many (or all) steps.
- However, this is not guaranteed; it’s possible we got “lucky” (a peaky enough branch was selected each time).

Next: verify empirically how many tokens survive top-p filtering at each generated step. This gives us an idea of what gets filtered for each `top_p` value and where it does.

Verification: nucleus size per step (first 20 steps)

Indeed, the original hypothesis was correct:

```
top_p=0.05 nucleus_sizes (first 20 steps) = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  min/mean/max = 1 / 1.0 / 1
top_p=0.1  nucleus_sizes (first 20 steps) = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  min/mean/max = 1 / 1.0 / 1
top_p=0.3  nucleus_sizes (first 20 steps) = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  min/mean/max = 1 / 1.0 / 1
top_p=0.5  nucleus_sizes (first 20 steps) = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1]  min/mean/max = 1 / 1.075 / 2
top_p=0.8  nucleus_sizes (first 20 steps) = [2, 1, 1, 1, 3, 2, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1]  min/mean/max = 1 / 1.25 / 3
```

So fundamentally, `top_p` controls how often the candidate set has more than one token given the current distribution (which temperature largely shapes). Therefore, `top_p` isn’t randomness by itself; it is another method of filtering candidates down to a set that sampling then draws from.

Min-p:

Now we move on to `min_p`.

- First we apply temperature.
- Compute `p_max` (the highest probability in the distribution).
- Multiply `p_max` by our chosen `min_p` value to get a threshold.
- Filter out the tail of the distribution below that threshold.
  - In other words, keep tokens whose temperature-normalized probabilities are >= `min_p * p_max`.
- Renormalize the remaining probabilities and sample from that set.

Fundamentally, we are keeping tokens whose probability is above a relative threshold with respect to the best (most probable) token.

Min-p experiment:

- Intuition: the bigger our `min_p` value, the closer `min_p * p_max` is to `p_max`, and therefore the more of the tail of the distribution is removed.
- Expected (opposite of `top_p`): larger `top_p` means more variation (more cumulative probability mass, more tokens included). Larger `min_p` means a threshold closer to `p_max`, so fewer tokens survive and we should see less variation.

We tested:

- `min_p`: `0.01`, `0.1`, `0.3`, `0.5`, `0.8`
- `temperature=0.5`
- 3 runs per setting

```python
import re
import torch

MIN_PS = [0.01, 0.1, 0.3, 0.5, 0.8]
RUNS_PER_P = 3
TEMPERATURE = 0.5
BASE_SEED = 1234

def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())

def jaccard(a: str, b: str) -> float:
    wa = set(re.findall(r"[a-zA-Z]+", a.lower()))
    wb = set(re.findall(r"[a-zA-Z]+", b.lower()))
    return len(wa & wb) / max(1, len(wa | wb))

def generate_once(
    seed: int,
    do_sample: bool,
    temperature: float | None = None,
    min_p: float | None = None,
):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    gen_kwargs = dict(
        max_new_tokens=80,
        do_sample=do_sample,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    if temperature is not None:
        gen_kwargs["temperature"] = temperature
    if min_p is not None:
        gen_kwargs["min_p"] = min_p

    with torch.inference_mode():
        out = model.generate(input_ids, **gen_kwargs)

    prompt_len = input_ids.shape[-1]
    gen_ids = out[0, prompt_len:].detach().to("cpu").tolist()
    return normalize(tokenizer.decode(gen_ids, skip_special_tokens=True))

def pairwise_stats(texts_a, texts_b):
    sims = []
    for a in texts_a:
        for b in texts_b:
            sims.append(jaccard(a, b))
    return {"min": min(sims), "mean": sum(sims) / len(sims), "max": max(sims)}

results = {}  # label -> list[str]
meta = {}     # label -> list[int]

seed_counter = 0

# Control: greedy (do_sample=False)
label = "control_greedy"
seeds = [BASE_SEED + seed_counter]
seed_counter += 1
results[label] = [generate_once(seed=seeds[0], do_sample=False)]
meta[label] = seeds

# min_p experiments (do_sample=True)
for p in MIN_PS:
    label = f"min_p={p}"
    seeds = []
    texts = []
    for _ in range(RUNS_PER_P):
        seed = BASE_SEED + seed_counter
        seed_counter += 1
        seeds.append(seed)
        texts.append(
            generate_once(
                seed=seed,
                do_sample=True,
                temperature=TEMPERATURE,
                min_p=p,
            )
        )
    results[label] = texts
    meta[label] = seeds

print("Unique outputs per setting:")
for label, texts in results.items():
    print(f"{label}: {len(set(texts))} of {len(texts)}")

print("\nWithin-setting similarity (Jaccard):")
for label, texts in results.items():
    print(f"{label}:", pairwise_stats(texts, texts))

print("\nBetween-setting similarity (Jaccard), adjacent min_p:")
for p1, p2 in zip(MIN_PS, MIN_PS[1:]):
    a = results[f"min_p={p1}"]
    b = results[f"min_p={p2}"]
    print(f"min_p={p1} vs min_p={p2}:", pairwise_stats(a, b))

print("\n--- Outputs ---")
for label, texts in results.items():
    print(f"\n===== {label} =====")
    for i, text in enumerate(texts, 1):
        seed = meta[label][i - 1]
        print(f"\n[run {i} seed={seed}]\n{text}")
```

Observation:

Indeed, as we increase `min_p`, we see less step-by-step variation (though there is still eventual branching). Also interesting: the greedy no-sample control matched exactly one of the highest (`min_p=0.8`) runs. That makes sense: as `min_p` approaches 1, we collapse toward the argmax token being the only viable option, which is what greedy decoding selects.

Results:

```
Unique outputs per setting:
control_greedy: 1 of 1
min_p=0.01: 3 of 3
min_p=0.1: 3 of 3
min_p=0.3: 3 of 3
min_p=0.5: 2 of 3
min_p=0.8: 3 of 3

Within-setting similarity (Jaccard):
control_greedy: {'min': 1.0, 'mean': 1.0, 'max': 1.0}
min_p=0.01: {'min': 0.2236842105263158, 'mean': 0.5695729222045012, 'max': 1.0}
min_p=0.1: {'min': 0.225, 'mean': 0.496343779677113, 'max': 1.0}
min_p=0.3: {'min': 0.375, 'mean': 0.6050580431177446, 'max': 1.0}
min_p=0.5: {'min': 0.45588235294117646, 'mean': 0.7581699346405228, 'max': 1.0}
min_p=0.8: {'min': 0.5373134328358209, 'mean': 0.744583360255002, 'max': 1.0}

Between-setting similarity (Jaccard), adjacent min_p:
min_p=0.01 vs min_p=0.1: {'min': 0.15, 'mean': 0.27724663262683347, 'max': 0.4307692307692308}
min_p=0.1 vs min_p=0.3: {'min': 0.18518518518518517, 'mean': 0.32819757520353154, 'max': 0.6666666666666666}
min_p=0.3 vs min_p=0.5: {'min': 0.3835616438356164, 'mean': 0.45480920705694, 'max': 0.7454545454545455}
min_p=0.5 vs min_p=0.8: {'min': 0.45588235294117646, 'mean': 0.6486541000393141, 'max': 1.0}
```

Selected outputs (snippets):

```
===== control_greedy =====

[run 1 seed=1234]
Cyanobacteria, as oxygenic photosynthetic prokaryotes, face intense light stress in aquatic and terrestrial environments...

===== min_p=0.8 =====

[run 2 seed=1248]
Cyanobacteria, as oxygenic photosynthetic prokaryotes, face intense light stress in aquatic and terrestrial environments...
```

Verification: candidate-set size under `min_p`

We can empirically see where branching/variation collapses as `min_p` increases toward 1. Per generation step, we can watch how many tokens survive filtering as the filter brings us closer and closer to permitting only the argmax token.

```
min_p=0.01  min/mean/max = 1 / 1.6 / 8
min_p=0.1   min/mean/max = 1 / 1.35 / 4
min_p=0.3   min/mean/max = 1 / 1.125 / 2
min_p=0.5   min/mean/max = 1 / 1.05 / 2
min_p=0.8   min/mean/max = 1 / 1.075 / 2
```

Max new tokens:

Next is a brief stop in the world of `max_new_tokens`. This is a simple concept: it describes how many new tokens can be produced after the prompt is encoded.

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
        max_new_tokens=250,  # focus of this experiment: max tokens after prompt
        do_sample=True,
        temperature=0.8,
        top_k=15,  # default is 50; accept only top-15 (can vary due to ties)
        top_p=0.9,  # take smallest set whose cumulative prob >= 0.9
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

prompt_len = input_ids.shape[-1]
gen_ids = out[0, prompt_len:].detach().to("cpu").tolist()
print(tokenizer.decode(gen_ids, skip_special_tokens=True))
```
Min new tokens:

Next, we similarly look at `min_new_tokens`. This forces the model to produce at least that many new tokens after the prompt is encoded before it is allowed to stop. It is generally paired with `max_new_tokens` (because without a max, it could run for a long time until EOS is produced).

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
        min_new_tokens=20,  # focus: minimum tokens before EOS can stop generation
        do_sample=True,
        temperature=0.8,  # baseline is 1.0
        top_k=15,  # default is 50; accept only top-15 (can vary due to ties)
        top_p=0.9,  # nucleus over the top-k filtered distribution
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

prompt_len = input_ids.shape[-1]
gen_ids = out[0, prompt_len:].detach().to("cpu").tolist()
print(tokenizer.decode(gen_ids, skip_special_tokens=True))
```

Cyanobacteria, as oxygenic photosynthetic prokaryotes, face intense light exposure in aquatic and terrestrial environments, which can lead to **photoinhibition** (damage to the photosynthetic apparatus) and **photodamage** (oxidative stress

```
