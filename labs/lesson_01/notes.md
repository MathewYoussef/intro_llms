2025-12-29 — Beginning Lesson 1.

I am beginning with a local-first approach as opposed to the cloud API. Specifically, I will be setting up vLLM for serving the model and running easy, web-UI-based tests, while also setting up a local download and lab via containerization and `docker-compose` on my Jetson Thor. This will be the primary box running the AI lessons unless we require a different, non-unified architecture for experiments, in which case we will transition to RTX GPUs. However, as it stands, I do not currently have the knowledge to differentiate between the pros and cons of either right now. We will begin by setting up containers on the Thor and a vLLM mode so we can work between the two.

We have added a `docker-compose.yml` to the repository to reliably and repeatably create environments for serving via vLLM and for running experiments in a lab-style environment.

In this workspace, any edits we make inside the container are saved back into our local files (via bind mounts).

To access JupyterLab for course work while hosting the model on Thor (and using Thor as the primary machine):

1) Start the lab container on Thor:

```bash
docker compose -f infra/compose.yml --env-file .env up lab
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
