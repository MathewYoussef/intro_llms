# Lesson 1 Tutorial Packet

## LLMs as Next-Token Predictors + Your First Experiments with NVIDIA Nemotron 3 Nano

**Course:** Introduction to LLMs & Generative AI (experiment-first)
**Primary model:** NVIDIA Nemotron 3 Nano
**Lesson format:** Learn -> Run -> Measure -> Reflect -> Assign

---

## 1) What you'll learn in Lesson 1

By the end of this lesson, you should be able to:

1. **Explain what an LLM does in plain language and in probability terms**

   * "It predicts the next token" -> more precisely: it models (P(\text{token}*t \mid \text{tokens}*{<t}))

2. **Trace the inference pipeline end-to-end**

   * Text -> tokens -> model logits -> probabilities -> decoding strategy (greedy/sampling) -> tokens -> text

3. **Run NVIDIA Nemotron 3 Nano and control its generation behavior**

   * Use an OpenAI-compatible API endpoint, or run locally with vLLM/Transformers

4. **Experiment with decoding knobs and interpret effects**

   * `temperature`, `top_p`, `max_tokens` (and optionally greedy vs sampling)

5. **Understand Nemotron's "reasoning mode" at a practical level**

   * Nemotron 3 Nano can optionally generate a reasoning trace before the final answer; the behavior is controlled via a chat-template flag (`enable_thinking`). ([NVIDIA API Documentation][1])

---

## 2) Key idea of Lesson 1

An LLM is not a database and not a rules engine. It's a **learned probability distribution over token sequences**, trained to **predict the next token**. At generation time, you're repeatedly asking:

> "Given everything so far, what token should come next?"

Then you either pick the most likely token (**greedy**) or sample (**stochastic decoding**) from the distribution. The "magic" is mostly "statistics + scale + good architecture," which you'll get comfortable with over the course.

---

## 3) What you need before you start

### Option A: Cloud API (recommended for Lesson 1)

This is the easiest path because you don't need GPU hardware locally.

* You need an NVIDIA API key (often formatted like `nvapi-...`)
* You'll call an **OpenAI-compatible** endpoint:
  `https://integrate.api.nvidia.com/v1/chat/completions` ([NVIDIA API Documentation][2])
* You'll use the model name: `nvidia/nemotron-3-nano-30b-a3b` ([NVIDIA API Documentation][1])

### Option B: Local vLLM server (GPU lab machines)

You can serve the model with vLLM and then call it via an OpenAI-style local endpoint. NVIDIA's model docs include a vLLM launch recipe and reasoning parser plugin. ([NVIDIA API Documentation][1])

### Option C: Local Transformers (GPU lab machines)

You can load the BF16 weights via Hugging Face Transformers:
`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` ([NVIDIA API Documentation][1])

> Note on context length: Nemotron 3 Nano supports **up to 1M context**, but the default Hugging Face config is **256k** due to VRAM requirements. ([NVIDIA API Documentation][1])
> (For Lesson 1, we'll stay far below that.)

---

## 4) Safety + privacy note (read this once, seriously)

* **Do not paste confidential data** into any hosted API (company secrets, customer data, private keys, personal medical info, etc.).
* Assume prompts may be logged by services unless you've verified otherwise.
* For local-only experiments later, you'll have more control.

---

## 5) Setup (works for Option A; also useful later)

### 5.1 Create a project folder + virtual environment

**macOS/Linux**

```bash
mkdir lesson1_nemotron
cd lesson1_nemotron
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```

**Windows (PowerShell)**

```powershell
mkdir lesson1_nemotron
cd lesson1_nemotron
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
```

### 5.2 Install dependencies

For the **API track**:

```bash
pip install -U openai python-dotenv requests
```

For **tokenization-only** demos (lightweight, no GPU required):

```bash
pip install -U transformers
```

---

## 6) Lesson walkthrough

### Part 1 -- First contact: "Hello, Nemotron"

You'll send a simple prompt and read the model response.

#### 6.1 Put your API key in a `.env` file (Option A)

Create a file named `.env` in your project folder:

```bash
NVIDIA_API_KEY=nvapi-your-key-here
```

#### 6.2 Create `hello_nemotron.py`

```python
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ["NVIDIA_API_KEY"],
)

resp = client.chat.completions.create(
    model="nvidia/nemotron-3-nano-30b-a3b",
    messages=[
        {"role": "user", "content": "Write a haiku about GPUs."}
    ],
    temperature=1.0,
    top_p=1.0,
    max_tokens=200,
)

print(resp.choices[0].message.content)
```

Run:

```bash
python hello_nemotron.py
```

**What to notice:**

* You got a response. Congrats -- your pipeline works.
* You just ran an autoregressive generation loop under the hood.

**Mini-checkpoint (write 2-3 sentences):**
What parts of the pipeline are visible here (messages in, text out), and what parts are hidden (tokenization, logits, decoding)?

---

### Part 2 -- The decoding knobs: determinism vs randomness

#### 6.3 Create `sampling_lab.py`

This script runs the same prompt multiple times under different decoding settings and saves results.

```python
import os, json, time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ["NVIDIA_API_KEY"],
)

PROMPT = "Invent a new ice cream flavor and describe it in one sentence."
RUNS_PER_SETTING = 5

settings = [
    {"name": "greedy-ish", "temperature": 0.0, "top_p": 1.0},
    {"name": "low-random", "temperature": 0.2, "top_p": 1.0},
    {"name": "balanced", "temperature": 0.7, "top_p": 0.95},
    {"name": "high-random", "temperature": 1.0, "top_p": 1.0},
]

def call_model(temp, top_p):
    resp = client.chat.completions.create(
        model="nvidia/nemotron-3-nano-30b-a3b",
        messages=[{"role": "user", "content": PROMPT}],
        temperature=temp,
        top_p=top_p,
        max_tokens=120,
    )
    return resp.choices[0].message.content.strip()

results = []
for s in settings:
    print(f"\n=== Setting: {s['name']} (T={s['temperature']}, top_p={s['top_p']}) ===")
    seen = set()
    for i in range(RUNS_PER_SETTING):
        t0 = time.time()
        out = call_model(s["temperature"], s["top_p"])
        dt = time.time() - t0
        seen.add(out)
        print(f"[{i+1}] ({dt:.2f}s) {out}")
        results.append({
            "setting": s["name"],
            "temperature": s["temperature"],
            "top_p": s["top_p"],
            "output": out,
            "latency_s": dt,
        })
    print(f"Unique outputs: {len(seen)}/{RUNS_PER_SETTING}")

with open("sampling_results.jsonl", "w", encoding="utf-8") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print("\nSaved: sampling_results.jsonl")
```

Run:

```bash
python sampling_lab.py
```

#### 6.4 What you're learning here (concept)

* **Temperature**: scales logits before softmax. Lower = sharper distribution (more "confident" / repetitive); higher = flatter (more diversity, more risk).
* **Top-p (nucleus sampling)**: restricts sampling to the smallest set of tokens whose cumulative probability >= p.

**Activity: Make a claim + defend it with evidence**
In your notes, answer:

1. Which setting produced the most diverse outputs?
2. Which setting produced the most "boring but consistent" outputs?
3. What surprised you?

---

### Part 3 -- Tokens: see what the model actually consumes

Even if you're using the cloud API, you can still explore the tokenizer locally (no GPU required).

#### 6.5 Create `tokenization_peek.py`

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    trust_remote_code=True
)

text = "LLMs generate text one token at a time."
enc = tok(text)

print("Text:", text)
print("Token IDs:", enc["input_ids"])
print("Tokens:", tok.convert_ids_to_tokens(enc["input_ids"]))
print("Decoded back:", tok.decode(enc["input_ids"]))
```

Run:

```bash
python tokenization_peek.py
```

**What to notice:**

* Tokens are often **subword pieces** (not "words").
* Tokenization is a major reason prompting can feel weirdly sensitive.

**Mini-checkpoint:**
Why does "the model predicts the next *token*" matter more than "the model predicts the next *word*"?

---

### Part 4 -- (Optional but recommended) Next-token probabilities with a tiny model

Nemotron is big; token-prob introspection is easiest on a small model for Lesson 1. This exercise teaches the concept without requiring heavy compute.

#### 6.6 Create `next_token_probs_demo.py`

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")
model.eval()

prompt = "The capital of France is"
inputs = tok(prompt, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits[0, -1]  # next-token logits

def show_topk(logits, temperature=1.0, k=10):
    scaled = logits / max(temperature, 1e-8)
    probs = torch.softmax(scaled, dim=-1)
    vals, idxs = torch.topk(probs, k=k)
    print(f"\nTop-{k} next tokens (temperature={temperature}):")
    for p, idx in zip(vals, idxs):
        token_str = tok.decode([idx.item()])
        print(f"{token_str!r:>12}  p={p.item():.4f}")

print("Prompt:", prompt)
show_topk(logits, temperature=0.5)
show_topk(logits, temperature=1.0)
show_topk(logits, temperature=1.5)
```

Run:

```bash
python next_token_probs_demo.py
```

**Takeaway:** temperature literally reshapes the probability landscape. Generation is just walking that landscape one token at a time.

---

## 7) Nemotron-specific feature: "Reasoning mode" (ON vs OFF)

Nemotron 3 Nano is described as a unified model for reasoning and non-reasoning tasks; it can generate a reasoning trace first and then a final answer. This can be configured via a flag in the chat template. ([NVIDIA API Documentation][1])

### 7.1 If you run locally with Transformers (Option C)

NVIDIA's quickstart shows:

* Reasoning ON is default
* To turn it OFF, pass `enable_thinking=False` to `apply_chat_template()` ([NVIDIA API Documentation][1])

It also notes:

* `temperature=1.0` and `top_p=1.0` recommended for reasoning tasks
* `temperature=0.6` and `top_p=0.95` recommended for tool calling ([NVIDIA API Documentation][1])

### 7.2 If you run locally with vLLM (Option B)

NVIDIA's docs show you can pass:

```json
"chat_template_kwargs": {"enable_thinking": false}
```

in your OpenAI-style request to disable reasoning. ([NVIDIA API Documentation][1])

> For the cloud API (Option A), the endpoint is OpenAI-compatible, but support for extra fields can vary by service. Start with the standard request first; if you later stand up your own vLLM server, the flag above is the cleanest control.

---

## 8) In-lesson activities (do these before homework)

### Activity A -- "Parameter sensitivity journal"

Pick **one** prompt you like (creative or technical). Run:

* T=0.0, top_p=1.0 (most deterministic)
* T=0.7, top_p=0.95 (balanced)
* T=1.0, top_p=1.0 (high freedom)

Write down:

1. One difference you observe in **style**
2. One difference you observe in **factuality or risk**
3. One difference you observe in **length**

### Activity B -- "The model is not a mind"

Run this prompt twice -- once with extra context, once without:

**Prompt 1 (no context):**

> "Write an email asking for a deadline extension."

**Prompt 2 (with context):**

> "You are a student in a graduate ML course. Write an email asking for a 48-hour deadline extension because your GPU instance died and you need to rerun experiments. Be polite, concise, and propose a plan."

Compare outputs. In 3-5 sentences, explain what "conditioning on context" means.

---

## 9) Homework assignment (submit next session)

### Deliverable 1 -- Code + output log

Submit:

* Your `sampling_lab.py` (or notebook)
* `sampling_results.jsonl`

### Deliverable 2 -- Mini lab report (1-2 pages)

Answer the following:

1. **Explain in your own words:**
   What does an LLM learn during training if it's trained on next-token prediction?

2. **Evidence-based decoding discussion:**
   Using your `sampling_results.jsonl`, argue (with examples) when you would use:

   * low temperature / greedy
   * moderate sampling
   * high sampling

3. **One failure mode you observed:**
   Examples: repetition, blandness, hallucinated facts, inconsistency, etc.
   Explain how decoding parameters contributed (or didn't).

### Deliverable 3 -- Short quiz (answer in plain text)

1. What is a "token" and why do we care?
2. What does temperature do conceptually?
3. What does `top_p` do conceptually?
4. Why can the same prompt produce different outputs?
5. What is the difference between "model weights" and "context"?

---

## 10) Grading rubric (simple and transparent)

* **40%** Correctly working code + outputs saved (reproducible run instructions included)
* **30%** Lab report shows real observations, not just definitions
* **20%** Quiz correctness
* **10%** Clarity and good experimental hygiene (label settings, keep logs, etc.)

---

## 11) What you should walk away with (the real takeaway)

By the end of Lesson 1, you're not "good at prompting" yet -- and that's fine. The win is deeper:

You've built the mental model that **generation is controlled randomness over tokens**, and you've built a tiny experiment harness that will carry the whole course.

Next lesson, we'll start tightening the screws: **tokenization** as a first-class object (not just a hidden step), and why it quietly shapes everything from RAG to fine-tuning.

[1]: https://docs.api.nvidia.com/nim/reference/nvidia-nemotron-3-nano-30b-a3b "nvidia / nemotron-3-nano-30b-a3b"
[2]: https://docs.api.nvidia.com/nim/reference/nvidia-nemotron-3-nano-30b-a3b-infer "Creates a model response for the given chat conversation."
