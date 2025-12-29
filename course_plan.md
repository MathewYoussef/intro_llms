# Introduction to LLMs and Generative AI — Course Plan (Nemotron 3 Nano)

This is a lab-driven curriculum designed to move from “what is a language model, mathematically?” to building and evaluating an agentic RAG system—while using **NVIDIA Nemotron 3 Nano** as the anchor model throughout (and smaller models as “practice dummies” when compute gets spicy).

## Design principles
- **18 lessons** (fits 18 weeks; can compress to 12–14 by pairing adjacent lessons).
- Each lesson includes:
  1) one core concept,  
  2) one controlled experiment,  
  3) one “engineering reality check.”
- **Two-model strategy**
  - **Nemotron 3 Nano** for: inference behavior, long-context experiments, MoE/hybrid architecture study, RAG/agents/evaluation, deployment benchmarking.
  - **Small model (1–8B)** for: “train it yourself” labs (toy pretraining, SFT, RLHF/DPO/GRPO), so students can iterate without needing a small datacenter.

## Tooling stack (choose per lab)
- **Hugging Face Transformers / Datasets / PEFT / TRL** (fast iteration; great for SFT/LoRA/DPO practice).
- **NVIDIA NeMo + NeMo Evaluator** (reproducible evaluation + NVIDIA ecosystem).
- **vLLM / TensorRT-LLM** for serving & throughput experiments.

## Course outcomes (what you’ll be able to do by the end)
- Explain how modern LLMs work from tokenization → embeddings → attention → transformers → scaling tricks.
- Run controlled experiments on LLM behavior (decoding, prompting, long-context, evaluation bias).
- Fine-tune models for targeted behaviors using SFT and parameter-efficient techniques (LoRA).
- Understand preference tuning and modern RL-for-reasoning concepts at a practical level.
- Build production-minded RAG and agent workflows with robust evaluation and logging.

---

# Lesson-by-lesson plan

## Lesson 1 — What an LLM is: a probability engine with amnesia
**Skills you’ll learn**
- Explain next-token prediction as a probabilistic modeling objective (and why it “looks intelligent”).
- Run **Nemotron 3 Nano** in inference mode and capture outputs, token counts, latency, throughput.
- Recognize hallucination vs uncertainty as a property of sampling + missing evidence.
- Build a minimal “prompt → output → metadata” logging harness (tokens, time, decoding params).

**What it’s about + takeaway**  
You’ll ground the entire course in one simple claim: an LLM is trained to predict the next token, repeatedly. Everything else—reasoning, instruction following, tool use—is an emergent or engineered layer on top. The takeaway is a mental model that stays true even when the hype gets loud: *LLMs are stochastic sequence predictors whose behavior you can probe, measure, and shape.*

**Experiment**
- Generate with a fixed prompt under multiple decoding settings; log variance and failure modes.

---

## Lesson 2 — NLP tasks, datasets, and what “good” even means
**Skills you’ll learn**
- Map common NLP tasks (classification, QA, summarization, extraction, translation) to metrics and failure modes.
- Design train/val/test splits that don’t leak.
- Implement a baseline (rules or n-gram) to calibrate expectations.
- Use a simple evaluation harness for correctness vs helpfulness tradeoffs.

**What it’s about + takeaway**  
Before transformers, before MoE, before RLHF: you need to know what you’re trying to optimize. This lesson builds the habit of defining tasks precisely and choosing metrics that reflect reality, not vibes. Takeaway: *if you can’t measure “better,” you’re not doing ML—you’re doing wishcasting.*

**Experiment**
- Create a mini benchmark for one task you care about (50–200 items) with clear scoring rules.

---

## Lesson 3 — Tokenization: the LLM’s “sensory organs”
**Skills you’ll learn**
- Explain why tokenization exists and how it shapes capability and cost.
- Compare BPE/WordPiece/SentencePiece behaviors on the same text.
- Compute token budgets for prompts and RAG contexts.
- Understand chat templates and special tokens (system/user/assistant delimiters).

**What it’s about + takeaway**  
Tokenization is where human language becomes model-digestible symbols. It affects multilingual performance, code handling, and how quickly you burn context length. Takeaway: *tokenization is not preprocessing trivia; it’s part of the model interface.*

**Experiment**
- Measure “tokenization tax” (tokens per character) across: English prose, math, code, and your domain texts.

---

## Lesson 4 — Embeddings: meaning as geometry
**Skills you’ll learn**
- Train a tiny word2vec-style embedding model (skip-gram / negative sampling) on a toy corpus.
- Compute cosine similarity and build nearest-neighbor queries.
- Explain the difference between token embeddings (inside the LLM) and text embeddings (for retrieval).
- Build a minimal semantic search over documents using embeddings + vector index.

**What it’s about + takeaway**  
Embeddings compress meaning into vectors so similarity becomes geometry. This lesson is the conceptual seed of RAG: retrieve relevant context by distance in embedding space. Takeaway: *retrieval is just “geometry + indexing,” and it’s one of the cleanest antidotes to hallucination.*

**Experiment**
- Build a “semantic search notebook” and test where it fails (polysemy, negation, jargon).

---

## Lesson 5 — Pre-transformer sequence models: RNNs and LSTMs as historical fossils (useful fossils)
**Skills you’ll learn**
- Implement a simple RNN/LSTM for sequence prediction.
- Understand vanishing/exploding gradients and why gating helped.
- Build an encoder-decoder seq2seq baseline.
- Diagnose why long-range dependencies are painful for recurrence.

**What it’s about + takeaway**  
You’re learning RNN/LSTM not because they’re trendy, but because they make the transformer’s design feel inevitable. Takeaway: *attention didn’t win because it’s fashionable; it won because it scales and parallelizes.*

**Experiment**
- Compare long-context performance of LSTM vs tiny transformer on the same task.

---

## Lesson 6 — Attention: learn the trick that ate deep learning
**Skills you’ll learn**
- Derive scaled dot-product attention and explain Q/K/V intuitively.
- Implement attention in PyTorch from scratch (including masking).
- Visualize attention maps and interpret basic patterns.
- Understand cross-attention vs self-attention.

**What it’s about + takeaway**  
Attention is differentiable retrieval: the model learns where to “look” inside its context. The takeaway is a concrete grasp of the mechanism that powers transformers and modern LLMs.

**Experiment**
- Build an attention-only model for a synthetic task (copy, reverse, bracket matching).

---

## Lesson 7 — Transformer architecture: the machine that made scale practical
**Skills you’ll learn**
- Explain residuals, layer norm, MLP blocks, and why depth works.
- Distinguish encoder-only, decoder-only, and encoder-decoder transformers.
- Implement a small decoder-only transformer and train on a tiny corpus.
- Understand causal masking and next-token training.

**What it’s about + takeaway**  
This is where the modern LLM becomes legible: a stack of attention + MLP blocks trained on next-token prediction. Takeaway: *you can build a toy GPT; once you do, production LLMs become “the same thing, but huge.”*

**Experiment**
- Train a tiny GPT to overfit a small dataset, then use it to study memorization vs generalization.

---

## Lesson 8 — Position information: absolute, learned, relative, and RoPE
**Skills you’ll learn**
- Compare sinusoidal vs learned absolute positional embeddings.
- Explain why naive absolute encodings can struggle to extrapolate.
- Understand RoPE (Rotary Position Embeddings) at a high level and why it’s popular.
- Run controlled tests: same prompt, shifted positions, and context-length stress.

**What it’s about + takeaway**  
Transformers don’t inherently know order; you have to inject it. RoPE is a clever way to bake relative position behavior into attention while keeping flexibility. Takeaway: *position encoding choices show up as real behavior in long-context tasks.*

**Experiment**
- Long-context “needle-in-haystack” probe at multiple context lengths.

---

## Lesson 9 — Efficient attention & inference: MHA vs MQA vs GQA, KV caching, and serving reality
**Skills you’ll learn**
- Explain multi-head attention (MHA) vs multi-query attention (MQA) vs grouped-query attention (GQA).
- Predict memory/latency differences using KV cache reasoning.
- Measure throughput/latency under batching and streaming.
- Understand modern serving tricks like paged KV caching.

**What it’s about + takeaway**  
This is the “systems” moment: most real-world cost is inference, and inference is dominated by attention + KV cache memory traffic. GQA and MQA exist because hardware is real and latency budgets are cruel. Takeaway: *architecture choices are often economics in disguise.*

**Experiment**
- Benchmark the same model under different batch sizes and sequence lengths; plot tokens/sec vs latency.

---

## Lesson 10 — Transformer families: BERT and its descendants, plus why “encoder vs decoder” matters
**Skills you’ll learn**
- Explain masked language modeling (BERT-style) vs causal modeling (GPT-style).
- Use encoder models for embeddings, classification, and reranking.
- Understand what “instruction tuning” changes behaviorally.
- Choose the right architecture for a task (not just the biggest model).

**What it’s about + takeaway**  
Not all transformers are “chat models.” BERT-like models shine for understanding tasks; decoder-only models shine for generation. Takeaway: *architecture and objective determine your default strengths.*

**Experiment**
- Build a reranker (encoder model) and show how it improves retrieval quality vs embedding-only retrieval.

---

## Lesson 11 — Mixture of Experts: conditional computation and the art of “being huge without paying for it”
**Skills you’ll learn**
- Explain the core MoE idea: route tokens to a subset of experts.
- Understand “active parameters” vs “total parameters” and why this matters.
- Recognize MoE training challenges (load balancing, routing collapse).
- Compare dense vs sparse scaling intuitions.

**What it’s about + takeaway**  
MoE is a scaling hack: you increase capacity without increasing per-token compute proportionally. It’s also a reliability and engineering challenge. Takeaway: *MoE gives you a bigger brain per FLOP, but you pay in complexity.*

**Experiment**
- Implement a toy MoE layer and visualize routing distributions.

---

## Lesson 12 — Nemotron 3 Nano deep dive: hybrid Mamba‑Transformer MoE in the wild
**Skills you’ll learn**
- Read and extract architectural facts from a model card (layers, routing, attention style).
- Explain what a hybrid Mamba‑Transformer MoE means conceptually.
- Understand Nemotron’s design intent (throughput + long context + agentic workloads).
- Run Nemotron with/without “reasoning trace” behaviors and compare outputs.

**What it’s about + takeaway**  
Now you zoom in on your “main character” model. The takeaway: *the model is an engineered compromise between accuracy, cost, and workflow integration—not a magical oracle.*

**Experiment**
- Toggle Nemotron reasoning-trace behaviors (via chat template conventions) on the same tasks and measure accuracy vs verbosity.

---

## Lesson 13 — Prompting, in-context learning, and decoding control
**Skills you’ll learn**
- Design prompts that are testable: explicit constraints, format contracts, stop conditions.
- Explain temperature, top‑p/top‑k, and why sampling changes “personality.”
- Apply in-context learning patterns: few-shot, exemplars, scratchpads, structured outputs.
- Use self-consistency as a decoding strategy for reasoning tasks.

**What it’s about + takeaway**  
Prompting is programming in a probabilistic language. Decoding is the “runtime.” This lesson turns prompt hacks into experimental practice: hypotheses, controlled variables, measured outcomes. Takeaway: *you don’t “try prompts,” you run experiments on a system.*

**Experiment**
- Implement self-consistency for a reasoning benchmark (sample N rationales → majority answer).

---

## Lesson 14 — Pretraining: where capabilities are born (and where bills are born too)
**Skills you’ll learn**
- Understand the pretraining pipeline: data → tokenizer → objective → optimization → eval.
- Learn the core scaling knobs: data size, model size, compute, context length.
- Explain training efficiency techniques conceptually (gradient checkpointing, mixed precision).
- Run a toy pretraining experiment on a small transformer to see loss curves and failure modes.

**What it’s about + takeaway**  
Pretraining is the phase that creates general capability; post-training shapes behavior. Even if you’ll never pretrain a 30B+ model yourself, understanding the pipeline explains why LLMs know what they know—and why they forget what they don’t. Takeaway: *capability is baked in; alignment is layered on.*

**Experiment**
- Pretrain a tiny model on a narrow corpus and observe domain overfitting vs generalization.

---

## Lesson 15 — Quantization & hardware optimization: making big models actually usable
**Skills you’ll learn**
- Explain quantization at a practical level (precision vs accuracy vs speed).
- Compare common approaches: 8-bit/4-bit weight quantization, quant-aware serving.
- Use an optimized inference engine workflow (vLLM or TensorRT‑LLM) conceptually.
- Measure impact: memory footprint, latency, throughput, output quality.

**What it’s about + takeaway**  
Most LLM work dies here: “the model works” but it’s too slow or too expensive. Quantization and inference optimizations translate research into deployment. Takeaway: *performance engineering is part of model quality.*

**Experiment**
- Quantize a smaller model and measure quality regressions vs speed gains; document the trade curve.

---

## Lesson 16 — Supervised fine-tuning: teaching the model your task (without destroying its brain)
**Skills you’ll learn**
- Design an instruction dataset: schema, formatting, negative examples, edge cases.
- Train an SFT run (small model) end-to-end; understand loss vs task metrics.
- Learn “what makes SFT fail”: bad data, leakage, overfitting, prompt-template mismatch.
- Evaluate before/after with an explicit benchmark and rubric.

**What it’s about + takeaway**  
SFT is the cleanest, most controllable way to specialize behavior. It’s also where people accidentally bake in contradictions and noise. Takeaway: *fine-tuning is data engineering disguised as ML.*

**Experiment**
- Fine-tune a small model to produce structured JSON outputs reliably; measure schema compliance.

---

## Lesson 17 — Parameter-efficient fine-tuning: LoRA/QLoRA as the “lever arm”
**Skills you’ll learn**
- Apply LoRA adapters and understand what “low-rank update” means.
- Use QLoRA ideas to fit bigger models into limited VRAM (conceptually and practically on smaller models).
- Choose adapter targets (attention vs MLP) and reason about effects.
- Ship and version adapters separately from base weights.

**What it’s about + takeaway**  
LoRA is the difference between “fine-tuning requires a supercomputer” and “fine-tuning is a weekly habit.” QLoRA pushes that further by training adapters on quantized bases. Takeaway: *you can specialize large models without retraining the whole beast.*

**Experiment**
- Run two LoRA variants (small rank vs larger rank) and compare: task gain vs overfitting.

---

## Lesson 18 — Preference tuning, reasoning RL, RAG, agents, and evaluation: the full stack integration
This final lesson is deliberately systems synthesis: you’ll connect everything into an end-to-end workflow and learn the last-mile techniques that make an LLM system reliable.

### Part A — Preference tuning: RLHF + DPO
**Skills you’ll learn**
- Explain RLHF’s pieces: SFT → reward model → RL optimization loop.
- Understand PPO-style alignment at a conceptual level.
- Implement DPO (Direct Preference Optimization) on a small model using preference pairs.
- Compare SFT-only vs preference-tuned behavior changes.

**What it’s about + takeaway**  
RLHF is powerful but complex; DPO shows you can get much of the alignment effect with a simpler objective. Takeaway: *alignment isn’t magic—it’s optimization under a preference signal.*

### Part B — Reasoning models & GRPO
**Skills you’ll learn**
- Understand “reasoning as a policy optimization problem.”
- Explain GRPO as a PPO variant that uses group-based baselines to reduce resource cost.
- Design reward functions for verifiable tasks (math, unit tests, tool success).

**What it’s about + takeaway**  
Reasoning improvements often come from RL plus better training environments. Takeaway: *reasoning can be trained—if you can define what “good reasoning outcomes” are.*

### Part C — Retrieval-Augmented Generation: advanced RAG techniques
**Skills you’ll learn**
- Build a RAG pipeline: chunking, embeddings, indexing, retrieval, prompt assembly.
- Apply advanced techniques: multi-query retrieval, reranking, hybrid search, citation grounding.
- Evaluate RAG quality: answer correctness, faithfulness, retrieval hit rate.

**What it’s about + takeaway**  
RAG is the pragmatic bridge between “model knowledge” and “your knowledge.” It’s how you keep outputs anchored to evidence and keep models current. Takeaway: *good RAG is mostly retrieval engineering + evaluation discipline.*

### Part D — Function calling & agents: ReAct + LangChain/LangGraph patterns
**Skills you’ll learn**
- Design tool schemas and robust structured outputs for tool calls.
- Implement a ReAct-style loop (reason → act → observe → answer).
- Build an agentic RAG system where the model decides when to retrieve.

**What it’s about + takeaway**  
Agents turn an LLM from “text generator” into “workflow orchestrator.” ReAct is one of the cleanest conceptual templates for this. Takeaway: *tools + retrieval + planning beats raw model size surprisingly often.*

### Part E — LLM-as-a-judge: evaluation that scales, and why it lies to you sometimes
**Skills you’ll learn**
- Build a judge prompt + rubric for pairwise comparisons.
- Recognize judge biases: position bias, verbosity bias, self-preference.
- Combine LLM judges with hard checks (unit tests, references, retrieval grounding).

**What it’s about + takeaway**  
LLM-as-judge is useful precisely because human eval doesn’t scale—but it introduces subtle bias patterns. Takeaway: *use LLM judges like scientific instruments: calibrate, cross-check, and never worship the readout.*

**Capstone deliverable**
- One working system: Nemotron-powered chat/agent app with RAG + at least one tool + an evaluation harness (including at least one LLM-judge rubric and one non-LLM metric).

---

# Suggested capstone directions (pick one, then go deep)
- **Domain research assistant**: RAG over a literature corpus + citation-grounded answers + uncertainty reporting.
- **Agentic workflow bot**: tool calling over APIs (tickets, databases, code execution) with audit logs.
- **Reasoning evaluation study**: quantify effects of self-consistency vs reasoning-mode vs tool-use.
- **Fine-tuning sprint**: build a small instruction dataset + LoRA tune a base model + compare against prompting-only.
- **Serving benchmark**: compare vLLM vs TensorRT‑LLM throughput/latency under realistic loads.

---

# Inspiration sources (high-level)
- Stanford CS224N (NLP foundations + transformers)
- Stanford CS25 (Transformers United)
- CMU 11-667 (Large Language Models: Methods & Applications)
- Rycolab “Large Language Models” (theory + modern methods)
- Hugging Face course (practical transformers)
- Full Stack Deep Learning LLM Bootcamp (engineering and deployment mindset)
