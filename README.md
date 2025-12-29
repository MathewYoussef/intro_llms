# LLMs & Generative AI — Experiment-Driven Course (Nemotron 3 Nano)

Welcome. This repository is a **hands-on course** on Large Language Models (LLMs) and Generative AI, built around **experiments** rather than memorization.

You’ll learn how modern LLMs work from the inside out (tokenization → embeddings → attention → transformers → scaling and alignment), and you’ll finish by building a **production-minded agentic RAG system** with real evaluation.

The anchor model for the course is **NVIDIA Nemotron 3 Nano**. When training compute is limited, we’ll use smaller open models for fine-tuning labs and transfer the lessons back to Nemotron for inference, long-context, RAG/agents, and deployment experiments.

---

## What you will learn

By the end, you should be able to:

- Explain the core mechanics of LLMs:
  - tokenization, embeddings, sequence modeling
  - attention and the transformer architecture
  - position encodings (including RoPE)
  - efficient attention variants (MHA / MQA / GQA) and KV caching
  - Mixture-of-Experts (MoE) and modern scaling tradeoffs

- Run **controlled experiments** on model behavior:
  - decoding settings (temperature, top-p, etc.)
  - prompt design and in-context learning
  - long-context probing
  - evaluation and error analysis

- Specialize models safely:
  - supervised fine-tuning (SFT)
  - parameter-efficient fine-tuning (LoRA / QLoRA)
  - preference tuning (RLHF overview, DPO)
  - reasoning-focused RL concepts (e.g., GRPO-level ideas)

- Build systems that don’t fall apart in reality:
  - Retrieval-Augmented Generation (RAG) including advanced techniques
  - tool / function calling
  - agent loops (ReAct-style)
  - “LLM-as-a-judge” evaluation, plus safeguards against judge bias

---

## How the course works

This repo is structured as a **sequence of 18 lessons**:

- `lessons/lesson_1.md` … `lessons/lesson_18.md`

Each lesson includes:
- a concept focus,
- a lab (the experiment),
- and required artifacts (notes + logs + evaluation).

### The most important rule
**You don’t pass by reading. You pass by measuring.**

Every “insight” should be backed by:
- a runnable experiment,
- logged configs,
- reproducible results,
- and a short written interpretation.

---

## The Teacher Agent (how you get help without losing the learning)

This course includes a “teacher behavior contract”:

- `AGENTS.md`

The teacher is not here to be a code vending machine. It is a **facilitator and evaluator**.

### What the teacher agent *will* do
- Explain concepts clearly.
- Help you design experiments and choose controls/baselines.
- Provide scaffolding (templates, TODO stubs, checklists).
- Help you debug (when you provide logs/errors).
- Assess your work and detect misconceptions early.
- Temporarily diverge from the plan to catch you up when needed.

### What the teacher agent *will not* do
- Write complete end-to-end solutions for the labs.
- Run experiments for you.
- “Just give the answer” to graded tasks.
- Produce final reports on your behalf.

If you try to outsource the core thinking, the teacher agent should refuse and switch to guided coaching.

---

## Recommended workflow (per lesson)

1) **Read the lesson file**  
   Example: `lessons/lesson_03.md`

2) **Start a teacher session**  
   Use your preferred LLM/chat interface. Provide it:
   - the contents of `AGENTS.md`
   - the current lesson file
   - your current repo state and what you’ve tried

   Suggested opening message:
   > “You are the course teacher. Follow AGENTS.md strictly.  
   > I’m working on lesson_03. Help me set up the lab and design the experiments.  
   > Do not write the full solution.”

3) **Set up the lab folder**  
   Create a working directory for the lesson:
   - `labs/lesson_03/`
   - `labs/lesson_03/notes.md`
   - `runs/` (raw logs + metrics)

4) **Run the experiments**  
   You run the commands. You collect evidence.

5) **Debrief and reflect**  
   In `labs/lesson_XX/notes.md`, write:
   - what you expected,
   - what happened,
   - what surprised you,
   - what you’d test next,
   - and what you now believe (with evidence).

6) **Self-check using the rubric**  
   The teacher will score your work on:
   - conceptual clarity,
   - implementation,
   - measurement quality,
   - reproducibility,
   - reflection.

---

## Project structure

```
.
├─ README.md
├─ course_plan.md
├─ AGENTS.md
├─ lessons/
│  ├─ lesson_1.md
│  ├─ ...
│  └─ lesson_18.md
├─ labs/
│  ├─ lesson_01/
│  │  ├─ notes.md
│  │  ├─ starter.py
│  │  └─ results/
│  └─ ...
├─ src/
│  ├─ data/
│  ├─ models/
│  ├─ eval/
│  └─ utils/
├─ runs/
└─ requirements.txt  (or environment.yml)
```

Not all folders may exist on day one. You’ll create them as you go.

---

## Getting started

### 1) Prerequisites
- Comfort writing Python.
- Basic linear algebra/probability helps, but the course is structured to teach what’s needed as you go.
- A GPU is helpful. Not required for every lesson, but strongly recommended for later labs.

### 2) Environment setup (typical)
Choose one:
- `venv` + `pip`
- `conda` / `mamba`

Install dependencies (once `requirements.txt` is provided):
```bash
pip install -r requirements.txt
```

### 3) Choose a model backend
You can complete the course using one or more of:
- Hugging Face Transformers (local inference / smaller-model training labs)
- vLLM or TensorRT-LLM (serving + throughput labs)
- a hosted endpoint (if you have one available)

The course is designed so the *concept* is what matters—the backend is just plumbing.

---

## Course checkpoints

### Mid-course checkpoint (after Lesson 9)
You should be able to:
- run controlled decoding experiments,
- measure latency/throughput and reason about KV cache,
- explain attention and transformers clearly,
- implement at least a toy transformer / attention module.

### Late-course checkpoint (after Lesson 17)
You should be able to:
- run SFT and LoRA on a small model,
- compare prompting vs tuning (with evidence),
- detect overfitting and data leakage,
- build an evaluation rubric and apply it consistently.

### Capstone (Lesson 18)
You will build a working system:
- Nemotron-powered chat/agent app
- RAG + at least one tool/function call
- logging + reproducibility
- evaluation including:
  - at least one LLM-judge rubric
  - at least one non-LLM metric (tests, exact match, retrieval hit rate, etc.)

---

## How to succeed in this course (the meta-skill)

- Treat every output as a hypothesis until measured.
- Keep a clean trail: configs, seeds, prompts, metrics, logs.
- Change one variable at a time.
- Use baselines and ablations.
- Write down what you believe *and what would change your mind*.

That last one sounds philosophical, but it’s also just good engineering.

---

## Files to read first
1) `course_plan.md` — the full curriculum and lesson flow  
2) `AGENTS.md` — the teacher behavior rules and assessment approach  
3) `lessons/lesson_1.md` — start here

---

## License / reuse
If you’re adapting this course for your own cohort, keep the spirit intact:
- labs stay lab-driven,
- assessment stays evidence-based,
- the teacher agent remains a facilitator, not a solution generator.
