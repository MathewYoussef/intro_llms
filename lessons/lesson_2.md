Lesson 2 Tutorial Packet
NLP Tasks, Datasets, and What "Good" Even Means

This packet is designed so a student can complete Lesson 2 end-to-end with learning + examples + guided activities + assignments. It includes a complete mini-lab (with starter dataset + starter code) and a graded-style assignment that becomes the backbone of later lessons (SFT, LoRA, RAG, eval, etc.).

0) Big idea of Lesson 2

An LLM is not "good" or "bad" in the abstract. It's good at a task, under a metric, on a dataset (with a particular distribution), under constraints (latency, cost, safety, format requirements).

Task + dataset + metric + constraints = what "good" means.

If you can't define those four, you can't reliably improve anything -- you can only vibe.

1) What you'll learn (skills checklist)

By the end of Lesson 2 you should be able to:

A. Task thinking

Convert a vague goal ("help customers faster") into a precise NLP task with inputs/outputs.

Choose the right task form: classification vs extraction vs QA vs summarization vs ranking vs generation.

B. Dataset design

Build a dataset schema (what fields exist, what labels mean, what "ground truth" is).

Write annotation guidelines so two humans would label examples the same way.

Construct train/val/test splits that avoid leakage (the silent killer).

C. Evaluation and metrics

Pick metrics that actually reflect what you care about (accuracy vs F1 vs exact match vs human rubric).

Build an evaluation harness that produces:

overall score,

per-class scores,

confusion matrix,

a curated error slice for analysis.

D. Baselines and comparisons

Implement a "dumb baseline" (majority class / keyword rules / regex).

Compare baseline vs LLM (Nemotron 3 Nano if available), and explain why each fails.

E. Reporting like a scientist

Produce a short "experiment report" with:

dataset description,

metric definition,

results,

error analysis,

next steps.

2) Prerequisites & setup
Required

Python 3.10+

A way to run code: Jupyter notebook or plain .py

Optional (for LLM evaluation)

You need some way to call Nemotron 3 Nano (choose one):

An inference endpoint your course provides (recommended for class).

Local model weights (only if you have serious GPU resources).

A smaller local model (fallback) for the lab if Nemotron access isn't available yet.

This lesson works even if you can't run an LLM: you'll still build the dataset + evaluation + baseline.

3) Core concepts (read this first)
3.1 Task types (what the model is supposed to do)

Here are common NLP task "shapes":

Classification
Input: text -> Output: one label from a small set
Example: route a support ticket: billing | technical | account | cancellation

Extraction / structured output
Input: text -> Output: fields (JSON)
Example: extract {order_id, refund_amount, reason}

Question Answering (QA)
Input: context + question -> Output: answer
Example: "Given policy text, what is the refund window?"

Summarization
Input: long text -> Output: shorter text
But "good" is tricky: short, accurate, covers key points, no hallucinations.

Retrieval / ranking
Input: query + documents -> Output: ranked list
Often evaluated with ranking metrics (e.g., recall@k).

Key move: Choose the task shape that makes evaluation easiest and most meaningful.

3.2 Datasets are not just "data" -- they are the task definition

A dataset is a specification:

What counts as input?

What counts as the correct output?

What edge cases exist?

What distribution you care about?

If your dataset is messy or ambiguous, your model will look "randomly wrong" because the target itself is fuzzy.

3.3 Leakage (a.k.a. "your eval is lying to you")

Leakage happens when information from the test set sneaks into training or prompt context.

Common leakage patterns:

Duplicate or near-duplicate examples across splits

Same customer/thread appears in both train and test

Time leakage: training includes future policy updates while test is earlier (or vice versa)

Labels embedded in the input ("Category: billing" inside the text)

Evaluation prompt includes the answer (surprisingly common)

Rule of thumb: If a human could "cheat" using information that wouldn't exist at deployment time, it's leakage.

3.4 Metrics: pick what you actually care about

Accuracy: "fraction correct" -- great when classes balanced and all errors equal.

Precision/Recall/F1: better for imbalance or when some errors are worse.

Exact Match: strict string match (great for extraction).

Token-level F1: forgiving overlap metric (common for QA).

ROUGE/BLEU: text overlap metrics for summarization/translation; can be misleading.

Human rubric: often necessary for generation tasks; needs careful design.

Important reality: For many LLM tasks, you need two layers:

Automatic checks (format, constraints, exact facts when possible)

Human or rubric-based judgment (quality, usefulness, tone, coverage)

3.5 Baselines: the sanity anchor

A baseline is not embarrassing. A baseline is how you avoid fooling yourself.

Examples:

Majority label baseline

Keyword routing (if contains "refund" -> billing)

Regex extraction (order IDs)

Simple retrieval baseline (BM25, keyword search)

A strong baseline can beat a weak LLM prompt. That's not a scandal. That's science.

4) Worked examples (quick, concrete)
Example A -- Classification: ticket routing

Labels: billing | technical | account | cancellation

Metric choice:

If label distribution is balanced -> accuracy is okay

If "cancellation" is rare but critical -> use per-class F1 and macro-F1

What error types matter?

Misrouting billing to technical causes longer resolution time

Misrouting cancellation to account might lose customers
So: you might weight errors differently later, but first measure cleanly.

Example B -- Extraction: refund amount

Input: "I was charged $24.99 twice--please refund one of them."

Output: { "refund_amount": 24.99 }

Metric choice:

Exact match on numeric value (or within tolerance)

Also measure parse success rate (did it output valid JSON?)

Example C -- Summarization: meeting notes

ROUGE might say two summaries are similar, but:

One could hallucinate action items (very bad)

One could omit the main decision (also bad)

So you use:

Format checks + factuality checks when possible

Human rubric with clear criteria (coverage, correctness, concision)

5) Guided Lab: Build a mini benchmark + baseline + evaluation harness

You'll do this in 5 parts.

Part 1 -- Create a tiny starter dataset (provided)

We'll use a toy helpdesk routing dataset. You can replace it later with your own domain.

Task

Given a helpdesk message, predict one label:

billing

technical

account

cancellation

Step 1: Paste and run this code (creates dataset in memory)
```python
toy = [
    {"id": 1, "text": "I was charged twice this month. Can you refund the extra payment?", "label": "billing"},
    {"id": 2, "text": "My card was declined but my bank says it's fine. What's going on?", "label": "billing"},
    {"id": 3, "text": "The app crashes every time I try to upload a file.", "label": "technical"},
    {"id": 4, "text": "I'm getting a 500 error when I log in from my laptop.", "label": "technical"},
    {"id": 5, "text": "I forgot my password and the reset link never arrives.", "label": "account"},
    {"id": 6, "text": "Please change the email on my account; I no longer have access to the old one.", "label": "account"},
    {"id": 7, "text": "I want to cancel my subscription at the end of the month.", "label": "cancellation"},
    {"id": 8, "text": "Stop renewing my plan. I'm done after this billing cycle.", "label": "cancellation"},
    {"id": 9, "text": "Why did my invoice include an extra seat? I didn't add anyone.", "label": "billing"},
    {"id": 10, "text": "Two-factor authentication isn't working; the code is always invalid.", "label": "account"},
    {"id": 11, "text": "Your website loads but the dashboard is blank.", "label": "technical"},
    {"id": 12, "text": "I need a receipt for last month's payment for my expense report.", "label": "billing"},
    {"id": 13, "text": "Please delete my account and cancel any active plans.", "label": "cancellation"},
    {"id": 14, "text": "I keep getting logged out and asked to sign in again every few minutes.", "label": "technical"},
    {"id": 15, "text": "I can't update my payment method-- it says 'invalid address' but it's correct.", "label": "billing"},
    {"id": 16, "text": "My profile shows the wrong name and I can't edit it.", "label": "account"},
]
labels = ["billing", "technical", "account", "cancellation"]
len(toy)
```

You should see 16.

Part 2 -- Split the data (and learn why splitting is not trivial)

For this toy dataset we'll do a simple random split. In real tasks you'll use smarter splits (by customer, by thread, by time).

```python
import random

def split_dataset(data, train_frac=0.7, seed=0):
    data = list(data)
    random.Random(seed).shuffle(data)
    n_train = int(len(data) * train_frac)
    return data[:n_train], data[n_train:]

train, test = split_dataset(toy, train_frac=0.7, seed=42)
(len(train), len(test), [x["id"] for x in test])
```

Checkpoint: You now have a test set. Treat it like radioactive material. Don't "peek and tweak" too much, or you'll overfit your choices to it.

Part 3 -- Build two baselines
Baseline 1: majority class

This is the dumbest baseline. It's still useful.

```python
from collections import Counter

def majority_baseline(train_data):
    counts = Counter(x["label"] for x in train_data)
    return counts.most_common(1)[0][0]

maj_label = majority_baseline(train)
maj_label
```

Predict that label for all test samples.

Baseline 2: keyword rules

A slightly smarter baseline that often surprises people.

```python
def keyword_baseline(text: str) -> str:
    t = text.lower()
    if any(w in t for w in ["refund", "invoice", "charged", "payment", "receipt", "billing", "card", "seat"]):
        return "billing"
    if any(w in t for w in ["crash", "error", "500", "bug", "blank", "dashboard", "upload", "website"]):
        return "technical"
    if any(w in t for w in ["password", "reset", "2fa", "two-factor", "email", "profile", "logged out", "login"]):
        return "account"
    if any(w in t for w in ["cancel", "cancellation", "stop renewing", "delete my account", "end of the month"]):
        return "cancellation"
    # fallback
    return "account"
```

Part 4 -- Build an evaluation harness (metrics + confusion matrix)
Why this matters

This harness will be reused later for:

SFT evaluation

LoRA comparisons

RAG system evaluations

judge-model experiments

Core evaluation code
```python
def confusion_matrix(y_true, y_pred, labels):
    idx = {lab:i for i,lab in enumerate(labels)}
    m = [[0 for _ in labels] for __ in labels]
    for t,p in zip(y_true, y_pred):
        m[idx[t]][idx[p]] += 1
    return m

def accuracy(y_true, y_pred):
    return sum(t==p for t,p in zip(y_true, y_pred)) / max(1, len(y_true))

def per_class_f1(y_true, y_pred, labels):
    # micro-utilities
    results = {}
    for lab in labels:
        tp = sum((t==lab and p==lab) for t,p in zip(y_true, y_pred))
        fp = sum((t!=lab and p==lab) for t,p in zip(y_true, y_pred))
        fn = sum((t==lab and p!=lab) for t,p in zip(y_true, y_pred))
        prec = tp / (tp + fp) if (tp+fp) else 0.0
        rec  = tp / (tp + fn) if (tp+fn) else 0.0
        f1   = (2*prec*rec / (prec+rec)) if (prec+rec) else 0.0
        results[lab] = {"precision": prec, "recall": rec, "f1": f1}
    macro_f1 = sum(results[lab]["f1"] for lab in labels) / len(labels)
    return results, macro_f1

def evaluate_classifier(test_data, predict_fn, labels):
    y_true = [x["label"] for x in test_data]
    y_pred = [predict_fn(x["text"]) for x in test_data]
    acc = accuracy(y_true, y_pred)
    pc, macro = per_class_f1(y_true, y_pred, labels)
    cm = confusion_matrix(y_true, y_pred, labels)
    return {"accuracy": acc, "macro_f1": macro, "per_class": pc, "confusion": cm, "y_true": y_true, "y_pred": y_pred}
```

Run evaluation
```python
# Majority baseline predictor
def predict_majority(_text):
    return maj_label

maj_results = evaluate_classifier(test, predict_majority, labels)
kw_results  = evaluate_classifier(test, keyword_baseline, labels)

maj_results["accuracy"], kw_results["accuracy"], kw_results["macro_f1"]
```

Pretty-print confusion matrix
```python
def print_confusion(cm, labels):
    header = "true\\pred".ljust(12) + " ".join(lab.ljust(12) for lab in labels)
    print(header)
    for i,lab in enumerate(labels):
        row = lab.ljust(12) + " ".join(str(cm[i][j]).ljust(12) for j in range(len(labels)))
        print(row)

print("Majority confusion:")
print_confusion(maj_results["confusion"], labels)

print("\nKeyword confusion:")
print_confusion(kw_results["confusion"], labels)
```

Checkpoint: You can now compare any two systems fairly, as long as they output a label per example.

Part 5 -- Optional: Evaluate an LLM (Nemotron 3 Nano) on the same test set
Important note about LLM evaluation

LLMs are stochastic. If you evaluate with temperature > 0, results vary. For "measurement mode," set:

temperature = 0 (or as low as possible)

consistent prompt template

strict output format

Prompt template (classification)

Use a prompt that forces a single label output (ideally JSON).

System message (recommended):

You are a routing classifier. Output only JSON.

User message:

Given the helpdesk message, classify into one of: billing, technical, account, cancellation.
Output JSON: {"label": "<one_of_labels>"}
Message: ...

Integration stub

Because your exact Nemotron calling method depends on your environment (endpoint vs local weights), the packet gives a single function you must implement once:

```python
import json, re

def call_llm_classify(text: str) -> str:
    """
    Return one of the labels: billing, technical, account, cancellation
    using YOUR Nemotron 3 Nano access method.
    """
    # TODO: replace this with your actual model call
    # response_text = ...
    response_text = '{"label": "account"}'  # placeholder

    # Parse JSON label robustly
    m = re.search(r'\{.*\}', response_text, flags=re.S)
    if not m:
        return "account"
    try:
        obj = json.loads(m.group(0))
        lab = obj.get("label", "").strip().lower()
        return lab if lab in labels else "account"
    except Exception:
        return "account"
```

Then evaluate:

```python
llm_results = evaluate_classifier(test, call_llm_classify, labels)
llm_results["accuracy"], llm_results["macro_f1"]
```

If you can't call Nemotron yet: skip this part. The main learning is the dataset + baseline + harness.

Part 6 -- Error analysis (the part most people skip... and the part that matters)

Pick 5 mistakes from your best baseline (or LLM if you ran it) and categorize them:

Error categories (use these labels):

Ambiguous label (dataset problem)

Missing keyword coverage (baseline limitation)

Multiple intents in one message (task definition issue)

Policy knowledge needed (requires retrieval / RAG later)

Formatting / parsing failure (LLM output not constrained)

Out-of-distribution (message type not represented in training)

Template:

Example ID:

Text:

True label:

Predicted:

Category:

Fix: (data? prompt? new label? tool? retrieval? better baseline?)

6) Activities (do during class or self-study)
Activity 1 -- Task reframing drills (10-15 minutes)

For each vague goal, write a precise task spec:

"Make meeting notes better."

"Reduce support costs."

"Help sales answer questions faster."

For each, define:

Input(s)

Output(s)

Label schema or output format

Metric(s)

Constraints (latency, cost, tone, citations)

Deliverable: 1-2 paragraphs per goal.

Activity 2 -- Leakage detective (15 minutes)

Decide whether each situation is leakage and explain why:

Two tickets from the same customer appear in train and test.

You deduplicate exact string matches, but not near-duplicates.

You evaluate on "test," but you tweak prompts until test accuracy improves.

Your examples contain "Category: billing" in the raw text.

You split randomly, but your deployment is time-ordered.

Deliverable: a short answer for each.

Activity 3 -- Metric matchmaker (15 minutes)

Choose the best metric for each task and justify:

Extracting an order ID

Detecting fraud messages (rare positives)

Summarizing a legal clause without hallucinations

Routing tickets where misrouting "cancellation" is very costly

Deliverable: metric + 2-3 sentence justification each.

7) Assignments (graded) -- "Build your benchmark"

You will create a benchmark that you will reuse for later lessons (SFT, LoRA, RAG).

Assignment 2.1 -- Define your task (written spec)

Deliverable: task_spec.md (1-2 pages)

Include:

Task name

Intended use case

Input schema

Output schema (labels or structured JSON)

Label definitions / rubric (clear enough for a second person to label consistently)

Primary metric + why

At least 5 "edge cases" you expect

Grading focus: clarity, measurability, and non-ambiguous definitions.

Assignment 2.2 -- Build your dataset (50-200 examples)

Deliverables:

data/train.jsonl

data/test.jsonl

data/README.md

Rules:

Must include an id, input, and label (or output) field.

Avoid leakage: no duplicates or near-duplicates across splits.

If there's grouping (same user/thread/doc), split by group.

Minimum quality bar:

At least 10 examples per label if classification (or justify imbalance).

At least 10 "hard cases" that would fool a naive baseline.

Assignment 2.3 -- Baseline + evaluation harness

Deliverables:

baseline.py (or notebook)

eval.py (or notebook)

A result summary table in results.md

Required outputs:

Accuracy (or appropriate metric)

Per-class F1 (if classification)

Confusion matrix (if classification)

At least 5 error analyses (as described in Part 6)

Baseline types allowed:

Majority class

Keyword/rules

Regex extraction

Simple retrieval baseline (keyword search)

Any non-LLM baseline

Assignment 2.4 -- LLM comparison (Nemotron 3 Nano preferred)

Deliverables:

llm_eval.py (or notebook)

prompt.txt (the prompt template you used)

results_llm.md

Requirements:

Use temperature 0 (or explain if not possible)

Strict output format (JSON strongly recommended)

Compare baseline vs LLM and explain:

where the LLM helps

where it fails

what you would try next (data? prompt? better metric? retrieval?)

If you don't have LLM access, submit a "planned experiment" section explaining how you would run it and what you'd measure.

Assignment 2.5 -- Mini "experiment report" (the scientist habit)

Deliverable: report.md (~ 1-2 pages)

Template:

What did you build?

Dataset summary (size, labels, examples)

Metric definition

Baseline results

LLM results (if run)

Error analysis (top 3 failure modes)

Next steps

8) Rubric (100 points)
Task spec (20)

(10) Clear schema + label definitions

(10) Metric + constraints are appropriate

Dataset (30)

(10) Enough examples + label coverage

(10) Low leakage risk + sensible split logic

(10) Includes challenging edge cases

Baseline + harness (25)

(10) Baseline implemented and explained

(10) Eval harness correctness + useful outputs

(5) Error analysis quality

LLM comparison (15)

(10) Prompting + output constraints + reproducibility

(5) Analysis: where LLM helps/fails

Report quality (10)

(5) Clear results presentation

(5) Thoughtful next steps

9) Knowledge checks (quick quiz)

Why can accuracy be misleading on imbalanced datasets?

Give two examples of data leakage.

What's the difference between a task definition and a dataset?

For extraction tasks, why is "parse success rate" often a metric?

Why is a baseline valuable even if it performs poorly?

(Answer key at the end.)

10) "Common failure modes" cheat sheet

Label ambiguity: if humans disagree, the model will look "randomly wrong."

Overfitting to test: tweaking prompts until test looks good = test is no longer test.

Metric mismatch: optimizing ROUGE while users care about factuality.

Hidden constraints: format requirements ignored until deployment breaks.

No baseline: you can't tell if your fancy approach is actually better than "if refund -> billing."

11) Answer key (for the quiz)

If one class dominates, predicting it always yields high accuracy while being useless for the rare class.

Duplicates across train/test; same customer/thread in both; time leakage; label text embedded in input; repeated prompt tuning on test.

Task definition specifies the input/output and what correctness means; dataset is an implementation (examples) of that spec and implicitly fixes the distribution.

If the model outputs malformed JSON or wrong format, downstream systems can't use it -- even if the "content" is right.

It anchors expectations, prevents self-deception, and tells you whether your "improvement" is real.

End-state of Lesson 2 (what you should have in your hands)

By the end, every student should have:

A benchmark dataset for one task they care about (50-200 examples)

A baseline and a reusable evaluation harness

A short report explaining what "good" means for that task

(Optional but ideal) a first Nemotron 3 Nano comparison run

This becomes your testbed for the rest of the course: when you learn SFT, LoRA, RAG, RLHF/DPO, and LLM-as-a-judge, you'll have a real way to tell whether you improved anything -- or just changed the vibes.
