# Lesson 02 — Notes

Use this folder for Lesson 02 notebooks, scripts, and artifacts.

Lesson 2. An LLM is good at a task under a specific metric on a particular dataset with a particular distribution under constraints.

We must therefore define:

1. dataset
2. the task
3. metric of good
4. constraints (latency, cost, safety, format)

These are musts for a worthy LLM project.

We will learn to create an NLP (natural language processing) task from a specific goal.

Choose the correct task form: extraction vs classification vs QA vs summarization vs ranking vs generation.

Build a dataset schema: what fields exist, what labels mean, what a ground truth is.

Write annotation guidelines so two humans can write labels the same way.

Construct a train/val/test split on data to avoid leakage.

Pick metrics for evaluation that make sense and relate to the goal.

overall score, per-class score, confusion matrix, curated error slice for analysis

Implement a "dumb baseline" (majority class / keyword rules / regex).

Compare that baseline vs LLM and explain why each fails.

We will then create an experimental report with dataset description, metric definition, results, error analysis, and next steps.

More notes:

Leakage is a critical point.

If we train on data and then test on that same data, that is leakage. The model is not proving it can generalize a solution, as it has already trained on that data (in supervised learning, it literally has seen the outcome already).

If we cluster or develop patterns using the data we will eventually test on, then the model is not proving it can solve a solution; it is overfitting.

Accidentally encoding the answer in the input is also leakage.

Overfitting is about training a model to recognize patterns in the training set and not being able to generalize outside of that set. Leakage refers to evaluation and test results lying with good outcomes because the training set was contaminated with solutions (or preprocessing was fit on all the data), or the training/validation set was contaminated with verbatim training data.

More notes:

F1 is the harmonic mean of precision and recall. It is common in classification, especially with class imbalance (one class is more important than the other).

We can calculate the per-class F1 normalized by number of classes. This shows us how good a particular class is compared to the rest.

Macro F1 is the overall score: the unweighted average of per-class F1 (so it is also normalized by class, so each class is treated equally and not by how common it is).

The equations for F1 are as follows:
"A" is a class -- treated as positive.

TP (true positive): model says A, it is A.
FP (false positive): model said A, but it's not A.
FN (false negative): model predicted not A, but it is A.
TN (true negative): model says not A and it is not A.

Precision = TP / (TP + FP)
It answers: "when the model says A, how often is it right?"

Recall = TP / (TP + FN)
It answers: "of all the true A examples, how many did the model find?"

F1 = 2 * (precision * recall) / (precision + recall)

We compare the predicted labels to the ground truth labels.

This is a part of evaluation.

By making each class the positive class, we reveal with macro F1 how well the model actually performed.

Spam vs Ham example: in a dataset of 100 emails, 90 are Ham and 10 are Spam. Ham is the majority class and Spam is the minority class.

The model predicts all 100 emails to be Ham emails (no spam emails detected).

We start by making Spam the positive class:

TP: 0 (never predicted spam)
FP: 0 (never predicted spam)
FN: 10 (missed all 10 spams)
Precision is undefined/0, recall is 0, F1 is 0.

For Ham as positive:
TP: 90
FP: 10 (it predicted ham for everything, but these were indeed spam)
Accuracy: 90/100 (looks really good on this class)
F1 = 2 * (0.9 * 1) / (0.9 + 1) = 1.8 / 1.9

But when we look at macro F1:

(F1_ham + F1_spam) / 2 = (0.95 + 0) / 2 = 0.47

It is revealed through macro F1 that the overall scoring of the model was bad.

A baseline is a reference method to compare the results of the model against. It is a minimum bar we pass to see if the model is actually improving anything (it can be a keyword searcher, a random guess, a heuristic, a logistic regression, small n-gram, smaller LLM, previous version of the model we are building).

It is important to identify the kinds of error types that matter for our specific task. If we need to route "cancellation" to billing but it is rare and also critical, we will likely use macro and per-class F1. Misrouting causes longer resolution times, so these errors can be weighted differently.

In an extraction question: "refund me $20.50" -- does it refund that exactly formatted in JSON or did it miss? And if it misses, then by what tolerance?

ROUGE is a family of automatic metrics used to evaluate generated text, typically used in summarization. It measures the overlap with a reference, commonly by n-gram overlap (token overlap).

Usually reported using precision/recall/F1 overlap scores between generated and reference text.

One must again be careful as it could 1) hallucinate items from the text or 2) miss the main points, so we can do formatting checks and factuality checks when possible.

A human rubric would come in handy here. A human can evaluate the decision of the LLM.
