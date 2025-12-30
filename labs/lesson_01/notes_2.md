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
