FROM nvcr.io/nvidia/vllm:25.09-py3

# Download tokenizer encodings needed by GPT-style models.
RUN apt-get update && \
    apt-get install -y curl ca-certificates && \
    mkdir -p /etc/encodings && \
    curl -L "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken" \
     -o /etc/encodings/cl100k_base.tiktoken && \
    curl -L "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken" \
     -o /etc/encodings/o200k_base.tiktoken && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Point tiktoken/vLLM to the local encodings.
ENV TIKTOKEN_ENCODINGS_BASE=/etc/encodings

WORKDIR /workspace
