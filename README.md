# Causal Trainer

Causal Trainer is a high-performance JAX platform for CausalLM model training on Cloud TPU infrastructure. It unifies pretraining, supervised fine-tuning, and parameter-efficient adaptation in a native multi-host execution stack, with distributed data preparation, TPU-optimized kernels, and scalable checkpointing.

## Capabilities

- Segment-isolated packing for attention, shifted loss, assistant-only supervision, and EndPrompt.
- Bounded-memory Hugging Face dataset streaming with online packing.
- A custom Efficient attention kernel that combines tiled execution, online softmax, and a custom backward path to reduce HBM traffic.
- Projection-fused Cut cross-entropy and chunked Pallas cross-entropy for reduced activation memory.
- Streaming multi-host checkpoint export with bounded host memory.

## Requirements

- Python 3.11 or later.
- JAX and JAXLIB 0.10.x with a compatible `libtpu` runtime.
- A supported checkpoint in Hugging Face tensor layout.
- A serialized fast tokenizer containing `tokenizer.json`.

The trainer reads Hugging Face configuration, tokenizer, and Safetensors artifacts directly. Transformers is not a runtime dependency.

## Installation

Create an isolated environment and install the TPU dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[tpu]'
```

The package installs the `causal-train` command. The equivalent module entry point is `python -m causal_trainer.train`.

## Quick start

### Raw-text pretraining

```bash
causal-train \
  --repo_id /path/to/model-or-hf-repo \
  --dataset_name /path/to/train.jsonl \
  --dataset_text_field text \
  --max_sequence_length 4096 \
  --packing True \
  --total_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 1 \
  --output_dir ./output-pt
```

Raw text is tokenized with `add_special_tokens=False`. EOS placement is preserved from the source records.

### Supervised fine-tuning

```bash
causal-train \
  --repo_id /path/to/model-or-hf-repo \
  --dataset_name /path/to/messages.jsonl \
  --dataset_text_field messages \
  --max_sequence_length 4096 \
  --packing True \
  --assistant_only_loss True \
  --total_batch_size 32 \
  --learning_rate 1e-5 \
  --num_train_epochs 1 \
  --output_dir ./output-sft
```

Assistant-only supervision requires a chat template that marks assistant generation spans with `{% generation %}` and `{% endgeneration %}`. `--assistant_only_loss True` supervises every marked assistant span. Use `--last_assistant_only_loss True` instead to supervise only the final contiguous marked assistant span in each source conversation. The two options are mutually exclusive and require `--dataset_text_field messages`. Records without a valid supervised target after truncation are filtered.

## Model and tokenizer sources

`--repo_id` accepts either a local checkpoint directory or a Hugging Face repository ID. Source checkpoints may contain sharded Safetensors weights. The configuration is validated against the architecture supported by this package before parameters are loaded.

By default, tokenizer assets are read from the model source. Use `--processor_repo_id` to select a separate local directory or repository. The source must contain `tokenizer.json`; standard companion assets are preserved when checkpoints are exported, including:

- `tokenizer_config.json`
- `special_tokens_map.json`
- `chat_template.jinja`
- `chat_templates/*.jinja`

Use the `HF_TOKEN` environment variable for authenticated repositories. The `--revision` option selects a specific model and tokenizer revision.

## Dataset inputs

`--dataset_name` accepts a Hugging Face dataset ID or a local JSON, JSONL, Parquet, CSV, or TSV source. `--dataset_split` defaults to `train`.

Raw-text records contain a string field, which defaults to `text`:

```json
{"text": "A training document."}
```

Conversational records contain a `messages` field:

```json
{
  "messages": [
    {"role": "user", "content": "Explain sequence packing."},
    {
      "role": "assistant",
      "content": "Sequence packing combines multiple samples into one row."
    }
  ]
}
```

Message records may also provide `tools` and `chat_template_kwargs` when the selected chat template requires them.

### Streaming datasets

Enable `--dataset_streaming True` to train from a Hugging Face dataset that is larger than local memory or disk. Streaming keeps only bounded working state in memory: the shuffle buffer, the active packing window, and the configured prefetched batches. It does not materialize or download the complete split before training.

```text
--dataset_streaming True \
--dataset_config_name optional-subset \
--dataset_revision dataset-commit \
--streaming_shuffle_buffer 10000
```

`--dataset_config_name` selects an optional Hugging Face dataset configuration. `--dataset_revision` pins the dataset independently of `--revision`, which continues to select the model and tokenizer revision.

One complete traversal of the source split is exactly one streaming epoch. Without an explicit positive `--max_steps`, `--num_train_epochs` must therefore be a positive integer. `--max_steps` may intentionally stop in the middle of an epoch.

When the Hugging Face split metadata provides its number of examples, epoch-driven training uses it only as an initial estimate for progress, ETA, and schedule horizon. The metadata count may be inaccurate: it never caps iteration, triggers retry, or decides whether training is complete. One epoch always ends at the stream's real EOF, and actual committed source examples are allowed to differ from the estimate. After each completed real epoch, the remaining progress total and ETA are re-estimated from observed consumption. Packing and filtering can make source examples differ from packed rows or optimizer steps. A source example is committed only after its corresponding device update completes successfully. If the metadata estimate is unavailable, progress begins without a percentage and obtains an observed estimate after the first real epoch. With an explicit `--max_steps`, the progress bar uses that exact step limit and source examples remain an additional metric.

For epoch-driven streaming with a metadata estimate, cosine and linear learning-rate decay use a scheduler horizon fixed at 105% of the metadata-derived source-example total; `warmup_steps` is converted to source-example units using the configured effective global batch. Within each real epoch, scheduler progress is capped at the original, unbuffered metadata row count. If the real stream contains more rows, every batch beyond that position keeps the same learning rate while the data continues to real EOF. The reserved 5% tail is not consumed, so epoch-driven streaming is not forced to reach `learning_rate_end`. Real EOF advances training to the next epoch but does not correct or extend the scheduler. Without a row-count estimate, epoch-driven training remains enabled and uses optimizer-step warmup followed by a constant learning rate because no defensible decay horizon exists.

Streaming shuffle uses `--streaming_shuffle_buffer` records and remains deterministic for the configured seed and epoch.

Transient Hugging Face network, transport, and server read failures are retried without an attempt limit. Retry delays begin at `--hf_retry_initial_delay` seconds and back off to at most `--hf_retry_max_delay` seconds. A disconnected stream is reopened at the same epoch, its delivered prefix is replayed, and a rolling digest rejects a changed prefix instead of mixing dataset revisions. A natural EOF completes the epoch regardless of the metadata estimate. Authentication or authorization failures, a missing dataset, malformed data, and preprocessing errors are not retried.

### Packing

Enable fixed-length packing with `--packing True`. Each source record receives an independent segment ID, and position IDs restart at the beginning of each segment. Attention and shifted language-model loss both enforce segment boundaries.

With `--last_assistant_only_loss True`, the final assistant span is selected independently in every source conversation before packing. A packed row therefore retains one final supervised span per trainable source record, while cross-record prediction edges remain masked by segment IDs.

`--packing_batch_size` controls the per-process packing window. The final short window is retained.

For streaming input, each completed packing window is emitted immediately. The final short window is flushed when the source epoch is exhausted, and packing state never crosses an epoch boundary.

### Distributed preprocessing

The default preprocessing mode is `shard_then_merge`:

```text
--preprocessing_mode shard_then_merge
```

For materialized datasets, source rows are divided among JAX processes, preprocessed locally, and merged over the initialized JAX network before training. Use `--preprocessing_mode replicated` to preprocess the complete dataset on every process. `--preprocessing_num_workers` controls tokenizer worker processes within each JAX process.

Streaming uses a deterministic replicated-read design: every JAX process opens the same deterministic stream and constructs the same global batches before JAX retains its addressable device slices. Pinning `--dataset_revision` to an immutable commit is strongly recommended. Before a batch is released to training, processes collectively compare `BATCH`/`EOF`/`ERROR` state and a digest of batch contents, shapes, dtypes, epoch, and source cursor; successful local device placement is part of this coordinated boundary. This avoids mixed-host batches and mismatched collective order. The design avoids materializing and merging the complete prepared dataset, while intentionally duplicating source reads across hosts. `--preprocessing_mode` is ignored for streaming input, and `--preprocessing_num_workers` is not supported. Source progress is counted once for the logical global stream, not once per process.

Streaming checkpoints store both committed source progress and a rolling digest of every trained global batch. Resume replays the bounded stream from the beginning, verifies the exact checkpoint boundary even when it coincides with EOF, and rejects changed historical data.

## Training modes

### Full-parameter training

Full-parameter training is the default. The base parameters and AdamW state use the configured parameter dtype, while numerically sensitive normalization, softmax, loss-reduction, and gradient-scratch operations use FP32 where required.

To keep the token embedding, language-model head, and every normalization scale frozen while training the remaining full-rank parameters:

```text
--frozen_parameters "lm_head|embed|norm"
```

Supported components are `lm_head`, `embed`, and `norm`; separate multiple components with `|` or `,`. The `norm` component includes the final normalization and both normalization scales in every decoder layer. When omitted, no full-rank parameters are frozen.

The default precision options are:

```text
--param_dtype bfloat16
--dtype bfloat16
--gradient_checkpointing nothing_saveable
```

### LoRA

Enable LoRA with:

```text
--lora True \
--lora_rank 256
```

Adapters are applied to the attention and gated-MLP projections. Embeddings and the language model head remain frozen by default.

To train the embedding and language model head alongside the adapters:

```text
--lora True \
--lora_train_embed_and_lm_head True
```

LoRA exports merged model weights by default. To write a PEFT-compatible adapter instead:

```text
--lora True \
--lora_save_adapter_only True
```

Adapter-only output contains `adapter_model.safetensors` and `adapter_config.json`. When embedding and head training is enabled, those tensors are included as PEFT modules to save.

### EndPrompt

EndPrompt appends a terminal prompt at an anchored logical position while preserving the source token stream. It supports raw text, messages, packing, assistant-only supervision, full-parameter training, and LoRA.

```text
--endprompt_enable True \
--endprompt_logical_length 2097152 \
--endprompt_logical_length_min 4096 \
--endprompt_prompts "This is the end of text, please pay attention here" \
--endprompt_context_loss_weight 1.0 \
--endprompt_prompt_loss_weight 0.1
```

Separate multiple terminal prompts with `||`. Selection and logical-position sampling are deterministic for each source record.

## Attention and loss backends

### Attention

The default attention backend is:

```text
--attn_mechanism efficient
```

The default `efficient` backend uses a tiled custom kernel with online softmax and a custom VJP, avoiding materialization of the full attention matrix. It supports packed segments and sequence parallelism.

Official JAX Splash is available as an independent backend selected with `--attn_mechanism splash`. The `--block_size_q` and `--block_size_k` options apply to Splash only. The `vanilla` backend provides a reference implementation for numerical checks.

### Cross-entropy

The default loss selection is:

```text
--loss_implementation auto
```

Available values are:

| Value | Description |
|---|---|
| `auto` | Selects Cut or Pallas according to mesh compatibility. |
| `cut` | Uses projection-fused cross-entropy on supported meshes. |
| `pallas` | Uses chunked projected logits with a TPU Pallas kernel. |
| `xla` | Uses the reference XLA implementation. |

`--loss_token_budget` controls the local projected-logits chunk used by the Pallas and XLA paths. Cut uses its own internal tiling.

### Execution controls

- `--mlp_chunk_size` controls static sequence tiling for the gated MLP. Use `0` to disable tiling. When omitted, the trainer selects a value from the training mode, context length, and sequence-parallel configuration.
- `--scan_layers` represents the decoder layer loop with `lax.scan`. `--no-scan-layers` selects the unrolled form. When neither is provided, the trainer applies its context-aware default.
- `--async_dispatch_steps` controls deferred host synchronization for completed step metrics.
- `--prefetch_batches` controls prepared global batches retained ahead of the active training step.

## Distributed training

The mesh axes are ordered as:

```text
(dp, fsdp, ep, tp, sp)
```

The default mesh is:

```text
--sharding_axis=-1,1,1,4,1
```

`-1` infers the data-parallel dimension from the available device count. The trainer requires `ep=1`. Set `sp>1` explicitly when sequence parallelism is needed. `--sharding_dcn_axis` may be used to provide the corresponding inter-host mesh shape.

Launch the same command on every host. For example:

```bash
eopod run --retry 1 --worker all \
  "cd /root/causal-trainer && \
   PYTHONPATH=src .venv/bin/causal-train <arguments>"
```

Standard Cloud TPU environments are detected automatically. Explicit cluster configuration is also supported:

```text
--coordinator_address host:port \
--num_processes N \
--process_id R \
--local_device_ids 0,1,...
```

`--total_batch_size` is the global micro-batch size. The effective batch per optimizer update is:

```text
total_batch_size * gradient_accumulation_steps
```

## Checkpoints and resume

Every process participates in checkpoint collectives; process zero writes the artifacts. All processes must observe the same completed checkpoint contents when resuming a multi-host run.

The trainer writes:

| Training mode | Default output |
|---|---|
| Full-parameter | `model.safetensors` with config and tokenizer assets. |
| LoRA | A merged `model.safetensors` with config and tokenizer assets. |
| Adapter-only LoRA | `adapter_model.safetensors` and `adapter_config.json`. |

Periodic checkpoints are configured with:

```text
--save_steps 100 \
--save_total_limit 1
```

Enable `--save_optimizer_state True` when exact optimizer-state resume is required. Adapter-only LoRA supports optimizer-state checkpoints; merged LoRA exports are final model artifacts rather than resumable adapter checkpoints.

At startup, the trainer discovers the latest compatible completed checkpoint under `output_dir` and resumes automatically.

## Logging

Enable Weights & Biases on process zero with:

```text
--use_wandb True \
--wandb_project causal-trainer \
--wandb_run_name my-run
```

Additional controls include `--logging_steps`, `--track_memory`, and `--weight_distribution_log_steps`.

## License

Causal Trainer is licensed under the [Apache License 2.0](LICENSE).
