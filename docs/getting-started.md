# Getting started

## What you need

- [x] A personal Google account
- [x] A modern browser (Chrome, Edge, Firefox or Safari)
- [ ] *Optional:* a free [Hugging Face](https://huggingface.co/join) account (faster model downloads; needed only if you swap in a gated model such as Llama)
- [ ] *Optional:* [Colab Pro](https://colab.research.google.com/signup) for the bonus track

Nothing needs to be installed on your laptop.

## Step 1: Open a lab and pick a GPU

1. Click **Open in Colab** on any module page.
2. In Colab choose **Runtime -> Change runtime type -> T4 GPU -> Save**.
3. Run the first cell (**Setup**). It downloads the course code and prints which track you're on:

```text
Runtime : Colab Free track
GPU     : Tesla T4  15.8 GB
bf16/fp8: False / False
```

!!! info "Free vs Pro: detected automatically"
    The notebooks read the GPU and pick model sizes for you.

    | GPU | Track | Typical availability |
    |---|---|---|
    | T4 (16 GB) | **Free**: core exercises | Colab Free; may be unavailable at peak times |
    | L4 (24 GB) | **Pro**: bonus, FP8 supported | Colab Pro |
    | A100 (40 GB) | **Pro**: bonus, fastest training | Colab Pro (compute units) |
    | No GPU | CPU fallback: tiny models, slow | Always |

## Step 2: Keep your work between sessions (recommended)

Colab deletes everything when the session ends. Outputs from Module 1 feed Modules 2 and 3, so set
`USE_DRIVE = True` in the setup cell to save to `MyDrive/llm-knowledge-workshop/`.
Colab will ask permission to access your Drive.

## Step 3 (optional): Hugging Face token

Every lab uses **ungated** Qwen models, so no token is required. A token is still useful:

- Downloads get faster rate limits (you may see "unauthenticated requests" warnings without one).
- If you want to try a **gated** model such as `meta-llama/Llama-3.1-8B-Instruct` in the Pro track,
  you must accept its licence on Hugging Face first.

To add one: create a **read** token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens),
then in Colab open the **key icon (Secrets)** in the left sidebar, add `HF_TOKEN`, and enable notebook
access. Never paste tokens directly into a notebook cell.

## Colab limits to know

| Limit | What to do |
|---|---|
| Idle disconnect after about 90 min | Labs are built to finish in about 45 min; keep the tab active |
| Session cap (about 12 h on Free, often less) | Save to Drive; each lab can resume from saved outputs |
| Free GPU not always available | Retry later, or run on CPU (slower, smaller models) |
| Packages reinstall every session | The setup cell handles it (about 2-3 min) |

## Cost guide for Pro

Colab Pro bills in *compute units*. Rough usage for the full bonus track is **50-120 units**
(about one month of Pro). An A100 uses roughly 5x more units per hour than an L4, so prefer
**L4** unless a lab says otherwise. Disconnect the runtime when you finish
(*Runtime -> Disconnect and delete runtime*).
