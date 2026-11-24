# Troubleshooting

??? question "\"Cannot connect to GPU backend\" / no T4 available"
    Free GPUs are shared and can run out at peak times. Options: retry in 30-60 minutes, run on CPU
    (the notebook switches to tiny models automatically; generation is slower), or use Colab Pro.

??? question "CUDA out of memory"
    *Runtime -> Restart session*, then re-run from the top. Lower `batch_size` in the generation cell.
    Make sure you didn't load two models at once (`del gen; import gc, torch; gc.collect(); torch.cuda.empty_cache()`).

??? question "`git clone` fails in the setup cell"
    Check that the `REPO_URL` in the first cell points at the course repository and that it is public.

??? question "`OSError: [E050] Can't find model 'en_core_web_sm'`"
    The spaCy model didn't finish downloading. Run `!python -m spacy download en_core_web_sm`, then
    *Runtime -> Restart session* and re-run the setup cells.

??? question "My check says FAIL but I think my code is right"
    Read the `hint:` line and print the intermediate DataFrame. Checks compare against an answer key
    built into the dataset (`data/raw/answer_key.json`). If you're sure it's a bug, set
    `USE_SOLUTIONS = True` to continue and open an issue on the course repo.

??? question "I lost my outputs when Colab disconnected"
    Set `USE_DRIVE = True` in the setup cell so everything is saved to Google Drive.

??? question "Hugging Face 401 / 403 when loading a model"
    The labs' Qwen models don't need a token. If you swapped in a gated model (e.g. Llama), accept its
    licence on its model page, add `HF_TOKEN` in Colab **Secrets**, and enable
    *Notebook access* for it. Approval can take a few minutes.

??? question "Out of memory during training (Modules 3-5)"
    The labs are sized for a 16 GB T4. If you hit an out-of-memory error: *Runtime -> Restart session*,
    make sure no other notebook is running in the same session, and re-run from the top. You can also
    halve `per_device_train_batch_size` and double `gradient_accumulation_steps`: the effective batch
    stays the same and memory drops.

??? question "Assessment pairs or model disappeared"
    Set `USE_DRIVE = True` in the Module 5 setup cell. Pairs are saved to
    `MyDrive/llm-knowledge-workshop/m05/preference_pairs.jsonl` and reloaded automatically. Delete
    the file to rebuild.
