# LLM Summarse

Description: A quick Rust script that will download a (quantized gguf) LLM model and


### Notes

 - For future work with other LLMs, be sure to import the appropriate `ModelWeights` from `candle_transformers::models` (specify the associated `quantized_{model}::ModelWeights`).
     - So for instance, to use a Qwen2.5 model, the import would be `use candle_transformers::models::quantized_qwen2::ModelWeights;` and for Llama3.2, it would be `use candle_transformers::models::quantized_llama::ModelWeights;`.
     - In future work, it would be a good idea to have some kind of parameterized imports based on the input model name (if possible).
     - For a list of what quantized models are available, see the crate [documentation](https://docs.rs/candle-transformers/latest/candle_transformers/all.html) for `candle-transformers`.
 - In the program, we sampled from one repo to get the gguf file we wanted (thankfully, Qwen has put all their gguf files for the same model in the same repo) and then we had to go to a different repo (taht was the same model) to get the tokenizer. Since the tokenizers are typically shared for the same model regardless of the model size (i.e. Llama 3.2 1B uses the same tokenizer as Llama 3.2 3B), this provided a way for us to get everything we needed.
     - We still need to work on how/where the tokenizer is cached to local storage (over the usual `~/.cache/huggingface/hub/` path).
