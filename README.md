# LLM Summarse

Description: A quick Rust script that will download a (quantized gguf) LLM model and


### Notes

 - For future work with other LLMs, be sure to import the appropriate `ModelWeights` from `candle_transformers::models` (specify the associated `quantized_{model}::ModelWeights`).
     - So for instance, to use a Qwen2.5 model, the import would be `use candle_transformers::models::quantized_qwen2::ModelWeights;` and for Llama3.2, it would be `use candle_transformers::models::quantized_llama::ModelWeights;`.
     - In future work, it would be a good idea to have some kind of parameterized imports based on the input model name (if possible).
     - For a list of what quantized models are available, see the crate [documentation](https://docs.rs/candle-transformers/latest/candle_transformers/all.html) for `candle-transformers`.