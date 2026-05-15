use anyhow::Result;
use candle_core::{Device, Tensor, Result as OtherResult};
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::quantized_llama::ModelWeights;
use hf_hub::{api::sync::Api, Repo, RepoType};
use std::fs;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

const MODEL_ID: &str = "Qwen/Qwen2.5-3B-Instruct-GGUF";
const MODEL_FILE: &str = "qwen2.5-3b-instruct-q4_k_m.gguf";

// Example alternatives:
//
// Gemma:
// MODEL_ID = "lmstudio-community/gemma-3-4b-it-GGUF"
//
// Phi:
// MODEL_ID = "microsoft/Phi-3-mini-4k-instruct-gguf"

fn detect_device(force_cpu: bool) -> OtherResult<Device> {
    if force_cpu {
        return Ok(Device::Cpu);
    }

    // 1. Try CUDA (NVIDIA GPUs)
    if let Ok(cuda_device) = Device::new_cuda(0) {
        println!("Using Device: CUDA");
        return Ok(cuda_device);
    }

    // 2. Try Metal (Apple Silicon / MPS equivalent)
    if let Ok(metal_device) = Device::new_metal(0) {
        println!("Using Device: Metal");
        return Ok(metal_device);
    }

    // 3. Fallback to CPU
    println!("Using Device: CPU");
    Ok(Device::Cpu)
}

fn download_if_needed(
    local_dir: &Path,
    model_id: &str,
    model_file: &str,
) -> Result<PathBuf> {
    fs::create_dir_all(local_dir)?;

    let local_model_path = local_dir.join(model_file);

    if local_model_path.exists() {
        println!(
            "Model already exists locally at: {}",
            local_model_path.display()
        );
        return Ok(local_model_path);
    }

    println!("Downloading model from HuggingFace...");

    let api = Api::new()?;
    let repo = api.repo(Repo::with_revision(
        model_id.to_string(),
        RepoType::Model,
        "main".to_string(),
    ));

    let downloaded_path = repo.get(model_file)?;

    fs::copy(&downloaded_path, &local_model_path)?;

    println!("Model downloaded to: {}", local_model_path.display());

    Ok(local_model_path)
}

fn main() -> Result<()> {
    // -------------------------------------------------------------------------
    // 1. Detect device
    // -------------------------------------------------------------------------
    let device = detect_device(false)?;

    // -------------------------------------------------------------------------
    // 2. Download model if necessary
    // -------------------------------------------------------------------------
    let local_model_dir = PathBuf::from("./models");

    let model_path = download_if_needed(
        &local_model_dir,
        MODEL_ID,
        MODEL_FILE,
    )?;

    // -------------------------------------------------------------------------
    // 3. Download tokenizer
    // -------------------------------------------------------------------------
    let api = Api::new()?;
    let repo = api.repo(Repo::new(
        MODEL_ID.to_string(),
        RepoType::Model,
    ));

    let tokenizer_path = repo.get("tokenizer.json")?;

    let tokenizer = Tokenizer::from_file(tokenizer_path)
        .map_err(anyhow::Error::msg)?;

    // -------------------------------------------------------------------------
    // 4. Load GGUF model
    // -------------------------------------------------------------------------
    println!("Loading model into memory...");

    let mut file = std::fs::File::open(&model_path)?;
    // let mut gguf_reader = candle_core::quantized::gguf_file::Reader::new(&mut file)?;
    let gguf_content = candle_core::quantized::gguf_file::Content::read(&mut file)?;
    // let model = ModelWeights::from_gguf(&mut file, &device)?;
    let mut model = ModelWeights::from_gguf(gguf_content, &mut file, &device)?;

    println!("Model loaded successfully.");

    // -------------------------------------------------------------------------
    // 5. Example long-ish text
    // -------------------------------------------------------------------------
    let text = r#"
Rust is a systems programming language focused on safety, speed, and concurrency.
It accomplishes these goals by being memory safe without using garbage collection.
Rust is commonly used for high-performance applications, backend systems, game engines,
embedded software, and increasingly for AI tooling and infrastructure.

One of the key features of Rust is ownership and borrowing, which ensures memory safety
at compile time. Developers can write highly efficient code while avoiding many classes
of bugs such as null pointer dereferencing, buffer overflows, and data races.

Recently, the Rust ecosystem has expanded rapidly in machine learning and AI. Projects
such as Candle provide lightweight tensor operations and model inference entirely in Rust,
allowing developers to build local-first AI applications that run efficiently across CPUs,
NVIDIA GPUs using CUDA, and Apple Silicon devices using Metal.
"#;

    let prompt = format!(
        "<|im_start|>system\nYou are a helpful summarization assistant.<|im_end|>\n\
         <|im_start|>user\nSummarize the following text:\n\n{}\n<|im_end|>\n\
         <|im_start|>assistant\n",
        text
    );

    // -------------------------------------------------------------------------
    // 6. Tokenize prompt
    // -------------------------------------------------------------------------
    let encoding = tokenizer
        .encode(prompt, true)
        .map_err(anyhow::Error::msg)?;

    let mut tokens = encoding.get_ids().to_vec();

    // -------------------------------------------------------------------------
    // 7. Text generation loop
    // -------------------------------------------------------------------------
    let temperature = 0.7;
    let top_p = Some(0.9);
    let seed = 42;

    let mut logits_processor =
        LogitsProcessor::new(seed, Some(temperature), top_p);

    let max_new_tokens = 120;

    println!("\nGenerating summary...\n");

    for _ in 0..max_new_tokens {
        let input = Tensor::new(&tokens[..], &device)?.unsqueeze(0)?;

        let logits = model.forward(&input, tokens.len())?;

        let logits = logits.squeeze(0)?;
        let next_token = logits_processor.sample(&logits)?;

        tokens.push(next_token);

        // EOS token for many Qwen models
        if next_token == 151645 {
            break;
        }
    }

    // -------------------------------------------------------------------------
    // 8. Decode output
    // -------------------------------------------------------------------------
    let output = tokenizer
        .decode(&tokens, true)
        .map_err(anyhow::Error::msg)?;

    // Try extracting assistant response only
    let summary = output
        .split("<|im_start|>assistant")
        .last()
        .unwrap_or(&output)
        .trim();

    // -------------------------------------------------------------------------
    // 9. Print results
    // -------------------------------------------------------------------------
    println!("================ ORIGINAL TEXT ================\n");
    println!("{}", text);

    println!("\n================ SUMMARY ================\n");
    println!("{}", summary);

    Ok(())
}