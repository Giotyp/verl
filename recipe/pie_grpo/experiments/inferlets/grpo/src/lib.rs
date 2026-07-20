//! GRPO rollout inferlet.
//!
//! Generates G completions for a prompt in parallel and returns tokens + text.
//! Log-probabilities are computed separately by the token-logprob inferlet.

use futures::future;
use inferlet::{Context, Result, model::Model, runtime, sample::Sampler};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Input {
    prompt: String,
    #[serde(default = "default_system")]
    system: String,
    #[serde(default = "default_n_samples")]
    n_samples: usize,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default = "default_temperature")]
    temperature: f32,
    #[serde(default = "default_top_p")]
    top_p: f32,
}

#[derive(Serialize)]
struct Sample {
    text: String,
    tokens: Vec<u32>,
}

#[derive(Serialize)]
struct GrpoOutput {
    samples: Vec<Sample>,
}

fn default_system() -> String {
    "You are a helpful assistant.".into()
}
fn default_n_samples() -> usize { 4 }
fn default_max_tokens() -> usize { 256 }
fn default_temperature() -> f32 { 0.8 }
fn default_top_p() -> f32 { 0.95 }

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let model_name = runtime::models().into_iter().next().ok_or("no models available")?;
    let model = Model::load(&model_name)?;

    // Build the shared prefix (system ONLY) and commit it to KV once. All
    // forks inherit these committed, page-aligned pages via O(1) GPU D2D copy.
    // The user prompt is intentionally NOT added here — see the note below.
    let mut base = Context::new(&model)?;
    base.system(&input.system);
    base.flush().await?;

    let forks = (0..input.n_samples)
        .map(|_| base.fork())
        .collect::<Result<Vec<_>>>()?;

    let prompt = input.prompt;
    let temperature = input.temperature;
    let top_p = input.top_p;
    let max_tokens = input.max_tokens;

    // Chat-template stop tokens (e.g. <|im_end|>). Without these the sampler
    // runs to max_tokens and emits hallucinated extra turns past the real
    // answer — junk that would otherwise get rewards/log-probs in GRPO.
    let stop = inferlet::chat::stop_tokens(&model);

    // Add the user turn INSIDE each fork (matches the proven parallel-generation
    // idiom): the prompt rides each fork's own buffer and is flushed by that
    // fork's generate(), instead of the fragile sub-page working-KV fork path
    // that silently drops a short prefix. The system prefix above is still
    // shared across forks, so the O(1) prefix-fork benefit is preserved.
    // Log-probs are NOT collected here — use token-logprob for exact scoring.
    let raw: Vec<Result<Vec<u32>>> = future::join_all(
        forks.into_iter().map(|mut ctx| {
            let prompt = prompt.clone();
            let stop = stop.clone();
            async move {
                ctx.user(&prompt);
                ctx.cue();
                ctx.generate(Sampler::TopP { temperature, p: top_p })
                    .max_tokens(max_tokens)
                    .stop(&stop)
                    .collect_tokens()
                    .await
            }
        }),
    )
    .await;

    let raw: Vec<Vec<u32>> = raw.into_iter().collect::<Result<_>>()?;

    let tok = model.tokenizer();
    let samples: Vec<Sample> = raw
        .into_iter()
        .map(|tokens| Sample {
            text: tok.decode(&tokens).unwrap_or_default(),
            tokens,
        })
        .collect();

    serde_json::to_string(&GrpoOutput { samples }).map_err(|e| e.to_string())
}
