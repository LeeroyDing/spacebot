//! Build full Google Gemini API requests with system prompt and tools.

use reqwest::RequestBuilder;
use rig::completion::CompletionRequest;

/// Build a fully configured Gemini API request from a CompletionRequest.
///
/// `base_url` is the provider's configured base URL (e.g. `https://generativelanguage.googleapis.com`).
/// The `/v1beta/models/{model}:generateContent` path is constructed automatically.
///
/// Ref: https://ai.google.dev/gemini-api/docs/get-started/tutorial?lang=rest
pub fn build_gemini_request(
    http_client: &reqwest::Client,
    api_key: &str,
    base_url: &str,
    model_name: &str,
    request: &CompletionRequest,
    is_stream: bool,
) -> RequestBuilder {
    // Documentation for method suffixes:
    // Regular: https://ai.google.dev/gemini-api/docs/get-started/tutorial?lang=rest#generate-text-from-text-input
    // Streaming: https://ai.google.dev/gemini-api/docs/get-started/tutorial?lang=rest#generate-a-text-stream
    let method_suffix = if is_stream {
        "streamGenerateContent?alt=sse"
    } else {
        "generateContent"
    };

    let model_id = model_name.strip_prefix("gemini/").unwrap_or(model_name);

    let url = format!(
        "{}/v1beta/models/{}:{}",
        base_url.trim_end_matches('/'),
        model_id,
        method_suffix
    );

    let mut body = serde_json::json!({
        "contents": crate::llm::model::convert_messages_to_gemini(&request.chat_history),
    });

    if let Some(preamble) = &request.preamble {
        // Ref: https://ai.google.dev/gemini-api/docs/system-instructions?lang=rest
        body["systemInstruction"] = serde_json::json!({
            "parts": [
                { "text": preamble }
            ]
        });
    }

    if !request.tools.is_empty() {
        // Ref: https://ai.google.dev/gemini-api/docs/function-calling?lang=rest
        let function_declarations: Vec<serde_json::Value> = request
            .tools
            .iter()
            .map(|t| {
                serde_json::json!({
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                })
            })
            .collect();

        body["tools"] = serde_json::json!([
            {
                "functionDeclarations": function_declarations,
            }
        ]);
    }

    let mut generation_config = serde_json::json!({});

    if let Some(temp) = request.temperature {
        generation_config["temperature"] = serde_json::json!(temp);
    }

    if let Some(max_tokens) = request.max_tokens {
        generation_config["maxOutputTokens"] = serde_json::json!(max_tokens);
    }

    if !generation_config.as_object().unwrap().is_empty() {
        body["generationConfig"] = generation_config;
    }

    http_client
        .post(&url)
        .header("x-goog-api-key", api_key)
        .header("content-type", "application/json")
        .json(&body)
}
