//! Native Google Gemini API integration using gemini-rust.

use crate::llm::manager::LlmManager;

use gemini_rust::{GeminiBuilder, Model, GenerationResponse};
use rig::completion::{CompletionRequest, CompletionResponse, CompletionError};
use rig::message::{AssistantContent, Message, UserContent, Text};
use rig::one_or_many::OneOrMany;
use rig::streaming::{RawStreamingChoice, StreamingCompletionResponse};
use crate::llm::model::RawStreamingResponse;
use futures::StreamExt;
use std::sync::Arc;

/// Build a Gemini native request and execute it.
pub async fn call_gemini_native(
    llm_manager: &Arc<LlmManager>,
    model_name: &str,
    request: &CompletionRequest,
) -> std::result::Result<CompletionResponse<crate::llm::model::RawResponse>, CompletionError> {
    let api_key = llm_manager
        .get_api_key("gemini")
        .map_err(|e| CompletionError::ProviderError(e.to_string()))?;

    let model = resolve_gemini_model(model_name);
    
    let client = GeminiBuilder::new(api_key)
        .with_model(model)
        .build()
        .map_err(|e| CompletionError::ProviderError(e.to_string()))?;

    let mut builder = client.generate_content();
    
    if let Some(preamble) = &request.preamble {
        builder = builder.with_system_instruction(preamble);
    }

    for message in request.chat_history.iter() {
        match message {
            Message::User { content } => {
                let text = content.iter().filter_map(|c| match c {
                    UserContent::Text(t) => Some(t.text.clone()),
                    _ => None,
                }).collect::<Vec<_>>().join("\n");
                builder = builder.with_user_message(text);
            }
            Message::Assistant { content, .. } => {
                let text = content.iter().filter_map(|c| match c {
                    AssistantContent::Text(t) => Some(t.text.clone()),
                    _ => None,
                }).collect::<Vec<_>>().join("\n");
                builder = builder.with_model_message(text);
            }
            Message::System { content } => {
                // If there is already a system instruction, we might want to append to it?
                // For now, let's treat it as a user message since we can't easily append.
                builder = builder.with_user_message(content);
            }
        }
    }

    let response: GenerationResponse = builder.execute()
        .await
        .map_err(|e| CompletionError::ProviderError(e.to_string()))?;

    let text = response.text();
    
    Ok(CompletionResponse {
        choice: OneOrMany::one(AssistantContent::Text(Text { text: text.clone() })),
        usage: rig::completion::Usage {
            input_tokens: 0,
            output_tokens: 0,
            total_tokens: 0,
            cached_input_tokens: 0,
        },
        message_id: None,
        raw_response: crate::llm::model::RawResponse {
            body: serde_json::json!({ "text": text }),
        },
    })
}

/// Execute a Gemini native streaming request.
pub async fn stream_gemini_native(
    llm_manager: &Arc<LlmManager>,
    model_name: &str,
    request: &CompletionRequest,
) -> std::result::Result<StreamingCompletionResponse<RawStreamingResponse>, CompletionError> {
    let api_key = llm_manager
        .get_api_key("gemini")
        .map_err(|e| CompletionError::ProviderError(e.to_string()))?;

    let model = resolve_gemini_model(model_name);
    
    let client = GeminiBuilder::new(api_key)
        .with_model(model)
        .build()
        .map_err(|e| CompletionError::ProviderError(e.to_string()))?;

    let mut builder = client.generate_content();
    
    if let Some(preamble) = &request.preamble {
        builder = builder.with_system_instruction(preamble);
    }

    for message in request.chat_history.iter() {
        match message {
            Message::User { content } => {
                let text = content.iter().filter_map(|c| match c {
                    UserContent::Text(t) => Some(t.text.clone()),
                    _ => None,
                }).collect::<Vec<_>>().join("\n");
                builder = builder.with_user_message(text);
            }
            Message::Assistant { content, .. } => {
                let text = content.iter().filter_map(|c| match c {
                    AssistantContent::Text(t) => Some(t.text.clone()),
                    _ => None,
                }).collect::<Vec<_>>().join("\n");
                builder = builder.with_model_message(text);
            }
            Message::System { content } => {
                builder = builder.with_user_message(content);
            }
        }
    }

    let stream = builder.execute_stream()
        .await
        .map_err(|e| CompletionError::ProviderError(e.to_string()))?;

    let mapped_stream = async_stream::stream! {
        let mut full_text = String::new();
        let mut stream = stream;
        while let Some(chunk_result) = stream.next().await {
            match chunk_result {
                Ok(chunk) => {
                    let text = chunk.text();
                    full_text.push_str(&text);
                    yield Ok(RawStreamingChoice::Message(text));
                }
                Err(e) => {
                    yield Err(CompletionError::ProviderError(e.to_string()));
                }
            }
        }
        
        yield Ok(RawStreamingChoice::FinalResponse(RawStreamingResponse {
            body: serde_json::json!({ "text": full_text }),
            usage: None,
        }));
    };

    Ok(StreamingCompletionResponse::stream(Box::pin(mapped_stream)))
}

fn resolve_gemini_model(model_name: &str) -> Model {
    // Standardize: lowercase and strip provider prefixes if present
    let model_lower = model_name.to_lowercase();
    let model = model_lower
        .strip_prefix("google/")
        .or_else(|| model_lower.strip_prefix("gemini/"))
        .unwrap_or(&model_lower);

    match model {
        "gemini-2.0-flash" | "gemini-2.0-flash-exp" | "gemini-1.5-flash" | "gemini-2.5-flash" | "gemini-2.0-flash-001" => Model::Gemini25Flash,
        "gemini-2.5-flash-lite" => Model::Gemini25FlashLite,
        "gemini-2.5-flash-image" => Model::Gemini25FlashImage,
        "gemini-2.0-pro" | "gemini-2.0-pro-exp" | "gemini-1.5-pro" | "gemini-2.5-pro" => Model::Gemini25Pro,
        "gemini-3-flash" | "gemini-3-flash-preview" | "gemini-3.0-flash" => Model::Gemini3Flash,
        "gemini-3-pro" | "gemini-3.0-pro" | "gemini-3.1-pro-preview" => Model::Gemini3Pro,
        "gemini-3-pro-image" => Model::Gemini3ProImage,
        "text-embedding-004" => Model::TextEmbedding004,
        _ => Model::Custom(model.to_string()),
    }
}
