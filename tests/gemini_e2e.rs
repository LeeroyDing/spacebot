//! End-to-end tests against the REAL Google Gemini API.
//!
//! Requires `GEMINI_API_KEY` environment variable. Skips otherwise.

use futures::StreamExt;
use rig::completion::{CompletionModel, CompletionRequest, ToolDefinition};
use rig::message::{Message, AssistantContent};
use rig::one_or_many::OneOrMany;
use serde_json::json;
use spacebot::config::{ApiType, LlmConfig, ProviderConfig};
use spacebot::llm::manager::LlmManager;
use spacebot::llm::model::SpacebotModel;
use std::collections::HashMap;
use std::sync::Arc;

/// Get the Gemini API key from environment.
fn get_api_key() -> Option<String> {
    std::env::var("GEMINI_API_KEY").ok()
}

/// Initialize LlmManager with Gemini configured.
async fn setup_gemini() -> Option<(Arc<LlmManager>, String)> {
    let api_key = get_api_key()?;
    let mut providers = HashMap::new();

    providers.insert(
        "gemini".to_string(),
        ProviderConfig {
            api_type: ApiType::Gemini,
            base_url: "https://generativelanguage.googleapis.com".to_string(),
            api_key: api_key.clone(),
            name: Some("Google Gemini".to_string()),
            use_bearer_auth: false,
            extra_headers: vec![],
        },
    );

    let config = LlmConfig {
        anthropic_key: None,
        openai_key: None,
        openrouter_key: None,
        kilo_key: None,
        zhipu_key: None,
        groq_key: None,
        together_key: None,
        fireworks_key: None,
        deepseek_key: None,
        xai_key: None,
        mistral_key: None,
        gemini_key: Some(api_key),
        ollama_key: None,
        ollama_base_url: None,
        opencode_zen_key: None,
        opencode_go_key: None,
        nvidia_key: None,
        minimax_key: None,
        minimax_cn_key: None,
        moonshot_key: None,
        zai_coding_plan_key: None,
        github_copilot_key: None,
        providers,
    };

    let manager = LlmManager::new(config).await.ok()?;
    Some((Arc::new(manager), "gemini/gemini-pro-latest".to_string()))
}

#[tokio::test]
async fn test_gemini_simple_completion() {
    let Some((manager, model_name)) = setup_gemini().await else {
        eprintln!("Skipping test_gemini_simple_completion: GEMINI_API_KEY not set");
        return;
    };

    let model = SpacebotModel::make(&manager, model_name.clone());

    let request = CompletionRequest {
        model: Some(model_name),
        chat_history: OneOrMany::many(vec![Message::user("What is 2+2? Please provide a brief explanation.")]).expect("OneOrMany failed"),
        preamble: None,
        tools: vec![],
        documents: vec![],
        temperature: None,
        max_tokens: Some(500),
        tool_choice: None,
        additional_params: None,
        output_schema: None,
    };

    let response = model
        .completion(request)
        .await
        .map_err(|e| {
            if let rig::completion::CompletionError::ResponseError(msg) = &e {
                 eprintln!("Response error: {}", msg);
            }
            e
        })
        .expect("Gemini completion failed");

    let mut text = String::new();
    for content in response.choice {
        match content {
            AssistantContent::Text(t) => text.push_str(&t.text),
            AssistantContent::Reasoning(r) => {
                for part in r.content {
                    if let rig::message::ReasoningContent::Text { text: r_text, .. } = part {
                        text.push_str(&r_text);
                    }
                }
            }
            _ => {}
        }
    }

    if !text.contains('4') {
        eprintln!("Combined text/reasoning: {}", text);
        eprintln!("Raw response: {}", serde_json::to_string_pretty(&response.raw_response.body).unwrap());
    }
    assert!(text.contains('4'), "Expected '4' in response, got: {}", text);
}

#[tokio::test]
async fn test_gemini_streaming() {
    let Some((manager, model_name)) = setup_gemini().await else {
        eprintln!("Skipping test_gemini_streaming: GEMINI_API_KEY not set");
        return;
    };

    let model = SpacebotModel::make(&manager, model_name.clone());

    let request = CompletionRequest {
        model: Some(model_name),
        chat_history: OneOrMany::many(vec![Message::user("Count from 1 to 5. One number per line.")]).expect("OneOrMany failed"),
        preamble: None,
        tools: vec![],
        documents: vec![],
        temperature: None,
        max_tokens: Some(50),
        tool_choice: None,
        additional_params: None,
        output_schema: None,
    };

    let mut stream = model
        .stream(request)
        .await
        .expect("Gemini stream failed");

    let mut chunk_count = 0;
    while let Some(chunk) = stream.next().await {
        let _ = chunk.expect("Stream chunk error");
        chunk_count += 1;
    }
    
    assert!(chunk_count > 0, "Stream yielded no chunks");
}

#[tokio::test]
async fn test_gemini_tool_use() {
    let Some((manager, model_name)) = setup_gemini().await else {
        eprintln!("Skipping test_gemini_tool_use: GEMINI_API_KEY not set");
        return;
    };

    let model = SpacebotModel::make(&manager, model_name.clone());

    let tool = ToolDefinition {
        name: "get_weather".to_string(),
        description: "Get the current weather in a given location".to_string(),
        parameters: json!({
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"]
                }
            },
            "required": ["location"]
        }),
    };

    let request = CompletionRequest {
        model: Some(model_name),
        chat_history: OneOrMany::many(vec![Message::user("What's the weather like in Tokyo?")]).expect("OneOrMany failed"),
        preamble: None,
        tools: vec![tool],
        documents: vec![],
        temperature: None,
        max_tokens: Some(500),
        tool_choice: None,
        additional_params: None,
        output_schema: None,
    };

    let response = model
        .completion(request)
        .await
        .map_err(|e| {
            if let rig::completion::CompletionError::ResponseError(msg) = &e {
                 eprintln!("Response error: {}", msg);
            }
            e
        })
        .expect("Gemini tool-use completion failed");

    let mut found_tool_call = false;
    for content in response.choice {
        if let AssistantContent::ToolCall(call) = content {
            assert_eq!(call.function.name, "get_weather");
            assert!(call.function.arguments.get("location").is_some());
            found_tool_call = true;
        }
    }

    if !found_tool_call {
        eprintln!("Raw response: {}", serde_json::to_string_pretty(&response.raw_response.body).unwrap());
    }
    assert!(found_tool_call, "Gemini failed to emit a tool call");
}

#[tokio::test]
async fn test_gemini_system_instruction() {
    let Some((manager, model_name)) = setup_gemini().await else {
        eprintln!("Skipping test_gemini_system_instruction: GEMINI_API_KEY not set");
        return;
    };

    let model = SpacebotModel::make(&manager, model_name.clone());

    let request = CompletionRequest {
        model: Some(model_name),
        chat_history: OneOrMany::many(vec![Message::user("Who are you?")]).expect("OneOrMany failed"),
        preamble: Some("You are a helpful assistant that MUST answer everything like a 1920s noir detective.".to_string()),
        tools: vec![],
        documents: vec![],
        temperature: None,
        max_tokens: Some(200),
        tool_choice: None,
        additional_params: None,
        output_schema: None,
    };

    let response = model
        .completion(request)
        .await
        .expect("Gemini system instruction completion failed");

    let mut text = String::new();
    for content in response.choice {
        if let AssistantContent::Text(t) = content {
            text.push_str(&t.text);
        }
    }

    eprintln!("Noir response: {}", text);
    // At least check it's not a generic "I am a helpful assistant".
    assert!(!text.contains("I am a large language model trained by Google"), "Response seems generic and ignored system instruction");
}

#[tokio::test]
async fn test_gemini_reasoning() {
    let Some((manager, model_name)) = setup_gemini().await else {
        eprintln!("Skipping test_gemini_reasoning: GEMINI_API_KEY not set");
        return;
    };

    let model = SpacebotModel::make(&manager, model_name.clone());

    let request = CompletionRequest {
        model: Some(model_name),
        chat_history: OneOrMany::many(vec![Message::user("Please reason about why 2+2=4. Output your reasoning steps.")]).expect("OneOrMany failed"),
        preamble: None,
        tools: vec![],
        documents: vec![],
        temperature: None,
        max_tokens: Some(1000),
        tool_choice: None,
        additional_params: None,
        output_schema: None,
    };

    let response = model
        .completion(request)
        .await
        .expect("Gemini reasoning completion failed");

    let mut found_reasoning = false;
    for content in response.choice {
        if let AssistantContent::Reasoning(_) = content {
            found_reasoning = true;
            break;
        }
    }

    if !found_reasoning {
        eprintln!("Raw response: {}", serde_json::to_string_pretty(&response.raw_response.body).unwrap());
    }
    assert!(found_reasoning, "Gemini failed to emit reasoning content");
}
