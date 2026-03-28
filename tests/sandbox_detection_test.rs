use spacebot::sandbox::{SandboxBackend, detect_backend};

#[tokio::test]
async fn test_backend_detection() {
    // This test just verifies the function exists and returns a value
    // Actual detection depends on the host environment
    let _backend = detect_backend().await;
    // Should not panic
}

#[test]
fn test_sandbox_backend_variants() {
    let bubblewrap = SandboxBackend::Bubblewrap;
    let sandbox_exec = SandboxBackend::SandboxExec;
    let none = SandboxBackend::None;

    // Verify all variants exist
    assert!(matches!(bubblewrap, SandboxBackend::Bubblewrap));
    assert!(matches!(sandbox_exec, SandboxBackend::SandboxExec));
    assert!(matches!(none, SandboxBackend::None));
}
