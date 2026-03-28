use tokio::process::Command;
use tracing::{info, warn};

/// Available sandbox backends detected at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SandboxBackend {
    /// Linux bubblewrap (bwrap) backend.
    Bubblewrap,
    /// macOS sandbox-exec backend.
    SandboxExec,
    /// No sandbox backend available.
    None,
}

/// Detect available sandbox backend by probing system binaries.
/// Runs preflight checks to ensure the backend actually works.
pub async fn detect_backend() -> SandboxBackend {
    // Try Linux bubblewrap first
    if let Ok(true) = check_bubblewrap().await {
        info!("Sandbox backend: bubblewrap");
        return SandboxBackend::Bubblewrap;
    }

    // Try macOS sandbox-exec
    if let Ok(true) = check_sandbox_exec().await {
        info!("Sandbox backend: sandbox-exec");
        return SandboxBackend::SandboxExec;
    }

    warn!("No sandbox backend available - commands will run unsandboxed");
    SandboxBackend::None
}

async fn check_bubblewrap() -> Result<bool, Box<dyn std::error::Error>> {
    // Check if bwrap exists
    let version_check = Command::new("bwrap").arg("--version").output().await?;

    if !version_check.status.success() {
        return Ok(false);
    }

    // Run preflight: try to use --proc flag (may fail in nested containers)
    let preflight = Command::new("bwrap")
        .args([
            "--ro-bind",
            "/",
            "/",
            "--proc",
            "/proc",
            "--",
            "/usr/bin/true",
        ])
        .output()
        .await?;

    Ok(preflight.status.success())
}

async fn check_sandbox_exec() -> Result<bool, Box<dyn std::error::Error>> {
    // Check if sandbox-exec exists at the hardcoded system path
    let metadata = tokio::fs::metadata("/usr/bin/sandbox-exec").await?;
    Ok(metadata.is_file())
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
