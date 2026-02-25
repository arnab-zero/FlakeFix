import { useState, useEffect } from "react";

function AnalysisProgress({ analysisId, repository, onBack }) {
  const [progress, setProgress] = useState({
    status: "initializing",
    currentTest: 0,
    totalTests: 0,
    tests: [],
  });
  const [error, setError] = useState(null);

  useEffect(() => {
    // Poll for progress updates every 2 seconds
    const interval = setInterval(async () => {
      try {
        const token = localStorage.getItem("github_token");
        const response = await fetch(
          `http://localhost:8000/analysis/${analysisId}/progress`,
          {
            headers: {
              Authorization: `Bearer ${token}`,
            },
          },
        );

        if (!response.ok) {
          throw new Error("Failed to fetch progress");
        }

        const data = await response.json();
        setProgress(data);

        // Stop polling if analysis is complete or failed
        if (data.status === "completed" || data.status === "failed") {
          clearInterval(interval);
        }
      } catch (err) {
        console.error("Error fetching progress:", err);
        setError(err.message);
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [analysisId]);

  const getStepIcon = (status) => {
    if (status === "completed") return "✓";
    if (status === "failed") return "✗";
    if (status === "in_progress") return "⟳";
    return "○";
  };

  const getStepClass = (status) => {
    if (status === "completed") return "step-completed";
    if (status === "failed") return "step-failed";
    if (status === "in_progress") return "step-in-progress";
    return "step-pending";
  };

  return (
    <div className="analysis-progress-container">
      <div className="progress-header">
        <button onClick={onBack} className="back-btn">
          ← Back
        </button>
        <h2>Analysis Progress</h2>
        <p className="repo-name">{repository}</p>
      </div>

      <div className="progress-summary">
        <div className="summary-item">
          <span className="summary-label">Status:</span>
          <span className={`summary-value status-${progress.status}`}>
            {progress.status.replace("_", " ").toUpperCase()}
          </span>
        </div>
        <div className="summary-item">
          <span className="summary-label">Progress:</span>
          <span className="summary-value">
            {progress.currentTest} / {progress.totalTests} tests
          </span>
        </div>
      </div>

      {error && (
        <div className="error-message">
          <strong>Error:</strong> {error}
        </div>
      )}

      <div className="tests-progress">
        {progress.tests.map((test, index) => (
          <div key={index} className="test-card">
            <div className="test-header">
              <h3>
                Test {index + 1}: {test.method}
              </h3>
              <span className={`test-status ${getStepClass(test.status)}`}>
                {test.status.replace("_", " ")}
              </span>
            </div>
            <p className="test-file">{test.file}</p>

            {test.steps && (
              <div className="pipeline-steps">
                <div className={`step ${getStepClass(test.steps.call_graph)}`}>
                  <span className="step-icon">
                    {getStepIcon(test.steps.call_graph)}
                  </span>
                  <span className="step-name">1. Call Graph Generation</span>
                  {test.steps.call_graph === "in_progress" && (
                    <span className="spinner"></span>
                  )}
                </div>

                <div className={`step ${getStepClass(test.steps.inlining)}`}>
                  <span className="step-icon">
                    {getStepIcon(test.steps.inlining)}
                  </span>
                  <span className="step-name">2. Method Inlining</span>
                  {test.steps.inlining === "in_progress" && (
                    <span className="spinner"></span>
                  )}
                </div>

                <div
                  className={`step ${getStepClass(test.steps.token_counting)}`}
                >
                  <span className="step-icon">
                    {getStepIcon(test.steps.token_counting)}
                  </span>
                  <span className="step-name">3. Token Counting</span>
                  {test.steps.token_counting === "in_progress" && (
                    <span className="spinner"></span>
                  )}
                </div>

                <div className={`step ${getStepClass(test.steps.detection)}`}>
                  <span className="step-icon">
                    {getStepIcon(test.steps.detection)}
                  </span>
                  <span className="step-name">4. Flaky Detection</span>
                  {test.steps.detection === "in_progress" && (
                    <span className="spinner"></span>
                  )}
                </div>

                <div className={`step ${getStepClass(test.steps.patch)}`}>
                  <span className="step-icon">
                    {getStepIcon(test.steps.patch)}
                  </span>
                  <span className="step-name">5. Patch Generation</span>
                  {test.steps.patch === "in_progress" && (
                    <span className="spinner"></span>
                  )}
                </div>
              </div>
            )}

            {test.result && (
              <div className="test-result">
                <h4>Result:</h4>
                <div className="result-details">
                  <p>
                    <strong>Is Flaky:</strong>{" "}
                    {test.result.is_flaky ? "Yes" : "No"}
                  </p>
                  {test.result.is_flaky && (
                    <>
                      <p>
                        <strong>Type:</strong> {test.result.label}
                      </p>
                      <p>
                        <strong>Confidence:</strong> {test.result.confidence}
                      </p>
                      {test.result.patch_generated && (
                        <p>
                          <strong>Patch:</strong> Generated successfully
                        </p>
                      )}
                    </>
                  )}
                </div>
              </div>
            )}

            {test.error && (
              <div className="test-error">
                <strong>Error:</strong> {test.error}
              </div>
            )}
          </div>
        ))}
      </div>

      {progress.status === "completed" && (
        <div className="completion-message">
          <h3>✓ Analysis Complete!</h3>
          <p>Results have been saved to the output/ directory on the server.</p>
        </div>
      )}
    </div>
  );
}

export default AnalysisProgress;
