import { useState } from "react";

function AnalysisForm({ user, onAnalysisStart }) {
  const [formData, setFormData] = useState({
    owner: "",
    repo: "",
    branch: "main",
    commitMessage: "",
  });
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const token = localStorage.getItem("github_token");
      const response = await fetch("http://localhost:8000/analyze", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify(formData),
      });

      if (!response.ok) {
        throw new Error(`Analysis failed: ${response.statusText}`);
      }

      const data = await response.json();
      setResult(data);

      // Navigate to progress view
      if (onAnalysisStart && data.analysis_id) {
        onAnalysisStart(data);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="analysis-container">
      <div className="analysis-card">
        <h2>Analyze Repository</h2>

        <form onSubmit={handleSubmit} className="analysis-form">
          <div className="form-group">
            <label htmlFor="owner">Repository Owner</label>
            <input
              type="text"
              id="owner"
              name="owner"
              value={formData.owner}
              onChange={handleChange}
              placeholder="e.g., facebook"
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="repo">Repository Name</label>
            <input
              type="text"
              id="repo"
              name="repo"
              value={formData.repo}
              onChange={handleChange}
              placeholder="e.g., react"
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="branch">Branch Name</label>
            <input
              type="text"
              id="branch"
              name="branch"
              value={formData.branch}
              onChange={handleChange}
              placeholder="e.g., main"
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="commitMessage">Commit Message (Optional)</label>
            <textarea
              id="commitMessage"
              name="commitMessage"
              value={formData.commitMessage}
              onChange={handleChange}
              placeholder="Describe the changes..."
              rows="3"
            />
          </div>

          <button type="submit" disabled={loading} className="submit-btn">
            {loading ? "Analyzing..." : "Start Analysis"}
          </button>
        </form>

        {error && (
          <div className="error-message">
            <strong>Error:</strong> {error}
          </div>
        )}

        {result && (
          <div className="result-message">
            <strong>Success!</strong> Analysis started for{" "}
            {result.test_methods_count} test method(s).
            <br />
            Results will be saved to the output/ directory.
            <br />
            Check the backend logs for progress.
          </div>
        )}
      </div>
    </div>
  );
}

export default AnalysisForm;
