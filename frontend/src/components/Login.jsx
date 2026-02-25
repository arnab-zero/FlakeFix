import { useState } from "react";

function Login({ onLogin }) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleGitHubLogin = async () => {
    setLoading(true);
    setError(null);

    try {
      // Redirect to backend OAuth endpoint
      window.location.href = "http://localhost:8000/auth/github";
    } catch (err) {
      setError("Failed to initiate GitHub login");
      setLoading(false);
    }
  };

  return (
    <div className="login-container">
      <div className="login-card">
        <h2>Sign in to FlakeFix</h2>
        <p>Authenticate with GitHub to analyze your repositories</p>

        {error && <div className="error-message">{error}</div>}

        <button
          onClick={handleGitHubLogin}
          disabled={loading}
          className="github-login-btn"
        >
          {loading ? "Connecting..." : "Sign in with GitHub"}
        </button>
      </div>
    </div>
  );
}

export default Login;
