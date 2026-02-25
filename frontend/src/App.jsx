import { useState, useEffect } from "react";
import "./App.css";
import Login from "./components/Login";
import AnalysisForm from "./components/AnalysisForm";
import AnalysisProgress from "./components/AnalysisProgress";
import "./components/AnalysisProgress.css";

function App() {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [analysisView, setAnalysisView] = useState(null); // { id, repository }

  useEffect(() => {
    // Check if user is already logged in
    const token = localStorage.getItem("github_token");
    const userData = localStorage.getItem("github_user");

    if (token && userData) {
      setUser(JSON.parse(userData));
    }

    // Handle OAuth callback
    const params = new URLSearchParams(window.location.search);
    const token_param = params.get("token");
    const user_param = params.get("user");
    const avatar_param = params.get("avatar");

    if (token_param && user_param) {
      const userData = {
        login: user_param,
        avatar_url: avatar_param,
      };
      localStorage.setItem("github_token", token_param);
      localStorage.setItem("github_user", JSON.stringify(userData));
      setUser(userData);

      // Clean up URL
      window.history.replaceState({}, document.title, "/");
    }

    setLoading(false);
  }, []);

  const handleLogin = (userData, token) => {
    localStorage.setItem("github_token", token);
    localStorage.setItem("github_user", JSON.stringify(userData));
    setUser(userData);
  };

  const handleLogout = () => {
    localStorage.removeItem("github_token");
    localStorage.removeItem("github_user");
    setUser(null);
    setAnalysisView(null);
  };

  const handleAnalysisStart = (analysisData) => {
    setAnalysisView({
      id: analysisData.analysis_id,
      repository: analysisData.repository,
    });
  };

  const handleBackToForm = () => {
    setAnalysisView(null);
  };

  if (loading) {
    return <div className="loading">Loading...</div>;
  }

  return (
    <div className="App">
      <header className="App-header">
        <h1>🔍 FlakeFix</h1>
        <p>Flaky Test Detection & Patch Generation</p>
      </header>

      {!user ? (
        <Login onLogin={handleLogin} />
      ) : analysisView ? (
        <AnalysisProgress
          analysisId={analysisView.id}
          repository={analysisView.repository}
          onBack={handleBackToForm}
        />
      ) : (
        <div>
          <div className="user-info">
            <img src={user.avatar_url} alt={user.login} className="avatar" />
            <span>Welcome, {user.login}!</span>
            <button onClick={handleLogout} className="logout-btn">
              Logout
            </button>
          </div>
          <AnalysisForm user={user} onAnalysisStart={handleAnalysisStart} />
        </div>
      )}
    </div>
  );
}

export default App;
