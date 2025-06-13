import React, { useState, useEffect } from 'react';
import './App.css';
import Login from './components/Login';
import Register from './components/Register';
import DocumentUpload from './components/DocumentUpload';
import QueryDocument from './components/QueryDocument';
import DocumentList from './components/DocumentList';

function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [showRegister, setShowRegister] = useState(false);
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(null);

  useEffect(() => {
    // Check if user is already logged in
    const savedToken = localStorage.getItem('access_token');
    const savedUser = localStorage.getItem('user_data');
    
    if (savedToken && savedUser) {
      setToken(savedToken);
      setUser(JSON.parse(savedUser));
      setIsAuthenticated(true);
    }
  }, []);

  const handleLogin = (userData, accessToken) => {
    setUser(userData);
    setToken(accessToken);
    setIsAuthenticated(true);
    
    // Save to localStorage
    localStorage.setItem('access_token', accessToken);
    localStorage.setItem('user_data', JSON.stringify(userData));
  };

  const handleLogout = () => {
    setUser(null);
    setToken(null);
    setIsAuthenticated(false);
    
    // Clear localStorage
    localStorage.removeItem('access_token');
    localStorage.removeItem('user_data');
  };

  const handleRegister = (userData, accessToken) => {
    handleLogin(userData, accessToken);
  };

  if (!isAuthenticated) {
    return (
      <div className="App">
        <header className="App-header">
          <h1>Document Query RAG System</h1>
          <p>Secure document management with AI-powered querying</p>
        </header>
        <main className="auth-container">
          {showRegister ? (
            <div>
              <Register onRegister={handleRegister} />
              <p style={{ textAlign: 'center', marginTop: '20px' }}>
                Already have an account?{' '}
                <button 
                  onClick={() => setShowRegister(false)}
                  style={{ background: 'none', border: 'none', color: '#007bff', cursor: 'pointer', textDecoration: 'underline' }}
                >
                  Login here
                </button>
              </p>
            </div>
          ) : (
            <div>
              <Login onLogin={handleLogin} />
              <p style={{ textAlign: 'center', marginTop: '20px' }}>
                Don't have an account?{' '}
                <button 
                  onClick={() => setShowRegister(true)}
                  style={{ background: 'none', border: 'none', color: '#007bff', cursor: 'pointer', textDecoration: 'underline' }}
                >
                  Register here
                </button>
              </p>
            </div>
          )}
        </main>
      </div>
    );
  }

  return (
    <div className="App">
      <header className="App-header">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', width: '100%' }}>
          <div>
            <h1>Document Query RAG System</h1>
            <p>Welcome, {user?.username}!</p>
          </div>
          <button 
            onClick={handleLogout}
            style={{
              padding: '10px 20px',
              backgroundColor: '#dc3545',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer'
            }}
          >
            Logout
          </button>
        </div>
      </header>
      <main className="main-content">
        <div className="content-grid">
          <div className="upload-section">
            <DocumentUpload token={token} />
          </div>
          <div className="query-section">
            <QueryDocument token={token} />
          </div>
          <div className="documents-section">
            <DocumentList token={token} />
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;