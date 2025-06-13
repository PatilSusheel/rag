import React, { useState } from 'react';
import axios from 'axios';

function QueryDocument({ token }) {
  const [query, setQuery] = useState('');
  const [response, setResponse] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [documentTitle, setDocumentTitle] = useState('');
  const [totalMatches, setTotalMatches] = useState(0);
  const [queryHistory, setQueryHistory] = useState([]);

  const handleQueryChange = (e) => {
    setQuery(e.target.value);
    setError(''); // Clear error when user types
  };

  const handleQuerySubmit = async () => {
    const trimmedQuery = query.trim();
    
    if (!trimmedQuery) {
      setError("Please enter a query");
      return;
    }

    if (!token) {
      setError("Authentication required. Please log in again.");
      return;
    }

    setLoading(true);
    setError('');
    setResponse('');

    try {
      const res = await axios.post('http://localhost:8000/query', 
        { query: trimmedQuery },
        {
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${token}`
          },
          timeout: 100000, // Increased timeout for complex queries
        }
      );
      
      if (res.data.response) {
        const newQueryResult = {
          id: Date.now(),
          query: trimmedQuery,
          response: res.data.response,
          documentTitle: res.data.document_title || '',
          totalMatches: res.data.total_matches || 0,
          timestamp: new Date().toLocaleString()
        };
        
        setResponse(res.data.response);
        setDocumentTitle(res.data.document_title || '');
        setTotalMatches(res.data.total_matches || 0);
        setQueryHistory(prev => [newQueryResult, ...prev.slice(0, 9)]); // Keep last 10 queries
        setQuery(''); // Clear the input after successful query
      } else {
        setError('No response received from server');
      }
      
    } catch (error) {
      console.error('Query error:', error);
      
      let errorMessage = 'Error fetching response';
      
      if (error.response) {
        errorMessage = error.response.data?.detail || `Server error: ${error.response.status}`;
        
        if (error.response.status === 401) {
          errorMessage = 'Authentication failed. Please log in again.';
        }
      } else if (error.request) {
        errorMessage = 'No response from server. Please check if the backend is running.';
      } else {
        errorMessage = error.message || 'Unknown error occurred';
      }
      
      setError(errorMessage);
      setResponse('');
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleQuerySubmit();
    }
  };

  const clearResults = () => {
    setResponse('');
    setError('');
    setDocumentTitle('');
    setTotalMatches(0);
  };

  const loadPreviousQuery = (queryItem) => {
    setQuery(queryItem.query);
    setResponse(queryItem.response);
    setDocumentTitle(queryItem.documentTitle);
    setTotalMatches(queryItem.totalMatches);
    setError('');
  };

  return (
    <div style={{ margin: '20px', padding: '20px', border: '1px solid #ddd', borderRadius: '8px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
        <h2 style={{ margin: 0 }}>Ask a Question</h2>
        {(response || error) && (
          <button 
            onClick={clearResults}
            style={{
              padding: '6px 12px',
              backgroundColor: '#6c757d',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer',
              fontSize: '12px'
            }}
          >
            Clear Results
          </button>
        )}
      </div>
      
      <div style={{ marginBottom: '15px' }}>
        <textarea
          value={query}
          onChange={handleQueryChange}
          onKeyPress={handleKeyPress}
          placeholder="Ask something about your documents... (Press Enter to submit, Shift+Enter for new line)"
          disabled={loading}
          rows="4"
          style={{
            width: '100%',
            padding: '12px',
            borderRadius: '6px',
            border: '2px solid #e9ecef',
            resize: 'vertical',
            fontSize: '14px',
            fontFamily: 'system-ui, -apple-system, sans-serif',
            lineHeight: '1.5',
            transition: 'border-color 0.2s',
            ':focus': {
              borderColor: '#007bff',
              outline: 'none'
            }
          }}
        />
        <small style={{ color: '#6c757d', fontSize: '12px' }}>
          💡 Try asking specific questions about your documents, request summaries, or ask for detailed explanations.
        </small>
      </div>
      
      <div style={{ display: 'flex', gap: '10px', marginBottom: '20px' }}>
        <button 
          onClick={handleQuerySubmit} 
          disabled={loading || !query.trim()}
          style={{
            padding: '12px 24px',
            backgroundColor: loading || !query.trim() ? '#ccc' : '#28a745',
            color: 'white',
            border: 'none',
            borderRadius: '6px',
            cursor: loading || !query.trim() ? 'not-allowed' : 'pointer',
            fontSize: '14px',
            fontWeight: '500',
            transition: 'all 0.2s'
          }}
        >
          {loading ? (
            <>
              <span style={{ marginRight: '8px' }}>⏳</span>
              Processing...
            </>
          ) : (
            <>
              <span style={{ marginRight: '8px' }}>🔍</span>
              Ask Question
            </>
          )}
        </button>
      </div>

      {error && (
        <div style={{
          marginBottom: '20px',
          padding: '16px',
          borderRadius: '8px',
          backgroundColor: '#f8d7da',
          color: '#721c24',
          border: '1px solid #f5c6cb',
          borderLeft: '4px solid #dc3545'
        }}>
          <div style={{ display: 'flex', alignItems: 'flex-start' }}>
            <span style={{ marginRight: '8px', fontSize: '16px' }}>❌</span>
            <div>
              <strong>Error:</strong> {error}
            </div>
          </div>
        </div>
      )}

      {response && (
        <div style={{
          marginBottom: '20px',
          borderRadius: '12px',
          backgroundColor: '#ffffff',
          border: '2px solid #28a745',
          boxShadow: '0 4px 12px rgba(40, 167, 69, 0.1)',
          overflow: 'hidden'
        }}>
          {/* Header */}
          <div style={{
            backgroundColor: '#28a745',
            color: 'white',
            padding: '16px 20px',
            borderBottom: '1px solid rgba(255,255,255,0.2)'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <h3 style={{ margin: 0, fontSize: '18px', fontWeight: '600' }}>
                <span style={{ marginRight: '8px' }}>💡</span>
                Answer
              </h3>
              {documentTitle && (
                <div style={{ fontSize: '12px', opacity: 0.9 }}>
                  📄 Source: {documentTitle}
                  {totalMatches > 0 && ` • ${totalMatches} matches`}
                </div>
              )}
            </div>
          </div>

          {/* Content */}
          <div style={{ padding: '24px' }}>
            <div style={{
              color: '#2d3748',
              lineHeight: '1.7',
              fontSize: '15px',
              fontFamily: 'system-ui, -apple-system, sans-serif',
              whiteSpace: 'pre-wrap',
              wordWrap: 'break-word',
              maxHeight: '600px',
              overflowY: 'auto',
              scrollbarWidth: 'thin',
              scrollbarColor: '#cbd5e0 #f7fafc'
            }}>
              {response.split('\n\n').map((paragraph, index) => (
                <div key={index} style={{ 
                  marginBottom: paragraph.trim() ? '16px' : '8px',
                  padding: paragraph.trim().startsWith('•') || paragraph.trim().startsWith('-') ? '0 0 0 16px' : '0'
                }}>
                  {paragraph.trim()}
                </div>
              ))}
            </div>
          </div>

          {/* Footer with metadata */}
          {(documentTitle || totalMatches > 0) && (
            <div style={{
              backgroundColor: '#f8f9fa',
              padding: '12px 20px',
              borderTop: '1px solid #e9ecef',
              fontSize: '13px',
              color: '#6c757d'
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                {documentTitle && (
                  <span>📋 Retrieved from: <strong>{documentTitle}</strong></span>
                )}
                {totalMatches > 0 && (
                  <span>🎯 {totalMatches} relevant sections found</span>
                )}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Query History */}
      {queryHistory.length > 0 && (
        <div style={{
          marginTop: '30px',
          padding: '20px',
          backgroundColor: '#f8f9fa',
          borderRadius: '8px',
          border: '1px solid #e9ecef'
        }}>
          <h4 style={{ margin: '0 0 15px 0', color: '#495057', fontSize: '16px' }}>
            📚 Recent Queries
          </h4>
          <div style={{ maxHeight: '300px', overflowY: 'auto' }}>
            {queryHistory.map((item) => (
              <div
                key={item.id}
                style={{
                  padding: '12px',
                  marginBottom: '8px',
                  backgroundColor: 'white',
                  borderRadius: '6px',
                  border: '1px solid #dee2e6',
                  cursor: 'pointer',
                  transition: 'all 0.2s'
                }}
                onClick={() => loadPreviousQuery(item)}
                onMouseEnter={(e) => {
                  e.target.style.backgroundColor = '#e3f2fd';
                  e.target.style.borderColor = '#2196f3';
                }}
                onMouseLeave={(e) => {
                  e.target.style.backgroundColor = 'white';
                  e.target.style.borderColor = '#dee2e6';
                }}
              >
                <div style={{ 
                  fontSize: '13px', 
                  fontWeight: '500', 
                  color: '#495057',
                  marginBottom: '4px',
                  display: '-webkit-box',
                  WebkitLineClamp: 2,
                  WebkitBoxOrient: 'vertical',
                  overflow: 'hidden'
                }}>
                  Q: {item.query}
                </div>
                <div style={{ 
                  fontSize: '11px', 
                  color: '#6c757d',
                  display: 'flex',
                  justifyContent: 'space-between'
                }}>
                  <span>{item.timestamp}</span>
                  {item.documentTitle && <span>📄 {item.documentTitle}</span>}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default QueryDocument;