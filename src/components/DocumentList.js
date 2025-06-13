import React, { useState, useEffect } from 'react';
import axios from 'axios';

function DocumentList({ token }) {
  const [documents, setDocuments] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [deleteLoading, setDeleteLoading] = useState({});

  useEffect(() => {
    fetchDocuments();
  }, [token]);

  const fetchDocuments = async () => {
    if (!token) return;

    setLoading(true);
    setError('');

    try {
      const response = await axios.get('http://localhost:8000/documents', {
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        }
      });

      setDocuments(response.data.documents || []);
    } catch (error) {
      console.error('Error fetching documents:', error);
      
      let errorMessage = 'Failed to load documents';
      if (error.response?.data?.detail) {
        errorMessage = error.response.data.detail;
      } else if (error.request) {
        errorMessage = 'Cannot connect to server. Please check if the backend is running.';
      }
      
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  const deleteDocument = async (documentId, documentTitle) => {
    if (!window.confirm(`Are you sure you want to delete "${documentTitle}"?`)) {
      return;
    }

    setDeleteLoading({ ...deleteLoading, [documentId]: true });

    try {
      await axios.delete(`http://localhost:8000/documents/${documentId}`, {
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        }
      });

      // Remove the document from the local state
      setDocuments(documents.filter(doc => doc.id !== documentId));
      
    } catch (error) {
      console.error('Error deleting document:', error);
      
      let errorMessage = 'Failed to delete document';
      if (error.response?.data?.detail) {
        errorMessage = error.response.data.detail;
      }
      
      alert(errorMessage);
    } finally {
      setDeleteLoading({ ...deleteLoading, [documentId]: false });
    }
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const formatDate = (dateString) => {
    try {
      return new Date(dateString).toLocaleString();
    } catch {
      return 'Unknown';
    }
  };

  const getFileTypeColor = (fileType) => {
    const colors = {
      'pdf': '#dc3545',
      'txt': '#28a745',
      'pptx': '#fd7e14',
      'ppt': '#fd7e14',
      'csv': '#20c997',
      'docx': '#0d6efd',
      'doc': '#0d6efd',
    };
    return colors[fileType?.toLowerCase()] || '#6c757d';
  };

  if (loading) {
    return (
      <div style={{ 
        margin: '20px', 
        padding: '20px', 
        border: '1px solid #ddd', 
        borderRadius: '8px',
        textAlign: 'center'
      }}>
        <h2>Your Documents</h2>
        <p>Loading documents...</p>
      </div>
    );
  }

  return (
    <div style={{ margin: '20px', padding: '20px', border: '1px solid #ddd', borderRadius: '8px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
        <h2>Your Documents</h2>
        <button
          onClick={fetchDocuments}
          style={{
            padding: '8px 16px',
            backgroundColor: '#007bff',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
            fontSize: '14px'
          }}
        >
          Refresh
        </button>
      </div>

      {error && (
        <div style={{
          marginBottom: '20px',
          padding: '10px',
          borderRadius: '4px',
          backgroundColor: '#f8d7da',
          color: '#721c24',
          border: '1px solid #f5c6cb'
        }}>
          <strong>Error:</strong> {error}
        </div>
      )}

      {documents.length === 0 ? (
        <div style={{
          textAlign: 'center',
          padding: '40px',
          color: '#6c757d'
        }}>
          <p>No documents uploaded yet.</p>
          <p>Upload your first document to get started!</p>
        </div>
      ) : (
        <div>
          <p style={{ marginBottom: '15px', color: '#6c757d' }}>
            Total documents: {documents.length}
          </p>
          
          <div style={{ display: 'grid', gap: '15px' }}>
            {documents.map((doc) => (
              <div
                key={doc.id}
                style={{
                  border: '1px solid #e9ecef',
                  borderRadius: '6px',
                  padding: '15px',
                  backgroundColor: '#f8f9fa'
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                  <div style={{ flex: 1 }}>
                    <h4 style={{ 
                      margin: '0 0 8px 0', 
                      color: '#495057',
                      wordBreak: 'break-word'
                    }}>
                      {doc.title}
                    </h4>
                    
                    <div style={{ display: 'flex', gap: '15px', marginBottom: '10px', flexWrap: 'wrap' }}>
                      <span style={{
                        backgroundColor: getFileTypeColor(doc.file_type),
                        color: 'white',
                        padding: '2px 8px',
                        borderRadius: '4px',
                        fontSize: '12px',
                        fontWeight: 'bold'
                      }}>
                        {doc.file_type?.toUpperCase() || 'UNKNOWN'}
                      </span>
                      
                      <span style={{ fontSize: '14px', color: '#6c757d' }}>
                        Size: {formatFileSize(doc.file_size)}
                      </span>
                      
                      <span style={{ fontSize: '14px', color: '#6c757d' }}>
                        Length: {doc.content_length?.toLocaleString()} characters
                      </span>
                    </div>
                    
                    {doc.upload_time && (
                      <p style={{ margin: '0 0 10px 0', fontSize: '13px', color: '#6c757d' }}>
                        Uploaded: {formatDate(doc.upload_time)}
                      </p>
                    )}
                    
                    {doc.content_preview && (
                      <div style={{
                        backgroundColor: '#fff',
                        padding: '10px',
                        borderRadius: '4px',
                        border: '1px solid #dee2e6',
                        fontSize: '13px',
                        color: '#495057',
                        fontFamily: 'monospace',
                        maxHeight: '100px',
                        overflow: 'hidden'
                      }}>
                        {doc.content_preview}
                      </div>
                    )}
                  </div>
                  
                  <button
                    onClick={() => deleteDocument(doc.id, doc.title)}
                    disabled={deleteLoading[doc.id]}
                    style={{
                      marginLeft: '15px',
                      padding: '6px 12px',
                      backgroundColor: deleteLoading[doc.id] ? '#ccc' : '#dc3545',
                      color: 'white',
                      border: 'none',
                      borderRadius: '4px',
                      cursor: deleteLoading[doc.id] ? 'not-allowed' : 'pointer',
                      fontSize: '12px'
                    }}
                  >
                    {deleteLoading[doc.id] ? 'Deleting...' : 'Delete'}
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default DocumentList;