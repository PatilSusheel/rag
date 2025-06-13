import React, { useState } from 'react';
import axios from 'axios';

function DocumentUpload({ token }) {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [message, setMessage] = useState('');
  const [messageType, setMessageType] = useState(''); // 'success' or 'error'

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
    setMessage(''); // Clear previous messages
    
    // Validate file type - Updated to match backend supported formats
    if (selectedFile) {
      const validTypes = ['.pdf', '.txt', '.pptx', '.ppt', '.csv', '.docx', '.doc'];
      const fileExtension = selectedFile.name.toLowerCase().substring(selectedFile.name.lastIndexOf('.'));
      
      if (!validTypes.includes(fileExtension)) {
        setMessage('Please select a valid file type: PDF, TXT, PPT, PPTX, CSV, DOC, DOCX');
        setMessageType('error');
        setFile(null);
        e.target.value = ''; // Reset file input
      }
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setMessage("Please choose a file");
      setMessageType('error');
      return;
    }

    if (!token) {
      setMessage("Authentication required. Please log in again.");
      setMessageType('error');
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      setUploading(true);
      setMessage("Uploading and processing document...");
      setMessageType('');
      
      // Make the POST request to the FastAPI backend with authentication
      const response = await axios.post('http://localhost:8000/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
          'Authorization': `Bearer ${token}`
        },
        timeout: 30000, // 30 second timeout
      });
      
      setMessage(response.data.message || 'Upload successful!');
      setMessageType('success');
      console.log('Upload response:', response.data);
      
      // Reset file input after successful upload
      setFile(null);
      document.querySelector('input[type="file"]').value = '';
      
    } catch (error) {
      console.error('Upload error:', error);
      
      let errorMessage = 'Error uploading file';
      
      if (error.response) {
        // Server responded with error status
        errorMessage = error.response.data?.detail || `Server error: ${error.response.status}`;
        
        // Handle authentication errors
        if (error.response.status === 401) {
          errorMessage = 'Authentication failed. Please log in again.';
        }
      } else if (error.request) {
        // Request made but no response received
        errorMessage = 'No response from server. Please check if the backend is running.';
      } else {
        // Something else happened
        errorMessage = error.message || 'Unknown error occurred';
      }
      
      setMessage(errorMessage);
      setMessageType('error');
    } finally {
      setUploading(false);
    }
  };

  return (
    <div style={{ margin: '20px', padding: '20px', border: '1px solid #ddd', borderRadius: '8px' }}>
      <h2>Upload Document</h2>
      <div style={{ marginBottom: '15px' }}>
        <input 
          type="file" 
          onChange={handleFileChange}
          accept=".pdf,.txt,.pptx,.ppt,.csv,.docx,.doc"
          disabled={uploading}
        />
        <small style={{ display: 'block', marginTop: '5px', color: '#666' }}>
          Supported formats: PDF, TXT, PPT, PPTX, CSV, DOC, DOCX
        </small>
      </div>
      
      <button 
        onClick={handleUpload} 
        disabled={uploading || !file}
        style={{
          padding: '10px 20px',
          backgroundColor: uploading || !file ? '#ccc' : '#007bff',
          color: 'white',
          border: 'none',
          borderRadius: '4px',
          cursor: uploading || !file ? 'not-allowed' : 'pointer',
          fontSize: '16px'
        }}
      >
        {uploading ? 'Uploading...' : 'Upload Document'}
      </button>
      
      {message && (
        <div style={{
          marginTop: '15px',
          padding: '10px',
          borderRadius: '4px',
          backgroundColor: messageType === 'success' ? '#d4edda' : messageType === 'error' ? '#f8d7da' : '#e2e3e5',
          color: messageType === 'success' ? '#155724' : messageType === 'error' ? '#721c24' : '#383d41',
          border: `1px solid ${messageType === 'success' ? '#c3e6cb' : messageType === 'error' ? '#f5c6cb' : '#d6d8db'}`
        }}>
          {message}
        </div>
      )}
    </div>
  );
}

export default DocumentUpload;