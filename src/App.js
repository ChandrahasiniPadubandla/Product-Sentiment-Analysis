import React, { useState } from 'react';
import { Upload, BarChart3, Brain, Activity, FileText, AlertCircle } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const API_URL = 'http://localhost:5000/api';

export default function SentimentAnalysisApp() {
  const [dataset, setDataset] = useState(null);
  const [preprocessed, setPreprocessed] = useState(false);
  const [models, setModels] = useState({});
  const [logs, setLogs] = useState([]);
  const [activeTab, setActiveTab] = useState('logs');
  const [testResults, setTestResults] = useState([]);
  const [loading, setLoading] = useState(false);

  const addLog = (message, type = 'info') => {
    setLogs(prev => [...prev, { 
      time: new Date().toLocaleTimeString(), 
      message,
      type 
    }]);
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setLoading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${API_URL}/upload`, {
        method: 'POST',
        body: formData
      });
      const data = await response.json();

      if (data.success) {
        setDataset(file.name);
        addLog(`✓ Dataset "${file.name}" uploaded successfully`, 'success');
        addLog(`✓ Total records: ${data.total_records}`, 'success');
        if (data.samples) {
          data.samples.forEach(sample => {
            addLog(`Sample: ${sample.text} - Rating: ${sample.rating}`, 'info');
          });
        }
      } else {
        addLog(`✗ Upload failed: ${data.error}`, 'error');
      }
    } catch (err) {
      addLog(`✗ Connection error: ${err.message}`, 'error');
    } finally {
      setLoading(false);
    }
  };

  const handlePreprocess = async () => {
    if (!dataset) {
      addLog('✗ Please upload a dataset first', 'error');
      return;
    }
    setLoading(true);
    try {
      const response = await fetch(`${API_URL}/preprocess`, { method: 'POST' });
      const data = await response.json();
      if (data.success) {
        setPreprocessed(true);
        addLog('✓ Preprocessing completed', 'success');
        addLog(`✓ Train: ${data.train_size}, Test: ${data.test_size}`, 'success');
      } else {
        addLog(`✗ Preprocessing failed: ${data.error}`, 'error');
      }
    } catch (err) {
      addLog(`✗ Error: ${err.message}`, 'error');
    } finally {
      setLoading(false);
    }
  };

  const trainModel = async (modelType, displayName) => {
    if (!preprocessed) {
      addLog('✗ Please preprocess the dataset first', 'error');
      return;
    }
    setLoading(true);
    addLog(`Training ${displayName}...`, 'info');
    try {
      const response = await fetch(`${API_URL}/train/${modelType}`, { method: 'POST' });
      const data = await response.json();
      if (data.success) {
        setModels(prev => ({
          ...prev,
          [data.model_name]: { accuracy: data.accuracy.toFixed(1) }
        }));
        addLog(`✓ ${displayName} - Accuracy: ${data.accuracy.toFixed(1)}%`, 'success');
        setActiveTab('results');
      } else {
        addLog(`✗ Training failed: ${data.error}`, 'error');
      }
    } catch (err) {
      addLog(`✗ Error: ${err.message}`, 'error');
    } finally {
      setLoading(false);
    }
  };

  const detectSentiment = async () => {
    if (Object.keys(models).length === 0) {
      addLog('✗ Please train at least one model first', 'error');
      return;
    }
    setLoading(true);
    try {
      const response = await fetch(`${API_URL}/detect_sentiment`, { method: 'POST' });
      const data = await response.json();
      if (data.success) {
        setTestResults(data.results);
        addLog(`✓ Sentiment detection completed`, 'success');
        setActiveTab('results');
      } else {
        addLog(`✗ Detection failed: ${data.error}`, 'error');
      }
    } catch (err) {
      addLog(`✗ Error: ${err.message}`, 'error');
    } finally {
      setLoading(false);
    }
  };

  const chartData = Object.entries(models).map(([name, data]) => ({
    name,
    accuracy: parseFloat(data.accuracy)
  }));

  return (
    <div style={{minHeight: '100vh', backgroundColor: '#00cccc', display: 'flex', flexDirection: 'column'}}>
      {/* Header */}
      <div style={{backgroundColor: '#6600cc', padding: '24px', textAlign: 'center'}}>
        <h1 style={{color: 'white', fontSize: '28px', fontWeight: 'bold', margin: 0}}>
          Sentiment Analysis of Customer Product Reviews using Machine Learning
        </h1>
      </div>

      {/* Main Content - Split Layout */}
      <div style={{flex: 1, display: 'flex', padding: '20px', gap: '20px'}}>
        
        {/* LEFT SIDE - Buttons */}
        <div style={{width: '420px', display: 'flex', flexDirection: 'column', gap: '12px'}}>
          
          {/* Upload Button */}
          <div style={{
            backgroundColor: 'white',
            borderRadius: '8px',
            padding: '16px 20px',
            fontWeight: 'bold',
            boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
            position: 'relative',
            cursor: 'pointer'
          }}>
            <div style={{display: 'flex', alignItems: 'center', gap: '12px'}}>
              <Upload size={20} />
              <span>Upload Amazon Reviews Dataset</span>
            </div>
            <input
              type="file"
              accept=".csv"
              onChange={handleFileUpload}
              disabled={loading}
              style={{
                position: 'absolute',
                top: 0,
                left: 0,
                width: '100%',
                height: '100%',
                opacity: 0,
                cursor: 'pointer'
              }}
            />
          </div>

          {/* Preprocess Button */}
          <button
            onClick={handlePreprocess}
            disabled={!dataset || preprocessed || loading}
            style={{
              backgroundColor: preprocessed ? '#7dd3c0' : 'white',
              borderRadius: '8px',
              padding: '16px 20px',
              fontWeight: 'bold',
              boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
              border: 'none',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '12px',
              opacity: (!dataset || preprocessed || loading) ? 0.5 : 1,
              transition: 'all 0.2s'
            }}
          >
            <Activity size={20} />
            <span>Preprocess Dataset</span>
          </button>

          {/* Run SVM */}
          <button
            onClick={() => trainModel('svm', 'SVM')}
            disabled={!preprocessed || loading}
            style={{
              backgroundColor: models.SVM ? '#7dd3c0' : 'white',
              borderRadius: '8px',
              padding: '16px 20px',
              fontWeight: 'bold',
              boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
              border: 'none',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '12px',
              opacity: (!preprocessed || loading) ? 0.5 : 1,
              transition: 'all 0.2s'
            }}
          >
            <Brain size={20} />
            <span>Run SVM Algorithm</span>
          </button>

          {/* Run Naive Bayes */}
          <button
            onClick={() => trainModel('naive_bayes', 'Naive Bayes')}
            disabled={!preprocessed || loading}
            style={{
              backgroundColor: models['Naive Bayes'] ? '#7dd3c0' : 'white',
              borderRadius: '8px',
              padding: '16px 20px',
              fontWeight: 'bold',
              boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
              border: 'none',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '12px',
              opacity: (!preprocessed || loading) ? 0.5 : 1,
              transition: 'all 0.2s'
            }}
          >
            <Brain size={20} />
            <span>Run Naive Bayes Algorithm</span>
          </button>

          {/* Run Decision Tree */}
          <button
            onClick={() => trainModel('decision_tree', 'Decision Tree')}
            disabled={!preprocessed || loading}
            style={{
              backgroundColor: models['Decision Tree'] ? '#7dd3c0' : 'white',
              borderRadius: '8px',
              padding: '16px 20px',
              fontWeight: 'bold',
              boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
              border: 'none',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '12px',
              opacity: (!preprocessed || loading) ? 0.5 : 1,
              transition: 'all 0.2s'
            }}
          >
            <Brain size={20} />
            <span>Run Decision Tree Algorithm</span>
          </button>

          {/* Detect Sentiment */}
          <button
            onClick={detectSentiment}
            disabled={Object.keys(models).length === 0 || loading}
            style={{
              backgroundColor: 'white',
              borderRadius: '8px',
              padding: '16px 20px',
              fontWeight: 'bold',
              boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
              border: 'none',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '12px',
              opacity: (Object.keys(models).length === 0 || loading) ? 0.5 : 1,
              transition: 'all 0.2s'
            }}
          >
            <FileText size={20} />
            <span>Detect Sentiment from Test Reviews</span>
          </button>

          {/* Accuracy Graph */}
          <button
            onClick={() => setActiveTab('graph')}
            disabled={Object.keys(models).length === 0}
            style={{
              backgroundColor: 'white',
              borderRadius: '8px',
              padding: '16px 20px',
              fontWeight: 'bold',
              boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
              border: 'none',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '12px',
              opacity: Object.keys(models).length === 0 ? 0.5 : 1,
              transition: 'all 0.2s'
            }}
          >
            <BarChart3 size={20} />
            <span>Accuracy Graph</span>
          </button>
        </div>

        {/* RIGHT SIDE - Results Area */}
        <div style={{flex: 1, backgroundColor: 'white', borderRadius: '12px', boxShadow: '0 4px 16px rgba(0,0,0,0.1)', display: 'flex', flexDirection: 'column', overflow: 'hidden'}}>
          
          {/* Tabs */}
          <div style={{display: 'flex', borderBottom: '2px solid #e5e7eb'}}>
            <button
              onClick={() => setActiveTab('logs')}
              style={{
                padding: '14px 24px',
                fontWeight: 'bold',
                fontSize: '15px',
                backgroundColor: activeTab === 'logs' ? '#3b82f6' : '#e5e7eb',
                color: activeTab === 'logs' ? 'white' : '#374151',
                border: 'none',
                cursor: 'pointer',
                transition: 'all 0.2s'
              }}
            >
              Logs
            </button>
            <button
              onClick={() => setActiveTab('results')}
              style={{
                padding: '14px 24px',
                fontWeight: 'bold',
                fontSize: '15px',
                backgroundColor: activeTab === 'results' ? '#3b82f6' : '#e5e7eb',
                color: activeTab === 'results' ? 'white' : '#374151',
                border: 'none',
                cursor: 'pointer',
                transition: 'all 0.2s'
              }}
            >
              Results
            </button>
            <button
              onClick={() => setActiveTab('graph')}
              style={{
                padding: '14px 24px',
                fontWeight: 'bold',
                fontSize: '15px',
                backgroundColor: activeTab === 'graph' ? '#3b82f6' : '#e5e7eb',
                color: activeTab === 'graph' ? 'white' : '#374151',
                border: 'none',
                cursor: 'pointer',
                transition: 'all 0.2s'
              }}
            >
              Graph
            </button>
          </div>

          {/* Content Area */}
          <div style={{flex: 1, padding: '24px', overflowY: 'auto'}}>
            
            {/* Logs Tab */}
            {activeTab === 'logs' && (
              <div style={{height: '100%'}}>
                {logs.length === 0 ? (
                  <div style={{display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: '#9ca3af'}}>
                    <div style={{textAlign: 'center'}}>
                      <AlertCircle size={64} style={{margin: '0 auto 16px', opacity: 0.4}} />
                      <p style={{fontSize: '18px'}}>No logs yet. Upload a dataset to begin.</p>
                    </div>
                  </div>
                ) : (
                  <div style={{fontFamily: 'monospace', fontSize: '14px'}}>
                    {logs.map((log, idx) => (
                      <div key={idx} style={{marginBottom: '8px'}}>
                        <span style={{color: '#6b7280'}}>[{log.time}]</span>{' '}
                        <span style={{
                          color: log.type === 'error' ? '#dc2626' : log.type === 'success' ? '#16a34a' : '#1f2937'
                        }}>{log.message}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {/* Results Tab */}
            {activeTab === 'results' && (
              <div>
                <h3 style={{fontSize: '24px', fontWeight: 'bold', marginBottom: '20px'}}>Model Performance</h3>
                {Object.keys(models).length === 0 ? (
                  <div style={{display: 'flex', alignItems: 'center', justifyContent: 'center', height: '400px', color: '#9ca3af'}}>
                    <div style={{textAlign: 'center'}}>
                      <Brain size={64} style={{margin: '0 auto 16px', opacity: 0.4}} />
                      <p style={{fontSize: '18px'}}>No models trained yet.</p>
                    </div>
                  </div>
                ) : (
                  <>
                    <div style={{display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '16px', marginBottom: '32px'}}>
                      {Object.entries(models).map(([name, data]) => (
                        <div key={name} style={{
                          background: 'linear-gradient(to bottom right, #dbeafe, #bfdbfe)',
                          borderRadius: '12px',
                          padding: '24px',
                          border: '2px solid #93c5fd',
                          boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
                        }}>
                          <h4 style={{fontWeight: 'bold', fontSize: '18px', marginBottom: '8px'}}>{name}</h4>
                          <p style={{fontSize: '40px', fontWeight: 'bold', color: '#2563eb', marginBottom: '4px'}}>{data.accuracy}%</p>
                          <p style={{fontSize: '14px', color: '#6b7280'}}>Accuracy</p>
                        </div>
                      ))}
                    </div>
                    {testResults.length > 0 && (
                      <div>
                        <h3 style={{fontSize: '20px', fontWeight: 'bold', marginBottom: '16px'}}>Test Predictions</h3>
                        <div style={{display: 'flex', flexDirection: 'column', gap: '12px'}}>
                          {testResults.map((result, idx) => (
                            <div key={idx} style={{
                              backgroundColor: '#f9fafb',
                              borderRadius: '8px',
                              padding: '16px',
                              border: '1px solid #e5e7eb'
                            }}>
                              <p style={{fontSize: '14px', marginBottom: '8px', color: '#374151'}}>{result.text}</p>
                              <span style={{
                                padding: '6px 12px',
                                borderRadius: '20px',
                                fontSize: '13px',
                                fontWeight: '600',
                                backgroundColor: result.predicted === 'Positive' ? '#dcfce7' : result.predicted === 'Negative' ? '#fee2e2' : '#fef3c7',
                                color: result.predicted === 'Positive' ? '#15803d' : result.predicted === 'Negative' ? '#991b1b' : '#854d0e'
                              }}>
                                {result.predicted}
                              </span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </>
                )}
              </div>
            )}

            {/* Graph Tab */}
            {activeTab === 'graph' && (
              <div>
                <h3 style={{fontSize: '24px', fontWeight: 'bold', marginBottom: '24px'}}>Accuracy Comparison Graph</h3>
                {chartData.length === 0 ? (
                  <div style={{display: 'flex', alignItems: 'center', justifyContent: 'center', height: '400px', color: '#9ca3af'}}>
                    <div style={{textAlign: 'center'}}>
                      <BarChart3 size={64} style={{margin: '0 auto 16px', opacity: 0.4}} />
                      <p style={{fontSize: '18px'}}>Train models to see comparison.</p>
                    </div>
                  </div>
                ) : (
                  <ResponsiveContainer width="100%" height={450}>
                    <BarChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="name" />
                      <YAxis domain={[0, 100]} label={{ value: 'Accuracy (%)', angle: -90, position: 'insideLeft' }} />
                      <Tooltip />
                      <Legend />
                      <Bar dataKey="accuracy" fill="#3B82F6" />
                    </BarChart>
                  </ResponsiveContainer>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}