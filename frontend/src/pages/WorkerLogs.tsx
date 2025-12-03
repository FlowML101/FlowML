import { useState } from 'react';
import { Card } from '../components/ui/Card';

export default function WorkerLogs() {
  const [selectedWorker, setSelectedWorker] = useState('worker-node-01');
  const [logLevel, setLogLevel] = useState('all');

  const workers = [
    { id: 'worker-node-01', name: 'Worker Node 01', status: 'active', jobs: 3 },
    { id: 'worker-node-02', name: 'Worker Node 02', status: 'active', jobs: 2 },
    { id: 'worker-node-03', name: 'Worker Node 03', status: 'idle', jobs: 0 },
    { id: 'worker-node-04', name: 'Worker Node 04', status: 'active', jobs: 1 }
  ];

  const logs = [
    {
      timestamp: '2025-12-03 14:23:45',
      level: 'info',
      worker: 'worker-node-01',
      job: 'job_abc123',
      message: 'Started training job for Customer Churn Prediction',
      details: 'Dataset: churn_data.csv (12,450 rows)'
    },
    {
      timestamp: '2025-12-03 14:24:12',
      level: 'info',
      worker: 'worker-node-01',
      job: 'job_abc123',
      message: 'Feature engineering completed',
      details: 'Generated 47 features from 12 original columns'
    },
    {
      timestamp: '2025-12-03 14:25:33',
      level: 'success',
      worker: 'worker-node-01',
      job: 'job_abc123',
      message: 'Model training epoch 10/100 completed',
      details: 'Accuracy: 82.4% | Loss: 0.3124 | Time: 45s'
    },
    {
      timestamp: '2025-12-03 14:18:22',
      level: 'warning',
      worker: 'worker-node-01',
      job: 'job_def456',
      message: 'High memory usage detected',
      details: 'RAM: 14.2GB / 16GB (88.75%) - Consider reducing batch size'
    },
    {
      timestamp: '2025-12-03 14:15:08',
      level: 'info',
      worker: 'worker-node-01',
      job: 'job_ghi789',
      message: 'Hyperparameter tuning initiated',
      details: 'Testing 48 parameter combinations using Optuna'
    },
    {
      timestamp: '2025-12-03 14:10:45',
      level: 'error',
      worker: 'worker-node-01',
      job: 'job_jkl012',
      message: 'Dataset validation failed',
      details: 'Missing values detected in 3 columns: age, income, tenure'
    },
    {
      timestamp: '2025-12-03 14:05:33',
      level: 'success',
      worker: 'worker-node-01',
      job: 'job_mno345',
      message: 'Model training completed successfully',
      details: 'Final accuracy: 94.2% | Total time: 2h 15m | Saved to storage'
    },
    {
      timestamp: '2025-12-03 14:02:18',
      level: 'info',
      worker: 'worker-node-01',
      job: null,
      message: 'Worker heartbeat sent to orchestrator',
      details: 'Status: Active | CPU: 45% | RAM: 8.2GB | GPU: 67%'
    }
  ];

  const filteredLogs = logs.filter(log => {
    if (selectedWorker !== 'all' && log.worker !== selectedWorker) return false;
    if (logLevel !== 'all' && log.level !== logLevel) return false;
    return true;
  });

  const getLevelColor = (level: string) => {
    switch (level) {
      case 'success': return '#10b981';
      case 'error': return '#ef4444';
      case 'warning': return '#f59e0b';
      case 'info': return '#3b82f6';
      default: return 'rgba(255,255,255,.5)';
    }
  };

  const getLevelBg = (level: string) => {
    switch (level) {
      case 'success': return 'rgba(16,185,129,.15)';
      case 'error': return 'rgba(239,68,68,.15)';
      case 'warning': return 'rgba(245,158,11,.15)';
      case 'info': return 'rgba(59,130,246,.15)';
      default: return 'rgba(255,255,255,.04)';
    }
  };

  return (
    <div className="page-content">
      <div className="page-header" style={{ marginBottom: '32px' }}>
        <div>
          <h1>Worker Logs</h1>
          <p className="page-subtitle">Complete history and activity logs from all workers</p>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '24px', marginBottom: '32px' }}>
        <Card variant="teal" decoration decorationSize="small">
          <div style={{ padding: '24px' }}>
            <div style={{ fontSize: '11px', letterSpacing: '1px', textTransform: 'uppercase', color: 'rgba(255,255,255,.55)', marginBottom: '12px' }}>Total Workers</div>
            <div className="metric-large" style={{ marginBottom: '8px' }}>{workers.length}</div>
            <div className="metric-subtitle">{workers.filter(w => w.status === 'active').length} active</div>
          </div>
        </Card>

        <Card variant="blue" decoration decorationSize="small">
          <div style={{ padding: '24px' }}>
            <div style={{ fontSize: '11px', letterSpacing: '1px', textTransform: 'uppercase', color: 'rgba(255,255,255,.55)', marginBottom: '12px' }}>Total Logs</div>
            <div className="metric-large" style={{ marginBottom: '8px' }}>{logs.length}</div>
            <div className="metric-subtitle">Last 24 hours</div>
          </div>
        </Card>

        <Card variant="purple" decoration decorationSize="small">
          <div style={{ padding: '24px' }}>
            <div style={{ fontSize: '11px', letterSpacing: '1px', textTransform: 'uppercase', color: 'rgba(255,255,255,.55)', marginBottom: '12px' }}>Errors</div>
            <div className="metric-large" style={{ marginBottom: '8px' }}>{logs.filter(l => l.level === 'error').length}</div>
            <div className="metric-subtitle">Requires attention</div>
          </div>
        </Card>
      </div>

      <Card variant="neutral" style={{ marginBottom: '32px' }}>
        <div className="card-header">
          <h3>Filter Options</h3>
          <button className="btn primary" style={{ padding: '8px 16px', fontSize: '12px' }}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
              <polyline points="7 10 12 15 17 10"/>
              <line x1="12" y1="15" x2="12" y2="3"/>
            </svg>
            Export Logs
          </button>
        </div>
        <div className="card-body" style={{ padding: '24px' }}>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
            <div>
              <label style={{ display: 'block', fontSize: '12px', color: 'rgba(255,255,255,.7)', marginBottom: '8px' }}>Worker Node</label>
              <select
                value={selectedWorker}
                onChange={(e) => setSelectedWorker(e.target.value)}
                style={{
                  width: '100%',
                  padding: '10px 12px',
                  background: 'rgba(255,255,255,.06)',
                  border: '1px solid rgba(255,255,255,.12)',
                  borderRadius: '6px',
                  color: 'var(--text)',
                  fontSize: '13px'
                }}
              >
                <option value="all">All Workers</option>
                {workers.map((worker) => (
                  <option key={worker.id} value={worker.id}>
                    {worker.name} ({worker.status})
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label style={{ display: 'block', fontSize: '12px', color: 'rgba(255,255,255,.7)', marginBottom: '8px' }}>Log Level</label>
              <select
                value={logLevel}
                onChange={(e) => setLogLevel(e.target.value)}
                style={{
                  width: '100%',
                  padding: '10px 12px',
                  background: 'rgba(255,255,255,.06)',
                  border: '1px solid rgba(255,255,255,.12)',
                  borderRadius: '6px',
                  color: 'var(--text)',
                  fontSize: '13px'
                }}
              >
                <option value="all">All Levels</option>
                <option value="info">Info</option>
                <option value="success">Success</option>
                <option value="warning">Warning</option>
                <option value="error">Error</option>
              </select>
            </div>
          </div>
        </div>
      </Card>

      <Card variant="glass">
        <div className="card-header">
          <h3>Activity Logs</h3>
          <span style={{ fontSize: '12px', color: 'rgba(255,255,255,.5)' }}>
            Showing {filteredLogs.length} of {logs.length} logs
          </span>
        </div>
        <div style={{ maxHeight: '600px', overflowY: 'auto' }}>
          {filteredLogs.map((log, index) => (
            <div
              key={index}
              style={{
                padding: '20px 24px',
                borderBottom: index < filteredLogs.length - 1 ? '1px solid rgba(255,255,255,.08)' : 'none',
                transition: 'background 0.2s ease'
              }}
              onMouseEnter={(e) => e.currentTarget.style.background = 'rgba(255,255,255,.02)'}
              onMouseLeave={(e) => e.currentTarget.style.background = 'transparent'}
            >
              <div style={{ display: 'flex', gap: '18px', alignItems: 'flex-start' }}>
                <div style={{ 
                  padding: '6px 10px',
                  background: getLevelBg(log.level),
                  border: `1px solid ${getLevelColor(log.level)}40`,
                  borderRadius: '6px',
                  fontSize: '10px',
                  fontWeight: 600,
                  color: getLevelColor(log.level),
                  textTransform: 'uppercase',
                  letterSpacing: '0.5px',
                  minWidth: '70px',
                  textAlign: 'center'
                }}>
                  {log.level}
                </div>

                <div style={{ flex: 1 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                    <div style={{ fontSize: '13px', fontWeight: 500, color: 'var(--text)' }}>{log.message}</div>
                    <div style={{ fontSize: '11px', color: 'rgba(255,255,255,.4)', whiteSpace: 'nowrap' }}>{log.timestamp}</div>
                  </div>

                  <div style={{ fontSize: '12px', color: 'rgba(255,255,255,.5)', marginBottom: '8px' }}>{log.details}</div>

                  <div style={{ display: 'flex', gap: '12px', fontSize: '11px', color: 'rgba(255,255,255,.4)' }}>
                    <span>Worker: {log.worker}</span>
                    {log.job && (
                      <>
                        <span>•</span>
                        <span>Job: {log.job}</span>
                      </>
                    )}
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}
