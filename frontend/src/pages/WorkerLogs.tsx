import { useState } from 'react';
import { Card } from '../components/ui/Card';

export default function WorkerLogs() {
  const [selectedWorker, setSelectedWorker] = useState('worker-node-01');
  const [logLevel, setLogLevel] = useState('all');

  const workers = [
    { id: 'worker-node-01', name: 'Worker Node 01', status: 'idle', jobs: 0 },
  ];

  const logs: any[] = [];

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

      <div className="grid" style={{ gap: 24, marginBottom: 32 }}>
        <div className="span-12">
          <Card variant="neutral">
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
        </div>
      </div>

      <div className="grid" style={{ gap: 24 }}>
        <div className="span-12">
          <Card variant="glass">
        <div className="card-header">
          <h3>Activity Logs</h3>
          <span style={{ fontSize: '12px', color: 'rgba(255,255,255,.5)' }}>
            Showing {filteredLogs.length} of {logs.length} logs
          </span>
        </div>
        <div style={{ maxHeight: '600px', overflowY: 'auto' }}>
          {filteredLogs.length === 0 ? (
            <div style={{ padding: '80px 40px', textAlign: 'center', opacity: 0.5 }}>
              <div style={{ fontSize: 64, marginBottom: 16 }}>📋</div>
              <div style={{ fontSize: 16, fontWeight: 600 }}>No logs found</div>
              <div style={{ fontSize: 13, marginTop: 8 }}>Logs will appear here when workers start processing jobs</div>
            </div>
          ) : filteredLogs.map((log, index) => (
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
          ))
        }
        </div>
      </Card>
        </div>
      </div>
    </div>
  );
}
