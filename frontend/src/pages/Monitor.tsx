import { useState } from 'react';
import { Card } from '../components/ui/Card';

export default function Monitor() {
  const [activeJobs] = useState([
    {
      id: 'job_abc123',
      name: 'Customer Churn Prediction',
      worker: 'worker-node-01',
      dataset: 'churn_data.csv',
      progress: 67,
      stage: 'Feature Engineering',
      epoch: 45,
      totalEpochs: 100,
      accuracy: 0.8421,
      loss: 0.3124,
      eta: '8m 23s',
      status: 'training'
    },
    {
      id: 'job_def456',
      name: 'Fraud Detection Model',
      worker: 'worker-node-02',
      dataset: 'transactions.parquet',
      progress: 34,
      stage: 'Hyperparameter Tuning',
      epoch: 12,
      totalEpochs: 50,
      accuracy: 0.7689,
      loss: 0.4521,
      eta: '15m 42s',
      status: 'training'
    },
    {
      id: 'job_ghi789',
      name: 'Sales Forecasting',
      worker: 'worker-node-03',
      dataset: 'sales_history.csv',
      progress: 89,
      stage: 'Model Validation',
      epoch: 178,
      totalEpochs: 200,
      accuracy: 0.9234,
      loss: 0.1456,
      eta: '2m 11s',
      status: 'training'
    }
  ]);

  return (
    <div className="page-content">
      <div className="page-header" style={{ marginBottom: '32px' }}>
        <div>
          <h1>Live Monitor</h1>
          <p className="page-subtitle">Real-time training progress and metrics</p>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '24px', marginBottom: '32px' }}>
        <Card variant="teal" decoration decorationSize="small">
          <div style={{ padding: '24px' }}>
            <div style={{ fontSize: '11px', letterSpacing: '1px', textTransform: 'uppercase', color: 'rgba(255,255,255,.55)', marginBottom: '12px' }}>Active Jobs</div>
            <div className="metric-large" style={{ marginBottom: '8px' }}>{activeJobs.length}</div>
            <div className="metric-subtitle">Currently training</div>
          </div>
        </Card>

        <Card variant="blue" decoration decorationSize="small">
          <div style={{ padding: '24px' }}>
            <div style={{ fontSize: '11px', letterSpacing: '1px', textTransform: 'uppercase', color: 'rgba(255,255,255,.55)', marginBottom: '12px' }}>Avg Progress</div>
            <div className="metric-large" style={{ marginBottom: '8px' }}>{Math.round(activeJobs.reduce((acc, job) => acc + job.progress, 0) / activeJobs.length)}%</div>
            <div className="metric-subtitle">Across all jobs</div>
          </div>
        </Card>

        <Card variant="purple" decoration decorationSize="small">
          <div style={{ padding: '24px' }}>
            <div style={{ fontSize: '11px', letterSpacing: '1px', textTransform: 'uppercase', color: 'rgba(255,255,255,.55)', marginBottom: '12px' }}>Best Accuracy</div>
            <div className="metric-large" style={{ marginBottom: '8px' }}>{(Math.max(...activeJobs.map(j => j.accuracy)) * 100).toFixed(1)}%</div>
            <div className="metric-subtitle">Highest performing model</div>
          </div>
        </Card>
      </div>

      <div style={{ display: 'grid', gap: '24px' }}>
        {activeJobs.map((job) => (
          <Card key={job.id} variant="neutral">
            <div className="card-header">
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                <div style={{ 
                  width: '8px', 
                  height: '8px', 
                  borderRadius: '50%', 
                  background: 'linear-gradient(135deg, #10b981 0%, #059669 100%)',
                  boxShadow: '0 0 12px rgba(16,185,129,.5)'
                }}/>
                <h3 style={{ fontSize: '14px', fontWeight: 600, color: 'var(--text)', textTransform: 'none', letterSpacing: '0' }}>{job.name}</h3>
              </div>
              <span style={{ fontSize: '12px', color: 'rgba(255,255,255,.5)' }}>{job.id}</span>
            </div>
            
            <div className="card-body" style={{ padding: '24px' }}>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px', marginBottom: '28px' }}>
                <div>
                  <div style={{ fontSize: '11px', color: 'rgba(255,255,255,.5)', marginBottom: '6px' }}>Worker Node</div>
                  <div style={{ fontSize: '13px', color: 'var(--text)' }}>{job.worker}</div>
                </div>
                <div>
                  <div style={{ fontSize: '11px', color: 'rgba(255,255,255,.5)', marginBottom: '6px' }}>Dataset</div>
                  <div style={{ fontSize: '13px', color: 'var(--text)' }}>{job.dataset}</div>
                </div>
                <div>
                  <div style={{ fontSize: '11px', color: 'rgba(255,255,255,.5)', marginBottom: '6px' }}>Current Stage</div>
                  <div style={{ fontSize: '13px', color: 'var(--text)' }}>{job.stage}</div>
                </div>
                <div>
                  <div style={{ fontSize: '11px', color: 'rgba(255,255,255,.5)', marginBottom: '6px' }}>ETA</div>
                  <div style={{ fontSize: '13px', color: 'var(--text)' }}>{job.eta}</div>
                </div>
              </div>

              <div style={{ marginBottom: '28px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '10px' }}>
                  <span style={{ fontSize: '12px', color: 'rgba(255,255,255,.7)' }}>Overall Progress</span>
                  <span style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text)' }}>{job.progress}%</span>
                </div>
                <div style={{ 
                  height: '8px', 
                  background: 'rgba(255,255,255,.08)', 
                  borderRadius: '4px', 
                  overflow: 'hidden',
                  border: '1px solid rgba(255,255,255,.1)'
                }}>
                  <div style={{ 
                    height: '100%', 
                    width: `${job.progress}%`,
                    background: 'linear-gradient(90deg, #5c7cfa 0%, #4c6ef5 100%)',
                    boxShadow: '0 0 8px rgba(92,124,250,.4)',
                    transition: 'width 0.3s ease'
                  }}/>
                </div>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '16px' }}>
                <div style={{ 
                  padding: '12px', 
                  background: 'rgba(255,255,255,.04)', 
                  borderRadius: '8px',
                  border: '1px solid rgba(255,255,255,.08)'
                }}>
                  <div style={{ fontSize: '10px', color: 'rgba(255,255,255,.5)', marginBottom: '4px' }}>Epoch</div>
                  <div style={{ fontSize: '16px', fontWeight: 600, color: 'var(--text)' }}>{job.epoch}/{job.totalEpochs}</div>
                </div>
                <div style={{ 
                  padding: '12px', 
                  background: 'rgba(255,255,255,.04)', 
                  borderRadius: '8px',
                  border: '1px solid rgba(255,255,255,.08)'
                }}>
                  <div style={{ fontSize: '10px', color: 'rgba(255,255,255,.5)', marginBottom: '4px' }}>Accuracy</div>
                  <div style={{ fontSize: '16px', fontWeight: 600, color: '#10b981' }}>{(job.accuracy * 100).toFixed(2)}%</div>
                </div>
                <div style={{ 
                  padding: '12px', 
                  background: 'rgba(255,255,255,.04)', 
                  borderRadius: '8px',
                  border: '1px solid rgba(255,255,255,.08)'
                }}>
                  <div style={{ fontSize: '10px', color: 'rgba(255,255,255,.5)', marginBottom: '4px' }}>Loss</div>
                  <div style={{ fontSize: '16px', fontWeight: 600, color: '#f59e0b' }}>{job.loss.toFixed(4)}</div>
                </div>
                <div style={{ 
                  padding: '12px', 
                  background: 'rgba(255,255,255,.04)', 
                  borderRadius: '8px',
                  border: '1px solid rgba(255,255,255,.08)'
                }}>
                  <div style={{ fontSize: '10px', color: 'rgba(255,255,255,.5)', marginBottom: '4px' }}>Status</div>
                  <div style={{ fontSize: '13px', fontWeight: 500, color: '#5c7cfa', textTransform: 'capitalize' }}>{job.status}</div>
                </div>
              </div>
            </div>
          </Card>
        ))}
      </div>
    </div>
  );
}
