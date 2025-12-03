import { Card } from '../components/ui/Card'
import { Link } from 'react-router-dom'

export default function Dashboard() {
  const stats = [
    { title: 'Total Datasets', value: 0, trend: 'No datasets yet', variant: 'teal' as const, icon: '📊' },
    { title: 'Active Jobs', value: 0, trend: 'No active jobs', variant: 'blue' as const, icon: '⚡' },
    { title: 'Workers Online', value: 0, trend: 'No workers connected', variant: 'purple' as const, icon: '🖥️' },
    { title: 'Completed Runs', value: 0, trend: 'No completed runs', variant: 'neutral' as const, icon: '✓' },
  ]

  const recentActivity: any[] = []

  const activeWorkers: any[] = []

  const upcomingTasks: any[] = []

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 28 }}>
      {/* Stats Grid */}
      <div className="grid" style={{ gap: 20 }}>
        {stats.map((stat, i) => (
          <div key={i} className="span-3">
            <Card variant={stat.variant} decoration decorationSize="small">
              <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between' }}>
                <div>
                  <div style={{ fontSize: 13, opacity: 0.7, marginBottom: 8 }}>{stat.title}</div>
                  <div className="metric-large" style={{ fontSize: 36, fontWeight: 700, marginBottom: 4 }}>{stat.value}</div>
                  <div style={{ fontSize: 12, opacity: 0.6, display: 'flex', alignItems: 'center', gap: 4 }}>
                    <span style={{ color: 'var(--accent-teal)' }}>↗</span> {stat.trend}
                  </div>
                </div>
                <div style={{ fontSize: 32, opacity: 0.3 }}>{stat.icon}</div>
              </div>
            </Card>
          </div>
        ))}
      </div>

      {/* Main Content Grid */}
      <div className="grid" style={{ gap: 20 }}>
        {/* Recent Activity - 2/3 width */}
        <div className="span-8">
          <Card 
            title="Recent Activity" 
            actions={
              <Link to="/jobs" style={{ fontSize: 13, color: 'var(--accent-teal)', textDecoration: 'none' }}>
                View All →
              </Link>
            }
            variant="neutral"
          >
            <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
              {recentActivity.length === 0 ? (
                <div style={{ padding: '40px', textAlign: 'center', opacity: 0.5 }}>
                  <div style={{ fontSize: 48, marginBottom: 12 }}>📋</div>
                  <div style={{ fontSize: 14 }}>No recent activity</div>
                  <div style={{ fontSize: 12, marginTop: 4 }}>Jobs will appear here once started</div>
                </div>
              ) : recentActivity.map((job) => (
                <div 
                  key={job.id}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    padding: 12,
                    background: 'rgba(255,255,255,0.02)',
                    borderRadius: 8,
                    border: '1px solid rgba(255,255,255,0.05)',
                  }}
                >
                  <div style={{ flex: 1 }}>
                    <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 4 }}>{job.dataset}</div>
                    <div style={{ fontSize: 12, opacity: 0.6 }}>
                      {job.id} • {job.worker}
                    </div>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
                    <div style={{ minWidth: 100 }}>
                      <div style={{ fontSize: 11, opacity: 0.6, marginBottom: 4 }}>Progress</div>
                      <div style={{ 
                        height: 6, 
                        background: 'rgba(255,255,255,0.1)', 
                        borderRadius: 3,
                        overflow: 'hidden'
                      }}>
                        <div style={{ 
                          height: '100%', 
                          width: `${job.progress}%`,
                          background: job.status === 'Failed' ? 'var(--error)' : 
                                    job.status === 'Completed' ? 'var(--accent-teal)' : 
                                    'var(--accent-blue)',
                          transition: 'width 0.3s ease'
                        }} />
                      </div>
                    </div>
                    <div style={{ minWidth: 80, textAlign: 'right' }}>
                      <span style={{ 
                        fontSize: 11,
                        padding: '4px 8px',
                        borderRadius: 4,
                        background: job.status === 'Failed' ? 'rgba(239, 68, 68, 0.1)' : 
                                  job.status === 'Completed' ? 'rgba(16, 185, 129, 0.1)' : 
                                  'rgba(59, 130, 246, 0.1)',
                        color: job.status === 'Failed' ? '#EF4444' : 
                              job.status === 'Completed' ? '#10B981' : 
                              '#3B82F6',
                      }}>
                        {job.status}
                      </span>
                    </div>
                    {job.eta !== '—' && (
                      <div style={{ fontSize: 12, opacity: 0.6, minWidth: 40 }}>
                        ETA {job.eta}
                      </div>
                    )}
                  </div>
                </div>
              ))
            }
            </div>
          </Card>
        </div>

        {/* Upcoming Tasks - 1/3 width */}
        <div className="span-4">
          <Card 
            title="Upcoming Tasks" 
            actions={
              <button style={{ 
                fontSize: 13, 
                background: 'var(--accent-teal)', 
                color: 'white',
                border: 'none',
                padding: '6px 12px',
                borderRadius: 6,
                cursor: 'pointer',
                fontWeight: 500
              }}>
                + New
              </button>
            }
            variant="neutral"
          >
            <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
              {upcomingTasks.length === 0 ? (
                <div style={{ padding: '40px', textAlign: 'center', opacity: 0.5 }}>
                  <div style={{ fontSize: 48, marginBottom: 12 }}>📝</div>
                  <div style={{ fontSize: 14 }}>No upcoming tasks</div>
                  <div style={{ fontSize: 12, marginTop: 4 }}>Create tasks to get started</div>
                </div>
              ) : upcomingTasks.map((task, i) => (
                <div 
                  key={i}
                  style={{
                    display: 'flex',
                    gap: 12,
                    padding: 10,
                    background: 'rgba(255,255,255,0.02)',
                    borderRadius: 6,
                    borderLeft: `3px solid ${
                      task.priority === 'high' ? 'var(--error)' :
                      task.priority === 'medium' ? 'var(--accent-blue)' :
                      'var(--accent-purple)'
                    }`,
                  }}
                >
                  <div style={{ 
                    minWidth: 8, 
                    height: 8, 
                    borderRadius: '50%', 
                    background: task.priority === 'high' ? 'var(--error)' :
                               task.priority === 'medium' ? 'var(--accent-blue)' :
                               'var(--text-secondary)',
                    marginTop: 4,
                    opacity: 0.6
                  }} />
                  <div style={{ flex: 1 }}>
                    <div style={{ fontSize: 13, fontWeight: 500, marginBottom: 4 }}>{task.title}</div>
                    <div style={{ fontSize: 11, opacity: 0.6 }}>{task.dataset}</div>
                    <div style={{ fontSize: 11, opacity: 0.5, marginTop: 4 }}>⏰ {task.time}</div>
                  </div>
                </div>
              ))
            }
            </div>
          </Card>
        </div>
      </div>

      {/* Bottom Grid */}
      <div className="grid" style={{ gap: 20 }}>
        {/* Active Workers - Full Width */}
        <div className="span-12">
          <Card 
            title="Active Workers" 
            actions={
              <Link to="/workers" style={{ fontSize: 13, color: 'var(--accent-teal)', textDecoration: 'none' }}>
                Manage →
              </Link>
            }
            variant="teal"
            decoration
            decorationSize="small"
          >
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 18 }}>
              {activeWorkers.length === 0 ? (
                <div style={{ gridColumn: '1 / -1', padding: '60px', textAlign: 'center', opacity: 0.5 }}>
                  <div style={{ fontSize: 64, marginBottom: 16 }}>🖥️</div>
                  <div style={{ fontSize: 16, fontWeight: 600 }}>No workers connected</div>
                  <div style={{ fontSize: 13, marginTop: 8 }}>Start worker nodes to begin processing jobs</div>
                </div>
              ) : activeWorkers.map((worker) => (
                <div 
                  key={worker.id}
                  style={{
                    padding: '20px',
                    background: 'rgba(255,255,255,0.03)',
                    borderRadius: 12,
                    border: '1px solid rgba(255,255,255,0.08)',
                  }}
                >
                  {/* Header */}
                  <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16 }}>
                    <div style={{ 
                      width: 48, 
                      height: 48, 
                      borderRadius: 12, 
                      background: worker.status === 'busy' ? 
                        'linear-gradient(135deg, var(--accent-teal), var(--accent-blue))' :
                        'rgba(255,255,255,0.1)',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      fontSize: 20
                    }}>
                      🖥️
                    </div>
                    <div style={{ flex: 1 }}>
                      <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 4 }}>
                        {worker.name}
                      </div>
                      <span style={{ 
                        fontSize: 10,
                        padding: '3px 8px',
                        borderRadius: 4,
                        background: worker.status === 'busy' ? 'rgba(16, 185, 129, 0.2)' : 'rgba(156, 163, 175, 0.2)',
                        color: worker.status === 'busy' ? '#10B981' : '#9CA3AF',
                        textTransform: 'uppercase',
                        fontWeight: 600,
                        letterSpacing: '0.5px'
                      }}>
                        {worker.status}
                      </span>
                    </div>
                  </div>

                  {/* Current Job */}
                  <div style={{ 
                    fontSize: 12, 
                    opacity: 0.6, 
                    marginBottom: 16,
                    minHeight: 36,
                    display: 'flex',
                    alignItems: 'center'
                  }}>
                    {worker.currentJob || 'Waiting for jobs...'}
                  </div>

                  {/* Resource Stats */}
                  <div style={{ 
                    display: 'grid', 
                    gridTemplateColumns: 'repeat(3, 1fr)', 
                    gap: 12,
                    padding: '12px 0',
                    borderTop: '1px solid rgba(255,255,255,0.08)'
                  }}>
                    <div style={{ textAlign: 'center' }}>
                      <div style={{ fontSize: 10, opacity: 0.5, marginBottom: 4, textTransform: 'uppercase', letterSpacing: '0.5px' }}>CPU</div>
                      <div style={{ fontSize: 16, fontWeight: 700, color: worker.cpuUsage > 80 ? '#f59e0b' : 'inherit' }}>
                        {worker.cpuUsage}%
                      </div>
                    </div>
                    <div style={{ textAlign: 'center' }}>
                      <div style={{ fontSize: 10, opacity: 0.5, marginBottom: 4, textTransform: 'uppercase', letterSpacing: '0.5px' }}>RAM</div>
                      <div style={{ fontSize: 16, fontWeight: 700, color: worker.memUsage > 70 ? '#f59e0b' : 'inherit' }}>
                        {worker.memUsage}%
                      </div>
                    </div>
                    <div style={{ textAlign: 'center' }}>
                      <div style={{ fontSize: 10, opacity: 0.5, marginBottom: 4, textTransform: 'uppercase', letterSpacing: '0.5px' }}>GPU</div>
                      <div style={{ fontSize: 16, fontWeight: 700 }}>
                        {worker.status === 'busy' ? '67%' : '0%'}
                      </div>
                    </div>
                  </div>
                </div>
              ))
            }
            </div>
          </Card>
        </div>

        {/* System Resources - Left */}
        <div className="span-6">
          <Card title="System Resources" variant="blue" decoration decorationSize="small">
            <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
              {/* Collective Resource Usage */}
              <div style={{ 
                display: 'flex', 
                alignItems: 'center', 
                justifyContent: 'center',
                padding: '20px 0'
              }}>
                <div style={{ position: 'relative', width: 180, height: 180 }}>
                  {/* Background Circle */}
                  <svg width="180" height="180" style={{ transform: 'rotate(-90deg)' }}>
                    <circle
                      cx="90"
                      cy="90"
                      r="75"
                      fill="none"
                      stroke="rgba(255,255,255,0.1)"
                      strokeWidth="12"
                    />
                    {/* Progress Circle */}
                    <circle
                      cx="90"
                      cy="90"
                      r="75"
                      fill="none"
                      stroke="url(#gradient)"
                      strokeWidth="12"
                      strokeLinecap="round"
                      strokeDasharray={`${2 * Math.PI * 75 * 0} ${2 * Math.PI * 75}`}
                    />
                    <defs>
                      <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" stopColor="var(--accent-teal)" />
                        <stop offset="100%" stopColor="var(--accent-blue)" />
                      </linearGradient>
                    </defs>
                  </svg>
                  {/* Center Text */}
                  <div style={{
                    position: 'absolute',
                    top: '50%',
                    left: '50%',
                    transform: 'translate(-50%, -50%)',
                    textAlign: 'center'
                  }}>
                    <div style={{ fontSize: 36, fontWeight: 700, marginBottom: 4 }}>0%</div>
                    <div style={{ fontSize: 11, opacity: 0.6, textTransform: 'uppercase', letterSpacing: '0.5px' }}>Avg Usage</div>
                  </div>
                </div>
              </div>

              {/* Resource Breakdown */}
              <div style={{ 
                display: 'grid', 
                gridTemplateColumns: 'repeat(3, 1fr)', 
                gap: 16,
                padding: '16px',
                background: 'rgba(255,255,255,0.03)',
                borderRadius: 10,
                border: '1px solid rgba(255,255,255,0.06)'
              }}>
                <div style={{ textAlign: 'center' }}>
                  <div style={{ fontSize: 11, opacity: 0.5, marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.5px' }}>CPU</div>
                  <div style={{ fontSize: 22, fontWeight: 700 }}>0%</div>
                  <div style={{ fontSize: 10, opacity: 0.5, marginTop: 2 }}>—</div>
                </div>
                <div style={{ textAlign: 'center' }}>
                  <div style={{ fontSize: 11, opacity: 0.5, marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.5px' }}>RAM</div>
                  <div style={{ fontSize: 22, fontWeight: 700 }}>0%</div>
                  <div style={{ fontSize: 10, opacity: 0.5, marginTop: 2 }}>—</div>
                </div>
                <div style={{ textAlign: 'center' }}>
                  <div style={{ fontSize: 11, opacity: 0.5, marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.5px' }}>GPU</div>
                  <div style={{ fontSize: 22, fontWeight: 700 }}>0%</div>
                  <div style={{ fontSize: 10, opacity: 0.5, marginTop: 2 }}>—</div>
                </div>
              </div>
            </div>
          </Card>
        </div>

        {/* System Overview - Right */}
        <div className="span-6">
          <Card title="System Overview" variant="purple" decoration decorationSize="small">
            <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
              {/* Storage */}
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 10 }}>
                  <span style={{ fontSize: 13, fontWeight: 500 }}>Storage Usage</span>
                  <span style={{ fontSize: 13, fontWeight: 700 }}>0 / 0 GB</span>
                </div>
                <div style={{ 
                  height: 10, 
                  background: 'rgba(255,255,255,0.1)', 
                  borderRadius: 5,
                  overflow: 'hidden'
                }}>
                  <div style={{ 
                    height: '100%', 
                    width: '0%',
                    background: 'linear-gradient(90deg, var(--accent-purple), var(--accent-blue))',
                    borderRadius: 5
                  }} />
                </div>
                <div style={{ fontSize: 11, opacity: 0.5, marginTop: 6 }}>No storage configured</div>
              </div>

              {/* Queue */}
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 10 }}>
                  <span style={{ fontSize: 13, fontWeight: 500 }}>Queue Depth</span>
                  <span style={{ fontSize: 13, fontWeight: 700 }}>0 pending</span>
                </div>
                <div style={{ 
                  height: 10, 
                  background: 'rgba(255,255,255,0.1)', 
                  borderRadius: 5,
                  overflow: 'hidden'
                }}>
                  <div style={{ 
                    height: '100%', 
                    width: '0%',
                    background: 'var(--accent-teal)',
                    borderRadius: 5
                  }} />
                </div>
                <div style={{ fontSize: 11, opacity: 0.5, marginTop: 6 }}>No jobs in queue</div>
              </div>

              {/* Quick Stats */}
              <div style={{ 
                display: 'grid',
                gridTemplateColumns: '1fr 1fr',
                gap: 16
              }}>
                <div style={{ 
                  padding: 14,
                  background: 'rgba(255,255,255,0.03)',
                  borderRadius: 8,
                  border: '1px solid rgba(255,255,255,0.06)'
                }}>
                  <div style={{ fontSize: 11, opacity: 0.5, marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.5px' }}>Avg Training</div>
                  <div style={{ fontSize: 20, fontWeight: 700 }}>—</div>
                </div>
                <div style={{ 
                  padding: 14,
                  background: 'rgba(255,255,255,0.03)',
                  borderRadius: 8,
                  border: '1px solid rgba(255,255,255,0.06)'
                }}>
                  <div style={{ fontSize: 11, opacity: 0.5, marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.5px' }}>Success Rate</div>
                  <div style={{ fontSize: 20, fontWeight: 700 }}>—</div>
                </div>
                <div style={{ 
                  padding: 14,
                  background: 'rgba(255,255,255,0.03)',
                  borderRadius: 8,
                  border: '1px solid rgba(255,255,255,0.06)'
                }}>
                  <div style={{ fontSize: 11, opacity: 0.5, marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.5px' }}>Models Trained</div>
                  <div style={{ fontSize: 20, fontWeight: 700 }}>0</div>
                </div>
                <div style={{ 
                  padding: 14,
                  background: 'rgba(255,255,255,0.03)',
                  borderRadius: 8,
                  border: '1px solid rgba(255,255,255,0.06)'
                }}>
                  <div style={{ fontSize: 11, opacity: 0.5, marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.5px' }}>Best Accuracy</div>
                  <div style={{ fontSize: 20, fontWeight: 700 }}>—</div>
                </div>
              </div>
            </div>
          </Card>
        </div>
      </div>
    </div>
  )
}
