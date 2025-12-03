import { Card } from '../components/ui/Card'

export default function Dashboard() {
  return (
    <div className="grid">
      <div className="span-4">
        <Card title="Workers Online" variant="teal" decoration decorationSize="small">
          <div className="metric-large">0</div>
          <div className="metric-subtitle">0 (wire backend)</div>
        </Card>
      </div>
      <div className="span-4">
        <Card title="Jobs In Progress" variant="blue" decoration decorationSize="small">
          <div className="metric-large">0</div>
        </Card>
      </div>
      <div className="span-4">
        <Card title="Storage Usage" variant="purple" decoration decorationSize="small">
          <div className="metric-large">—</div>
        </Card>
      </div>
      <div className="span-8">
        <Card title="Recent Jobs" actions={<a className="btn" href="/jobs">View All</a>} variant="neutral">
          <table className="table">
            <thead>
              <tr><th>ID</th><th>Dataset</th><th>Status</th><th>Updated</th></tr>
            </thead>
            <tbody>
              <tr><td>—</td><td>—</td><td>—</td><td>—</td></tr>
            </tbody>
          </table>
        </Card>
      </div>
      <div className="span-4">
        <Card title="Alerts" variant="neutral">
          <ul style={{ listStyle: 'none', padding: 0, margin: 0 }}>
            <li style={{ fontSize: 13, color: 'rgba(255,255,255,.6)', padding: '8px 0', lineHeight: 1.6 }}>• No alerts. Connect broker/storage to populate.</li>
          </ul>
        </Card>
      </div>
    </div>
  )
}
