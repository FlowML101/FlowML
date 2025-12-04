import { Card } from '../components/ui/Card'

export default function Runs() {
  return (
    <div className="page-content">
      <div className="page-header" style={{ marginBottom: '32px' }}>
        <div>
          <h1>Runs</h1>
          <p className="page-subtitle">View training runs and artifacts</p>
        </div>
      </div>

      <div className="grid">
        <div className="span-12">
          <Card title="Runs & Artifacts" actions={<button className="btn">Refresh</button>} variant="purple" decoration>
            <table className="table">
              <thead>
                <tr><th>Run ID</th><th>Model</th><th>Metric</th><th>Value</th><th>Actions</th></tr>
              </thead>
              <tbody>
                <tr><td>—</td><td>—</td><td>—</td><td>—</td><td><a className="btn" href="#">Download</a></td></tr>
              </tbody>
            </table>
          </Card>
        </div>
      </div>
    </div>
  )
}
