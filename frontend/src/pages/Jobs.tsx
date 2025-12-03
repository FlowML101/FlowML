import { Card } from '../components/ui/Card'

export default function Jobs() {
  return (
    <div className="grid">
        <div className="span-12">
          <Card title="New Job" actions={<button className="btn primary">Start</button>} variant="purple" decoration>
            <form className="grid">
              <div className="span-6">
                <label>Dataset</label>
                <select><option>—</option></select>
              </div>
              <div className="span-6">
                <label>Target Column</label>
                <input type="text" placeholder="e.g. label" />
              </div>
              <div className="span-6">
                <label>Task Type</label>
                <select><option>classification</option><option>regression</option></select>
              </div>
              <div className="span-6">
                <label>Constraints</label>
                <input type="text" placeholder="min_vram_gb=6, cpu_only=false" />
              </div>
              <div className="span-6">
                <label>HPO Budget</label>
                <input type="text" placeholder="trials=50 or time=30m" />
              </div>
              <div className="span-6">
                <label>Priority</label>
                <select><option>normal</option><option>high</option></select>
              </div>
            </form>
          </Card>
        </div>
        <div className="span-12">
          <Card title="Jobs" actions={<button className="btn">Refresh</button>} variant="blue">
            <table className="table">
              <thead>
                <tr><th>ID</th><th>Dataset</th><th>Status</th><th>Queue</th><th>Actions</th></tr>
              </thead>
              <tbody>
                <tr><td>—</td><td>—</td><td>—</td><td>—</td><td><button className="btn">Cancel</button></td></tr>
              </tbody>
            </table>
          </Card>
        </div>
    </div>
  )
}
