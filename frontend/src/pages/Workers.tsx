import { Card } from '../components/ui/Card'

export default function Workers() {
  return (
    <div className="grid">
        <div className="span-12">
          <Card title="Workers" actions={<button className="btn">Refresh</button>} variant="teal" decoration>
            <table className="table">
              <thead>
                <tr><th>Name</th><th>CPU</th><th>RAM</th><th>GPU</th><th>VRAM</th><th>Queues</th><th>Heartbeat</th><th>Actions</th></tr>
              </thead>
              <tbody>
                <tr><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td><button className="btn">Drain</button></td></tr>
              </tbody>
            </table>
          </Card>
        </div>
    </div>
  )
}
