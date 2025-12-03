import { Card } from '../components/ui/Card'

export default function Settings() {
  return (
    <div className="grid">
        <div className="span-12">
          <Card title="Settings" actions={<button className="btn primary">Save</button>} variant="neutral">
            <form className="grid">
              <div className="span-6">
                <label>API Base</label>
                <input type="text" placeholder="http://localhost:8000" />
              </div>
              <div className="span-6">
                <label>Broker URL</label>
                <input type="text" placeholder="redis://localhost:6379" />
              </div>
              <div className="span-6">
                <label>Storage (S3/MinIO)</label>
                <input type="text" placeholder="http://localhost:9000" />
              </div>
              <div className="span-6">
                <label>MLflow Enabled</label>
                <select><option>off</option><option>on</option></select>
              </div>
            </form>
          </Card>
        </div>
        <div className="span-12">
          <Card title="About" variant="teal">
            <p style={{ fontSize: 14, lineHeight: 1.7, color: 'rgba(255,255,255,.85)', margin: '0 0 12px' }}>FlowML Studio — Privacy-first AutoML on consumer hardware.</p>
            <p style={{ fontSize: 14, lineHeight: 1.7, color: 'rgba(255,255,255,.7)', margin: 0 }}>Final Year Project | Team FlowML101</p>
          </Card>
        </div>
    </div>
  )
}
