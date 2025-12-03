import { Card } from '../components/ui/Card'

export default function Copilot() {
  return (
    <div className="grid">
        <div className="span-12">
          <Card title="Copilot" actions={<button className="btn primary">Generate</button>} variant="blue" decoration>
            <div className="grid">
              <div className="span-12">
                <label>Prompt</label>
                <textarea placeholder="e.g., Clean missing values in Age column" rows={6} style={{ width: '100%' }} />
              </div>
              <div className="span-12">
                <Card title="Preview Diff" variant="neutral">
                  <pre style={{ fontSize: 13, color: 'rgba(255,255,255,.7)', margin: 0 }}>// Polars code preview here</pre>
                </Card>
              </div>
              <div className="span-12" style={{ display: 'flex', gap: 8 }}>
                <button className="btn">Preview</button>
                <button className="btn primary">Apply</button>
              </div>
            </div>
          </Card>
        </div>
    </div>
  )
}
