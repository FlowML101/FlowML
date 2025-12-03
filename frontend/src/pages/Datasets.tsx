import { Card } from '../components/ui/Card'

export default function Datasets() {
  return (
    <div className="grid">
        <div className="span-12">
          <Card title="Upload Dataset" actions={<button className="btn primary">Upload</button>} variant="teal" decoration>
            <form className="grid">
              <div className="span-6">
                <label>Name</label>
                <input type="text" placeholder="dataset name" />
              </div>
              <div className="span-6">
                <label>Tags</label>
                <input type="text" placeholder="comma,separated" />
              </div>
              <div className="span-12">
                <input type="file" accept=".csv,.parquet" />
              </div>
            </form>
          </Card>
        </div>
        <div className="span-12">
          <Card title="Datasets" actions={<button className="btn">Refresh</button>} variant="blue">
            <table className="table">
              <thead>
                <tr><th>Name</th><th>Version</th><th>Rows</th><th>Columns</th><th>Actions</th></tr>
              </thead>
              <tbody>
                <tr><td>—</td><td>—</td><td>—</td><td>—</td><td><a className="btn" href="#">View</a></td></tr>
              </tbody>
            </table>
          </Card>
        </div>
    </div>
  )
}
