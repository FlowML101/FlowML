type Props = { dark: boolean; onToggleDark: () => void }

export function Header({ dark, onToggleDark }: Props) {
  return (
    <header className="header">
      <div className="title">FlowML Studio</div>
      <div className="actions">
        <button className="btn" onClick={onToggleDark}>{dark ? 'Light' : 'Dark'} Mode</button>
        <a className="btn primary" href="/jobs">New Job</a>
        <a className="btn" href="/datasets">Upload Dataset</a>
      </div>
    </header>
  )
}
