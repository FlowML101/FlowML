import { NavLink } from 'react-router-dom'

const linkClass = ({ isActive }: { isActive: boolean }) =>
  isActive ? 'active' : ''

function Icon({ name }: { name: 'dashboard'|'datasets'|'jobs'|'workers'|'runs'|'monitor'|'modellab'|'workerlogs'|'copilot'|'settings' }) {
  const common = { width: 18, height: 18, fill: 'none', stroke: 'currentColor', strokeWidth: 1.6 } as const
  switch (name) {
    case 'dashboard': return (<svg {...common} viewBox="0 0 24 24"><path d="M4 13h7v7H4z"/><path d="M13 4h7v7h-7z"/><path d="M13 13h7v7h-7z"/><path d="M4 4h7v7H4z"/></svg>)
    case 'datasets': return (<svg {...common} viewBox="0 0 24 24"><ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v6c0 1.7 3.6 3 8 3s8-1.3 8-3V5"/></svg>)
    case 'jobs': return (<svg {...common} viewBox="0 0 24 24"><path d="M3 7h18v12H3z"/><path d="M9 7V5h6v2"/></svg>)
    case 'workers': return (<svg {...common} viewBox="0 0 24 24"><circle cx="7" cy="8" r="3"/><circle cx="17" cy="8" r="3"/><path d="M2 20c0-3 3-5 5-5h0"/><path d="M17 15c2 0 5 2 5 5"/></svg>)
    case 'runs': return (<svg {...common} viewBox="0 0 24 24"><path d="M4 12h16"/><path d="M12 4v16"/></svg>)
    case 'monitor': return (<svg {...common} viewBox="0 0 24 24"><path d="M3 3v18h18"/><path d="M18 17l-3-3-4 4-4-4-4 4"/></svg>)
    case 'modellab': return (<svg {...common} viewBox="0 0 24 24"><path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/></svg>)
    case 'workerlogs': return (<svg {...common} viewBox="0 0 24 24"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/></svg>)
    case 'copilot': return (<svg {...common} viewBox="0 0 24 24"><path d="M12 3l4 7H8l4-7z"/><path d="M4 20h16l-4-7H8z"/></svg>)
    case 'settings': return (<svg {...common} viewBox="0 0 24 24"><path d="M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8z"/><path d="M3 12h3"/><path d="M18 12h3"/><path d="M12 3v3"/><path d="M12 18v3"/></svg>)
  }
}

export function Sidebar() {
  return (
    <aside className="sidebar">
      <div className="logo">FlowML</div>
      <nav>
        <NavLink to="/dashboard" className={linkClass}><span style={{ display:'inline-flex', gap:8, alignItems:'center' }}><Icon name="dashboard"/>Dashboard</span></NavLink>
        <NavLink to="/datasets" className={linkClass}><span style={{ display:'inline-flex', gap:8, alignItems:'center' }}><Icon name="datasets"/>Datasets</span></NavLink>
        <NavLink to="/jobs" className={linkClass}><span style={{ display:'inline-flex', gap:8, alignItems:'center' }}><Icon name="jobs"/>Jobs</span></NavLink>
        <NavLink to="/workers" className={linkClass}><span style={{ display:'inline-flex', gap:8, alignItems:'center' }}><Icon name="workers"/>Workers</span></NavLink>
        <NavLink to="/runs" className={linkClass}><span style={{ display:'inline-flex', gap:8, alignItems:'center' }}><Icon name="runs"/>Runs</span></NavLink>
        <NavLink to="/monitor" className={linkClass}><span style={{ display:'inline-flex', gap:8, alignItems:'center' }}><Icon name="monitor"/>Live Monitor</span></NavLink>
        <NavLink to="/model-lab" className={linkClass}><span style={{ display:'inline-flex', gap:8, alignItems:'center' }}><Icon name="modellab"/>Model Lab</span></NavLink>
        <NavLink to="/worker-logs" className={linkClass}><span style={{ display:'inline-flex', gap:8, alignItems:'center' }}><Icon name="workerlogs"/>Worker Logs</span></NavLink>
        <NavLink to="/copilot" className={linkClass}><span style={{ display:'inline-flex', gap:8, alignItems:'center' }}><Icon name="copilot"/>Copilot</span></NavLink>
        <NavLink to="/settings" className={linkClass}><span style={{ display:'inline-flex', gap:8, alignItems:'center' }}><Icon name="settings"/>Settings</span></NavLink>
      </nav>
    </aside>
  )
}
