import { Sidebar } from './Sidebar'
import { Header } from './Header'
import './appshell.css'
import { PropsWithChildren, useEffect, useState } from 'react'

export function AppShell({ children }: PropsWithChildren) {
  const prefersDark = typeof window !== 'undefined' && window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches
  const [dark, setDark] = useState<boolean>(() => {
    const saved = typeof window !== 'undefined' ? localStorage.getItem('theme') : null
    if (saved === 'dark') return true
    if (saved === 'light') return false
    return prefersDark
  })

  useEffect(() => {
    localStorage.setItem('theme', dark ? 'dark' : 'light')
  }, [dark])

  return (
    <div className={dark ? 'app' : 'app theme-light'}>
      <Sidebar />
      <div className="content">
        <Header dark={dark} onToggleDark={() => setDark((v) => !v)} />
        <main className="main">{children}</main>
      </div>
    </div>
  )
}
