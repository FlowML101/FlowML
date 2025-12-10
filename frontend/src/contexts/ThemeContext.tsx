import { createContext, useContext, useEffect, useState, ReactNode } from 'react'

type Theme = 'light' | 'dark'
type AccentColor = 'purple' | 'blue' | 'green' | 'orange'

interface ThemeContextType {
  theme: Theme
  accentColor: AccentColor
  toggleTheme: () => void
  setAccentColor: (color: AccentColor) => void
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined)

const accentColors = {
  purple: {
    primary: 'hsl(263.4 70% 50.4%)',
    primaryForeground: 'hsl(0 0% 100%)',
    hover: 'hsl(263.4 70% 45%)',
  },
  blue: {
    primary: 'hsl(217.2 91.2% 59.8%)',
    primaryForeground: 'hsl(0 0% 100%)',
    hover: 'hsl(217.2 91.2% 54%)',
  },
  green: {
    primary: 'hsl(142.1 76.2% 36.3%)',
    primaryForeground: 'hsl(0 0% 100%)',
    hover: 'hsl(142.1 76.2% 31%)',
  },
  orange: {
    primary: 'hsl(24.6 95% 53.1%)',
    primaryForeground: 'hsl(0 0% 100%)',
    hover: 'hsl(24.6 95% 48%)',
  },
}

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [theme, setTheme] = useState<Theme>(() => {
    const saved = localStorage.getItem('flowml-theme')
    return (saved as Theme) || 'dark'
  })

  const [accentColor, setAccentColorState] = useState<AccentColor>(() => {
    const saved = localStorage.getItem('flowml-accent')
    return (saved as AccentColor) || 'purple'
  })

  useEffect(() => {
    const root = document.documentElement
    
    // Apply theme
    root.classList.remove('light', 'dark')
    root.classList.add(theme)
    localStorage.setItem('flowml-theme', theme)

    // Apply accent color
    const colors = accentColors[accentColor]
    root.style.setProperty('--primary', colors.primary)
    root.style.setProperty('--primary-foreground', colors.primaryForeground)
    root.style.setProperty('--primary-hover', colors.hover)
    localStorage.setItem('flowml-accent', accentColor)
  }, [theme, accentColor])

  const toggleTheme = () => {
    setTheme(prev => prev === 'dark' ? 'light' : 'dark')
  }

  const setAccentColor = (color: AccentColor) => {
    setAccentColorState(color)
  }

  return (
    <ThemeContext.Provider value={{ theme, accentColor, toggleTheme, setAccentColor }}>
      {children}
    </ThemeContext.Provider>
  )
}

export function useTheme() {
  const context = useContext(ThemeContext)
  if (!context) {
    throw new Error('useTheme must be used within ThemeProvider')
  }
  return context
}
