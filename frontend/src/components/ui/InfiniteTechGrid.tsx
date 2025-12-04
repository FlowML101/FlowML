import { useEffect, useState } from 'react'

// Tech Stack Icons as SVG components
const PythonIcon = ({ className }: { className?: string }) => (
  <svg viewBox="0 0 24 24" fill="currentColor" className={className}>
    <path d="M14.25.18l.9.2.73.26.59.3.45.32.34.34.25.34.16.33.1.3.04.26.02.2-.01.13V8.5l-.05.63-.13.55-.21.46-.26.38-.3.31-.33.25-.35.19-.35.14-.33.1-.3.07-.26.04-.21.02H8.77l-.69.05-.59.14-.5.22-.41.27-.33.32-.27.35-.2.36-.15.37-.1.35-.07.32-.04.27-.02.21v3.06H3.17l-.21-.03-.28-.07-.32-.12-.35-.18-.36-.26-.36-.36-.35-.46-.32-.59-.28-.73-.21-.88-.14-1.05-.05-1.23.06-1.22.16-1.04.24-.87.32-.71.36-.57.4-.44.42-.33.42-.24.4-.16.36-.1.32-.05.24-.01h.16l.06.01h8.16v-.83H6.18l-.01-2.75-.02-.37.05-.34.11-.31.17-.28.25-.26.31-.23.38-.2.44-.18.51-.15.58-.12.64-.1.71-.06.77-.04.84-.02 1.27.05zm-6.3 1.98l-.23.33-.08.41.08.41.23.34.33.22.41.09.41-.09.33-.22.23-.34.08-.41-.08-.41-.23-.33-.33-.22-.41-.09-.41.09zm13.09 3.95l.28.06.32.12.35.18.36.27.36.35.35.47.32.59.28.73.21.88.14 1.04.05 1.23-.06 1.23-.16 1.04-.24.86-.32.71-.36.57-.4.45-.42.33-.42.24-.4.16-.36.09-.32.05-.24.02-.16-.01h-8.22v.82h5.84l.01 2.76.02.36-.05.34-.11.31-.17.29-.25.25-.31.24-.38.2-.44.17-.51.15-.58.13-.64.09-.71.07-.77.04-.84.01-1.27-.04-1.07-.14-.9-.2-.73-.25-.59-.3-.45-.33-.34-.34-.25-.34-.16-.33-.1-.3-.04-.25-.02-.2.01-.13v-5.34l.05-.64.13-.54.21-.46.26-.38.3-.32.33-.24.35-.2.35-.14.33-.1.3-.06.26-.04.21-.02.13-.01h5.84l.69-.05.59-.14.5-.21.41-.28.33-.32.27-.35.2-.36.15-.36.1-.35.07-.32.04-.28.02-.21V6.07h2.09l.14.01.21.03zm-6.47 14.25l-.23.33-.08.41.08.41.23.33.33.23.41.08.41-.08.33-.23.23-.33.08-.41-.08-.41-.23-.33-.33-.23-.41-.08-.41.08z"/>
  </svg>
)

const DockerIcon = ({ className }: { className?: string }) => (
  <svg viewBox="0 0 24 24" fill="currentColor" className={className}>
    <path d="M13.983 11.078h2.119a.186.186 0 00.186-.185V9.006a.186.186 0 00-.186-.186h-2.119a.185.185 0 00-.185.185v1.888c0 .102.083.185.185.185m-2.954-5.43h2.118a.186.186 0 00.186-.186V3.574a.186.186 0 00-.186-.185h-2.118a.185.185 0 00-.185.185v1.888c0 .102.082.185.185.185m0 2.716h2.118a.187.187 0 00.186-.186V6.29a.186.186 0 00-.186-.185h-2.118a.185.185 0 00-.185.185v1.887c0 .102.082.186.185.186m-2.93 0h2.12a.186.186 0 00.184-.186V6.29a.185.185 0 00-.185-.185H8.1a.185.185 0 00-.185.185v1.887c0 .102.083.186.185.186m-2.964 0h2.119a.186.186 0 00.185-.186V6.29a.185.185 0 00-.185-.185H5.136a.186.186 0 00-.186.185v1.887c0 .102.084.186.186.186m5.893 2.715h2.118a.186.186 0 00.186-.185V9.006a.186.186 0 00-.186-.186h-2.118a.185.185 0 00-.185.185v1.888c0 .102.082.185.185.185m-2.93 0h2.12a.185.185 0 00.184-.185V9.006a.185.185 0 00-.184-.186h-2.12a.185.185 0 00-.184.185v1.888c0 .102.083.185.185.185m-2.964 0h2.119a.185.185 0 00.185-.185V9.006a.185.185 0 00-.184-.186h-2.12a.186.186 0 00-.186.186v1.887c0 .102.084.185.186.185m-2.92 0h2.12a.185.185 0 00.184-.185V9.006a.185.185 0 00-.184-.186h-2.12a.185.185 0 00-.184.185v1.888c0 .102.082.185.185.185M23.763 9.89c-.065-.051-.672-.51-1.954-.51-.338 0-.676.03-1.01.09-.248-1.827-1.419-2.69-2.896-3.138l-.485-.185-.308.437c-.401.607-.64 1.362-.7 2.165-.023.307-.01.615.052.918.09.438.26.85.502 1.215.366.558.975 1.023 1.756 1.345a7.4 7.4 0 01-1.238.088H.457a.44.44 0 00-.442.454c.003.302.03.605.094.907.308 1.648 1.022 3.11 2.125 4.349a7.6 7.6 0 003.744 2.257c.806.225 1.631.34 2.459.343 1.019.006 2.038-.132 3.027-.413 1.196-.34 2.326-.87 3.35-1.573a12.25 12.25 0 003.35-3.208 13.66 13.66 0 001.755-3.483h.233c1.135 0 1.843-.412 2.233-.801.296-.296.53-.66.68-1.06l.121-.326-.356-.244z"/>
  </svg>
)

const ReactIcon = ({ className }: { className?: string }) => (
  <svg viewBox="0 0 24 24" fill="currentColor" className={className}>
    <circle cx="12" cy="12" r="2" />
    <path d="M12,2C6.48,2,2,6.48,2,12s4.48,10,10,10s10-4.48,10-10S17.52,2,12,2z M12,4c1.93,0,3.68,0.78,4.95,2.05 C18.22,7.32,19,9.07,19,11c0,1.93-0.78,3.68-2.05,4.95C15.68,17.22,13.93,18,12,18s-3.68-0.78-4.95-2.05C5.78,14.68,5,12.93,5,11 c0-1.93,0.78-3.68,2.05-4.95C8.32,4.78,10.07,4,12,4z M12,6C9.79,6,8,7.79,8,10s1.79,4,4,4s4-1.79,4-4S14.21,6,12,6z"/>
    <ellipse cx="12" cy="12" rx="10" ry="4" fill="none" stroke="currentColor" strokeWidth="1.5" transform="rotate(60 12 12)" />
    <ellipse cx="12" cy="12" rx="10" ry="4" fill="none" stroke="currentColor" strokeWidth="1.5" transform="rotate(-60 12 12)" />
  </svg>
)

const ScikitIcon = ({ className }: { className?: string }) => (
  <svg viewBox="0 0 24 24" fill="currentColor" className={className}>
    <path d="M12 2L2 7v10l10 5 10-5V7L12 2zm0 2.2L19.5 7.5 12 11.3 4.5 7.5 12 4.2zm-8 4.1l7.5 3.8v7.4L4 15.7V8.3zm9.5 3.8l7.5-3.8v7.4l-7.5 3.8v-7.4z"/>
  </svg>
)

const RedisIcon = ({ className }: { className?: string }) => (
  <svg viewBox="0 0 24 24" fill="currentColor" className={className}>
    <path d="M10.5 2.661l-7.5 3v7l7.5 3.5 7.5-3.5v-7l-7.5-3zm0 1.732l5.268 2.268-5.268 2.268-5.268-2.268 5.268-2.268zm-6 3.732l6 2.732v5.268l-6-2.732v-5.268zm7.5 2.732l6-2.732v5.268l-6 2.732v-5.268z"/>
    <path d="M22 10.661l-7.5 3.5v2l7.5-3.5v-2zm-19.5 0v2l7.5 3.5v-2l-7.5-3.5z"/>
  </svg>
)

const FastAPIIcon = ({ className }: { className?: string }) => (
  <svg viewBox="0 0 24 24" fill="currentColor" className={className}>
    <path d="M12 1L3 5v6c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V5l-9-4zm0 2.18l7 3.12v5.2c0 4.52-3.13 8.75-7 9.88-3.87-1.13-7-5.36-7-9.88v-5.2l7-3.12z"/>
    <path d="M13.5 7L11 14h2l.5 3 2.5-7h-2l-.5-3z"/>
  </svg>
)

const TailwindIcon = ({ className }: { className?: string }) => (
  <svg viewBox="0 0 24 24" fill="currentColor" className={className}>
    <path d="M12 6c-2.67 0-4.33 1.33-5 4 1-1.33 2.17-1.83 3.5-1.5.76.19 1.31.74 1.91 1.35.98 1 2.09 2.15 4.59 2.15 2.67 0 4.33-1.33 5-4-1 1.33-2.17 1.83-3.5 1.5-.76-.19-1.3-.74-1.91-1.35C15.61 7.15 14.5 6 12 6m-5 6c-2.67 0-4.33 1.33-5 4 1-1.33 2.17-1.83 3.5-1.5.76.19 1.3.74 1.91 1.35C8.39 16.85 9.5 18 12 18c2.67 0 4.33-1.33 5-4-1 1.33-2.17 1.83-3.5 1.5-.76-.19-1.3-.74-1.91-1.35C10.61 13.15 9.5 12 7 12z"/>
  </svg>
)

// Tech logo configuration
const techLogos = [
  { Icon: PythonIcon, color: 'text-blue-400', name: 'Python' },
  { Icon: DockerIcon, color: 'text-blue-500', name: 'Docker' },
  { Icon: ReactIcon, color: 'text-cyan-400', name: 'React' },
  { Icon: ScikitIcon, color: 'text-orange-400', name: 'Scikit' },
  { Icon: RedisIcon, color: 'text-red-400', name: 'Redis' },
  { Icon: FastAPIIcon, color: 'text-green-400', name: 'FastAPI' },
  { Icon: TailwindIcon, color: 'text-cyan-500', name: 'Tailwind' },
]

interface FloatingLogo {
  id: number
  Icon: React.ComponentType<{ className?: string }>
  color: string
  x: number
  y: number
  delay: number
  duration: number
}

export function InfiniteTechGrid() {
  const [logos, setLogos] = useState<FloatingLogo[]>([])

  useEffect(() => {
    // Generate random positions for logos
    const generatedLogos: FloatingLogo[] = []
    const logoCount = 24 // Number of floating logos

    for (let i = 0; i < logoCount; i++) {
      const tech = techLogos[i % techLogos.length]
      generatedLogos.push({
        id: i,
        Icon: tech.Icon,
        color: tech.color,
        x: Math.random() * 100, // Random X position (0-100%)
        y: Math.random() * 200, // Random Y position (0-200% to cover scrolling area)
        delay: Math.random() * 20, // Random animation delay
        duration: 40 + Math.random() * 20, // Random duration (40-60s)
      })
    }

    setLogos(generatedLogos)
  }, [])

  return (
    <div className="absolute inset-0 overflow-hidden pointer-events-none">
      {/* Animated Grid Background */}
      <div
        className="absolute inset-0 opacity-30"
        style={{
          backgroundImage: `
            linear-gradient(to right, rgba(255, 255, 255, 0.05) 1px, transparent 1px),
            linear-gradient(to bottom, rgba(255, 255, 255, 0.05) 1px, transparent 1px)
          `,
          backgroundSize: '80px 80px',
          animation: 'gridScroll 30s linear infinite',
          willChange: 'transform',
        }}
      />

      {/* Floating Tech Logos */}
      <div className="absolute inset-0">
        {logos.map((logo) => (
          <div
            key={logo.id}
            className="absolute"
            style={{
              left: `${logo.x}%`,
              top: `${logo.y}%`,
              animation: `floatUp ${logo.duration}s linear infinite`,
              animationDelay: `-${logo.delay}s`,
              willChange: 'transform',
            }}
          >
            <logo.Icon className={`w-12 h-12 ${logo.color} opacity-20`} />
          </div>
        ))}
      </div>

      {/* Radial Fade Mask - Center Clear */}
      <div
        className="absolute inset-0"
        style={{
          background: `
            radial-gradient(ellipse 100% 80% at 50% 45%, transparent 0%, transparent 40%, rgba(9, 9, 11, 0.4) 70%, rgba(9, 9, 11, 0.9) 100%)
          `,
        }}
      />

      {/* Top Fade */}
      <div
        className="absolute top-0 left-0 right-0 h-32"
        style={{
          background: 'linear-gradient(to bottom, rgba(9, 9, 11, 1) 0%, transparent 100%)',
        }}
      />

      {/* Bottom Fade */}
      <div
        className="absolute bottom-0 left-0 right-0 h-48"
        style={{
          background: 'linear-gradient(to top, rgba(9, 9, 11, 1) 0%, transparent 100%)',
        }}
      />

      {/* CSS Animations */}
      <style>{`
        @keyframes gridScroll {
          0% {
            transform: translateY(0);
          }
          100% {
            transform: translateY(80px);
          }
        }

        @keyframes floatUp {
          0% {
            transform: translateY(0) rotate(0deg);
            opacity: 0;
          }
          5% {
            opacity: 0.2;
          }
          95% {
            opacity: 0.2;
          }
          100% {
            transform: translateY(-100vh) rotate(360deg);
            opacity: 0;
          }
        }

        @media (prefers-reduced-motion: reduce) {
          * {
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.01ms !important;
          }
        }
      `}</style>
    </div>
  )
}
