import { PropsWithChildren } from 'react'

type Props = PropsWithChildren<{ 
  title?: string
  actions?: React.ReactNode
  variant?: 'glass' | 'solid' | 'teal' | 'blue' | 'purple' | 'neutral'
  decoration?: boolean
  decorationSize?: 'normal' | 'small'
}>

export function Card({ title, actions, children, variant = 'glass', decoration = false, decorationSize = 'normal' }: Props) {
  const variantClass = variant === 'glass' ? 'card-glass' : variant === 'solid' ? 'card-solid' : `card-${variant}`
  return (
    <section className={`card ${variantClass}`}>
      {decoration && <div className={`card-decoration ${decorationSize === 'small' ? 'small' : ''}`} />}
      {(title || actions) && (
        <div className="card-header">
          <h3>{title}</h3>
          <div className="card-actions">{actions}</div>
        </div>
      )}
      <div className="card-body">{children}</div>
    </section>
  )
}
