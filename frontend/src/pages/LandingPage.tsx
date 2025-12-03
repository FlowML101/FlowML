import { Hero } from '@/features/marketing/Hero'
import { FeaturesGrid } from '@/features/marketing/FeaturesGrid'
import { CTA } from '@/features/marketing/CTA'

export function LandingPage() {
  return (
    <div>
      <Hero />
      <FeaturesGrid />
      <CTA />
    </div>
  )
}
