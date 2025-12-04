import { Hero } from '@/features/marketing/Hero'
import { FeaturesGrid } from '@/features/marketing/FeaturesGrid'
import { CTA } from '@/features/marketing/CTA'
import { InfiniteTechGrid } from '@/components/ui/InfiniteTechGrid'

export function LandingPage() {
  return (
    <div className="relative">
      {/* Living Tech Grid Background */}
      <InfiniteTechGrid />
      
      {/* Content */}
      <div className="relative z-10">
        <Hero />
        <FeaturesGrid />
        <CTA />
      </div>
    </div>
  )
}
