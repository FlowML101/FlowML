import { motion } from 'framer-motion'
import { Network, Shield, Zap, Database, Cpu } from 'lucide-react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'

const features = [
  {
    icon: Network,
    title: 'Distributed Power',
    description: 'Harness compute from multiple laptops simultaneously. Scale horizontally without cloud costs.',
    gradient: 'from-purple-600 to-purple-400',
  },
  {
    icon: Shield,
    title: 'Privacy Native',
    description: 'Your data never leaves your network. Train models on-premise with end-to-end encryption.',
    gradient: 'from-blue-600 to-blue-400',
  },
  {
    icon: Zap,
    title: 'AutoML Engine',
    description: 'Automatically discovers the best model architecture and hyperparameters for your dataset.',
    gradient: 'from-purple-600 to-blue-600',
  },
  {
    icon: Zap,
    title: 'Zero Latency',
    description: 'Real-time inference on your local LAN. No internet required, no API delays.',
    gradient: 'from-yellow-600 to-orange-600',
  },
  {
    icon: Database,
    title: 'Data Copilot',
    description: 'Ask natural language questions about your data. Get instant insights and visualizations.',
    gradient: 'from-purple-500 to-pink-500',
  },
  {
    icon: Cpu,
    title: 'GPU Accelerated',
    description: 'Automatic GPU detection and utilization across all worker nodes in your mesh.',
    gradient: 'from-blue-500 to-cyan-500',
  },
]

export function FeaturesGrid() {
  return (
    <section id="features" className="py-24 relative overflow-hidden">
      {/* Background */}
      <div className="absolute inset-0 bg-gradient-to-b from-zinc-950 via-zinc-900 to-zinc-950"></div>

      <div className="container mx-auto px-6 relative z-10">
        {/* Section Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          viewport={{ once: true }}
          className="text-center max-w-3xl mx-auto mb-16"
        >
          <h2 className="text-5xl font-bold mb-6">
            Built for the{' '}
            <span className="bg-gradient-to-r from-purple-400 to-blue-400 bg-clip-text text-transparent">
              Privacy Era
            </span>
          </h2>
          <p className="text-xl text-zinc-400">
            Everything you need to train production-ready models without touching the cloud.
          </p>
        </motion.div>

        {/* Features Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {features.map((feature, index) => (
            <motion.div
              key={feature.title}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
              viewport={{ once: true }}
            >
              <Card className="group hover:border-purple-600/50 transition-all duration-300 h-full bg-zinc-900/100 backdrop-blur-sm">
                <CardHeader>
                  <div className={`w-12 h-12 rounded-lg bg-gradient-to-br ${feature.gradient} flex items-center justify-center mb-4 group-hover:scale-110 transition-transform`}>
                    <feature.icon className="w-6 h-6 text-white" />
                  </div>
                  <CardTitle className="text-xl">{feature.title}</CardTitle>
                </CardHeader>
                <CardContent>
                  <CardDescription className="text-zinc-400 text-base">
                    {feature.description}
                  </CardDescription>
                </CardContent>
              </Card>
            </motion.div>
          ))}
        </div>

        {/* Bottom CTA */}
        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          transition={{ delay: 0.5 }}
          viewport={{ once: true }}
          className="mt-16 text-center"
        >
          <p className="text-zinc-500 text-sm">
            Trusted by data scientists who value privacy and performance
          </p>
        </motion.div>
      </div>
    </section>
  )
}
