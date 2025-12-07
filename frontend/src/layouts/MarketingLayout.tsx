import { Link, Outlet } from 'react-router-dom'
import { motion } from 'framer-motion'
import { Rocket, Github, Twitter } from 'lucide-react'
import { Button } from '@/components/ui/button'

export function MarketingLayout() {
  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-50">
      {/* Navbar */}
      <motion.nav
        initial={{ y: -100 }}
        animate={{ y: 0 }}
        className="fixed top-0 left-0 right-0 z-50 glass-strong border-b border-zinc-800"
      >
        <div className="container mx-auto px-6 h-16 flex items-center justify-between">
          <Link to="/" className="flex items-center gap-2 font-bold text-xl">
            <div className="w-8 h-8 bg-gradient-to-br from-purple-600 to-blue-600 rounded-lg flex items-center justify-center">
              <Rocket className="w-5 h-5 text-white" />
            </div>
            <span className="bg-gradient-to-r from-purple-400 to-blue-400 bg-clip-text text-transparent">
              FlowML Studio
            </span>
          </Link>

          <div className="flex items-center gap-6">
            <Link to="/" className="text-sm hover:text-purple-400 transition-colors">
              Home
            </Link>
            <Link to="/#features" className="text-sm hover:text-purple-400 transition-colors">
              Features
            </Link>
            <a href="https://github.com" target="_blank" rel="noopener noreferrer" className="text-sm hover:text-purple-400 transition-colors">
              Docs
            </a>
            <Link to="/app">
              <Button className="bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700">
                Launch Studio
              </Button>
            </Link>
          </div>
        </div>
      </motion.nav>

      {/* Main Content */}
      <main className="pt-16">
        <Outlet />
      </main>

      {/* Footer */}
      <footer className="border-t border-zinc-800 bg-zinc-900/50">
        <div className="container mx-auto px-6 py-12">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
            <div>
              <div className="flex items-center gap-2 font-bold text-lg mb-4">
                <div className="w-6 h-6 bg-gradient-to-br from-purple-600 to-blue-600 rounded flex items-center justify-center">
                  <Rocket className="w-4 h-4 text-white" />
                </div>
                FlowML Studio
              </div>
              <p className="text-sm text-zinc-400">
                Transform idle laptops into a mesh supercomputer for training AI models.
              </p>
            </div>

            <div>
              <h3 className="font-semibold mb-4">Product</h3>
              <ul className="space-y-2 text-sm text-zinc-400">
                <li><Link to="/#features" className="hover:text-purple-400 transition-colors">Features</Link></li>
                <li><a href="#" className="hover:text-purple-400 transition-colors">Pricing</a></li>
                <li><a href="#" className="hover:text-purple-400 transition-colors">Documentation</a></li>
              </ul>
            </div>

            <div>
              <h3 className="font-semibold mb-4">Creators</h3>
              <ul className="space-y-2 text-sm text-zinc-400">
                <li><a href="#" className="hover:text-purple-400 transition-colors">About Us</a></li>
              </ul>
            </div>

            <div>
              <h3 className="font-semibold mb-4">Connect</h3>
              <div className="flex gap-4">
                <a href="#" className="w-9 h-9 rounded-lg bg-zinc-800 flex items-center justify-center hover:bg-purple-600 transition-colors">
                  <Github className="w-4 h-4" />
                </a>
                <a href="#" className="w-9 h-9 rounded-lg bg-zinc-800 flex items-center justify-center hover:bg-blue-600 transition-colors">
                  <Twitter className="w-4 h-4" />
                </a>
              </div>
            </div>
          </div>

          <div className="border-t border-zinc-800 mt-8 pt-8 text-center text-sm text-zinc-400">
            © 2025 FlowML Studio. Privacy-first AutoML for everyone.
          </div>
        </div>
      </footer>
    </div>
  )
}
