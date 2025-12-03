import { useEffect } from 'react'
import { toast } from 'sonner'

// Mock system events that occur randomly for demo
const systemEvents = [
  {
    type: 'success',
    title: 'Dataset Uploaded',
    description: 'customer_churn.csv processed successfully (4,521 rows)',
  },
  {
    type: 'info',
    title: 'Worker Node Connected',
    description: 'worker-laptop-04 joined the cluster (VRAM: 6GB)',
  },
  {
    type: 'success',
    title: 'Model Training Complete',
    description: 'XGBoost achieved 94.2% accuracy',
  },
  {
    type: 'info',
    title: 'Resource Optimization',
    description: 'Workload redistributed to 3 active workers',
  },
  {
    type: 'error',
    title: 'Training Job Failed',
    description: 'Neural Network: OOM error on worker-02',
  },
  {
    type: 'info',
    title: 'Inference Request',
    description: 'Processed 47 predictions in 23ms',
  },
  {
    type: 'success',
    title: 'Model Exported',
    description: 'catboost_v1.pkl downloaded successfully',
  },
]

export function useSystemEvents() {
  useEffect(() => {
    // Show initial welcome toast
    toast.info('System Online', {
      description: 'FlowML Studio connected to distributed cluster',
    })

    // Set up interval for random events every 30 seconds
    const interval = setInterval(() => {
      const randomEvent = systemEvents[Math.floor(Math.random() * systemEvents.length)]
      
      switch (randomEvent.type) {
        case 'success':
          toast.success(randomEvent.title, {
            description: randomEvent.description,
          })
          break
        case 'info':
          toast.info(randomEvent.title, {
            description: randomEvent.description,
          })
          break
        case 'error':
          toast.error(randomEvent.title, {
            description: randomEvent.description,
          })
          break
      }
    }, 30000) // 30 seconds

    return () => clearInterval(interval)
  }, [])
}
