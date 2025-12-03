# FlowML Studio - Frontend

Production-grade React frontend for FlowML Studio, a privacy-first distributed AutoML platform.

## Tech Stack

- **Framework:** React 18 + TypeScript + Vite
- **UI Components:** ShadCN/UI (Radix UI + Tailwind CSS)
- **Styling:** Tailwind CSS (Dark Mode)
- **Icons:** Lucide React
- **Charts:** Recharts
- **Animations:** Framer Motion
- **Routing:** React Router DOM v6

## Project Structure

```
src/
├── layouts/
│   ├── MarketingLayout.tsx      # Navbar + Footer for landing page
│   └── DashboardLayout.tsx      # Sidebar + Topbar for app
├── features/
│   ├── marketing/               # Landing page components
│   │   ├── Hero.tsx
│   │   ├── FeaturesGrid.tsx
│   │   └── CTA.tsx
│   ├── dashboard/               # Dashboard widgets
│   │   ├── ClusterHealth.tsx
│   │   ├── ResourceGauges.tsx
│   │   └── ActiveJobs.tsx
│   ├── datasets/                # Data studio components
│   │   ├── UploadZone.tsx
│   │   ├── DataPreview.tsx
│   │   └── CopilotSidebar.tsx
│   └── jobs/                    # Training job components
├── pages/
│   ├── LandingPage.tsx          # Marketing site
│   ├── DashboardHome.tsx        # App home
│   ├── DataStudio.tsx           # Dataset management
│   ├── TrainingConfig.tsx       # Model training setup
│   ├── LiveMonitor.tsx          # Real-time training
│   ├── WorkersManager.tsx       # Cluster management
│   └── Results.tsx              # Model results & inference
├── components/ui/               # ShadCN primitives
└── App.tsx                      # Router configuration
```

## Getting Started

### Install Dependencies

```bash
npm install
```

### Development Server

```bash
npm run dev
```

Open [http://localhost:5173](http://localhost:5173)

### Build for Production

```bash
npm run build
```

### Preview Production Build

```bash
npm run preview
```

## Features

### Marketing Zone (/)
- Animated hero section with glassmorphism
- Feature cards with hover effects
- Call-to-action sections
- Responsive navbar and footer

### Dashboard Zone (/app)
- **Dashboard Home:** Cluster status, resource gauges, active jobs
- **Data Studio:** CSV upload, table preview, AI copilot
- **Training Config:** Target selection, time budget, compute mode
- **Live Monitor:** Real-time charts, terminal logs, progress tracking
- **Workers Manager:** Node table with status and actions
- **Results:** Model leaderboard, inference lab

## Design System

- **Colors:**
  - Background: Zinc-950
  - Foreground: Zinc-50
  - Primary: Purple-600
  - Secondary: Blue-600
- **Typography:** System font stack
- **Border Radius:** 0.5rem
- **Animations:** Framer Motion for page transitions

## Development Guidelines

### Feature-Based Architecture
- Each feature has its own folder
- Components are co-located with their feature
- Shared UI components in `/components/ui`

### State Management
- Currently using React hooks (useState)
- Ready for Context API or Zustand integration

### Type Safety
- Full TypeScript coverage
- Strict mode enabled
- Props interfaces for all components

## Mock Data

All pages use mock data for demonstration:
- Titanic dataset (891 rows)
- Training metrics and logs
- Worker node statistics
- Model performance metrics

## Browser Support

- Chrome/Edge (latest)
- Firefox (latest)
- Safari (latest)

## License

Proprietary - FlowML Studio

## Team

Built for distributed AutoML by the FlowML team.
