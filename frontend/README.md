# FlowML Frontend

Professional dashboard-first UI built with React + TypeScript.

## Features
- Dashboard as home with system cards and quick actions
- Datasets, Jobs, Workers, Runs, Copilot, Settings routes
- Reusable components: AppShell, Sidebar, Header, Cards, Tables, Forms
- Theming via CSS variables; responsive layout; dark mode toggle

## Getting Started (Windows PowerShell)

```powershell
# Install dependencies
npm install

# Start dev server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

## Structure
- `src/App.tsx`: Router and layout
- `src/components/`: UI components
- `src/pages/`: Page views
- `src/styles/`: Global styles and theme
- `src/lib/api.ts`: API client stubs (wire to FastAPI later)

## Wiring Backend
Set `VITE_API_BASE` in `.env` to your orchestrator URL (e.g., `http://localhost:8000`).
