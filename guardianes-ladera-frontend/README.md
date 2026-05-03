# Guardianes de la Ladera Frontend

This frontend is the interactive dashboard for the Guardianes de la Ladera project. It is currently implemented as a Vite + React + TypeScript single-page application that visualizes landslide risk conditions by municipality and zone through a map, charts, operational indicators, and an analyst-oriented summary panel.

The frontend now loads its operational data from the FastAPI backend through a typed API layer. It keeps the existing dashboard behavior, but municipalities, zones, overlays, source freshness, latest-run metadata, and zone explanations are now fetched from backend endpoints instead of local browser-generated state. If an explanation is unavailable, the UI reports that directly instead of fabricating local fallback text.

The current frontend contract is also more permissive than the original prototype: zone types are backend-driven strings, zone exposure values may be `null`, and road-segment risk labels may contain source-provided text such as `Sin clasificar`.

## Current Scope

- Display municipalities, zones, roads, rainfall overlays, historical events, and UNGRD records.
- Show risk levels, trend changes, confidence, and explanatory drivers per zone.
- Read latest-run metadata and source freshness from the backend.
- Export dashboard content to PNG and export the analyst panel to PDF.
- Provide a map-centric operational view for risk managers.

## Technology Stack

- `React 19`
- `TypeScript`
- `Vite`
- `Leaflet` + `react-leaflet`
- `Recharts`
- `html2canvas`
- `jsPDF`
- `lucide-react`

## Current Architecture Summary

- Entry point: [`src/main.tsx`](./src/main.tsx)
- Page shell: [`src/App.tsx`](./src/App.tsx)
- Typed API client and DTO adapters: [`src/api.ts`](./src/api.ts)
- Data loading hooks: [`src/useDashboardApi.ts`](./src/useDashboardApi.ts)
- Dashboard view-model hook: [`src/useDashboardViewModel.ts`](./src/useDashboardViewModel.ts)
- On-demand export helpers: [`src/exporters.ts`](./src/exporters.ts)
- Legacy non-runtime fixture dataset: [`src/mockData.ts`](./src/mockData.ts)
- Global styles: [`src/index.css`](./src/index.css)
- Build configuration: [`vite.config.ts`](./vite.config.ts)

The application is currently implemented as a single-page dashboard without routing. `src/App.tsx` now focuses on page-shell rendering and action wiring, while selection/filter/countdown logic and derived dashboard selectors live in `src/useDashboardViewModel.ts`. The map and analyst panels are lazy-loaded, and export libraries are dynamically imported through `src/exporters.ts` so they no longer inflate the initial bundle. `src/mockData.ts` remains available as a legacy fixture file, but it is not a valid runtime source and the browser no longer synthesizes explanation content locally.

## Project Structure

```text
guardianes-ladera-frontend/
  docs/
  public/
  src/
    App.tsx
    api.ts
    exporters.ts
    index.css
    main.tsx
    mockData.ts
    useDashboardApi.ts
    useDashboardViewModel.ts
  index.html
  package.json
  tsconfig.json
  vite.config.ts
```

## Local Development

Install dependencies:

```bash
npm install
```

Start the backend first so the dashboard can reach the API:

```bash
cd ../guardianes-ladera-backend
uv run uvicorn app.main:app --reload
```

In the Docker workflow, remember that the backend API will not become healthy until `runtime-bootstrap` finishes importing the official structural bundle and completing the first official ingestion cycle.

Then start the frontend development server:

```bash
npm run dev
```

By default Vite proxies `/api/*` to `http://127.0.0.1:8000/*`. To point the frontend somewhere else, set `VITE_API_BASE_URL`.

Build the production bundle:

```bash
npm run build
```

Run the frontend tests:

```bash
npm test -- --run
```

Preview the production build locally:

```bash
npm run preview
```

## Current Limitations

- The dashboard now depends on the backend being available during local development unless `VITE_API_BASE_URL` is pointed at another running API.
- `App.tsx` is slimmer than before, but it is still the shell for a single large screen.
- There is no client-side router yet.
- Frontend tests are configured, but coverage is still narrow.
- The next-run countdown is an estimated window derived from the latest backend run timestamp.
- If the backend has no persisted run or no explanation for a selected zone, the frontend now surfaces that unavailable state instead of falling back to mock content.
- If the backend is still inside `runtime-bootstrap` or live provider ingestion is delayed, the frontend will wait on the API instead of inventing provisional dashboard data.

## Recommended Next Frontend Refactors

- Keep splitting the remaining page-shell responsibilities in `App.tsx` into smaller feature modules.
- Keep expanding the typed data-access layer around backend contracts.
- Add broader component and integration coverage for loading, partial-data, and export states.
