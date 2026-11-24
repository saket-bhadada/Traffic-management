import React, { useEffect, useState } from 'react';
import { GitBranch, Video } from 'lucide-react';
import LiveDashboard from './pages/LiveDashboard';
import SimulationPage from './pages/SimulationPage';
import './App.css';

function resolveRoute() {
  const path = window.location.pathname.replace(/\/+$/, '') || '/';
  const hash = window.location.hash;
  if (path === '/simulation' || hash === '#/simulation') {
    return '/simulation';
  }
  return '/';
}

export default function App() {
  const [route, setRoute] = useState(resolveRoute);

  useEffect(() => {
    const onPopState = () => setRoute(resolveRoute());
    window.addEventListener('popstate', onPopState);
    window.addEventListener('hashchange', onPopState);
    return () => {
      window.removeEventListener('popstate', onPopState);
      window.removeEventListener('hashchange', onPopState);
    };
  }, []);

  const navigate = (targetRoute) => {
    if (route === targetRoute) return;
    window.history.pushState({}, '', targetRoute);
    setRoute(targetRoute);
  };

  return (
    <div className="dashboard-container">
      <header className="header">
        <div className="header-left">
          <span className="sys-id-badge">CHOWK-04</span>
          <div>
            <h1>Traffic Signal Operations Console</h1>
            <p className="subtitle">
              4-Way Junction Control &middot; YOLOv8 Detection &middot; LSTM Queue Forecast
            </p>
          </div>
        </div>

        <nav className="top-nav" aria-label="Primary navigation">
          <button
            type="button"
            aria-current={route === '/' ? 'page' : undefined}
            className={`nav-link ${route === '/' ? 'active' : ''}`}
            onClick={() => navigate('/')}
          >
            <Video size={16} strokeWidth={1.75} /> Live Camera Feed
          </button>
          <button
            type="button"
            aria-current={route === '/simulation' ? 'page' : undefined}
            className={`nav-link ${route === '/simulation' ? 'active' : ''}`}
            onClick={() => navigate('/simulation')}
          >
            <GitBranch size={16} strokeWidth={1.75} /> Signal Simulation
          </button>
        </nav>
      </header>

      <main>
        {route === '/simulation' ? <SimulationPage /> : <LiveDashboard />}
      </main>
    </div>
  );
}
