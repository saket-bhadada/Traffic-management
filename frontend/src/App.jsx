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
          <div className="sys-id-badge">
            <span className="sys-dot" />
            CHOWK-04
          </div>
          <div>
            <h1>Adaptive Signal Control &amp; Preemption Console</h1>
            <p className="subtitle">
              YOLOv8s &middot; 2L-LSTM &middot; 5-State Deterministic Controller
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
            <Video size={15} /> Live Camera Feed
          </button>
          <button
            type="button"
            aria-current={route === '/simulation' ? 'page' : undefined}
            className={`nav-link ${route === '/simulation' ? 'active' : ''}`}
            onClick={() => navigate('/simulation')}
          >
            <GitBranch size={15} /> Signal Simulation
          </button>
        </nav>
      </header>

      <main>
        {route === '/simulation' ? <SimulationPage /> : <LiveDashboard />}
      </main>
    </div>
  );
}
