import React, { useEffect, useState } from 'react';
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

  const headerLeft = (
    <div className="topbar-left">
      <span className="brand-title">Traffic Control</span>
      <nav className="top-nav" aria-label="Primary navigation">
        <button
          type="button"
          aria-current={route === '/' ? 'page' : undefined}
          className={`nav-tab ${route === '/' ? 'active' : ''}`}
          onClick={() => navigate('/')}
        >
          Live
        </button>
        <button
          type="button"
          aria-current={route === '/simulation' ? 'page' : undefined}
          className={`nav-tab ${route === '/simulation' ? 'active' : ''}`}
          onClick={() => navigate('/simulation')}
        >
          Simulation
        </button>
      </nav>
    </div>
  );

  return (
    <div className="app-frame">
      {route === '/simulation' ? (
        <SimulationPage headerLeft={headerLeft} />
      ) : (
        <LiveDashboard headerLeft={headerLeft} />
      )}
    </div>
  );
}
