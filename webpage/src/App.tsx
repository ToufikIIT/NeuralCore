import { NavLink, Outlet } from 'react-router-dom'
import './App.css'

function App() {
  return (
    <div className="site-shell">
      <header className="topbar">
        <div className="brand-block">
          <NavLink className="brand-pill" to="/">
            NeuralCore
          </NavLink>
          <p className="brand-subtitle">A modern C++ deep learning framework</p>
        </div>
        <nav className="main-nav" aria-label="Primary">
          <NavLink to="/docs">Docs</NavLink>
          <NavLink to="/api">API</NavLink>
          <NavLink to="/tutorials">Tutorials</NavLink>
          <NavLink className="cta-link" to="/docs">
            Start Building
          </NavLink>
        </nav>
      </header>

      <main className="page-main">
        <Outlet />
      </main>

      <footer className="footer">
        <p>NeuralCore Docs</p>
        <p>Designed for researchers and builders who want full control in C++.</p>
      </footer>
    </div>
  )
}

export default App
