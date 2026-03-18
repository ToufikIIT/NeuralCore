import { useEffect, useState } from 'react'
import { NavLink, Outlet, useLocation } from 'react-router-dom'
import './App.css'

function App() {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false)
  const location = useLocation()

  useEffect(() => {
    setIsMobileMenuOpen(false)
  }, [location.pathname, location.search])

  function closeMobileMenu() {
    setIsMobileMenuOpen(false)
  }

  return (
    <div className="site-shell">
      <header className="topbar">
        <div className="brand-block">
          <NavLink className="brand-pill" to="/">
            NeuralCore
          </NavLink>
          <p className="brand-subtitle">A modern C++ deep learning framework</p>
        </div>
        <button
          className="nav-toggle"
          type="button"
          aria-expanded={isMobileMenuOpen}
          aria-controls="primary-nav"
          onClick={() => setIsMobileMenuOpen((open) => !open)}
        >
          {isMobileMenuOpen ? 'Close' : 'Menu'}
        </button>
        <nav
          id="primary-nav"
          className={`main-nav ${isMobileMenuOpen ? 'open' : ''}`}
          aria-label="Primary"
        >
          <NavLink to="/docs" onClick={closeMobileMenu}>
            Docs
          </NavLink>
          <NavLink to="/api" onClick={closeMobileMenu}>
            API
          </NavLink>
          <NavLink to="/tutorials" onClick={closeMobileMenu}>
            Tutorials
          </NavLink>
          <NavLink className="cta-link" to="/docs" onClick={closeMobileMenu}>
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
