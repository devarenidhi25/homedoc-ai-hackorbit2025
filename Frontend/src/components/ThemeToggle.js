"use client"

import { useTheme } from "../contexts/ThemeContext"
import "./ThemeToggle.css"

const ThemeToggle = () => {
  const { isDarkMode, toggleTheme } = useTheme()

  return (
    <button className="theme-toggle" onClick={toggleTheme} aria-label="Toggle theme">
      <div className="toggle-track">
        <div className="toggle-thumb">
          <i className={`fas ${isDarkMode ? "fa-moon" : "fa-sun"}`}></i>
        </div>
      </div>
    </button>
  )
}

export default ThemeToggle
