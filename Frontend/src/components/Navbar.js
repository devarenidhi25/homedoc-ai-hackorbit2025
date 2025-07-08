"use client"

import { useState, useEffect } from "react"
import { useNavigate } from "react-router-dom"
import { useTheme } from "../contexts/ThemeContext"
import { useAuth } from "../contexts/AuthContext"
import ThemeToggle from "./ThemeToggle"
import "./Navbar.css"

const Navbar = () => {
  const [isMenuOpen, setIsMenuOpen] = useState(false)
  const [isScrolled, setIsScrolled] = useState(false)
  const [isProfileMenuOpen, setIsProfileMenuOpen] = useState(false)
  const { isDarkMode } = useTheme()
  const { isAuthenticated, getInitials, getDisplayName, logout } = useAuth()
  const navigate = useNavigate()

  useEffect(() => {
    const handleScroll = () => {
      const scrollTop = window.scrollY
      setIsScrolled(scrollTop > 50)
    }

    window.addEventListener("scroll", handleScroll)
    return () => window.removeEventListener("scroll", handleScroll)
  }, [])

  // Close profile menu when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (!event.target.closest(".user-profile-container")) {
        setIsProfileMenuOpen(false)
      }
    }

    document.addEventListener("click", handleClickOutside)
    return () => document.removeEventListener("click", handleClickOutside)
  }, [])

  const scrollToSection = (sectionId) => {
    if (window.location.pathname !== "/") {
      navigate("/")
      setTimeout(() => {
        const element = document.getElementById(sectionId)
        if (element) {
          element.scrollIntoView({ behavior: "smooth" })
        }
      }, 100)
    } else {
      const element = document.getElementById(sectionId)
      if (element) {
        element.scrollIntoView({ behavior: "smooth" })
      }
    }
    setIsMenuOpen(false)
  }

  const handleLogoClick = () => {
    navigate("/")
    window.scrollTo({ top: 0, behavior: "smooth" })
  }

  const handleSignIn = () => {
    navigate("/signin")
    setIsMenuOpen(false)
  }

  const handleSignUp = () => {
    navigate("/signup")
    setIsMenuOpen(false)
  }

  const handleLogout = async () => {
    try {
      await logout()
      navigate("/")
      setIsProfileMenuOpen(false)
      setIsMenuOpen(false)
    } catch (error) {
      console.error("Error logging out:", error)
    }
  }

  const handleProfileClick = () => {
    setIsProfileMenuOpen(!isProfileMenuOpen)
  }

  const navigateToProfile = () => {
    navigate("/profile")
    setIsProfileMenuOpen(false)
    setIsMenuOpen(false)
  }

  return (
    <nav className={`navbar ${isScrolled ? "scrolled" : ""} ${isDarkMode ? "dark" : ""}`}>
      <div className="nav-container">
        <div className="nav-logo" onClick={handleLogoClick}>
          <img src="/images/fav-icon.png" alt="HomeDoc Logo" className="logo-image" />
          <span className="logo-text">HomeDoc AI</span>
        </div>

        <div className={`nav-menu ${isMenuOpen ? "active" : ""}`}>
          <button className="nav-link" onClick={() => scrollToSection("home")}>
            Home
          </button>
          <button className="nav-link" onClick={() => scrollToSection("about")}>
            About
          </button>
          <button className="nav-link" onClick={() => scrollToSection("contact")}>
            Contact Us
          </button>

          {isAuthenticated ? (
            <div className="user-profile-container">
              <div className="user-profile" onClick={handleProfileClick}>
                <div className="profile-avatar">{getInitials()}</div>
                <span className="profile-name">{getDisplayName()}</span>
                <i className={`fas fa-chevron-down profile-arrow ${isProfileMenuOpen ? "rotated" : ""}`}></i>
              </div>

              {isProfileMenuOpen && (
                <div className="profile-dropdown">
                  <div className="profile-dropdown-header">
                    <div className="profile-avatar-large">{getInitials()}</div>
                    <div className="profile-info">
                      <span className="profile-display-name">{getDisplayName()}</span>
                      <span className="profile-email">Welcome back!</span>
                    </div>
                  </div>
                  <div className="profile-dropdown-divider"></div>
                  {/* <button className="profile-dropdown-item" onClick={navigateToProfile}>
                    <i className="fas fa-user"></i>
                    My Profile
                  </button> */}
                  {/* <button className="profile-dropdown-item" onClick={() => navigate("/dashboard")}>
                    <i className="fas fa-tachometer-alt"></i>
                    Dashboard
                  </button> */}
                  {/* <button className="profile-dropdown-item" onClick={() => navigate("/settings")}>
                    <i className="fas fa-cog"></i>
                    Settings
                  </button> */}
                  {/* <div className="profile-dropdown-divider"></div>
                  <button className="profile-dropdown-item logout-item" onClick={handleLogout}>
                    <i className="fas fa-sign-out-alt"></i>
                    Sign Out
                  </button> */}
                  <button className="profile-dropdown-item logout-item" onClick={handleLogout}>
                    <i className="fas fa-sign-out-alt"></i>
                    Sign Out
                  </button>
                </div>
              )}
            </div>
          ) : (
            <div className="auth-buttons">
              <button className="btn-signin" onClick={handleSignIn}>
                <i className="fas fa-sign-in-alt"></i>
                Sign In
              </button>
              <button className="btn-signup" onClick={handleSignUp}>
                <i className="fas fa-user-plus"></i>
                Sign Up
              </button>
            </div>
          )}

          <ThemeToggle />
        </div>

        <div className="hamburger" onClick={() => setIsMenuOpen(!isMenuOpen)}>
          <span></span>
          <span></span>
          <span></span>
        </div>
      </div>
    </nav>
  )
}

export default Navbar
