"use client"

import { useAuth } from "../contexts/AuthContext"
import { useNavigate } from "react-router-dom"
import "./Profile.css"

const Profile = () => {
  const { user, userData, getDisplayName, logout } = useAuth()
  const navigate = useNavigate()

  const handleLogout = async () => {
    try {
      await logout()
      navigate("/")
    } catch (error) {
      console.error("Error logging out:", error)
    }
  }

  if (!user) {
    navigate("/signin")
    return null
  }

  return (
    <div className="profile-container">
      <div className="profile-card">
        <div className="profile-header">
          <div className="profile-avatar-xl">
            {userData?.firstName?.charAt(0) || user.email?.charAt(0) || "U"}
            {userData?.lastName?.charAt(0) || ""}
          </div>
          <h1>{getDisplayName()}</h1>
          <p>{user.email}</p>
        </div>

        <div className="profile-details">
          <div className="detail-item">
            <label>First Name</label>
            <span>{userData?.firstName || "Not provided"}</span>
          </div>
          <div className="detail-item">
            <label>Last Name</label>
            <span>{userData?.lastName || "Not provided"}</span>
          </div>
          <div className="detail-item">
            <label>Email</label>
            <span>{user.email}</span>
          </div>
          <div className="detail-item">
            <label>Member Since</label>
            <span>{userData?.createdAt?.toDate?.()?.toLocaleDateString() || "Unknown"}</span>
          </div>
        </div>

        <div className="profile-actions">
          <button className="btn-edit">Edit Profile</button>
          <button className="btn-logout" onClick={handleLogout}>
            Sign Out
          </button>
        </div>
      </div>
    </div>
  )
}

export default Profile
