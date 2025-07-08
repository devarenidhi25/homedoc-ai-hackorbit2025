"use client"

import { createContext, useContext, useState, useEffect } from "react"
import { onAuthStateChanged, signOut } from "firebase/auth"
import { doc, getDoc } from "firebase/firestore"
import { auth, db } from "../firebase"

const AuthContext = createContext()

export const useAuth = () => {
  const context = useContext(AuthContext)
  if (!context) {
    throw new Error("useAuth must be used within an AuthProvider")
  }
  return context
}

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null)
  const [userData, setUserData] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, async (user) => {
      if (user) {
        setUser(user)
        // Fetch user data from Firestore
        try {
          const userDoc = await getDoc(doc(db, "users", user.uid))
          if (userDoc.exists()) {
            setUserData(userDoc.data())
          }
        } catch (error) {
          console.error("Error fetching user data:", error)
        }
      } else {
        setUser(null)
        setUserData(null)
      }
      setLoading(false)
    })

    return unsubscribe
  }, [])

  const logout = async () => {
    try {
      await signOut(auth)
      setUser(null)
      setUserData(null)
    } catch (error) {
      console.error("Error signing out:", error)
    }
  }

  const getInitials = () => {
    if (userData?.firstName && userData?.lastName) {
      return `${userData.firstName.charAt(0)}${userData.lastName.charAt(0)}`.toUpperCase()
    }
    if (user?.email) {
      return user.email.charAt(0).toUpperCase()
    }
    return "U"
  }

  const getDisplayName = () => {
    if (userData?.firstName && userData?.lastName) {
      return `${userData.firstName} ${userData.lastName}`
    }
    if (user?.email) {
      return user.email
    }
    return "User"
  }

  const value = {
    user,
    userData,
    loading,
    logout,
    getInitials,
    getDisplayName,
    isAuthenticated: !!user,
  }

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}
