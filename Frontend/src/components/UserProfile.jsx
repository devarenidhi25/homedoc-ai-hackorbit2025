import { useEffect, useState } from "react";
import { auth, db } from "../firebase";
import { doc, getDoc } from "firebase/firestore";

export default function UserProfile() {
  const [profile, setProfile] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchProfile = async () => {
      const user = auth.currentUser;
      if (user) {
        const docRef = doc(db, "users", user.uid);
        const docSnap = await getDoc(docRef);
        if (docSnap.exists()) {
          setProfile(docSnap.data());
        } else {
          console.warn("User profile not found in Firestore.");
        }
      }
      setLoading(false);
    };
    fetchProfile();
  }, []);

  if (loading) return <p>Loading profile...</p>;
  if (!profile) return <p>No profile data found.</p>;

  return (
    <div style={{ padding: 20, maxWidth: 500 }}>
      <h2>👤 User Profile</h2>
      <p><strong>Email:</strong> {profile.email}</p>
      <p><strong>Joined:</strong> {profile.createdAt?.toDate().toLocaleString()}</p>
    </div>
  );
}