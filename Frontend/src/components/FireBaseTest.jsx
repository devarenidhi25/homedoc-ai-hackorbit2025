import React, { useEffect } from "react";
import { collection, getDocs } from "firebase/firestore";
import { db } from "../firebase";

const FirebaseTest = () => {
  useEffect(() => {
    const fetchData = async () => {
      try {
        const querySnapshot = await getDocs(collection(db, "testCollection"));
        querySnapshot.forEach((doc) => {
          console.log(`${doc.id} =>`, doc.data());
        });
        alert("✅ Firebase is working! Check console.");
      } catch (error) {
        console.error("Firebase test failed:", error);
        alert("❌ Firebase test failed. See console for details.");
      }
    };

    fetchData();
  }, []);

  return <p>Testing Firebase connection... (Check the console)</p>;
};

export default FirebaseTest;
