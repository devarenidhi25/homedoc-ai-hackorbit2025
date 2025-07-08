import { initializeApp } from "firebase/app";
import { getAuth } from "firebase/auth";
import { getFirestore } from "firebase/firestore";

const firebaseConfig = {
  apiKey: "AIzaSyDI8lEwIA2Ivm4ZQuZ2p82XdSvPK98x32c",
  authDomain: "homedoc-ai.firebaseapp.com",
  projectId: "homedoc-ai",
  storageBucket: "homedoc-ai.appspot.com", // ✅ FIXED
  messagingSenderId: "612839037933",
  appId: "1:612839037933:web:446be2ffa125abe3f665e4",
  measurementId: "G-8QXQWFJY6B"
};

const app = initializeApp(firebaseConfig);
export const auth = getAuth(app);
export const db = getFirestore(app);