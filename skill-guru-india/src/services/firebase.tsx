import { initializeApp } from "firebase/app";
import { getAuth, GoogleAuthProvider, signInWithPopup, User } from "firebase/auth";
import { getAnalytics } from "firebase/analytics";

const firebaseConfig = {
  apiKey: "AIzaSyChxBEVwAhJp1mK4b1exWwjsh4Q8YFr02M",
  authDomain: "ai-advisor-86f45.firebaseapp.com",
  projectId: "ai-advisor-86f45",
  storageBucket: "ai-advisor-86f45.firebasestorage.app",
  messagingSenderId: "545392985407",
  appId: "1:545392985407:web:30fd79c349cf15da7d7f8a",
  measurementId: "G-7YVH4XEP5Y"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const analytics = getAnalytics(app);

// ---- Initialize Auth ----
const auth = getAuth(app);
const provider = new GoogleAuthProvider();

// ---- Export Google Sign-In function ----
export const signInWithGoogle = async (): Promise<{ user: User; idToken: string }> => {
  const result = await signInWithPopup(auth, provider);
  const user = result.user;
  const idToken = await user.getIdToken();
  return { user, idToken };
};

export { auth };
