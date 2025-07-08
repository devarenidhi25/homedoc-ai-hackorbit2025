import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import { ThemeProvider } from "./contexts/ThemeContext";
import { AuthProvider } from "./contexts/AuthContext"
import Navbar from "./components/Navbar";
import Home from "./pages/Home";
import Chatbot from "./pages/Chatbot";
import Predictor from "./pages/Predictor";
import ReportScanner from "./pages/ReportScanner";
import SignIn from "./pages/SignIn";
import SignUp from "./pages/SignUp";
import UserProfile from "./components/UserProfile"; // ✅ NEW
import Footer from "./components/Footer";
import "./App.css";
function App() {
  return (
    <ThemeProvider>
      <AuthProvider>
        <Router>
          <div className="App">
            <Navbar />
            <main className="main-content">
              <Routes>
                <Route path="/" element={<Home />} />
                <Route path="/chatbot" element={<Chatbot />} />
                <Route path="/predictor" element={<Predictor />} />
                <Route path="/report-scanner" element={<ReportScanner />} />
                <Route path="/signin" element={<SignIn />} />
                <Route path="/signup" element={<SignUp />} />
                <Route path="/profile" element={<UserProfile />} /> {/* ✅ NEW */}
              </Routes>
            </main>
            <Footer />
          </div>
        </Router>
      </AuthProvider>
    </ThemeProvider>
  );
}

export default App;
