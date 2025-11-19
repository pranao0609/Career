import { Button } from "@/components/ui/button";
import { Users } from "lucide-react";
import { Link } from "react-router-dom";

const JoinCommunityButton = () => {
  return (
    <>
      <link rel="preconnect" href="https://fonts.googleapis.com" />
      <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
      <link 
        href="https://fonts.googleapis.com/css2?family=Google+Sans:wght@300;400;500;700&family=Roboto:wght@300;400;500;700&display=swap" 
        rel="stylesheet" 
      />
      <link 
        href="https://fonts.googleapis.com/icon?family=Material+Icons" 
        rel="stylesheet" 
      />

      <Link to="/community">
        <div className="fixed bottom-4 right-4 z-30 group">
          {/* Main Button with Enhanced Gemini Border Effect */}
          <div className="community-gemini-border-wrapper">
            <Button 
              className="community-gemini-gradient-border rounded-3xl h-12 px-5 shadow-xl hover:shadow-2xl transition-all duration-300 border-2 border-gray-200 hover:border-transparent group-hover:scale-105"
              size="lg"
              style={{ 
                fontFamily: 'Google Sans, sans-serif',
                backgroundColor: '#ffffff',
                color: '#1f2937'
              }}
            >
              <span className="material-icons text-blue-600 text-lg mr-2">
                groups
              </span>
              <span className="font-medium text-sm">Join Community</span>
            </Button>
          </div>

          {/* Activity Indicator */}
          <div className="absolute -top-1 -right-1 w-5 h-5 bg-gradient-to-r from-green-500 to-green-600 rounded-full border-2 border-white shadow-lg">
            <div className="w-1.5 h-1.5 bg-white rounded-full mx-auto mt-1 animate-pulse" />
          </div>

          {/* Tooltip */}
          <div className="absolute bottom-full right-0 mb-2 opacity-0 group-hover:opacity-100 transition-opacity duration-200">
            <div className="bg-gray-900 text-white text-xs px-2 py-1.5 rounded-lg shadow-lg whitespace-nowrap">
              <span style={{ fontFamily: 'Roboto, sans-serif' }}>
                Connect with students
              </span>
              <div className="absolute top-full right-3 w-0 h-0 border-l-3 border-r-3 border-t-3 border-transparent border-t-gray-900" />
            </div>
          </div>
        </div>
      </Link>

      {/* FIXED CSS - Positioned to Complete Right Corner Without Overlap */}
      <style>{`
        /* SCOPED Gemini Effect - Only for Community Join Button */
        :root {
          /* ENHANCED Google Gemini Gradient Colors - More Vibrant */
          --community-gemini-orange: #FF6B35;
          --community-gemini-pink: #E91E63;
          --community-gemini-purple: #9C27B0;
          --community-gemini-blue: #2196F3;
          --community-gemini-cyan: #00BCD4;
          --community-gemini-green: #4CAF50;
          --community-gemini-yellow: #FFC107;
        }

        /* Button wrapper - FIXED positioning to avoid all overlaps */
        .community-gemini-border-wrapper {
          position: relative;
          border-radius: 1.5rem;
          transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
          padding: 0;
        }

        /* ENHANCED: Much more visible gradient border overlay */
        .community-gemini-border-wrapper::before {
          content: '';
          position: absolute;
          top: -4px;
          left: -4px;
          right: -4px;
          bottom: -4px;
          background: linear-gradient(
            45deg,
            var(--community-gemini-orange) 0%,
            var(--community-gemini-pink) 14%,
            var(--community-gemini-purple) 28%,
            var(--community-gemini-blue) 42%,
            var(--community-gemini-cyan) 57%,
            var(--community-gemini-green) 71%,
            var(--community-gemini-yellow) 85%,
            var(--community-gemini-orange) 100%
          );
          background-size: 300% 300%;
          border-radius: inherit;
          opacity: 0;
          z-index: -1;
          transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
          filter: saturate(1.2) brightness(1.1);
        }

        /* ENHANCED: Much more visible gradient border on hover */
        .community-gemini-border-wrapper:hover::before {
          opacity: 1;
          animation: community-gemini-gradient-rotate 3s linear infinite;
          filter: saturate(1.4) brightness(1.2);
          top: -5px;
          left: -5px;
          right: -5px;
          bottom: -5px;
        }

        /* ENHANCED: Stronger glow effect */
        .community-gemini-border-wrapper::after {
          content: '';
          position: absolute;
          top: -8px;
          left: -8px;
          right: -8px;
          bottom: -8px;
          background: linear-gradient(
            45deg,
            var(--community-gemini-orange) 0%,
            var(--community-gemini-pink) 14%,
            var(--community-gemini-purple) 28%,
            var(--community-gemini-blue) 42%,
            var(--community-gemini-cyan) 57%,
            var(--community-gemini-green) 71%,
            var(--community-gemini-yellow) 85%,
            var(--community-gemini-orange) 100%
          );
          background-size: 300% 300%;
          border-radius: inherit;
          filter: blur(10px);
          opacity: 0;
          z-index: -2;
          transition: opacity 0.4s ease;
        }

        .community-gemini-border-wrapper:hover::after {
          opacity: 0.6;
          animation: community-gemini-gradient-rotate 3s linear infinite;
        }

        /* Button stays visible and maintains white background */
        .community-gemini-gradient-border {
          background-color: #ffffff !important;
          background: #ffffff !important;
          color: #1f2937 !important;
          position: relative;
          z-index: 1;
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }

        .community-gemini-gradient-border:hover {
          background-color: #f9fafb !important;
          background: #f9fafb !important;
          color: #1f2937 !important;
          border-color: transparent !important;
          transform: translateY(-2px);
          box-shadow: 
            0 12px 35px rgba(255, 107, 53, 0.15),
            0 8px 25px rgba(233, 30, 99, 0.1),
            0 4px 12px rgba(0, 0, 0, 0.1);
        }

        .community-gemini-gradient-border:focus {
          background-color: #ffffff !important;
          background: #ffffff !important;
          color: #1f2937 !important;
        }

        .community-gemini-gradient-border:active {
          background-color: #ffffff !important;
          background: #ffffff !important;
          color: #1f2937 !important;
          transform: translateY(0px) scale(0.98);
        }

        /* ENHANCED: Faster, more visible animation */
        @keyframes community-gemini-gradient-rotate {
          0% {
            background-position: 0% 50%;
            filter: hue-rotate(0deg) saturate(1.2) brightness(1.1);
          }
          25% {
            background-position: 100% 0%;
            filter: hue-rotate(90deg) saturate(1.4) brightness(1.2);
          }
          50% {
            background-position: 100% 100%;
            filter: hue-rotate(180deg) saturate(1.3) brightness(1.15);
          }
          75% {
            background-position: 0% 100%;
            filter: hue-rotate(270deg) saturate(1.4) brightness(1.2);
          }
          100% {
            background-position: 0% 50%;
            filter: hue-rotate(360deg) saturate(1.2) brightness(1.1);
          }
        }

        /* SCOPED focus states */
        .community-gemini-gradient-border:focus-visible {
          outline: 3px solid #2196F3;
          outline-offset: 3px;
          background-color: #ffffff !important;
          background: #ffffff !important;
        }

        /* SCOPED active states */
        .community-gemini-border-wrapper:active {
          transform: scale(0.96);
        }

        /* SCOPED reduced motion support */
        @media (prefers-reduced-motion: reduce) {
          .community-gemini-border-wrapper:hover::before,
          .community-gemini-border-wrapper:hover::after {
            animation: none !important;
          }
          
          .community-gemini-gradient-border:hover {
            transform: translateY(-1px);
          }
        }

        /* Ensure activity indicator and tooltip stay on top */
        .community-gemini-border-wrapper ~ div {
          z-index: 3;
          position: relative;
        }

        /* FIXED: Complete right corner positioning to avoid ALL overlaps */
        .fixed.bottom-4.right-4.z-30 {
          bottom: 1rem !important;
          right: 1rem !important;
          z-index: 30 !important;
          max-width: fit-content;
          max-height: fit-content;
        }

        /* Responsive positioning - even tighter to corner */
        @media (max-width: 1200px) {
          .fixed.bottom-4.right-4.z-30 {
            bottom: 0.75rem !important;
            right: 0.75rem !important;
          }
        }

        @media (max-width: 768px) {
          .fixed.bottom-4.right-4.z-30 {
            bottom: 0.5rem !important;
            right: 0.5rem !important;
          }
        }

        @media (max-width: 640px) {
          .fixed.bottom-4.right-4.z-30 {
            bottom: 0.5rem !important;
            right: 0.25rem !important;
          }
        }

        /* Ensure it doesn't interfere with any other components */
        .fixed.bottom-4.right-4.z-30 {
          pointer-events: auto;
          isolation: isolate;
        }

        /* Make button more compact to avoid overlaps */
        .community-gemini-gradient-border {
          min-width: auto;
          width: fit-content;
        }
      `}</style>
    </>
  );
};

export default JoinCommunityButton;
