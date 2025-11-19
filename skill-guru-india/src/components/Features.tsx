import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Link } from "react-router-dom";
import { 
  Brain, 
  Target, 
  FileText, 
  Users, 
  TrendingUp, 
  Bell,
  ArrowRight,
  Sparkles,
  BookOpen,
  MessageSquare
} from "lucide-react";

const Features = () => {
  const features = [
    {
      icon: "psychology",
      title: "AI Career Matching",
      description: "Get personalized career recommendations based on your skills, interests, and market trends using advanced AI algorithms.",
      color: "blue",
      bgGradient: "from-blue-500 to-blue-600"
    },
    {
      icon: "gps_fixed",
      title: "Skills Gap Analysis", 
      description: "Identify skill gaps and get targeted learning recommendations to bridge the gap between your current abilities and dream job.",
      color: "green",
      bgGradient: "from-green-500 to-green-600"
    },
    {
      icon: "description",
      title: "Smart Resume Builder",
      description: "Create ATS-optimized resumes with AI-powered content suggestions tailored for specific job roles and industries.",
      color: "yellow",
      bgGradient: "from-yellow-500 to-orange-500"
    },
    {
      icon: "people",
      title: "Expert Mentorship",
      description: "Connect with industry professionals and mentors for personalized guidance and career advice through 1:1 sessions.",
      color: "purple",
      bgGradient: "from-purple-500 to-purple-600"
    },
    {
      icon: "trending_up",
      title: "Job Market Insights",
      description: "Stay updated with real-time job market trends, salary insights, and demand forecasting for different career paths.",
      color: "indigo",
      bgGradient: "from-indigo-500 to-indigo-600"
    },
    {
      icon: "notifications_active",
      title: "Smart Job Alerts",
      description: "Receive personalized job notifications from top platforms like Naukri, LinkedIn, and Indeed based on your profile.",
      color: "red",
      bgGradient: "from-red-500 to-pink-500"
    },
  ];

  return (
    <>
      {/* Google Fonts Import */}
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

      <section className="py-24 bg-gradient-to-br from-gray-50 via-blue-50 to-white relative overflow-hidden">
        {/* Google-style Background Elements */}
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          <div className="absolute top-1/4 left-10 w-64 h-64 rounded-full bg-gradient-to-br from-blue-400/10 to-green-400/10 blur-3xl" />
          <div className="absolute bottom-1/4 right-10 w-80 h-80 rounded-full bg-gradient-to-br from-purple-400/8 to-yellow-400/8 blur-3xl" />
          
          {/* Google-style geometric pattern */}
          <div className="absolute top-20 right-1/3 opacity-5">
            <div className="grid grid-cols-4 gap-4">
              {[...Array(16)].map((_, i) => (
                <div
                  key={i}
                  className={`w-3 h-3 rounded-full ${
                    i % 4 === 0 ? 'bg-blue-500' :
                    i % 4 === 1 ? 'bg-green-500' :
                    i % 4 === 2 ? 'bg-red-500' : 'bg-yellow-500'
                  }`}
                />
              ))}
            </div>
          </div>
        </div>

        <div className="container px-6 lg:px-8 relative z-10">
          {/* Google Material Header */}
          <div className="text-center space-y-6 mb-20">
            {/* Google-style Badge */}
            <div className="inline-flex items-center gap-2 px-4 py-2 bg-white border border-blue-200 rounded-full text-sm font-medium text-blue-700 shadow-sm">
              <span className="material-icons text-base">auto_awesome</span>
              <span style={{ fontFamily: 'Google Sans, sans-serif' }}>
                Powered by Google AI
              </span>
            </div>

            <h2 
              className="text-4xl lg:text-6xl font-normal text-gray-900 leading-tight max-w-4xl mx-auto"
              style={{ fontFamily: 'Google Sans, sans-serif' }}
            >
              Everything You Need to
              <span className="block bg-gradient-to-r from-blue-600 via-purple-600 to-green-600 bg-clip-text text-transparent font-medium mt-2">
                Accelerate Your Career
              </span>
            </h2>

            <p 
              className="text-xl lg:text-2xl text-gray-600 max-w-3xl mx-auto leading-relaxed"
              style={{ fontFamily: 'Roboto, sans-serif' }}
            >
              From career discovery to landing your dream job, our comprehensive platform 
              provides all the tools and guidance you need for <span className="font-medium text-blue-600">career success</span>.
            </p>
          </div>

          {/* Google Material Design Cards Grid */}
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8 mb-20">
            {features.map((feature, index) => (
              <Card 
                key={index} 
                className="group relative overflow-hidden bg-white hover:shadow-2xl transition-all duration-500 border-0 rounded-3xl transform hover:-translate-y-2"
                style={{
                  boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
                }}
              >
                {/* Hover gradient overlay */}
                <div className={`absolute inset-0 bg-gradient-to-br ${feature.bgGradient} opacity-0 group-hover:opacity-5 transition-opacity duration-500`} />
                
                <CardHeader className="space-y-6 p-8">
                  {/* Google Material Icon Container */}
                  <div className={`relative w-16 h-16 rounded-2xl bg-gradient-to-br ${feature.bgGradient} flex items-center justify-center group-hover:scale-110 transition-all duration-300 shadow-lg`}>
                    <span 
                      className="material-icons text-2xl text-white"
                      style={{ 
                        textShadow: '0 2px 4px rgba(0,0,0,0.1)',
                      }}
                    >
                      {feature.icon}
                    </span>
                    
                    {/* Google-style glow effect */}
                    <div className={`absolute inset-0 rounded-2xl bg-gradient-to-br ${feature.bgGradient} opacity-20 blur-xl group-hover:opacity-40 transition-opacity duration-300`} />
                  </div>

                  <div>
                    <CardTitle 
                      className="text-2xl font-medium text-gray-900 group-hover:text-blue-600 transition-colors duration-300"
                      style={{ fontFamily: 'Google Sans, sans-serif' }}
                    >
                      {feature.title}
                    </CardTitle>
                  </div>
                </CardHeader>

                <CardContent className="px-8 pb-8">
                  <CardDescription 
                    className="text-gray-600 leading-relaxed text-base"
                    style={{ fontFamily: 'Roboto, sans-serif' }}
                  >
                    {feature.description}
                  </CardDescription>
                </CardContent>

                {/* Google Material ripple effect on hover */}
                <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white to-transparent opacity-0 group-hover:opacity-10 -skew-x-12 transform translate-x-full group-hover:translate-x-[-200%] transition-transform duration-1000" />
              </Card>
            ))}
          </div>

          {/* Google Material Bottom CTA Section */}
          <div className="relative overflow-hidden bg-white rounded-3xl p-12 lg:p-16 text-center space-y-8 shadow-xl border border-gray-100">
            {/* Background gradient */}
            <div className="absolute inset-0 bg-gradient-to-br from-blue-50 via-purple-50 to-green-50 opacity-50" />
            
            {/* Content */}
            <div className="relative z-10">
              <div className="flex justify-center mb-6">
                <div className="flex items-center gap-3 px-4 py-2 bg-blue-100 rounded-full">
                  <span className="material-icons text-blue-600">rocket_launch</span>
                  <span 
                    className="text-sm font-medium text-blue-700"
                    style={{ fontFamily: 'Google Sans, sans-serif' }}
                  >
                    Ready to Get Started?
                  </span>
                </div>
              </div>

              <h3 
                className="text-3xl lg:text-4xl font-medium text-gray-900 mb-4"
                style={{ fontFamily: 'Google Sans, sans-serif' }}
              >
                Transform Your Career Journey Today
              </h3>

              <p 
                className="text-xl text-gray-600 max-w-2xl mx-auto mb-10"
                style={{ fontFamily: 'Roboto, sans-serif' }}
              >
                Join thousands of students and professionals who are already using AI to 
                <span className="font-medium text-blue-600"> accelerate their careers</span>.
              </p>

              {/* Google Material Action Buttons */}
              <div className="flex flex-col sm:flex-row gap-4 justify-center">
                {/* Start Free Trial Button - Original styling */}
                <Button
                  asChild
                  size="lg"
                  className="group h-14 px-10 bg-blue-600 hover:bg-blue-700 text-white rounded-full shadow-lg hover:shadow-2xl transition-all duration-300 transform hover:-translate-y-1"
                  style={{ fontFamily: 'Google Sans, sans-serif' }}
                >
                  <Link to="/sign-up">
                    <span className="material-icons mr-3">play_arrow</span>
                    Start Free Trial
                    <ArrowRight className="ml-3 h-5 w-5 transition-transform duration-300 group-hover:translate-x-1" />
                  </Link>
                </Button>

                {/* Learn More Button - WITH Gemini Border Effect */}
                <div className="features-gemini-border-wrapper">
                  <Button               
                    size="lg" 
                    className="features-gemini-gradient-border h-14 px-10 rounded-full transition-all duration-300 border-2 border-gray-300 hover:border-transparent relative"
                    style={{ 
                      fontFamily: 'Google Sans, sans-serif',
                      backgroundColor: '#ffffff',
                      color: '#374151'
                    }}
                  >
                    <span className="material-icons mr-3">menu_book</span>
                    <a 
                      href="https://en.wikipedia.org/wiki/StudentAdvisor" 
                      target="_blank" 
                      rel="noopener noreferrer"
                      className="inline-block text-inherit no-underline"
                    >
                      Learn More
                    </a>
                  </Button>
                </div>
              </div>

              {/* Google-style trust indicators */}
              <div className="flex justify-center items-center gap-8 mt-12 pt-8 border-t border-gray-200">
                <div className="flex items-center gap-2 text-gray-500">
                  <span className="material-icons text-green-500">verified</span>
                  <span className="text-sm font-medium">Trusted by 50K+ students</span>
                </div>
                <div className="flex items-center gap-2 text-gray-500">
                  <span className="material-icons text-blue-500">security</span>
                  <span className="text-sm font-medium">Enterprise-grade security</span>
                </div>
              </div>
            </div>

            {/* Decorative elements */}
            <div className="absolute top-4 right-4 w-20 h-20 rounded-full bg-gradient-to-br from-yellow-400/20 to-orange-400/20 blur-2xl" />
            <div className="absolute bottom-4 left-4 w-16 h-16 rounded-full bg-gradient-to-br from-green-400/20 to-blue-400/20 blur-2xl" />
          </div>
        </div>
      </section>

      {/* Enhanced CSS with Subtle Gemini Border Effect for Learn More Button */}
      <style>{`
        /* SCOPED Gemini Effect - Only for Features Learn More Button */
        :root {
          /* Subtle Google Gemini Gradient Colors */
          --features-gemini-orange: #FF8A80;
          --features-gemini-pink: #FF80AB;
          --features-gemini-purple: #EA80FC;
          --features-gemini-blue: #8C9EFF;
          --features-gemini-cyan: #84FFFF;
          --features-gemini-green: #B9F6CA;
          --features-gemini-yellow: #FFFF8D;
        }

        /* Button wrapper with minimal padding */
        .features-gemini-border-wrapper {
          position: relative;
          border-radius: 9999px;
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
          padding: 0;
        }

        /* SUBTLE: Small gradient border overlay */
        .features-gemini-border-wrapper::before {
          content: '';
          position: absolute;
          top: -1px;
          left: -1px;
          right: -1px;
          bottom: -1px;
          background: linear-gradient(
            45deg,
            var(--features-gemini-orange) 0%,
            var(--features-gemini-pink) 14%,
            var(--features-gemini-purple) 28%,
            var(--features-gemini-blue) 42%,
            var(--features-gemini-cyan) 57%,
            var(--features-gemini-green) 71%,
            var(--features-gemini-yellow) 85%,
            var(--features-gemini-orange) 100%
          );
          background-size: 400% 400%;
          border-radius: inherit;
          opacity: 0;
          z-index: -1;
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
          filter: saturate(0.8) brightness(0.9);
        }

        /* SUBTLE: Small gradient border on hover */
        .features-gemini-border-wrapper:hover::before {
          opacity: 0.7;
          animation: features-gemini-gradient-rotate 5s linear infinite;
          filter: saturate(1.0) brightness(1.0);
          top: -2px;
          left: -2px;
          right: -2px;
          bottom: -2px;
        }

        /* Button maintains white background with subtle border */
        .features-gemini-gradient-border {
          background-color: #ffffff !important;
          background: #ffffff !important;
          color: #374151 !important;
          position: relative;
          z-index: 1;
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }

        .features-gemini-gradient-border:hover {
          background-color: #f9fafb !important;
          background: #f9fafb !important;
          color: #2563eb !important;
          border-color: transparent !important;
          transform: translateY(-1px);
          box-shadow: 
            0 6px 20px rgba(0, 0, 0, 0.08),
            0 3px 10px rgba(0, 0, 0, 0.04);
        }

        .features-gemini-gradient-border:focus {
          background-color: #ffffff !important;
          background: #ffffff !important;
          color: #374151 !important;
        }

        .features-gemini-gradient-border:active {
          background-color: #ffffff !important;
          background: #ffffff !important;
          color: #374151 !important;
          transform: translateY(0px) scale(0.98);
        }

        /* SUBTLE: Slower, more subtle animation */
        @keyframes features-gemini-gradient-rotate {
          0% {
            background-position: 0% 50%;
            filter: hue-rotate(0deg) saturate(0.8) brightness(0.9);
          }
          50% {
            background-position: 100% 50%;
            filter: hue-rotate(180deg) saturate(1.0) brightness(1.0);
          }
          100% {
            background-position: 0% 50%;
            filter: hue-rotate(360deg) saturate(0.8) brightness(0.9);
          }
        }

        /* SCOPED focus states */
        .features-gemini-gradient-border:focus-visible {
          outline: 2px solid #8C9EFF;
          outline-offset: 2px;
          background-color: #ffffff !important;
          background: #ffffff !important;
        }

        /* SCOPED active states */
        .features-gemini-border-wrapper:active {
          transform: scale(0.98);
        }

        /* SCOPED reduced motion support */
        @media (prefers-reduced-motion: reduce) {
          .features-gemini-border-wrapper:hover::before {
            animation: none !important;
          }
          
          .features-gemini-gradient-border:hover {
            transform: translateY(-1px);
          }
        }

        /* Ensure link styling inside button */
        .features-gemini-gradient-border a {
          color: inherit !important;
          text-decoration: none !important;
        }

        .features-gemini-gradient-border:hover a {
          color: inherit !important;
        }
      `}</style>
    </>
  );
};

export default Features;
