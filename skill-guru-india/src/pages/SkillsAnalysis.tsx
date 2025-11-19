import React, { useState, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import Header from "@/components/Header";
import { useAuth } from "@/contexts/AuthContext";
import { useNavigate } from 'react-router-dom';
import { 
  Brain,
  Target,
  TrendingUp,
  BookOpen,
  Award,
  Clock,
  CheckCircle,
  AlertCircle,
  ArrowRight,
  Star,
  PlayCircle,
  FileText,
  Users,
  Zap,
  RefreshCw,
  BarChart3,
  PieChart,
  Activity,
  Filter,
  Trophy,
  Medal,
  Flame,
  Rocket,
  Shield,
  Crown,
  Gem,
  Sparkles,
  Bolt,
  Lock,
  Unlock
} from "lucide-react";

// Chart.js imports
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  RadialLinearScale,
  ArcElement,
  BarElement,
  Filler
} from 'chart.js';

import {
  Radar,
  Pie,
  Bar,
  Line,
  Doughnut
} from 'react-chartjs-2';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  RadialLinearScale,
  ArcElement,
  BarElement,
  Filler
);

// Authentication Guard Component
const AuthenticationRequired: React.FC = () => {
  const navigate = useNavigate();

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

      <Header />
      
      <section className="min-h-screen bg-gradient-to-br from-gray-50 via-blue-50 to-white flex items-center justify-center p-6">
        {/* Background decoration */}
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          <div className="absolute top-1/4 left-10 w-96 h-96 rounded-full bg-gradient-to-br from-blue-400/10 to-purple-400/10 blur-3xl" />
          <div className="absolute bottom-1/4 right-10 w-80 h-80 rounded-full bg-gradient-to-br from-green-400/8 to-yellow-400/8 blur-3xl" />
        </div>

        {/* Authentication Required Card */}
        <Card className="w-full max-w-2xl border-0 rounded-3xl shadow-2xl bg-white relative z-10">
          <CardHeader className="text-center space-y-6 p-12">
            {/* Google-style skills icon */}
            <div className="w-20 h-20 mx-auto bg-gradient-to-br from-green-500 to-green-600 rounded-2xl flex items-center justify-center shadow-lg">
              <span className="material-icons text-white text-3xl">psychology</span>
            </div>
            
            <div className="space-y-4">
              <CardTitle 
                className="text-3xl font-medium text-gray-900"
                style={{ fontFamily: 'Google Sans, sans-serif' }}
              >
                Unlock Your Skills Analysis
              </CardTitle>
              <CardDescription 
                className="text-xl text-gray-600 leading-relaxed max-w-lg mx-auto"
                style={{ fontFamily: 'Roboto, sans-serif' }}
              >
                Sign in to take AI-powered assessments, track your progress, and get personalized learning recommendations.
              </CardDescription>
            </div>
          </CardHeader>

          <CardContent className="space-y-8 p-12 pt-0">
            {/* Skills Features */}
            <div className="grid md:grid-cols-2 gap-6">
              <div className="flex items-start space-x-4 p-4 rounded-2xl bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-100">
                <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-blue-500 to-blue-600 shadow-md">
                  <span className="material-icons text-white text-lg">quiz</span>
                </div>
                <div>
                  <h4 
                    className="font-semibold text-gray-900 mb-1"
                    style={{ fontFamily: 'Google Sans, sans-serif' }}
                  >
                    AI Skills Assessment
                  </h4>
                  <p 
                    className="text-sm text-gray-600"
                    style={{ fontFamily: 'Roboto, sans-serif' }}
                  >
                    Comprehensive evaluation of technical and soft skills
                  </p>
                </div>
              </div>

              <div className="flex items-start space-x-4 p-4 rounded-2xl bg-gradient-to-r from-green-50 to-emerald-50 border border-green-100">
                <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-green-500 to-green-600 shadow-md">
                  <span className="material-icons text-white text-lg">bar_chart</span>
                </div>
                <div>
                  <h4 
                    className="font-semibold text-gray-900 mb-1"
                    style={{ fontFamily: 'Google Sans, sans-serif' }}
                  >
                    Visual Analytics
                  </h4>
                  <p 
                    className="text-sm text-gray-600"
                    style={{ fontFamily: 'Roboto, sans-serif' }}
                  >
                    Interactive charts and progress tracking
                  </p>
                </div>
              </div>

              <div className="flex items-start space-x-4 p-4 rounded-2xl bg-gradient-to-r from-purple-50 to-violet-50 border border-purple-100">
                <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-purple-500 to-purple-600 shadow-md">
                  <span className="material-icons text-white text-lg">emoji_events</span>
                </div>
                <div>
                  <h4 
                    className="font-semibold text-gray-900 mb-1"
                    style={{ fontFamily: 'Google Sans, sans-serif' }}
                  >
                    Achievement Badges
                  </h4>
                  <p 
                    className="text-sm text-gray-600"
                    style={{ fontFamily: 'Roboto, sans-serif' }}
                  >
                    Earn XP and unlock badges for your progress
                  </p>
                </div>
              </div>

              <div className="flex items-start space-x-4 p-4 rounded-2xl bg-gradient-to-r from-orange-50 to-yellow-50 border border-orange-100">
                <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-orange-500 to-orange-600 shadow-md">
                  <span className="material-icons text-white text-lg">menu_book</span>
                </div>
                <div>
                  <h4 
                    className="font-semibold text-gray-900 mb-1"
                    style={{ fontFamily: 'Google Sans, sans-serif' }}
                  >
                    Learning Paths
                  </h4>
                  <p 
                    className="text-sm text-gray-600"
                    style={{ fontFamily: 'Roboto, sans-serif' }}
                  >
                    Personalized recommendations to improve skills
                  </p>
                </div>
              </div>
            </div>

            {/* Preview Assessment */}
            <div className="p-6 rounded-2xl bg-gradient-to-r from-gray-50 to-blue-50 border border-gray-200">
              <div className="flex items-center space-x-3 mb-4">
                <span className="material-icons text-green-600">preview</span>
                <h4 
                  className="font-semibold text-gray-900"
                  style={{ fontFamily: 'Google Sans, sans-serif' }}
                >
                  What's Included in Your Assessment:
                </h4>
              </div>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {[
                  { label: "Technical Skills", icon: "code", color: "text-blue-600" },
                  { label: "Soft Skills", icon: "people", color: "text-purple-600" },
                  { label: "Industry Knowledge", icon: "business", color: "text-green-600" },
                  { label: "Problem Solving", icon: "lightbulb", color: "text-orange-600" }
                ].map((item, index) => (
                  <div key={index} className="text-center">
                    <span className={`material-icons text-2xl ${item.color} mb-2 block`}>
                      {item.icon}
                    </span>
                    <span 
                      className="text-sm text-gray-700 font-medium"
                      style={{ fontFamily: 'Roboto, sans-serif' }}
                    >
                      {item.label}
                    </span>
                  </div>
                ))}
              </div>
            </div>

            {/* Gamification Preview */}
            <div className="p-6 rounded-2xl bg-gradient-to-r from-yellow-50 to-orange-50 border border-yellow-200">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center space-x-3">
                  <span className="material-icons text-yellow-600">emoji_events</span>
                  <h4 
                    className="font-semibold text-gray-900"
                    style={{ fontFamily: 'Google Sans, sans-serif' }}
                  >
                    Level Up Your Skills
                  </h4>
                </div>
                <div className="flex items-center space-x-2">
                  <span className="material-icons text-purple-500 text-lg">auto_awesome</span>
                  <span 
                    className="text-sm font-bold text-purple-600"
                    style={{ fontFamily: 'Google Sans, sans-serif' }}
                  >
                    Earn XP & Badges
                  </span>
                </div>
              </div>
              <p 
                className="text-sm text-gray-600 mb-4"
                style={{ fontFamily: 'Roboto, sans-serif' }}
              >
                Complete assessments, improve skills, and unlock achievement badges to level up your profile!
              </p>
              <div className="flex space-x-2">
                {['First Steps', 'Skill Master', 'Perfectionist', 'Tech Guru'].map((badge, index) => (
                  <Badge key={index} variant="outline" className="text-xs bg-white border-gray-300">
                    {badge}
                  </Badge>
                ))}
              </div>
            </div>

            {/* Action Buttons */}
            <div className="flex flex-col sm:flex-row gap-4">
              <Button 
                onClick={() => navigate('/sign-in')}
                className="flex-1 h-14 bg-gradient-to-r from-green-600 to-green-700 hover:from-green-700 hover:to-green-800 text-white rounded-2xl shadow-lg hover:shadow-xl transition-all duration-300 text-base font-medium"
                style={{ fontFamily: 'Google Sans, sans-serif' }}
              >
                <span className="material-icons mr-3">login</span>
                Sign In to Start Assessment
              </Button>

              <Button 
                onClick={() => navigate('/sign-up')}
                variant="outline"
                className="flex-1 h-14 border-2 border-gray-300 hover:bg-gray-50 hover:border-gray-400 transition-all duration-200 rounded-2xl text-base font-medium"
                style={{ fontFamily: 'Google Sans, sans-serif' }}
              >
                <span className="material-icons mr-3">person_add</span>
                Create Account
              </Button>
            </div>

            {/* Additional Info */}
            <div className="text-center pt-4 border-t border-gray-200">
              <p 
                className="text-sm text-gray-500"
                style={{ fontFamily: 'Roboto, sans-serif' }}
              >
                Join thousands of professionals who've improved their skills with our AI-powered assessments
              </p>
            </div>
          </CardContent>
        </Card>

        {/* Google-style Feature Notice */}
        <div className="absolute bottom-8 left-1/2 transform -translate-x-1/2">
          <div className="flex items-center space-x-2 text-sm text-gray-500 bg-white/80 backdrop-blur-sm px-4 py-2 rounded-full border border-gray-200 shadow-sm">
            <span className="material-icons text-green-500 text-sm">psychology</span>
            <span style={{ fontFamily: 'Roboto, sans-serif' }}>
              AI-powered skills analysis with machine learning insights
            </span>
          </div>
        </div>
      </section>
    </>
  );
};

const SkillsAnalysis = () => {
  const { user, isAuthenticated, isLoading } = useAuth();
  
  // State variables
  const [hasCompletedQuiz, setHasCompletedQuiz] = useState(false);
  const [isQuizLoading, setIsQuizLoading] = useState(false);
  const [skillsScore, setSkillsScore] = useState(68);
  const [strongSkillsCount, setStrongSkillsCount] = useState(8);
  const [skillGapsCount, setSkillGapsCount] = useState(6);
  const [criticalAreasCount, setCriticalAreasCount] = useState(4);
  const [quizStats, setQuizStats] = useState({
    questionsAnswered: 45,
    timeSpent: "18m 32s",
    accuracy: 78,
    completionDate: "Today"
  });
  const [activeTab, setActiveTab] = useState("overview");
  const [selectedCategory, setSelectedCategory] = useState("all");
  const [chartType, setChartType] = useState("radar");

  // Badge system states
  const [userBadges, setUserBadges] = useState([]);
  const [showBadgeAnimation, setShowBadgeAnimation] = useState(false);
  const [newlyEarnedBadge, setNewlyEarnedBadge] = useState(null);
  const [userStats, setUserStats] = useState({
    totalXP: 1250,
    level: 5,
    currentStreak: 7,
    totalAssessments: 12,
    perfectScores: 3,
    improvementStreak: 5
  });

  // Google Material Icon mapping for badges
  const materialIconMap = {
    Brain: "psychology",
    Trophy: "emoji_events",
    Star: "star",
    Flame: "local_fire_department",
    Bolt: "bolt",
    Rocket: "rocket_launch",
    Users: "people",
    Shield: "security",
    Crown: "workspace_premium",
    Gem: "diamond"
  };

  // Badge definitions with Google Material styling
  const badgeDefinitions = [
    {
      id: 'first_assessment',
      title: 'First Steps',
      description: 'Complete your first skills assessment',
      icon: "psychology",
      color: 'bg-blue-500',
      gradient: 'from-blue-500 to-blue-600',
      xpReward: 50,
      rarity: 'common',
      category: 'milestone',
      condition: (stats, skills) => stats.totalAssessments >= 1
    },
    {
      id: 'skill_master',
      title: 'Skill Master',
      description: 'Achieve 90% or higher in any skill category',
      icon: "emoji_events",
      color: 'bg-yellow-500',
      gradient: 'from-yellow-500 to-orange-500',
      xpReward: 200,
      rarity: 'rare',
      category: 'achievement',
      condition: (stats, skills) => skills.some(cat => cat.overall >= 90)
    },
    {
      id: 'perfectionist',
      title: 'Perfectionist',
      description: 'Score 100% on any assessment',
      icon: "star",
      color: 'bg-purple-500',
      gradient: 'from-purple-500 to-pink-500',
      xpReward: 300,
      rarity: 'epic',
      category: 'achievement',
      condition: (stats, skills) => stats.perfectScores >= 1
    },
    {
      id: 'streak_warrior',
      title: 'Streak Warrior',
      description: 'Complete assessments for 7 consecutive days',
      icon: "local_fire_department",
      color: 'bg-red-500',
      gradient: 'from-red-500 to-orange-500',
      xpReward: 150,
      rarity: 'uncommon',
      category: 'engagement',
      condition: (stats, skills) => stats.currentStreak >= 7
    },
    {
      id: 'rapid_learner',
      title: 'Rapid Learner',
      description: 'Improve 5 skills in a single session',
      icon: "bolt",
      color: 'bg-yellow-400',
      gradient: 'from-yellow-400 to-yellow-500',
      xpReward: 175,
      rarity: 'uncommon',
      category: 'progress',
      condition: (stats, skills) => stats.improvementStreak >= 5
    },
    {
      id: 'tech_guru',
      title: 'Tech Guru',
      description: 'Master all technical skills (80%+ average)',
      icon: "rocket_launch",
      color: 'bg-green-500',
      gradient: 'from-green-500 to-blue-500',
      xpReward: 250,
      rarity: 'rare',
      category: 'mastery',
      condition: (stats, skills) => {
        const techSkills = skills.find(cat => cat.name === 'Technical Skills');
        return techSkills && techSkills.overall >= 80;
      }
    },
    {
      id: 'communication_ace',
      title: 'Communication Ace',
      description: 'Excel in all soft skills (85%+ average)',
      icon: "people",
      color: 'bg-pink-500',
      gradient: 'from-pink-500 to-red-500',
      xpReward: 250,
      rarity: 'rare',
      category: 'mastery',
      condition: (stats, skills) => {
        const softSkills = skills.find(cat => cat.name === 'Soft Skills');
        return softSkills && softSkills.overall >= 85;
      }
    },
    {
      id: 'industry_expert',
      title: 'Industry Expert',
      description: 'Demonstrate deep industry knowledge (90%+ average)',
      icon: "security",
      color: 'bg-indigo-500',
      gradient: 'from-indigo-500 to-purple-500',
      xpReward: 300,
      rarity: 'epic',
      category: 'mastery',
      condition: (stats, skills) => {
        const industrySkills = skills.find(cat => cat.name === 'Industry Knowledge');
        return industrySkills && industrySkills.overall >= 90;
      }
    },
    {
      id: 'all_rounder',
      title: 'All-Rounder',
      description: 'Score 75%+ in all skill categories',
      icon: "workspace_premium",
      color: 'bg-gradient-to-r from-yellow-400 to-orange-500',
      gradient: 'from-yellow-400 via-orange-500 to-red-500',
      xpReward: 400,
      rarity: 'legendary',
      category: 'mastery',
      condition: (stats, skills) => skills.every(cat => cat.overall >= 75)
    },
    {
      id: 'xp_collector',
      title: 'XP Collector',
      description: 'Earn 1000+ experience points',
      icon: "diamond",
      color: 'bg-cyan-500',
      gradient: 'from-cyan-500 to-blue-500',
      xpReward: 100,
      rarity: 'uncommon',
      category: 'progression',
      condition: (stats, skills) => stats.totalXP >= 1000
    }
  ];

  const skillCategories = [
    {
      name: "Technical Skills",
      overall: 75,
      skills: [
        { name: "JavaScript", level: 85, status: "strong", trend: "up", improvement: 15 },
        { name: "React", level: 80, status: "strong", trend: "up", improvement: 12 },
        { name: "Node.js", level: 70, status: "good", trend: "stable", improvement: 5 },
        { name: "Python", level: 45, status: "gap", trend: "down", improvement: -8 },
        { name: "TypeScript", level: 35, status: "gap", trend: "down", improvement: -5 },
        { name: "AWS", level: 25, status: "critical", trend: "down", improvement: -10 }
      ]
    },
    {
      name: "Soft Skills",
      overall: 68,
      skills: [
        { name: "Communication", level: 78, status: "good", trend: "up", improvement: 8 },
        { name: "Leadership", level: 65, status: "good", trend: "stable", improvement: 2 },
        { name: "Problem Solving", level: 82, status: "strong", trend: "up", improvement: 18 },
        { name: "Team Collaboration", level: 70, status: "good", trend: "up", improvement: 10 },
        { name: "Time Management", level: 55, status: "gap", trend: "stable", improvement: 0 },
        { name: "Presentation", level: 45, status: "gap", trend: "down", improvement: -12 }
      ]
    },
    {
      name: "Industry Knowledge",
      overall: 60,
      skills: [
        { name: "Agile Methodology", level: 70, status: "good", trend: "up", improvement: 14 },
        { name: "DevOps", level: 40, status: "gap", trend: "stable", improvement: 1 },
        { name: "System Design", level: 35, status: "gap", trend: "down", improvement: -6 },
        { name: "Database Design", level: 75, status: "strong", trend: "up", improvement: 20 },
        { name: "Security Best Practices", level: 30, status: "critical", trend: "down", improvement: -15 },
        { name: "Testing", level: 65, status: "good", trend: "stable", improvement: 3 }
      ]
    }
  ];

  const learningPaths = [
    {
      title: "Python for Full Stack Development",
      description: "Bridge your Python knowledge gap with hands-on projects",
      priority: "High",
      duration: "4 weeks",
      modules: 8,
      skillsGained: ["Python", "Django", "Flask", "API Development"],
      difficulty: "Intermediate",
      rating: 4.8,
      gradient: "from-blue-500 to-blue-600"
    },
    {
      title: "AWS Cloud Fundamentals",
      description: "Learn cloud computing essentials for modern development",
      priority: "High", 
      duration: "6 weeks",
      modules: 12,
      skillsGained: ["AWS", "EC2", "S3", "Lambda", "RDS"],
      difficulty: "Beginner",
      rating: 4.9,
      gradient: "from-yellow-500 to-orange-500"
    },
    {
      title: "Advanced TypeScript",
      description: "Master type-safe JavaScript for better code quality",
      priority: "Medium",
      duration: "3 weeks", 
      modules: 6,
      skillsGained: ["TypeScript", "Advanced Types", "Generics", "Decorators"],
      difficulty: "Advanced",
      rating: 4.7,
      gradient: "from-green-500 to-green-600"
    }
  ];

  // Google Material Badge Components
  const BadgeCard = ({ badge, earned = false, showAnimation = false }) => {
    return (
      <div className={`relative p-6 rounded-3xl border-2 transition-all duration-300 ${
        earned 
          ? 'border-yellow-300 bg-gradient-to-br from-yellow-50 to-orange-50 shadow-lg scale-105' 
          : 'border-gray-200 bg-gray-50 opacity-60'
      } ${showAnimation ? 'animate-bounce' : ''}`}>
        
        <div className={`w-16 h-16 mx-auto mb-4 rounded-2xl flex items-center justify-center ${
          earned ? `bg-gradient-to-br ${badge.gradient}` : 'bg-gray-300'
        } shadow-lg`}>
          <span 
            className={`material-icons text-2xl ${earned ? 'text-white' : 'text-gray-500'}`}
          >
            {badge.icon}
          </span>
        </div>

        <div className="text-center">
          <h3 
            className={`font-medium text-lg mb-2 ${earned ? 'text-gray-800' : 'text-gray-500'}`}
            style={{ fontFamily: 'Google Sans, sans-serif' }}
          >
            {badge.title}
          </h3>
          <p 
            className={`text-sm mb-4 ${earned ? 'text-gray-600' : 'text-gray-400'} leading-relaxed`}
            style={{ fontFamily: 'Roboto, sans-serif' }}
          >
            {badge.description}
          </p>
          
          <div className="space-y-2">
            <Badge 
              variant={earned ? 'default' : 'secondary'} 
              className={`text-xs font-medium rounded-full px-3 py-1 ${getRarityColor(badge.rarity)}`}
            >
              {badge.rarity.toUpperCase()}
            </Badge>
            
            {earned && (
              <div className="flex items-center justify-center space-x-2">
                <span className="material-icons text-yellow-500 text-lg">auto_awesome</span>
                <span 
                  className="text-sm font-semibold text-yellow-600"
                  style={{ fontFamily: 'Google Sans, sans-serif' }}
                >
                  +{badge.xpReward} XP
                </span>
              </div>
            )}
          </div>
        </div>

        <div className="absolute top-3 right-3">
          {earned ? (
            <span className="material-icons text-green-500 text-xl">check_circle</span>
          ) : (
            <span className="material-icons text-gray-400 text-lg">lock</span>
          )}
        </div>

        {earned && badge.rarity === 'legendary' && (
          <div className="absolute inset-0 rounded-3xl bg-gradient-to-r from-yellow-400 via-orange-500 to-red-500 opacity-20 animate-pulse" />
        )}
      </div>
    );
  };

  const BadgeUnlockAnimation = ({ badge, onClose }) => {
    useEffect(() => {
      const timer = setTimeout(() => {
        onClose();
      }, 3000);
      return () => clearTimeout(timer);
    }, [onClose]);

    return (
      <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
        <div className="bg-white rounded-3xl p-10 max-w-md mx-4 text-center animate-bounce shadow-2xl">
          <div className="mb-6">
            <span className="material-icons text-yellow-500 text-5xl mb-4 block animate-spin">auto_awesome</span>
            <h2 
              className="text-3xl font-medium text-gray-800 mb-2"
              style={{ fontFamily: 'Google Sans, sans-serif' }}
            >
              Badge Unlocked!
            </h2>
          </div>
          
          <div className={`w-24 h-24 mx-auto mb-6 rounded-2xl flex items-center justify-center bg-gradient-to-br ${badge.gradient} shadow-xl`}>
            <span className="material-icons text-white text-3xl">{badge.icon}</span>
          </div>
          
          <h3 
            className="text-2xl font-medium text-gray-800 mb-3"
            style={{ fontFamily: 'Google Sans, sans-serif' }}
          >
            {badge.title}
          </h3>
          <p 
            className="text-gray-600 mb-6 leading-relaxed"
            style={{ fontFamily: 'Roboto, sans-serif' }}
          >
            {badge.description}
          </p>
          
          <div className="flex items-center justify-center space-x-2 mb-6">
            <span className="material-icons text-yellow-500 text-xl">diamond</span>
            <span 
              className="text-xl font-bold text-yellow-600"
              style={{ fontFamily: 'Google Sans, sans-serif' }}
            >
              +{badge.xpReward} XP
            </span>
          </div>
          
          <Button 
            onClick={onClose}
            className="bg-gradient-to-r from-yellow-500 to-orange-500 hover:from-yellow-600 hover:to-orange-600 text-white rounded-full px-8 py-3 font-medium"
            style={{ fontFamily: 'Google Sans, sans-serif' }}
          >
            Awesome!
          </Button>
        </div>
      </div>
    );
  };

  const XPProgressBar = ({ currentXP, level }) => {
    const xpForNextLevel = level * 300;
    const xpProgress = (currentXP % 300) / 300 * 100;
    
    return (
      <div className="bg-white rounded-2xl p-6 border border-gray-200 shadow-sm">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center space-x-2">
            <span className="material-icons text-yellow-500 text-xl">workspace_premium</span>
            <span 
              className="font-medium text-gray-800 text-lg"
              style={{ fontFamily: 'Google Sans, sans-serif' }}
            >
              Level {level}
            </span>
          </div>
          <span 
            className="text-sm text-gray-600"
            style={{ fontFamily: 'Roboto, sans-serif' }}
          >
            {currentXP} XP
          </span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-3 mb-2">
          <div 
            className="bg-gradient-to-r from-blue-500 to-purple-500 h-3 rounded-full transition-all duration-1000 ease-out"
            style={{ width: `${xpProgress}%` }}
          />
        </div>
        <div 
          className="text-xs text-gray-500 text-center"
          style={{ fontFamily: 'Roboto, sans-serif' }}
        >
          {Math.round((300 - (currentXP % 300)))} XP to Level {level + 1}
        </div>
      </div>
    );
  };

  // Chart data preparation
  const prepareChartData = () => {
    const allSkills = skillCategories.flatMap(category => 
      category.skills.map(skill => ({
        ...skill,
        category: category.name
      }))
    );

    return selectedCategory === "all" 
      ? allSkills 
      : skillCategories.find(cat => cat.name === selectedCategory)?.skills || [];
  };

  const chartData = prepareChartData();

  // Chart.js data configurations with Google colors
  const getRadarData = () => ({
    labels: chartData.map(skill => skill.name),
    datasets: [
      {
        label: 'Your Level',
        data: chartData.map(skill => skill.level),
        backgroundColor: 'rgba(26, 115, 232, 0.2)',
        borderColor: '#1A73E8',
        borderWidth: 3,
        pointBackgroundColor: '#1A73E8',
        pointBorderColor: '#fff',
        pointHoverBackgroundColor: '#fff',
        pointHoverBorderColor: '#1A73E8'
      },
      {
        label: 'Industry Average',
        data: chartData.map(() => 75),
        backgroundColor: 'rgba(52, 168, 83, 0.1)',
        borderColor: '#34A853',
        borderWidth: 2,
        borderDash: [5, 5],
        pointBackgroundColor: '#34A853',
        pointBorderColor: '#fff'
      }
    ]
  });

  const getBarData = () => ({
    labels: chartData.map(skill => skill.name),
    datasets: [
      {
        label: 'Skill Level (%)',
        data: chartData.map(skill => skill.level),
        backgroundColor: chartData.map(skill => {
          switch(skill.status) {
            case 'strong': return '#34A853';
            case 'good': return '#1A73E8';
            case 'gap': return '#FBBC04';
            case 'critical': return '#EA4335';
            default: return '#9AA0A6';
          }
        }),
        borderColor: chartData.map(skill => {
          switch(skill.status) {
            case 'strong': return '#137333';
            case 'good': return '#1557B0';
            case 'gap': return '#E8710A';
            case 'critical': return '#C5221F';
            default: return '#80868B';
          }
        }),
        borderWidth: 2,
        borderRadius: 8,
        borderSkipped: false,
      }
    ]
  });

  const getPieData = () => {
    const statusCounts = {
      strong: chartData.filter(s => s.status === 'strong').length,
      good: chartData.filter(s => s.status === 'good').length,
      gap: chartData.filter(s => s.status === 'gap').length,
      critical: chartData.filter(s => s.status === 'critical').length
    };

    return {
      labels: ['Strong', 'Good', 'Gap', 'Critical'],
      datasets: [
        {
          data: [statusCounts.strong, statusCounts.good, statusCounts.gap, statusCounts.critical],
          backgroundColor: ['#34A853', '#1A73E8', '#FBBC04', '#EA4335'],
          borderColor: ['#137333', '#1557B0', '#E8710A', '#C5221F'],
          borderWidth: 2,
          hoverOffset: 4
        }
      ]
    };
  };

  const getLineData = () => ({
    labels: chartData.map(skill => skill.name),
    datasets: [
      {
        label: 'Improvement Trend',
        data: chartData.map(skill => skill.improvement),
        fill: true,
        backgroundColor: 'rgba(26, 115, 232, 0.2)',
        borderColor: '#1A73E8',
        borderWidth: 3,
        pointBackgroundColor: chartData.map(skill => 
          skill.improvement > 0 ? '#34A853' : '#EA4335'
        ),
        pointBorderColor: '#fff',
        pointBorderWidth: 2,
        pointRadius: 6,
        tension: 0.4
      }
    ]
  });

  // Google Material Chart options
  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top' as const,
        labels: {
          usePointStyle: true,
          padding: 20,
          font: {
            family: 'Google Sans, sans-serif',
            size: 12
          }
        }
      },
      tooltip: {
        backgroundColor: 'rgba(32, 33, 36, 0.9)',
        titleColor: 'white',
        bodyColor: 'white',
        borderColor: 'rgba(255, 255, 255, 0.1)',
        borderWidth: 1,
        cornerRadius: 12,
        displayColors: true,
        titleFont: {
          family: 'Google Sans, sans-serif'
        },
        bodyFont: {
          family: 'Roboto, sans-serif'
        }
      }
    },
    scales: {
      y: {
        beginAtZero: true,
        max: 100,
        grid: {
          color: 'rgba(0, 0, 0, 0.1)'
        },
        ticks: {
          font: {
            family: 'Roboto, sans-serif'
          }
        }
      },
      x: {
        grid: {
          display: false
        },
        ticks: {
          font: {
            family: 'Roboto, sans-serif'
          }
        }
      }
    }
  };

  const radarOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top' as const,
        labels: {
          font: {
            family: 'Google Sans, sans-serif'
          }
        }
      }
    },
    scales: {
      r: {
        beginAtZero: true,
        max: 100,
        ticks: {
          stepSize: 20,
          font: {
            family: 'Roboto, sans-serif'
          }
        },
        grid: {
          color: 'rgba(0, 0, 0, 0.1)'
        }
      }
    }
  };

  // Progress Circle Component with Google styling
  const ProgressCircle = ({ percentage, size = 120, strokeWidth = 8, color = "#1A73E8" }) => {
    const radius = (size - strokeWidth) / 2;
    const circumference = radius * 2 * Math.PI;
    const offset = circumference - (percentage / 100) * circumference;

    return (
      <div className="relative inline-flex items-center justify-center">
        <svg
          width={size}
          height={size}
          className="transform -rotate-90"
        >
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            stroke="currentColor"
            strokeWidth={strokeWidth}
            fill="transparent"
            className="text-gray-200"
          />
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            stroke={color}
            strokeWidth={strokeWidth}
            fill="transparent"
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            strokeLinecap="round"
            className="transition-all duration-1000 ease-out"
          />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center">
          <span 
            className="text-2xl font-bold text-gray-700"
            style={{ fontFamily: 'Google Sans, sans-serif' }}
          >
            {percentage}%
          </span>
        </div>
      </div>
    );
  };

  // Helper functions
  const getRarityColor = (rarity) => {
    switch(rarity) {
      case 'common': return 'bg-gray-100 text-gray-700 border-gray-200';
      case 'uncommon': return 'bg-green-100 text-green-700 border-green-200';
      case 'rare': return 'bg-blue-100 text-blue-700 border-blue-200';
      case 'epic': return 'bg-purple-100 text-purple-700 border-purple-200';
      case 'legendary': return 'bg-gradient-to-r from-yellow-400 to-orange-500 text-white border-yellow-400';
      default: return 'bg-gray-100 text-gray-700 border-gray-200';
    }
  };

  const checkBadgeUnlocks = (stats, skills) => {
    const newBadges = [];
    
    badgeDefinitions.forEach(badge => {
      const alreadyEarned = userBadges.some(earned => earned.id === badge.id);
      
      if (!alreadyEarned && badge.condition(stats, skills)) {
        newBadges.push({
          ...badge,
          unlockedAt: new Date(),
          xpEarned: badge.xpReward
        });
      }
    });

    if (newBadges.length > 0) {
      setUserBadges(prev => [...prev, ...newBadges]);
      setUserStats(prev => ({
        ...prev,
        totalXP: prev.totalXP + newBadges.reduce((sum, badge) => sum + badge.xpReward, 0)
      }));
      
      if (newBadges[0]) {
        setNewlyEarnedBadge(newBadges[0]);
        setShowBadgeAnimation(true);
      }
    }
  };

  // Handler functions
  const handleStartQuiz = async () => {
    setIsQuizLoading(true);
    setTimeout(() => {
      console.log(`Starting comprehensive skills assessment for ${user?.firstName || user?.name || 'user'}...`);
      setIsQuizLoading(false);
      setTimeout(() => {
        handleQuizCompletion();
      }, 2000);
    }, 2000);
  };

  const handleQuizCompletion = () => {
    setSkillsScore(72);
    setStrongSkillsCount(9);
    setSkillGapsCount(5);
    setCriticalAreasCount(3);
    setQuizStats({
      questionsAnswered: 48,
      timeSpent: "17m 45s",
      accuracy: 81,
      completionDate: "Just now"
    });
    setHasCompletedQuiz(true);
    
    const newStats = {
      ...userStats,
      totalAssessments: userStats.totalAssessments + 1,
      perfectScores: userStats.perfectScores + (skillsScore === 100 ? 1 : 0)
    };
    setUserStats(newStats);
    
    setTimeout(() => {
      checkBadgeUnlocks(newStats, skillCategories);
    }, 1000);
  };

  const handleRetakeQuiz = () => {
    setHasCompletedQuiz(false);
    setSkillsScore(0);
    setIsQuizLoading(false);
  };

  const handleCategoryQuiz = (categoryName) => {
    console.log(`Starting ${categoryName} specific assessment for ${user?.firstName || user?.name || 'user'}...`);
  };

  const getStatusColor = (status) => {
    switch (status) {
      case "strong": return "text-green-600";
      case "good": return "text-blue-600";
      case "gap": return "text-yellow-600";
      case "critical": return "text-red-600";
      default: return "text-gray-500";
    }
  };

  const getStatusBadge = (status) => {
    switch (status) {
      case "strong": return <Badge className="bg-green-100 text-green-700 border-green-200 rounded-full">Strong</Badge>;
      case "good": return <Badge className="bg-blue-100 text-blue-700 border-blue-200 rounded-full">Good</Badge>;
      case "gap": return <Badge className="bg-yellow-100 text-yellow-700 border-yellow-200 rounded-full">Gap</Badge>;
      case "critical": return <Badge className="bg-red-100 text-red-700 border-red-200 rounded-full">Critical</Badge>;
      default: return null;
    }
  };

  const getTrendIcon = (trend) => {
    switch (trend) {
      case "up": return <span className="material-icons text-green-600 text-lg">trending_up</span>;
      case "down": return <span className="material-icons text-red-600 text-lg">trending_down</span>;
      case "stable": return <span className="material-icons text-gray-500 text-lg">trending_flat</span>;
      default: return null;
    }
  };

  // Show loading spinner while checking authentication
  if (isLoading) {
    return (
      <>
        <Header />
        <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-gray-50 via-blue-50 to-white">
          <div className="text-center">
            <div className="google-loading-spinner mb-4"></div>
            <p 
              className="text-gray-600"
              style={{ fontFamily: 'Roboto, sans-serif' }}
            >
              Loading skills analysis...
            </p>
          </div>
        </div>
        
        <style>{`
          .google-loading-spinner {
            width: 32px;
            height: 32px;
            border: 3px solid #e0e0e0;
            border-top: 3px solid #4285f4;
            border-radius: 50%;
            animation: google-spin 1s linear infinite;
            margin: 0 auto;
          }

          @keyframes google-spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
          }
        `}</style>
      </>
    );
  }

  // Show authentication required screen if user is not authenticated
  if (!isAuthenticated) {
    return <AuthenticationRequired />;
  }

  // Show skills analysis if user is authenticated
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

      <div className="min-h-screen bg-gradient-to-br from-gray-50 via-blue-50 to-white">
        <Header />
        
        {/* Badge Unlock Animation */}
        {showBadgeAnimation && newlyEarnedBadge && (
          <BadgeUnlockAnimation 
            badge={newlyEarnedBadge}
            onClose={() => {
              setShowBadgeAnimation(false);
              setNewlyEarnedBadge(null);
            }}
          />
        )}
        
        <main className="pt-8">
          {/* Google Material Hero Section with personalized welcome */}
          <section className="py-20 lg:py-28 relative overflow-hidden">
            {/* Google-style Background Elements */}
            <div className="absolute inset-0 overflow-hidden pointer-events-none">
              <div className="absolute top-1/4 left-10 w-96 h-96 rounded-full bg-gradient-to-br from-blue-400/10 to-purple-400/10 blur-3xl" />
              <div className="absolute bottom-1/4 right-10 w-80 h-80 rounded-full bg-gradient-to-br from-green-400/8 to-yellow-400/8 blur-3xl" />
              
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
              <div className="text-center space-y-8 max-w-4xl mx-auto">
                {/* Google-style Badge with personalization */}
                <div className="inline-flex items-center gap-2 px-4 py-2 bg-white border border-blue-200 rounded-full text-sm font-medium text-blue-700 shadow-sm">
                  <span className="material-icons text-base">psychology</span>
                  <span style={{ fontFamily: 'Google Sans, sans-serif' }}>
                    Welcome back, {user?.firstName || user?.name?.split(' ')[0] || 'User'}! AI Skills Assessment
                  </span>
                </div>

                <h1 
                  className="text-5xl lg:text-7xl font-normal text-gray-900 leading-tight"
                  style={{ fontFamily: 'Google Sans, sans-serif' }}
                >
                  Analyze & Improve
                  <span className="block bg-gradient-to-r from-blue-600 via-purple-600 to-green-600 bg-clip-text text-transparent font-medium mt-2">
                    Your Skills
                  </span>
                </h1>

                <p 
                  className="text-xl lg:text-2xl text-gray-600 max-w-3xl mx-auto leading-relaxed"
                  style={{ fontFamily: 'Roboto, sans-serif' }}
                >
                  Get detailed insights into your skill levels, identify gaps, and receive 
                  personalized learning recommendations powered by <span className="font-medium text-blue-600">Google AI</span>.
                </p>
                
                {/* XP Progress Bar */}
                <div className="max-w-md mx-auto mt-8">
                  <XPProgressBar currentXP={userStats.totalXP} level={userStats.level} />
                </div>
              </div>
            </div>
          </section>

          {/* Skills Analysis */}
          <section className="py-20 bg-white">
            <div className="container px-6 lg:px-8">
              <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-8">
                <div className="text-center">
                  <TabsList className="grid w-full max-w-2xl mx-auto grid-cols-4 bg-gray-100 p-1 rounded-full">
                    <TabsTrigger 
                      value="overview"
                      className="rounded-full data-[state=active]:bg-white data-[state=active]:shadow-sm"
                      style={{ fontFamily: 'Google Sans, sans-serif' }}
                    >
                      Overview
                    </TabsTrigger>
                    <TabsTrigger 
                      value="detailed"
                      className="rounded-full data-[state=active]:bg-white data-[state=active]:shadow-sm"
                      style={{ fontFamily: 'Google Sans, sans-serif' }}
                    >
                      Detailed
                    </TabsTrigger>
                    <TabsTrigger 
                      value="badges"
                      className="rounded-full data-[state=active]:bg-white data-[state=active]:shadow-sm"
                      style={{ fontFamily: 'Google Sans, sans-serif' }}
                    >
                      Badges
                    </TabsTrigger>
                    <TabsTrigger 
                      value="learning"
                      className="rounded-full data-[state=active]:bg-white data-[state=active]:shadow-sm"
                      style={{ fontFamily: 'Google Sans, sans-serif' }}
                    >
                      Learning
                    </TabsTrigger>
                  </TabsList>
                </div>

                {/* Overview Tab */}
                <TabsContent value="overview" className="space-y-8">
                  <div className="text-center space-y-6 mb-16">
                    <h2 
                      className="text-4xl lg:text-5xl font-normal text-gray-900"
                      style={{ fontFamily: 'Google Sans, sans-serif' }}
                    >
                      Your Skills Overview
                    </h2>
                    <p 
                      className="text-xl text-gray-600 max-w-3xl mx-auto leading-relaxed"
                      style={{ fontFamily: 'Roboto, sans-serif' }}
                    >
                      {!hasCompletedQuiz 
                        ? `${user?.firstName || 'Take'} our AI-powered assessment to analyze your skills across different categories.`
                        : `Based on your AI assessment, here's your comprehensive skills analysis, ${user?.firstName || 'User'}.`
                      }
                    </p>
                  </div>

                  {/* Quiz section - before/after */}
                  {!hasCompletedQuiz ? (
                    <Card className="max-w-3xl mx-auto border-0 rounded-3xl shadow-lg">
                      <CardHeader className="text-center p-10">
                        <div className="w-20 h-20 mx-auto mb-6 bg-blue-50 rounded-2xl flex items-center justify-center">
                          <span className="material-icons text-blue-600 text-3xl">psychology</span>
                        </div>
                        <CardTitle 
                          className="text-3xl font-medium text-gray-900 mb-3"
                          style={{ fontFamily: 'Google Sans, sans-serif' }}
                        >
                          Skills Assessment
                        </CardTitle>
                        <CardDescription 
                          className="text-lg text-gray-600 leading-relaxed"
                          style={{ fontFamily: 'Roboto, sans-serif' }}
                        >
                          Take our comprehensive AI-powered quiz to evaluate your skills and get personalized insights
                        </CardDescription>
                      </CardHeader>
                      <CardContent className="space-y-8 text-center p-10 pt-0">
                        <div className="space-y-6">
                          <div className="flex items-center justify-center space-x-8 text-gray-600">
                            <div className="flex items-center space-x-2">
                              <span className="material-icons text-blue-600">schedule</span>
                              <span style={{ fontFamily: 'Roboto, sans-serif' }}>15-20 minutes</span>
                            </div>
                            <div className="flex items-center space-x-2">
                              <span className="material-icons text-green-600">quiz</span>
                              <span style={{ fontFamily: 'Roboto, sans-serif' }}>50+ questions</span>
                            </div>
                            <div className="flex items-center space-x-2">
                              <span className="material-icons text-purple-600">gps_fixed</span>
                              <span style={{ fontFamily: 'Roboto, sans-serif' }}>All skill areas</span>
                            </div>
                          </div>
                          
                          <div className="bg-gray-50 rounded-2xl p-6 space-y-3">
                            <h4 
                              className="font-medium text-gray-900"
                              style={{ fontFamily: 'Google Sans, sans-serif' }}
                            >
                              Assessment Includes:
                            </h4>
                            <div className="flex flex-wrap justify-center gap-2">
                              <Badge variant="secondary" className="bg-blue-100 text-blue-700 border-blue-200 rounded-full px-3 py-1">Technical Skills</Badge>
                              <Badge variant="secondary" className="bg-green-100 text-green-700 border-green-200 rounded-full px-3 py-1">Soft Skills</Badge>
                              <Badge variant="secondary" className="bg-purple-100 text-purple-700 border-purple-200 rounded-full px-3 py-1">Industry Knowledge</Badge>
                              <Badge variant="secondary" className="bg-orange-100 text-orange-700 border-orange-200 rounded-full px-3 py-1">Problem Solving</Badge>
                            </div>
                          </div>
                        </div>

                        <Button 
                          size="lg" 
                          className="w-full max-w-xs mx-auto h-12 bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 text-white rounded-full shadow-lg hover:shadow-xl transition-all duration-300"
                          onClick={handleStartQuiz}
                          disabled={isQuizLoading}
                          style={{ fontFamily: 'Google Sans, sans-serif' }}
                        >
                          {isQuizLoading ? (
                            <>
                              <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin mr-3"></div>
                              Preparing Quiz for {user?.firstName || 'You'}...
                            </>
                          ) : (
                            <>
                              <span className="material-icons mr-3">psychology</span>
                              Take Skills Quiz
                            </>
                          )}
                        </Button>

                        <p 
                          className="text-sm text-gray-500"
                          style={{ fontFamily: 'Roboto, sans-serif' }}
                        >
                          Your responses will be analyzed by our ML model to provide accurate skill assessments
                        </p>
                      </CardContent>
                    </Card>
                  ) : (
                    <>
                      {/* Quiz completed - show results */}
                      <Card className="max-w-3xl mx-auto border-0 rounded-3xl shadow-lg">
                        <CardHeader className="text-center p-10">
                          <div className="w-20 h-20 mx-auto mb-6 bg-green-50 rounded-2xl flex items-center justify-center">
                            <span className="material-icons text-green-600 text-3xl">check_circle</span>
                          </div>
                          <CardTitle 
                            className="text-3xl font-medium text-gray-900 mb-3"
                            style={{ fontFamily: 'Google Sans, sans-serif' }}
                          >
                            Assessment Complete, {user?.firstName || 'User'}!
                          </CardTitle>
                          <CardDescription 
                            className="text-lg text-gray-600"
                            style={{ fontFamily: 'Roboto, sans-serif' }}
                          >
                            Your comprehensive skill assessment across all categories
                          </CardDescription>
                        </CardHeader>
                        <CardContent className="space-y-8 p-10 pt-0">
                          <div className="text-center">
                            <div 
                              className="text-6xl font-bold text-blue-600 mb-2"
                              style={{ fontFamily: 'Google Sans, sans-serif' }}
                            >
                              {skillsScore}
                            </div>
                            <div 
                              className="text-lg text-gray-500 mb-4"
                              style={{ fontFamily: 'Roboto, sans-serif' }}
                            >
                              out of 100
                            </div>
                            <Badge 
                              className={`rounded-full px-4 py-2 font-medium ${
                                skillsScore >= 80 ? 'bg-green-100 text-green-700 border-green-200' :
                                skillsScore >= 60 ? 'bg-blue-100 text-blue-700 border-blue-200' :
                                skillsScore >= 40 ? 'bg-yellow-100 text-yellow-700 border-yellow-200' :
                                'bg-red-100 text-red-700 border-red-200'
                              }`}
                            >
                              {skillsScore >= 80 ? 'Expert Level' :
                               skillsScore >= 60 ? 'Advanced' :
                               skillsScore >= 40 ? 'Intermediate' : 'Beginner'}
                            </Badge>
                          </div>

                          {/* Custom Google-style progress bar */}
                          <div className="relative">
                            <div className="w-full bg-gray-200 rounded-full h-4">
                              <div 
                                className="bg-gradient-to-r from-blue-500 to-blue-600 h-4 rounded-full transition-all duration-1000 ease-out relative overflow-hidden"
                                style={{ width: `${skillsScore}%` }}
                              >
                                <div className="absolute inset-0 bg-white/20 animate-pulse" />
                              </div>
                            </div>
                          </div>
                          
                          <div className="grid grid-cols-3 gap-6 text-center">
                            <div className="p-4 bg-green-50 rounded-2xl border border-green-200">
                              <div 
                                className="text-3xl font-bold text-green-600 mb-1"
                                style={{ fontFamily: 'Google Sans, sans-serif' }}
                              >
                                {strongSkillsCount}
                              </div>
                              <div 
                                className="text-sm text-gray-600"
                                style={{ fontFamily: 'Roboto, sans-serif' }}
                              >
                                Strong Skills
                              </div>
                            </div>
                            <div className="p-4 bg-yellow-50 rounded-2xl border border-yellow-200">
                              <div 
                                className="text-3xl font-bold text-yellow-600 mb-1"
                                style={{ fontFamily: 'Google Sans, sans-serif' }}
                              >
                                {skillGapsCount}
                              </div>
                              <div 
                                className="text-sm text-gray-600"
                                style={{ fontFamily: 'Roboto, sans-serif' }}
                              >
                                Skill Gaps
                              </div>
                            </div>
                            <div className="p-4 bg-red-50 rounded-2xl border border-red-200">
                              <div 
                                className="text-3xl font-bold text-red-600 mb-1"
                                style={{ fontFamily: 'Google Sans, sans-serif' }}
                              >
                                {criticalAreasCount}
                              </div>
                              <div 
                                className="text-sm text-gray-600"
                                style={{ fontFamily: 'Roboto, sans-serif' }}
                              >
                                Critical Areas
                              </div>
                            </div>
                          </div>

                          <div className="flex flex-col sm:flex-row gap-4 pt-4">
                            <Button 
                              variant="outline" 
                              className="flex-1 h-12 rounded-full border-2 border-gray-300 hover:bg-gray-50"
                              onClick={handleRetakeQuiz}
                              style={{ fontFamily: 'Google Sans, sans-serif' }}
                            >
                              <span className="material-icons mr-2">refresh</span>
                              Retake Assessment
                            </Button>
                            <Button 
                              className="flex-1 h-12 bg-gradient-to-r from-green-600 to-green-700 hover:from-green-700 hover:to-green-800 text-white rounded-full shadow-lg"
                              onClick={() => setActiveTab('learning')}
                              style={{ fontFamily: 'Google Sans, sans-serif' }}
                            >
                              <span className="material-icons mr-2">menu_book</span>
                              View Learning Path
                            </Button>
                          </div>
                        </CardContent>
                      </Card>

                      {/* Category Overview */}
                      <div className="grid md:grid-cols-3 gap-8">
                        {skillCategories.map((category, index) => (
                          <Card key={index} className="border-0 rounded-3xl shadow-lg bg-white hover:shadow-xl transition-all duration-500">
                            <CardHeader className="p-8">
                              <CardTitle 
                                className="flex items-center space-x-3 text-xl font-medium text-gray-900"
                                style={{ fontFamily: 'Google Sans, sans-serif' }}
                              >
                                <span className="material-icons text-blue-600">gps_fixed</span>
                                <span>{category.name}</span>
                              </CardTitle>
                            </CardHeader>
                            <CardContent className="space-y-6 p-8 pt-0">
                              <div className="text-center">
                                <ProgressCircle 
                                  percentage={category.overall} 
                                  size={120} 
                                  color={
                                    category.overall >= 80 ? '#34A853' :
                                    category.overall >= 60 ? '#1A73E8' :
                                    '#FBBC04'
                                  }
                                />
                              </div>
                              <div className="space-y-3">
                                {category.skills.slice(0, 3).map((skill, skillIndex) => (
                                  <div key={skillIndex} className="flex justify-between items-center p-3 bg-gray-50 rounded-xl">
                                    <span 
                                      className="text-sm font-medium text-gray-700"
                                      style={{ fontFamily: 'Roboto, sans-serif' }}
                                    >
                                      {skill.name}
                                    </span>
                                    <div className="flex items-center space-x-2">
                                      <span 
                                        className={`text-sm font-semibold ${getStatusColor(skill.status)}`}
                                        style={{ fontFamily: 'Google Sans, sans-serif' }}
                                      >
                                        {skill.level}%
                                      </span>
                                      {getTrendIcon(skill.trend)}
                                    </div>
                                  </div>
                                ))}
                              </div>
                              <Button 
                                size="sm" 
                                variant="outline" 
                                className="w-full h-10 rounded-full border-2 border-gray-300 hover:bg-blue-50 hover:border-blue-300 hover:text-blue-600"
                                onClick={() => handleCategoryQuiz(category.name)}
                                style={{ fontFamily: 'Google Sans, sans-serif' }}
                              >
                                <span className="material-icons mr-2 text-base">psychology</span>
                                Retake {category.name}
                              </Button>
                            </CardContent>
                          </Card>
                        ))}
                      </div>
                    </>
                  )}
                </TabsContent>

                {/* Detailed Tab */}
                <TabsContent value="detailed" className="space-y-8">
                  <div className="text-center space-y-6 mb-16">
                    <h2 
                      className="text-4xl lg:text-5xl font-normal text-gray-900"
                      style={{ fontFamily: 'Google Sans, sans-serif' }}
                    >
                      Advanced Skills Analytics
                    </h2>
                    <p 
                      className="text-xl text-gray-600 max-w-3xl mx-auto"
                      style={{ fontFamily: 'Roboto, sans-serif' }}
                    >
                      Interactive data visualizations showing comprehensive analysis of your skills.
                    </p>
                  </div>

                  {/* Control Panel */}
                  <Card className="mb-8 border-0 rounded-3xl shadow-lg">
                    <CardHeader className="p-8">
                      <CardTitle 
                        className="flex items-center gap-3 text-2xl font-medium text-gray-900"
                        style={{ fontFamily: 'Google Sans, sans-serif' }}
                      >
                        <span className="material-icons text-blue-600">bar_chart</span>
                        Visualization Controls
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="p-8 pt-0">
                      <div className="flex flex-wrap gap-6 items-center">
                        <div className="flex items-center gap-3">
                          <span className="material-icons text-gray-500">filter_alt</span>
                          <select 
                            value={selectedCategory}
                            onChange={(e) => setSelectedCategory(e.target.value)}
                            className="px-4 py-2 rounded-full border border-gray-300 bg-white text-sm font-medium focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                            style={{ fontFamily: 'Roboto, sans-serif' }}
                          >
                            <option value="all">All Categories</option>
                            {skillCategories.map(cat => (
                              <option key={cat.name} value={cat.name}>{cat.name}</option>
                            ))}
                          </select>
                        </div>

                        <div className="flex items-center gap-3">
                          <span className="material-icons text-gray-500">show_chart</span>
                          <div className="flex gap-2">
                            {[
                              { type: 'radar', icon: 'gps_fixed', label: 'Radar' },
                              { type: 'bar', icon: 'bar_chart', label: 'Bar Chart' },
                              { type: 'pie', icon: 'pie_chart', label: 'Distribution' },
                              { type: 'line', icon: 'trending_up', label: 'Trends' }
                            ].map(({ type, icon, label }) => (
                              <Button
                                key={type}
                                size="sm"
                                variant={chartType === type ? "default" : "outline"}
                                onClick={() => setChartType(type)}
                                className={`text-sm rounded-full px-4 py-2 font-medium transition-all duration-200 ${
                                  chartType === type 
                                    ? 'bg-blue-600 text-white shadow-lg' 
                                    : 'border-gray-300 hover:bg-blue-50 hover:border-blue-300 hover:text-blue-600'
                                }`}
                                style={{ fontFamily: 'Google Sans, sans-serif' }}
                              >
                                <span className="material-icons mr-1 text-sm">{icon}</span>
                                {label}
                              </Button>
                            ))}
                          </div>
                        </div>
                      </div>
                    </CardContent>
                  </Card>

                  {/* Main Chart */}
                  <Card className="border-0 rounded-3xl shadow-lg">
                    <CardHeader className="p-8">
                      <CardTitle 
                        className="text-2xl font-medium text-gray-900"
                        style={{ fontFamily: 'Google Sans, sans-serif' }}
                      >
                        Skills Analysis - {chartType.charAt(0).toUpperCase() + chartType.slice(1)} View
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="p-8 pt-0">
                      <div style={{ height: '500px' }} className="bg-gray-50 rounded-2xl p-6">
                        {chartData.length > 0 ? (
                          <>
                            {chartType === 'radar' && <Radar data={getRadarData()} options={radarOptions} />}
                            {chartType === 'bar' && <Bar data={getBarData()} options={chartOptions} />}
                            {chartType === 'pie' && <Pie data={getPieData()} options={{ responsive: true, maintainAspectRatio: false }} />}
                            {chartType === 'line' && <Line data={getLineData()} options={chartOptions} />}
                          </>
                        ) : (
                          <div className="flex items-center justify-center h-full">
                            <p 
                              className="text-gray-500 text-lg"
                              style={{ fontFamily: 'Roboto, sans-serif' }}
                            >
                              No data available
                            </p>
                          </div>
                        )}
                      </div>
                    </CardContent>
                  </Card>
                </TabsContent>

                {/* Badges Tab */}
                <TabsContent value="badges" className="space-y-8">
                  <div className="text-center space-y-6 mb-16">
                    <h2 
                      className="text-4xl lg:text-5xl font-normal text-gray-900"
                      style={{ fontFamily: 'Google Sans, sans-serif' }}
                    >
                      Achievement Badges
                    </h2>
                    <p 
                      className="text-xl text-gray-600 max-w-3xl mx-auto leading-relaxed"
                      style={{ fontFamily: 'Roboto, sans-serif' }}
                    >
                      Unlock badges by completing assessments, improving skills, and reaching milestones. 
                      Each badge rewards you with <span className="font-medium text-blue-600">XP to level up</span> your profile!
                    </p>
                  </div>

                  {/* Stats Dashboard */}
                  <div className="grid md:grid-cols-4 gap-6 mb-12">
                    <Card className="border-0 rounded-3xl shadow-lg bg-white">
                      <CardContent className="p-8 text-center">
                        <span className="material-icons text-yellow-500 text-4xl mb-3 block">emoji_events</span>
                        <div 
                          className="text-3xl font-bold text-gray-900 mb-1"
                          style={{ fontFamily: 'Google Sans, sans-serif' }}
                        >
                          {userBadges.length}
                        </div>
                        <div 
                          className="text-sm text-gray-600"
                          style={{ fontFamily: 'Roboto, sans-serif' }}
                        >
                          Badges Earned
                        </div>
                      </CardContent>
                    </Card>
                    <Card className="border-0 rounded-3xl shadow-lg bg-white">
                      <CardContent className="p-8 text-center">
                        <span className="material-icons text-purple-500 text-4xl mb-3 block">auto_awesome</span>
                        <div 
                          className="text-3xl font-bold text-gray-900 mb-1"
                          style={{ fontFamily: 'Google Sans, sans-serif' }}
                        >
                          {userStats.totalXP}
                        </div>
                        <div 
                          className="text-sm text-gray-600"
                          style={{ fontFamily: 'Roboto, sans-serif' }}
                        >
                          Total XP
                        </div>
                      </CardContent>
                    </Card>
                    <Card className="border-0 rounded-3xl shadow-lg bg-white">
                      <CardContent className="p-8 text-center">
                        <span className="material-icons text-blue-500 text-4xl mb-3 block">workspace_premium</span>
                        <div 
                          className="text-3xl font-bold text-gray-900 mb-1"
                          style={{ fontFamily: 'Google Sans, sans-serif' }}
                        >
                          {userStats.level}
                        </div>
                        <div 
                          className="text-sm text-gray-600"
                          style={{ fontFamily: 'Roboto, sans-serif' }}
                        >
                          Current Level
                        </div>
                      </CardContent>
                    </Card>
                    <Card className="border-0 rounded-3xl shadow-lg bg-white">
                      <CardContent className="p-8 text-center">
                        <span className="material-icons text-red-500 text-4xl mb-3 block">local_fire_department</span>
                        <div 
                          className="text-3xl font-bold text-gray-900 mb-1"
                          style={{ fontFamily: 'Google Sans, sans-serif' }}
                        >
                          {userStats.currentStreak}
                        </div>
                        <div 
                          className="text-sm text-gray-600"
                          style={{ fontFamily: 'Roboto, sans-serif' }}
                        >
                          Day Streak
                        </div>
                      </CardContent>
                    </Card>
                  </div>

                  {/* Badge Categories */}
                  <div className="space-y-12">
                    {['milestone', 'achievement', 'mastery', 'engagement', 'progression'].map(category => (
                      <div key={category}>
                        <h3 
                          className="text-2xl font-medium text-gray-900 mb-8 capitalize flex items-center"
                          style={{ fontFamily: 'Google Sans, sans-serif' }}
                        >
                          <span className="material-icons mr-3 text-blue-600">military_tech</span>
                          {category} Badges
                        </h3>
                        <div className="grid md:grid-cols-3 lg:grid-cols-4 gap-8">
                          {badgeDefinitions
                            .filter(badge => badge.category === category)
                            .map(badge => {
                              const earned = userBadges.some(earned => earned.id === badge.id);
                              return (
                                <BadgeCard 
                                  key={badge.id} 
                                  badge={badge} 
                                  earned={earned}
                                />
                              );
                            })
                          }
                        </div>
                      </div>
                    ))}
                  </div>

                  {/* Progress Towards Next Badges */}
                  <Card className="border-0 rounded-3xl shadow-lg">
                    <CardHeader className="p-8">
                      <CardTitle 
                        className="flex items-center gap-3 text-2xl font-medium text-gray-900"
                        style={{ fontFamily: 'Google Sans, sans-serif' }}
                      >
                        <span className="material-icons text-blue-600">gps_fixed</span>
                        Progress Towards Next Badges
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="p-8 pt-0">
                      <div className="space-y-4">
                        {badgeDefinitions
                          .filter(badge => !userBadges.some(earned => earned.id === badge.id))
                          .slice(0, 3)
                          .map(badge => (
                            <div key={badge.id} className="flex items-center justify-between p-6 border border-gray-200 rounded-2xl bg-gray-50 hover:bg-gray-100 transition-colors duration-200">
                              <div className="flex items-center space-x-4">
                                <span className="material-icons text-gray-400 text-2xl">{badge.icon}</span>
                                <div>
                                  <div 
                                    className="font-medium text-gray-900 text-lg"
                                    style={{ fontFamily: 'Google Sans, sans-serif' }}
                                  >
                                    {badge.title}
                                  </div>
                                  <div 
                                    className="text-sm text-gray-600 mt-1"
                                    style={{ fontFamily: 'Roboto, sans-serif' }}
                                  >
                                    {badge.description}
                                  </div>
                                </div>
                              </div>
                              <Badge variant="outline" className={getRarityColor(badge.rarity)}>
                                {badge.rarity}
                              </Badge>
                            </div>
                          ))
                        }
                      </div>
                    </CardContent>
                  </Card>
                </TabsContent>

                {/* Learning Paths Tab */}
                <TabsContent value="learning" className="space-y-8">
                  <div className="text-center space-y-6 mb-16">
                    <h2 
                      className="text-4xl lg:text-5xl font-normal text-gray-900"
                      style={{ fontFamily: 'Google Sans, sans-serif' }}
                    >
                      Personalized Learning Paths
                    </h2>
                    <p 
                      className="text-xl text-gray-600 max-w-3xl mx-auto leading-relaxed"
                      style={{ fontFamily: 'Roboto, sans-serif' }}
                    >
                      AI-curated learning paths designed specifically for {user?.firstName || 'you'} to address your skill gaps and 
                      <span className="font-medium text-blue-600"> career goals</span>.
                    </p>
                  </div>

                  <div className="grid lg:grid-cols-2 gap-8">
                    {learningPaths.map((path, index) => (
                      <Card key={index} className="group border-0 rounded-3xl shadow-lg bg-white hover:shadow-2xl transition-all duration-500 transform hover:-translate-y-2">
                        <CardHeader className="space-y-6 p-8">
                          <div className="flex items-start justify-between">
                            <div className={`w-16 h-16 rounded-2xl bg-gradient-to-br ${path.gradient} flex items-center justify-center group-hover:scale-110 transition-all duration-300 shadow-lg`}>
                              <span className="material-icons text-white text-2xl">menu_book</span>
                            </div>
                            <Badge 
                              variant={path.priority === "High" ? "destructive" : "secondary"}
                              className={`rounded-full px-3 py-1 font-medium ${
                                path.priority === "High" 
                                  ? 'bg-red-100 text-red-700 border-red-200' 
                                  : 'bg-gray-100 text-gray-700 border-gray-200'
                              }`}
                            >
                              {path.priority} Priority
                            </Badge>
                          </div>
                          
                          <div>
                            <CardTitle 
                              className="text-2xl font-medium text-gray-900 group-hover:text-blue-600 transition-colors duration-300 mb-3"
                              style={{ fontFamily: 'Google Sans, sans-serif' }}
                            >
                              {path.title}
                            </CardTitle>
                            <CardDescription 
                              className="text-gray-600 text-base leading-relaxed"
                              style={{ fontFamily: 'Roboto, sans-serif' }}
                            >
                              {path.description}
                            </CardDescription>
                          </div>
                        </CardHeader>

                        <CardContent className="space-y-8 p-8 pt-0">
                          <div className="grid grid-cols-2 gap-4">
                            <div className="flex items-center space-x-2 text-sm text-gray-600">
                              <span className="material-icons text-blue-600">schedule</span>
                              <span style={{ fontFamily: 'Roboto, sans-serif' }}>{path.duration}</span>
                            </div>
                            <div className="flex items-center space-x-2 text-sm text-gray-600">
                              <span className="material-icons text-green-600">library_books</span>
                              <span style={{ fontFamily: 'Roboto, sans-serif' }}>{path.modules} Modules</span>
                            </div>
                            <div className="flex items-center space-x-2 text-sm text-gray-600">
                              <span className="material-icons text-purple-600">bolt</span>
                              <span style={{ fontFamily: 'Roboto, sans-serif' }}>{path.difficulty}</span>
                            </div>
                            <div className="flex items-center space-x-2 text-sm text-gray-600">
                              <span className="material-icons text-yellow-600">star</span>
                              <span style={{ fontFamily: 'Roboto, sans-serif' }}>{path.rating}/5.0</span>
                            </div>
                          </div>

                          <div className="space-y-4">
                            <h4 
                              className="text-lg font-medium text-gray-900"
                              style={{ fontFamily: 'Google Sans, sans-serif' }}
                            >
                              Skills You'll Gain
                            </h4>
                            <div className="flex flex-wrap gap-2">
                              {path.skillsGained.map((skill, skillIndex) => (
                                <Badge 
                                  key={skillIndex} 
                                  variant="secondary" 
                                  className="bg-blue-50 text-blue-700 border border-blue-200 rounded-full px-3 py-1 text-xs font-medium"
                                >
                                  {skill}
                                </Badge>
                              ))}
                            </div>
                          </div>

                          <div className="flex space-x-4">
                            <Button 
                              className={`flex-1 h-12 bg-gradient-to-r ${path.gradient} hover:shadow-lg text-white rounded-full shadow-md transition-all duration-300 group`}
                              style={{ fontFamily: 'Google Sans, sans-serif' }}
                            >
                              <span className="material-icons mr-3">play_arrow</span>
                              Start Learning
                              <ArrowRight className="ml-3 h-5 w-5 transition-transform duration-300 group-hover:translate-x-1" />
                            </Button>
                            <Button 
                              variant="outline" 
                              size="icon"
                              className="h-12 w-12 rounded-full border-2 border-gray-300 hover:bg-gray-50 transition-all duration-300"
                            >
                              <span className="material-icons">play_circle</span>
                            </Button>
                          </div>
                        </CardContent>
                      </Card>
                    ))}
                  </div>

                  <div className="text-center space-y-8 bg-gradient-to-br from-blue-50 to-purple-50 rounded-3xl p-12 border border-blue-100">
                    <h3 
                      className="text-3xl font-medium text-gray-900"
                      style={{ fontFamily: 'Google Sans, sans-serif' }}
                    >
                      Ready to Bridge Your Skill Gaps, {user?.firstName || 'User'}?
                    </h3>
                    <p 
                      className="text-gray-600 max-w-2xl mx-auto leading-relaxed text-lg"
                      style={{ fontFamily: 'Roboto, sans-serif' }}
                    >
                      Take action on your skill development with our personalized learning paths and 
                      <span className="font-medium text-blue-600"> expert guidance</span>.
                    </p>
                    <div className="flex flex-col sm:flex-row gap-4 justify-center">
                      <Button 
                        size="lg" 
                        className="h-12 px-8 bg-gradient-to-r from-green-600 to-green-700 hover:from-green-700 hover:to-green-800 text-white rounded-full shadow-lg hover:shadow-xl transition-all duration-300"
                        style={{ fontFamily: 'Google Sans, sans-serif' }}
                      >
                        <span className="material-icons mr-3">check_circle</span>
                        Create Learning Plan
                      </Button>
                      <Button 
                        size="lg" 
                        variant="outline"
                        className="h-12 px-8 border-2 border-gray-300 hover:bg-white hover:border-blue-300 hover:text-blue-600 rounded-full transition-all duration-300"
                        style={{ fontFamily: 'Google Sans, sans-serif' }}
                      >
                        <span className="material-icons mr-3">people</span>
                        Get Expert Guidance
                      </Button>
                    </div>
                  </div>
                </TabsContent>
              </Tabs>
            </div>
          </section>
        </main>
      </div>
    </>
  );
};

export default SkillsAnalysis;
