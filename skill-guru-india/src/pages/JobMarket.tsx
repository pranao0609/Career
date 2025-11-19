import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import Header from "@/components/Header";
import { 
  MapPin, 
  Briefcase, 
  TrendingUp, 
  Clock,
  Users,
  DollarSign,
  Search,
  Filter,
  Star,
  ExternalLink,
  Calendar,
  BarChart3,
  Target,
  Building,
  Globe,
  ArrowUpRight,
  RefreshCw,
  Heart,
  Share2,
  Bookmark,
  Eye,
  ChevronRight,
  Navigation,
  CheckCircle,
  FileText,
  UserCheck,
  Network,
  BookOpen,
  Lightbulb,
  Newspaper,
  TrendingDown
} from "lucide-react";

// Interfaces remain the same...
interface Job {
  id: string;
  title: string;
  company: string;
  location: string;
  distance?: string;
  salary: string;
  type: string;
  postedTime: string;
  description: string;
  skills: string[];
  logo?: string;
  urgent?: boolean;
  remote?: boolean;
}

interface MarketTrend {
  id: string;
  title: string;
  description: string;
  trend: 'up' | 'down' | 'stable';
  percentage: number;
  category: string;
  timeframe: string;
}

interface NewsItem {
  id: string;
  title: string;
  summary: string;
  source: string;
  publishedAt: string;
  imageUrl?: string;
  category: string;
  readTime: string;
}

interface JobPlatform {
  id: string;
  name: string;
  description: string;
  url: string;
  logoUrl: string;
  speciality: string;
  activeJobs: string;
  color: string;
}

interface JobSearchTip {
  id: string;
  title: string;
  description: string;
  icon: string;
  category: string;
}

const JobMarket = () => {
  const [roleBasedJobs, setRoleBasedJobs] = useState<Job[]>([]);
  const [marketTrends, setMarketTrends] = useState<MarketTrend[]>([]);
  const [marketNews, setMarketNews] = useState<NewsItem[]>([]);
  const [jobPlatforms, setJobPlatforms] = useState<JobPlatform[]>([]);
  const [jobSearchTips, setJobSearchTips] = useState<JobSearchTip[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedRole, setSelectedRole] = useState("All Roles");
  const [selectedLocation, setSelectedLocation] = useState("Current Location");
  const [searchQuery, setSearchQuery] = useState("");

  useEffect(() => {
    fetchJobMarketData();
  }, []);

  const fetchJobMarketData = async () => {
    try {
      setLoading(true);
      
      // Simulate API calls with sample data
      setTimeout(() => {
        setRoleBasedJobs(sampleRoleBasedJobs);
        setMarketTrends(sampleMarketTrends);
        setMarketNews(sampleMarketNews);
        setJobPlatforms(sampleJobPlatforms);
        setJobSearchTips(sampleJobSearchTips);
        setLoading(false);
      }, 1500);
      
    } catch (error) {
      console.error('Error fetching job market data:', error);
      setLoading(false);
    }
  };

  // Sample data for job platforms with real logos
  const sampleJobPlatforms: JobPlatform[] = [
    {
      id: '1',
      name: 'Naukri.com',
      description: 'India\'s leading job portal with 50M+ registered users',
      url: 'https://naukri.com',
      logoUrl: 'https://static.naukimg.com/s/0/0/i/naukri_Logo_120x30.gif',
      speciality: 'All Industries',
      activeJobs: '2.5M+',
      color: 'from-blue-500 to-blue-600'
    },
    {
      id: '2',
      name: 'LinkedIn',
      description: 'Professional networking platform with global opportunities',
      url: 'https://linkedin.com/jobs',
      logoUrl: 'https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png',
      speciality: 'Professional Network',
      activeJobs: '15M+',
      color: 'from-blue-700 to-blue-800'
    },
    {
      id: '3',
      name: 'Indeed',
      description: 'World\'s largest job search engine',
      url: 'https://indeed.co.in',
      logoUrl: 'https://upload.wikimedia.org/wikipedia/commons/f/fc/Indeed_logo.svg',
      speciality: 'Global Jobs',
      activeJobs: '10M+',
      color: 'from-blue-600 to-indigo-600'
    },
    {
      id: '4',
      name: 'Glassdoor',
      description: 'Company reviews, salaries, and job listings',
      url: 'https://glassdoor.co.in',
      logoUrl: 'https://media.glassdoor.com/lst2x/glassdoor_logo_80x80.png',
      speciality: 'Company Insights',
      activeJobs: '1M+',
      color: 'from-green-500 to-emerald-600'
    },
    {
      id: '5',
      name: 'AngelList',
      description: 'Startup jobs and equity opportunities',
      url: 'https://angel.co',
      logoUrl: 'https://images.crunchbase.com/image/upload/c_lpad,f_auto,q_auto:eco,dpr_1/erkxwhl1gd48xfhe2yld',
      speciality: 'Startups & Tech',
      activeJobs: '100K+',
      color: 'from-black to-gray-700'
    }
  ];

  // Sample data for job search tips
  const sampleJobSearchTips: JobSearchTip[] = [
    {
      id: '1',
      title: 'Optimize Your Resume',
      description: 'Use ATS-friendly formats, relevant keywords, and quantify achievements with numbers and metrics.',
      icon: 'description',
      category: 'Resume'
    },
    {
      id: '2',
      title: 'Build Your Network',
      description: 'Connect with professionals in your field, attend industry events, and leverage referrals.',
      icon: 'people',
      category: 'Networking'
    },
    {
      id: '3',
      title: 'Research Companies',
      description: 'Study company culture, values, recent news, and interview processes before applying.',
      icon: 'business',
      category: 'Research'
    },
    {
      id: '4',
      title: 'Prepare for Interviews',
      description: 'Practice common questions, prepare examples using STAR method, and research interviewers.',
      icon: 'psychology',
      category: 'Interview'
    },
    {
      id: '5',
      title: 'Use Multiple Platforms',
      description: 'Apply through various job boards, company websites, and professional networks simultaneously.',
      icon: 'hub',
      category: 'Strategy'
    },
    {
      id: '6',
      title: 'Follow Up Professionally',
      description: 'Send thank-you emails after interviews and follow up on applications appropriately.',
      icon: 'mail',
      category: 'Communication'
    }
  ];

  const sampleRoleBasedJobs: Job[] = [
    {
      id: '4',
      title: 'Full Stack Developer',
      company: 'Startup Hub',
      location: 'Bangalore',
      salary: '₹10-15 LPA',
      type: 'Full-time',
      postedTime: '3 hours ago',
      description: 'Build scalable web applications using modern tech stack.',
      skills: ['React', 'Node.js', 'MongoDB', 'AWS'],
      remote: true
    },
    {
      id: '5',
      title: 'UI/UX Designer',
      company: 'Design Studio',
      location: 'Pune',
      salary: '₹8-12 LPA',
      type: 'Full-time',
      postedTime: '6 hours ago',
      description: 'Create beautiful and intuitive user experiences.',
      skills: ['Figma', 'Adobe XD', 'Prototyping', 'User Research']
    },
    {
      id: '6',
      title: 'Data Analyst',
      company: 'Analytics Corp',
      location: 'Hyderabad',
      salary: '₹6-10 LPA',
      type: 'Full-time',
      postedTime: '1 day ago',
      description: 'Analyze data to drive business insights and decisions.',
      skills: ['Python', 'SQL', 'Tableau', 'Statistics'],
      remote: true
    }
  ];

  const sampleMarketTrends: MarketTrend[] = [
    {
      id: '1',
      title: 'AI/ML Jobs',
      description: 'Strong growth in AI and ML positions',
      trend: 'up',
      percentage: 35,
      category: 'Technology',
      timeframe: 'Last 3 months'
    },
    {
      id: '2',
      title: 'Remote Work',
      description: 'Remote positions increasing across sectors',
      trend: 'up',
      percentage: 42,
      category: 'Work Culture',
      timeframe: 'Last 6 months'
    }
  ];

  const sampleMarketNews: NewsItem[] = [
    {
      id: '1',
      title: 'Tech Giants Announce Major Hiring Drive for 2025',
      summary: 'Major technology companies plan to hire over 100,000 new employees across India.',
      source: 'TechNews Today',
      publishedAt: '2 hours ago',
      category: 'Technology',
      readTime: '3 min read'
    },
    {
      id: '2',
      title: 'Remote Work Policies: What Employees Can Expect',
      summary: 'How companies are adapting remote work policies in the post-pandemic era.',
      source: 'Career Insights',
      publishedAt: '5 hours ago',
      category: 'Work Culture',
      readTime: '5 min read'
    }
  ];

  const JobCard: React.FC<{ job: Job; showDistance?: boolean }> = ({ job, showDistance = false }) => (
    <Card className="group border-0 rounded-2xl shadow-md bg-white hover:shadow-lg transition-all duration-300">
      <CardHeader className="p-5">
        <div className="flex items-start justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl flex items-center justify-center shadow-sm">
              <Building className="w-5 h-5 text-white" />
            </div>
            <div className="flex-1">
              <CardTitle 
                className="text-lg font-medium text-gray-900 group-hover:text-blue-600 transition-colors duration-300 mb-1"
                style={{ fontFamily: 'Google Sans, sans-serif' }}
              >
                {job.title}
              </CardTitle>
              <p className="text-gray-600 text-sm" style={{ fontFamily: 'Roboto, sans-serif' }}>
                {job.company}
              </p>
            </div>
          </div>
          <div className="flex items-center space-x-2">
            {job.urgent && (
              <Badge className="bg-red-100 text-red-700 text-xs rounded-full px-2 py-1">
                Urgent
              </Badge>
            )}
            {job.remote && (
              <Badge className="bg-green-100 text-green-700 text-xs rounded-full px-2 py-1">
                Remote
              </Badge>
            )}
          </div>
        </div>
      </CardHeader>
      
      <CardContent className="p-5 pt-0 space-y-3">
        <div className="flex items-center justify-between text-sm text-gray-600">
          <div className="flex items-center space-x-3">
            <div className="flex items-center space-x-1">
              <MapPin className="w-4 h-4" />
              <span>{job.location}</span>
            </div>
            <div className="flex items-center space-x-1">
              <Clock className="w-4 h-4" />
              <span>{job.postedTime}</span>
            </div>
          </div>
          <div className="flex items-center space-x-1">
            <DollarSign className="w-4 h-4 text-green-600" />
            <span className="font-medium text-green-600">{job.salary}</span>
          </div>
        </div>

        <p className="text-gray-700 text-sm leading-relaxed" style={{ fontFamily: 'Roboto, sans-serif' }}>
          {job.description}
        </p>

        <div className="flex flex-wrap gap-2">
          {job.skills.slice(0, 4).map((skill, index) => (
            <Badge key={index} variant="secondary" className="bg-blue-50 text-blue-700 rounded-full px-2 py-1 text-xs">
              {skill}
            </Badge>
          ))}
        </div>

        <div className="flex items-center justify-between pt-3">
          <Badge variant="outline" className="rounded-full px-2 py-1 text-xs">
            {job.type}
          </Badge>
          <div className="flex items-center space-x-2">
            <Button variant="ghost" size="icon" className="h-8 w-8 rounded-full hover:bg-gray-100">
              <Bookmark className="h-4 w-4" />
            </Button>
            <Button variant="ghost" size="icon" className="h-8 w-8 rounded-full hover:bg-gray-100">
              <Share2 className="h-4 w-4" />
            </Button>
            <Button 
              className="bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-full px-4 py-2 text-sm hover:shadow-md transition-all duration-300"
              style={{ fontFamily: 'Google Sans, sans-serif' }}
            >
              Apply Now
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  );

  const TipCard: React.FC<{ tip: JobSearchTip }> = ({ tip }) => (
    <Card className="border-0 rounded-xl shadow-sm bg-white hover:shadow-md transition-all duration-300 group cursor-pointer">
      <CardContent className="p-4">
        <div className="flex items-start space-x-3">
          <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl flex items-center justify-center shadow-sm flex-shrink-0">
            <span className="material-icons text-white text-lg">{tip.icon}</span>
          </div>
          <div className="flex-1">
            <div className="flex items-center justify-between mb-2">
              <h4 className="font-semibold text-base text-gray-900 group-hover:text-blue-600 transition-colors duration-300" 
                  style={{ fontFamily: 'Google Sans, sans-serif' }}>
                {tip.title}
              </h4>
              <Badge variant="outline" className="text-xs rounded-full">
                {tip.category}
              </Badge>
            </div>
            <p className="text-gray-600 text-sm leading-relaxed" style={{ fontFamily: 'Roboto, sans-serif' }}>
              {tip.description}
            </p>
          </div>
        </div>
      </CardContent>
    </Card>
  );

  const PlatformCard: React.FC<{ platform: JobPlatform }> = ({ platform }) => (
    <Card className="border-0 rounded-xl shadow-sm bg-white hover:shadow-md transition-all duration-300 group cursor-pointer">
      <CardContent className="p-4">
        <div className="flex items-center space-x-3 mb-3">
          <div className="w-12 h-12 bg-white border border-gray-200 rounded-xl flex items-center justify-center shadow-sm overflow-hidden">
            <img 
              src={platform.logoUrl} 
              alt={`${platform.name} logo`}
              className="w-8 h-8 object-contain"
              onError={(e) => {
                const target = e.target as HTMLImageElement;
                target.style.display = 'none';
                target.nextElementSibling!.classList.remove('hidden');
              }}
            />
            <div className={`hidden w-8 h-8 bg-gradient-to-br ${platform.color} rounded-lg flex items-center justify-center text-white text-xs font-bold`}>
              {platform.name.substring(0, 2).toUpperCase()}
            </div>
          </div>
          <div className="flex-1">
            <h4 className="font-semibold text-base text-gray-900 group-hover:text-blue-600 transition-colors duration-300" 
                style={{ fontFamily: 'Google Sans, sans-serif' }}>
              {platform.name}
            </h4>
            <div className="flex items-center space-x-2 text-xs text-gray-500">
              <span>{platform.speciality}</span>
              <span>•</span>
              <span className="font-medium text-green-600">{platform.activeJobs} jobs</span>
            </div>
          </div>
          <ExternalLink className="w-4 h-4 text-gray-400 group-hover:text-blue-600 transition-colors duration-300" />
        </div>
        <p className="text-gray-600 text-sm leading-relaxed mb-3" style={{ fontFamily: 'Roboto, sans-serif' }}>
          {platform.description}
        </p>
        <Button 
          className="w-full bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-full text-sm hover:shadow-md transition-all duration-300"
          style={{ fontFamily: 'Google Sans, sans-serif' }}
          onClick={() => window.open(platform.url, '_blank')}
        >
          Visit Platform
        </Button>
      </CardContent>
    </Card>
  );

  const TrendCard: React.FC<{ trend: MarketTrend }> = ({ trend }) => (
    <Card className="border-0 rounded-xl shadow-sm bg-gradient-to-br from-gray-50 to-blue-50 hover:shadow-md transition-all duration-300 hover:from-blue-50 hover:to-indigo-50">
      <CardContent className="p-5">
        <div className="flex items-center justify-between mb-4">
          <div className={`flex items-center space-x-3 ${
            trend.trend === 'up' ? 'text-green-600' :
            trend.trend === 'down' ? 'text-red-600' : 'text-gray-600'
          }`}>
            <div className={`w-10 h-10 rounded-full flex items-center justify-center ${
              trend.trend === 'up' ? 'bg-green-100' : 
              trend.trend === 'down' ? 'bg-red-100' : 'bg-gray-100'
            }`}>
              {trend.trend === 'up' ? (
                <TrendingUp className="w-5 h-5" />
              ) : trend.trend === 'down' ? (
                <TrendingDown className="w-5 h-5" />
              ) : (
                <BarChart3 className="w-5 h-5" />
              )}
            </div>
            <span className="font-bold text-2xl">{trend.percentage}%</span>
          </div>
          <Badge variant="outline" className="text-xs rounded-full bg-white">
            {trend.category}
          </Badge>
        </div>
        <h4 className="font-semibold text-lg text-gray-900 mb-2" style={{ fontFamily: 'Google Sans, sans-serif' }}>
          {trend.title}
        </h4>
        <p className="text-gray-600 text-sm mb-3 leading-relaxed" style={{ fontFamily: 'Roboto, sans-serif' }}>
          {trend.description}
        </p>
        <div className="flex items-center text-xs text-gray-500">
          <Clock className="w-3 h-3 mr-1" />
          <span>{trend.timeframe}</span>
        </div>
      </CardContent>
    </Card>
  );

  const NewsCard: React.FC<{ news: NewsItem }> = ({ news }) => (
    <div className="bg-white border-l-4 border-l-orange-400 rounded-r-xl shadow-sm hover:shadow-md transition-all duration-300 group cursor-pointer">
      <div className="p-5">
        <div className="flex items-start justify-between mb-3">
          <div className="flex items-center space-x-2">
            <div className="w-8 h-8 bg-orange-100 rounded-full flex items-center justify-center">
              <Newspaper className="w-4 h-4 text-orange-600" />
            </div>
            <Badge className="bg-orange-50 text-orange-700 text-xs rounded-full px-2 py-1">
              {news.category}
            </Badge>
          </div>
          <div className="text-xs text-gray-500 flex items-center">
            <Clock className="w-3 h-3 mr-1" />
            {news.publishedAt}
          </div>
        </div>
        
        <h4 className="font-bold text-lg text-gray-900 mb-2 group-hover:text-orange-600 transition-colors duration-300 leading-tight" 
            style={{ fontFamily: 'Google Sans, sans-serif' }}>
          {news.title}
        </h4>
        
        <p className="text-gray-700 text-sm leading-relaxed mb-4" style={{ fontFamily: 'Roboto, sans-serif' }}>
          {news.summary}
        </p>
        
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3 text-xs text-gray-500">
            <div className="flex items-center">
              <Globe className="w-3 h-3 mr-1" />
              <span className="font-medium">{news.source}</span>
            </div>
            <span>•</span>
            <div className="flex items-center">
              <Eye className="w-3 h-3 mr-1" />
              <span>{news.readTime}</span>
            </div>
          </div>
          <div className="flex items-center space-x-2">
            <Button variant="ghost" size="icon" className="h-7 w-7 rounded-full hover:bg-orange-50">
              <Heart className="h-3 w-3 text-gray-400 hover:text-orange-500" />
            </Button>
            <Button variant="ghost" size="icon" className="h-7 w-7 rounded-full hover:bg-orange-50">
              <Share2 className="h-3 w-3 text-gray-400 hover:text-orange-500" />
            </Button>
            <ArrowUpRight className="w-4 h-4 text-gray-400 group-hover:text-orange-600 transition-colors duration-300" />
          </div>
        </div>
      </div>
    </div>
  );

  if (loading) {
    return (
      <>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link 
          href="https://fonts.googleapis.com/css2?family=Google+Sans:wght@300;400;500;700&family=Roboto:wght@300;400;500;700&display=swap" 
          rel="stylesheet" 
        />
        
        <div className="min-h-screen bg-gradient-to-br from-gray-50 via-blue-50 to-white flex items-center justify-center">
          <div className="text-center space-y-4">
            <div className="relative">
              <div className="w-12 h-12 border-4 border-blue-200 border-t-blue-600 rounded-full animate-spin mx-auto"></div>
            </div>
            <div>
              <h3 className="text-lg font-medium text-gray-900 mb-2" style={{ fontFamily: 'Google Sans, sans-serif' }}>
                Loading Job Market Data
              </h3>
              <p className="text-sm text-gray-600" style={{ fontFamily: 'Roboto, sans-serif' }}>
                Fetching the latest opportunities...
              </p>
            </div>
          </div>
        </div>
      </>
    );
  }

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
        
        {/* Hero Section */}
        <section className="py-16 relative overflow-hidden">
          <div className="absolute inset-0 overflow-hidden pointer-events-none">
            <div className="absolute top-1/4 left-10 w-72 h-72 rounded-full bg-gradient-to-br from-blue-400/10 to-purple-400/10 blur-3xl" />
            <div className="absolute bottom-1/4 right-10 w-60 h-60 rounded-full bg-gradient-to-br from-green-400/8 to-yellow-400/8 blur-3xl" />
          </div>

          <div className="container px-6 lg:px-8 relative z-10">
            <div className="text-center space-y-6 max-w-4xl mx-auto">
              <div className="inline-flex items-center gap-2 px-4 py-2 bg-white border border-blue-200 rounded-full text-sm font-medium text-blue-700 shadow-sm">
                <span className="material-icons text-base">trending_up</span>
                <span style={{ fontFamily: 'Google Sans, sans-serif' }}>
                  Live Job Market Intelligence
                </span>
              </div>

              <h1 
                className="text-4xl lg:text-5xl font-normal text-gray-900 leading-tight"
                style={{ fontFamily: 'Google Sans, sans-serif' }}
              >
                Explore the
                <span className="block bg-gradient-to-r from-blue-600 via-purple-600 to-green-600 bg-clip-text text-transparent font-medium mt-2">
                  Job Market
                </span>
              </h1>

              <p 
                className="text-lg lg:text-xl text-gray-600 max-w-3xl mx-auto leading-relaxed"
                style={{ fontFamily: 'Roboto, sans-serif' }}
              >
                Discover opportunities, learn job search strategies, and stay updated with 
                the latest <span className="font-medium text-blue-600">industry insights</span>.
              </p>
            </div>
          </div>
        </section>

        {/* Main Content */}
        <section className="py-8">
          <div className="container px-6 lg:px-8">
            <div className="grid lg:grid-cols-3 gap-8">
              
              {/* Left Column - Main Content */}
              <div className="lg:col-span-2 space-y-8">
                
                {/* How to Find Jobs Section */}
                <div>
                  <div className="flex items-center justify-between mb-6">
                    <h2 
                      className="text-2xl font-medium text-gray-900 flex items-center"
                      style={{ fontFamily: 'Google Sans, sans-serif' }}
                    >
                      <Lightbulb className="w-6 h-6 mr-2 text-yellow-500" />
                      How to Find Jobs
                    </h2>
                    <Button variant="ghost" className="text-blue-600 hover:text-blue-800 text-sm">
                      View All Tips <ChevronRight className="w-4 h-4 ml-1" />
                    </Button>
                  </div>
                  <div className="grid md:grid-cols-2 gap-4">
                    {jobSearchTips.map((tip) => (
                      <TipCard key={tip.id} tip={tip} />
                    ))}
                  </div>
                </div>

               

                {/* Market News Section - New Design */}
                <div>
                  <div className="flex items-center justify-between mb-6">
                    <h2 
                      className="text-2xl font-medium text-gray-900 flex items-center"
                      style={{ fontFamily: 'Google Sans, sans-serif' }}
                    >
                      <Newspaper className="w-6 h-6 mr-2 text-orange-500" />
                      Market News
                    </h2>
                    <Button variant="ghost" size="icon" className="h-8 w-8 rounded-full">
                      <RefreshCw className="w-4 h-4" />
                    </Button>
                  </div>
                  <div className="space-y-4">
                    {marketNews.map((news) => (
                      <NewsCard key={news.id} news={news} />
                    ))}
                  </div>
                </div>

                 {/* Market Trends Section */}
                <div>
                  <div className="flex items-center justify-between mb-6">
                    <h2 
                      className="text-2xl font-medium text-gray-900 flex items-center"
                      style={{ fontFamily: 'Google Sans, sans-serif' }}
                    >
                      <BarChart3 className="w-6 h-6 mr-2 text-blue-600" />
                      Market Trends
                    </h2>
                  </div>
                  <div className="grid md:grid-cols-2 gap-4">
                    {marketTrends.map((trend) => (
                      <TrendCard key={trend.id} trend={trend} />
                    ))}
                  </div>
                </div>
              </div>

              {/* Right Column - Job Platforms and Market Overview */}
              <div className="space-y-6">
                
                {/* Top Job Platforms */}
                <Card className="border-0 rounded-2xl shadow-md bg-white sticky top-6">
                  <CardHeader className="p-5">
                    <CardTitle 
                      className="flex items-center text-lg font-medium text-gray-900"
                      style={{ fontFamily: 'Google Sans, sans-serif' }}
                    >
                      <Network className="w-5 h-5 mr-2 text-purple-600" />
                      Top Job Platforms
                    </CardTitle>
                    <CardDescription className="text-sm" style={{ fontFamily: 'Roboto, sans-serif' }}>
                      Popular job search platforms in India
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="p-5 pt-0 space-y-3">
                    {jobPlatforms.map((platform) => (
                      <PlatformCard key={platform.id} platform={platform} />
                    ))}
                  </CardContent>
                </Card>

            
                
              </div>
            </div>
          </div>
        </section>
      </div>

      {/* Custom Styles */}
      <style>{`
        .line-clamp-2 {
          display: -webkit-box;
          -webkit-line-clamp: 2;
          -webkit-box-orient: vertical;
          overflow: hidden;
        }
      `}</style>
    </>
  );
};

export default JobMarket;
