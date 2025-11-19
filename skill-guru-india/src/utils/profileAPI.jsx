import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import Header from "@/components/Header";
import { useToast } from "@/hooks/use-toast";
import { profileAPI } from "@/utils/profileAPI";

// ... (keep all the existing interfaces and initial profile state)

const Profile = () => {
  const { toast } = useToast();
  const [isEditing, setIsEditing] = useState(false);
  const [editSection, setEditSection] = useState(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);

  // For demo purposes - in production, get this from auth context
  const userId = "demo-user-123";

  // Initialize with empty profile structure
  const [profile, setProfile] = useState({
    personalInfo: {
      name: "",
      email: "",
      phone: "",
      location: "",
      bio: "",
      avatar: "",
      headline: ""
    },
    careerInfo: {
      currentRole: "",
      experience: "",
      industry: "",
      expectedSalary: "",
      preferredLocation: ""
    },
    skills: [],
    education: [],
    achievements: [],
    careerGoals: {
      shortTerm: "",
      longTerm: "",
      interests: []
    }
  });

  // State for adding new items
  const [newSkill, setNewSkill] = useState({ name: "", level: 50, category: 'technical' });
  const [newEducation, setNewEducation] = useState({ degree: "", institution: "", year: "", grade: "" });
  const [newAchievement, setNewAchievement] = useState({ title: "", description: "", date: "" });

  // Load profile data on component mount
  useEffect(() => {
    loadProfile();
  }, []);

  const loadProfile = async () => {
    try {
      setLoading(true);
      const profileData = await profileAPI.getProfile(userId);
      
      if (profileData && profileData.profile) {
        setProfile(profileData.profile);
        toast({
          title: "Profile Loaded",
          description: "Your profile has been loaded successfully.",
        });
      }
    } catch (error) {
      if (error.message.includes('404')) {
        // Profile doesn't exist yet - that's ok, user can create one
        console.log("No existing profile found, user can create new one");
      } else {
        toast({
          title: "Error Loading Profile",
          description: "Failed to load your profile. Please try again.",
          variant: "destructive",
        });
        console.error("Failed to load profile:", error);
      }
    } finally {
      setLoading(false);
    }
  };

  const saveProfile = async () => {
    try {
      setSaving(true);
      
      // Check if profile exists first
      const existsResponse = await profileAPI.checkProfileExists(userId);
      
      if (existsResponse.exists) {
        // Update existing profile
        await profileAPI.updateProfile(userId, profile);
        toast({
          title: "Profile Updated",
          description: "Your profile has been updated successfully.",
        });
      } else {
        // Create new profile
        await profileAPI.createProfile(userId, profile);
        toast({
          title: "Profile Created",
          description: "Your profile has been created successfully.",
        });
      }
    } catch (error) {
      toast({
        title: "Save Failed",
        description: "Failed to save your profile. Please try again.",
        variant: "destructive",
      });
      console.error("Failed to save profile:", error);
    } finally {
      setSaving(false);
    }
  };

  const handleSave = async (section) => {
    await saveProfile();
    setEditSection(null);
  };

  // Auto-save when profile changes (debounced)
  useEffect(() => {
    if (loading) return; // Don't auto-save during initial load

    const saveTimeout = setTimeout(() => {
      saveProfile();
    }, 2000); // Auto-save after 2 seconds of no changes

    return () => clearTimeout(saveTimeout);
  }, [profile]); // This will trigger whenever profile state changes

  // ... (keep all existing helper functions like addSkill, removeSkill, etc.)
  const addSkill = () => {
    if (newSkill.name.trim()) {
      setProfile(prev => ({
        ...prev,
        skills: [...prev.skills, { ...newSkill }]
      }));
      setNewSkill({ name: "", level: 50, category: 'technical' });
    }
  };

  const removeSkill = (index) => {
    setProfile(prev => ({
      ...prev,
      skills: prev.skills.filter((_, i) => i !== index)
    }));
  };

  const addEducation = () => {
    if (newEducation.degree.trim() && newEducation.institution.trim()) {
      setProfile(prev => ({
        ...prev,
        education: [...prev.education, { ...newEducation }]
      }));
      setNewEducation({ degree: "", institution: "", year: "", grade: "" });
    }
  };

  const addAchievement = () => {
    if (newAchievement.title.trim()) {
      setProfile(prev => ({
        ...prev,
        achievements: [...prev.achievements, { ...newAchievement }]
      }));
      setNewAchievement({ title: "", description: "", date: "" });
    }
  };

  const EditableField = ({ 
    value, 
    onChange, 
    type = "text", 
    multiline = false,
    placeholder = ""
  }) => {
    if (multiline) {
      return (
        <Textarea
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder={placeholder}
          className="min-h-20 rounded-2xl border-2 border-gray-200 focus:border-blue-500 focus:ring-0 bg-white shadow-sm"
          style={{ fontFamily: 'Roboto, sans-serif' }}
        />
      );
    }
    return (
      <Input
        type={type}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        className="h-12 rounded-2xl border-2 border-gray-200 focus:border-blue-500 focus:ring-0 bg-white shadow-sm"
        style={{ fontFamily: 'Roboto, sans-serif' }}
      />
    );
  };

  const getSkillColor = (level) => {
    if (level >= 80) return "bg-gradient-to-r from-green-500 to-green-600";
    if (level >= 60) return "bg-gradient-to-r from-blue-500 to-blue-600";
    if (level >= 40) return "bg-gradient-to-r from-yellow-500 to-orange-500";
    return "bg-gradient-to-r from-red-500 to-red-600";
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-50 via-blue-50 to-white flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-xl text-gray-600">Loading your profile...</p>
        </div>
      </div>
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
        
        {/* Save Status Indicator */}
        {saving && (
          <div className="fixed top-4 right-4 z-50 bg-blue-600 text-white px-4 py-2 rounded-full shadow-lg">
            <div className="flex items-center gap-2">
              <div className="animate-spin h-4 w-4 border-2 border-white border-t-transparent rounded-full"></div>
              Saving...
            </div>
          </div>
        )}

        {/* Hero Section */}
        <div className="relative overflow-hidden bg-gradient-to-br from-blue-600 to-purple-700">
          {/* Background decoration */}
          <div className="absolute inset-0 overflow-hidden">
            <div className="absolute top-1/4 left-10 w-96 h-96 rounded-full bg-white/10 blur-3xl" />
            <div className="absolute bottom-1/4 right-10 w-80 h-80 rounded-full bg-white/5 blur-3xl" />
          </div>
          
          <div className="relative container mx-auto px-8 py-16">
            <div className="flex flex-col lg:flex-row items-center gap-12">
              {/* Profile Picture & Basic Info */}
              <div className="flex flex-col items-center text-center lg:text-left">
                <div className="relative group mb-6">
                  <div className="absolute -inset-4 bg-white/20 rounded-full blur-xl group-hover:blur-2xl transition-all duration-300" />
                  <Avatar className="relative h-40 w-40 ring-4 ring-white/30 shadow-2xl">
                    <AvatarImage src={profile.personalInfo.avatar} />
                    <AvatarFallback 
                      className="text-3xl bg-gradient-to-br from-white to-gray-100 text-blue-600 shadow-lg"
                      style={{ fontFamily: 'Google Sans, sans-serif' }}
                    >
                      {profile.personalInfo.name ? profile.personalInfo.name.split(' ').map(n => n[0]).join('') : 'U'}
                    </AvatarFallback>
                  </Avatar>
                  {isEditing && (
                    <Button 
                      size="icon" 
                      className="absolute -bottom-2 -right-2 h-12 w-12 rounded-full bg-white text-blue-600 hover:bg-blue-50 shadow-lg"
                    >
                      <span className="material-icons">photo_camera</span>
                    </Button>
                  )}
                </div>
                
                <div className="space-y-3 text-white">
                  <h1 
                    className="text-4xl lg:text-5xl font-medium"
                    style={{ fontFamily: 'Google Sans, sans-serif' }}
                  >
                    {profile.personalInfo.name || "Your Name"}
                  </h1>
                  <p 
                    className="text-xl text-blue-100 font-medium"
                    style={{ fontFamily: 'Roboto, sans-serif' }}
                  >
                    {profile.personalInfo.headline || "Your Professional Headline"}
                  </p>
                  <div className="flex items-center gap-2 text-blue-100 justify-center lg:justify-start">
                    <span className="material-icons text-lg">location_on</span>
                    <span style={{ fontFamily: 'Roboto, sans-serif' }}>
                      {profile.personalInfo.location || "Your Location"}
                    </span>
                  </div>
                </div>
              </div>

              {/* Quick Stats Cards */}
              <div className="flex-1 grid grid-cols-1 md:grid-cols-3 gap-6 w-full">
                <div className="bg-white/10 backdrop-blur-md rounded-3xl p-8 text-center border border-white/20 shadow-lg">
                  <div 
                    className="text-4xl font-bold text-white mb-2"
                    style={{ fontFamily: 'Google Sans, sans-serif' }}
                  >
                    {profile.careerInfo.experience || "0+"}
                  </div>
                  <div 
                    className="text-blue-100"
                    style={{ fontFamily: 'Roboto, sans-serif' }}
                  >
                    Experience
                  </div>
                </div>
                
                <div className="bg-white/10 backdrop-blur-md rounded-3xl p-8 text-center border border-white/20 shadow-lg">
                  <div 
                    className="text-4xl font-bold text-white mb-2"
                    style={{ fontFamily: 'Google Sans, sans-serif' }}
                  >
                    {profile.skills.length}
                  </div>
                  <div 
                    className="text-blue-100"
                    style={{ fontFamily: 'Roboto, sans-serif' }}
                  >
                    Skills
                  </div>
                </div>
                
                <div className="bg-white/10 backdrop-blur-md rounded-3xl p-8 text-center border border-white/20 shadow-lg">
                  <div 
                    className="text-4xl font-bold text-white mb-2"
                    style={{ fontFamily: 'Google Sans, sans-serif' }}
                  >
                    {profile.achievements.length}
                  </div>
                  <div 
                    className="text-blue-100"
                    style={{ fontFamily: 'Roboto, sans-serif' }}
                  >
                    Achievements
                  </div>
                </div>
              </div>

              {/* Action Button */}
              <div className="lg:ml-8">
                <Button 
                  onClick={() => setIsEditing(!isEditing)}
                  size="lg"
                  className={`h-14 px-8 rounded-full shadow-xl hover:shadow-2xl transition-all duration-300 ${
                    isEditing 
                      ? 'bg-white text-blue-600 hover:bg-gray-50' 
                      : 'bg-white text-blue-600 hover:bg-blue-50'
                  }`}
                  style={{ fontFamily: 'Google Sans, sans-serif' }}
                >
                  {isEditing ? (
                    <>
                      <span className="material-icons mr-2">close</span>
                      Cancel Editing
                    </>
                  ) : (
                    <>
                      <span className="material-icons mr-2">edit</span>
                      Edit Profile
                    </>
                  )}
                </Button>
              </div>
            </div>
          </div>
        </div>

        {/* Main Content */}
        <div className="container mx-auto px-8 py-12">
          <Tabs defaultValue="overview" className="space-y-8">
            <TabsList className="grid w-full max-w-2xl mx-auto grid-cols-4 bg-gray-100 p-1 rounded-full">
              <TabsTrigger 
                value="overview"
                className="rounded-full data-[state=active]:bg-white data-[state=active]:shadow-sm"
                style={{ fontFamily: 'Google Sans, sans-serif' }}
              >
                <span className="material-icons mr-2 text-base">dashboard</span>
                Overview
              </TabsTrigger>
              <TabsTrigger 
                value="skills"
                className="rounded-full data-[state=active]:bg-white data-[state=active]:shadow-sm"
                style={{ fontFamily: 'Google Sans, sans-serif' }}
              >
                <span className="material-icons mr-2 text-base">psychology</span>
                Skills
              </TabsTrigger>
              <TabsTrigger 
                value="experience"
                className="rounded-full data-[state=active]:bg-white data-[state=active]:shadow-sm"
                style={{ fontFamily: 'Google Sans, sans-serif' }}
              >
                <span className="material-icons mr-2 text-base">work</span>
                Experience
              </TabsTrigger>
              <TabsTrigger 
                value="goals"
                className="rounded-full data-[state=active]:bg-white data-[state=active]:shadow-sm"
                style={{ fontFamily: 'Google Sans, sans-serif' }}
              >
                <span className="material-icons mr-2 text-base">flag</span>
                Goals
              </TabsTrigger>
            </TabsList>

            {/* Keep all the existing tab content with the same structure but using the profile state */}
            {/* ... rest of the component stays the same ... */}
          </Tabs>
        </div>
      </div>
    </>
  );
};

export default Profile;