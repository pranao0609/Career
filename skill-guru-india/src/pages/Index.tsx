import Header from "@/components/Header";
import Hero from "@/components/Hero";
import Features from "@/components/Features";
import Footer from "@/components/Footer";
import JoinCommunityButton from "../components/JoinCommunityButton";


const Index = () => {
  return (
    <div className="min-h-screen bg-background">
      <Header />
      <main>
        <Hero />
        <Features />
        <JoinCommunityButton />
      </main>
      <Footer />
      
    </div>
  );
};

export default Index;