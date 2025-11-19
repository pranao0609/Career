import { useState } from "react";
import { PersonalInfoForm } from "@/components/PersonalInfoForm";
import { CareerInfoForm } from "@/components/CareerInfoForm";
import { DocumentUpload } from "@/components/DocumentUpload";
import Profile from "./Profile";
import axios from "axios";

type OnboardingStep = "personal" | "career" | "documents" | "completed";

const Onboarding = () => {
  const [currentStep, setCurrentStep] = useState<OnboardingStep>("personal");
  const [userEmail, setUserEmail] = useState<string | null>(null); // ✅ Changed to userEmail

  const handleStepComplete = (nextStep: OnboardingStep) => {
    setCurrentStep(nextStep);
  };

  const handleStepBack = (prevStep: OnboardingStep) => {
    setCurrentStep(prevStep);
  };

  const renderCurrentStep = () => {
    switch (currentStep) {
      case "personal":
        return (
          <PersonalInfoForm
            onNext={(email: string) => {
              console.log("📧 Setting user email:", email);
              setUserEmail(email); // ✅ Save user email globally
              handleStepComplete("career");
            }}
            onBack={() => {}}
          />
        );

      case "career":
        return (
          <CareerInfoForm
            userEmail={userEmail || ""} // ✅ Pass userEmail correctly
            onNext={() => handleStepComplete("documents")}
            onBack={() => handleStepBack("personal")}
          />
        );

      case "documents":
        return (
          <DocumentUpload
            userEmail={userEmail || ""} // ✅ Pass userEmail correctly (update DocumentUpload props too)
            onNext={() => handleStepComplete("completed")}
            onBack={() => handleStepBack("career")}
          />
        );

      case "completed":
        return <Profile userEmail={userEmail} />; // ✅ Pass userEmail to Profile if needed
    }
  };

  // Debug info (remove in production)
  console.log("🔍 Current step:", currentStep, "User Email:", userEmail);

  return (
    <div>
      {/* Debug panel - remove in production */}
      {userEmail && (
        <div className="fixed top-4 right-4 bg-blue-100 p-2 rounded text-xs z-50">
          <strong>Debug:</strong> {userEmail}
        </div>
      )}
      {renderCurrentStep()}
    </div>
  );
};

export default Onboarding;