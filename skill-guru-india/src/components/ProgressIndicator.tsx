import { Progress } from "@/components/ui/progress";

interface ProgressIndicatorProps {
  currentStep: number;
  totalSteps: number;
  stepLabels?: string[];
}

export const ProgressIndicator = ({ 
  currentStep, 
  totalSteps, 
  stepLabels = [] 
}: ProgressIndicatorProps) => {
  const progress = (currentStep / totalSteps) * 100;
  
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

      <div className="w-full max-w-4xl mx-auto mb-12">
        {/* Google Material Progress Header */}
        <div className="flex justify-between items-center mb-6">
          <div className="flex items-center space-x-3">
            <div className="flex items-center space-x-2 bg-white px-4 py-2 rounded-full border border-gray-200 shadow-sm">
              <span className="material-icons text-blue-600 text-base">schedule</span>
              <span 
                className="text-sm font-medium text-gray-700"
                style={{ fontFamily: 'Google Sans, sans-serif' }}
              >
                Step {currentStep} of {totalSteps}
              </span>
            </div>
          </div>
          
          <div className="flex items-center space-x-2 bg-white px-4 py-2 rounded-full border border-green-200 shadow-sm">
            <span className="material-icons text-green-600 text-base">trending_up</span>
            <span 
              className="text-sm font-bold text-green-700"
              style={{ fontFamily: 'Google Sans, sans-serif' }}
            >
              {Math.round(progress)}% Complete
            </span>
          </div>
        </div>
        
        {/* Google Material Progress Bar */}
        <div className="relative">
          {/* Background Track */}
          <div className="w-full h-3 bg-gray-200 rounded-full overflow-hidden shadow-inner">
            {/* Progress Fill with Gradient */}
            <div 
              className="h-full bg-gradient-to-r from-blue-500 via-blue-600 to-green-500 rounded-full transition-all duration-700 ease-out relative overflow-hidden"
              style={{ width: `${progress}%` }}
            >
              {/* Animated Shimmer Effect */}
              <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent animate-pulse" />
            </div>
          </div>
          
          {/* Progress Dots */}
          <div className="absolute top-1/2 transform -translate-y-1/2 w-full flex justify-between px-1">
            {Array.from({ length: totalSteps }, (_, index) => {
              const stepNumber = index + 1;
              const isCompleted = stepNumber <= currentStep;
              const isActive = stepNumber === currentStep;
              
              return (
                <div 
                  key={index}
                  className={`w-4 h-4 rounded-full border-2 transition-all duration-300 ${
                    isCompleted 
                      ? 'bg-gradient-to-br from-blue-500 to-blue-600 border-blue-600 shadow-lg' 
                      : 'bg-white border-gray-300 shadow-sm'
                  }`}
                >
                  {isCompleted && (
                    <div className="w-full h-full flex items-center justify-center">
                      <span className="material-icons text-white text-xs">check</span>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
        
        {/* Google Material Step Labels */}
        {stepLabels.length > 0 && (
          <div className="flex justify-between mt-6 px-2">
            {stepLabels.map((label, index) => {
              const stepNumber = index + 1;
              const isCompleted = stepNumber < currentStep;
              const isActive = stepNumber === currentStep;
              const isUpcoming = stepNumber > currentStep;
              
              return (
                <div key={index} className="flex flex-col items-center space-y-2 max-w-[120px]">
                  {/* Step Icon */}
                  <div className={`w-10 h-10 rounded-full flex items-center justify-center transition-all duration-300 ${
                    isCompleted 
                      ? 'bg-gradient-to-br from-green-500 to-green-600 shadow-lg' 
                      : isActive 
                        ? 'bg-gradient-to-br from-blue-600 to-blue-700 shadow-lg ring-4 ring-blue-100' 
                        : 'bg-gray-200 border-2 border-gray-300'
                  }`}>
                    {isCompleted ? (
                      <span className="material-icons text-white text-lg">check</span>
                    ) : (
                      <span className={`font-medium text-sm ${
                        isActive ? 'text-white' : isUpcoming ? 'text-gray-500' : 'text-gray-600'
                      }`} style={{ fontFamily: 'Google Sans, sans-serif' }}>
                        {stepNumber}
                      </span>
                    )}
                  </div>
                  
                  {/* Step Label */}
                  <span
                    className={`text-sm text-center leading-tight font-medium transition-colors duration-300 ${
                      isCompleted 
                        ? "text-green-600" 
                        : isActive 
                          ? "text-blue-600" 
                          : "text-gray-500"
                    }`}
                    style={{ fontFamily: 'Roboto, sans-serif' }}
                  >
                    {label}
                  </span>
                  
                  {/* Status Indicator */}
                  <div className={`w-2 h-2 rounded-full transition-all duration-300 ${
                    isCompleted 
                      ? 'bg-green-400' 
                      : isActive 
                        ? 'bg-blue-500 animate-pulse' 
                        : 'bg-gray-300'
                  }`} />
                </div>
              );
            })}
          </div>
        )}

        {/* Google Material Motivational Message */}
        <div className="mt-8 text-center">
          <div className="inline-flex items-center space-x-2 bg-gradient-to-r from-blue-50 to-purple-50 px-6 py-3 rounded-full border border-blue-200">
            <span className="material-icons text-blue-600 text-lg">emoji_events</span>
            <span 
              className="text-sm font-medium text-blue-700"
              style={{ fontFamily: 'Roboto, sans-serif' }}
            >
              {progress < 50 
                ? "Great start! Keep going to complete your profile." 
                : progress < 100 
                  ? "You're almost there! Just a few more steps." 
                  : "Congratulations! Your profile is complete."
              }
            </span>
          </div>
        </div>
      </div>
    </>
  );
};
