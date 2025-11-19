import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Mail, ArrowLeft } from "lucide-react";

interface ForgotPasswordDialogProps {
  children: React.ReactNode;
}

const ForgotPasswordDialog = ({ children }: ForgotPasswordDialogProps) => {
  const [email, setEmail] = useState("");
  const [isSubmitted, setIsSubmitted] = useState(false);
  const [isOpen, setIsOpen] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    try {
      // Simulate API call for password reset
      await new Promise(resolve => setTimeout(resolve, 1000));
      setIsSubmitted(true);
    } catch (error) {
      console.error("Password reset failed:", error);
    }
  };

  const handleClose = () => {
    setIsOpen(false);
    setIsSubmitted(false);
    setEmail("");
  };

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

      <Dialog open={isOpen} onOpenChange={setIsOpen}>
        <DialogTrigger asChild onClick={() => setIsOpen(true)}>
          {children}
        </DialogTrigger>
        
        <DialogContent className="sm:max-w-lg border-0 rounded-3xl shadow-2xl bg-white p-0 overflow-hidden">
          {/* Google-style Header */}
          <DialogHeader className="p-8 pb-0">
            <div className="flex flex-col items-center text-center space-y-4">
              {/* Icon */}
              <div className={`w-16 h-16 rounded-2xl flex items-center justify-center shadow-lg ${
                isSubmitted 
                  ? 'bg-gradient-to-br from-green-500 to-green-600' 
                  : 'bg-gradient-to-br from-blue-500 to-blue-600'
              }`}>
                <span className="material-icons text-white text-2xl">
                  {isSubmitted ? 'mark_email_read' : 'lock_reset'}
                </span>
              </div>
              
              {/* Title */}
              <DialogTitle 
                className="text-2xl font-medium text-gray-900"
                style={{ fontFamily: 'Google Sans, sans-serif' }}
              >
                {isSubmitted ? "Check Your Email" : "Reset Password"}
              </DialogTitle>
              
              {/* Description */}
              <DialogDescription 
                className="text-base text-gray-600 leading-relaxed max-w-md"
                style={{ fontFamily: 'Roboto, sans-serif' }}
              >
                {isSubmitted ? (
                  <>
                    We've sent password reset instructions to{" "}
                    <span className="font-medium text-gray-900">{email}</span>.
                    Check your inbox and follow the link to reset your password.
                  </>
                ) : (
                  "Enter your email address and we'll send you a secure link to reset your password."
                )}
              </DialogDescription>
            </div>
          </DialogHeader>

          <div className="p-8">
            {!isSubmitted ? (
              <form onSubmit={handleSubmit} className="space-y-6">
                <div className="space-y-3">
                  <Label 
                    htmlFor="reset-email"
                    className="text-base font-medium text-gray-900"
                    style={{ fontFamily: 'Google Sans, sans-serif' }}
                  >
                    Email Address
                  </Label>
                  <div className="relative">
                    <span className="material-icons absolute left-4 top-1/2 transform -translate-y-1/2 text-gray-400 text-xl">
                      email
                    </span>
                    <Input
                      id="reset-email"
                      type="email"
                      placeholder="Enter your email"
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      className="pl-12 h-14 rounded-2xl border-2 border-gray-200 focus:border-blue-500 focus:ring-0 bg-white shadow-sm text-base transition-all duration-200"
                      style={{ fontFamily: 'Roboto, sans-serif' }}
                      required
                    />
                  </div>
                </div>
                
                <DialogFooter className="flex-col space-y-3 pt-4">
                  <Button 
                    type="submit" 
                    className="w-full h-12 bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 text-white rounded-2xl shadow-lg hover:shadow-xl transition-all duration-300 text-base font-medium"
                    style={{ fontFamily: 'Google Sans, sans-serif' }}
                  >
                    <span className="material-icons mr-2">send</span>
                    Send Reset Link
                  </Button>
                  
                  <Button 
                    type="button"
                    variant="ghost"
                    onClick={handleClose}
                    className="w-full h-12 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-2xl transition-all duration-200"
                    style={{ fontFamily: 'Google Sans, sans-serif' }}
                  >
                    Cancel
                  </Button>
                </DialogFooter>
              </form>
            ) : (
              <div className="space-y-6">
                {/* Success State Content */}
                <div className="bg-green-50 border border-green-200 rounded-2xl p-6">
                  <div className="flex items-start space-x-3">
                    <span className="material-icons text-green-600 text-xl mt-0.5">info</span>
                    <div>
                      <h4 
                        className="font-medium text-green-900 mb-2"
                        style={{ fontFamily: 'Google Sans, sans-serif' }}
                      >
                        What's Next?
                      </h4>
                      <ul 
                        className="text-sm text-green-700 space-y-1"
                        style={{ fontFamily: 'Roboto, sans-serif' }}
                      >
                        <li className="flex items-center space-x-2">
                          <span className="material-icons text-xs">check_circle</span>
                          <span>Check your email inbox</span>
                        </li>
                        <li className="flex items-center space-x-2">
                          <span className="material-icons text-xs">check_circle</span>
                          <span>Click the reset link in the email</span>
                        </li>
                        <li className="flex items-center space-x-2">
                          <span className="material-icons text-xs">check_circle</span>
                          <span>Create a new password</span>
                        </li>
                      </ul>
                    </div>
                  </div>
                </div>

                {/* Additional Help */}
                <div className="bg-blue-50 border border-blue-200 rounded-2xl p-4">
                  <div className="flex items-center space-x-2 text-sm text-blue-700">
                    <span className="material-icons text-blue-600 text-base">help_outline</span>
                    <span style={{ fontFamily: 'Roboto, sans-serif' }}>
                      Didn't receive the email? Check your spam folder or try again.
                    </span>
                  </div>
                </div>
                
                <DialogFooter className="flex-col space-y-3">
                  <Button 
                    onClick={handleClose}
                    className="w-full h-12 bg-gradient-to-r from-green-600 to-green-700 hover:from-green-700 hover:to-green-800 text-white rounded-2xl shadow-lg hover:shadow-xl transition-all duration-300 text-base font-medium"
                    style={{ fontFamily: 'Google Sans, sans-serif' }}
                  >
                    <span className="material-icons mr-2">arrow_back</span>
                    Back to Sign In
                  </Button>
                  
                  <Button 
                    onClick={() => {
                      setIsSubmitted(false);
                      setEmail("");
                    }}
                    variant="ghost"
                    className="w-full h-12 text-blue-600 hover:text-blue-700 hover:bg-blue-50 rounded-2xl transition-all duration-200"
                    style={{ fontFamily: 'Google Sans, sans-serif' }}
                  >
                    <span className="material-icons mr-2">refresh</span>
                    Try Different Email
                  </Button>
                </DialogFooter>
              </div>
            )}
          </div>

          {/* Google-style Security Footer */}
          <div className="bg-gray-50 border-t border-gray-100 p-6">
            <div className="flex items-center justify-center space-x-2 text-sm text-gray-500">
              <span className="material-icons text-green-500 text-sm">shield</span>
              <span style={{ fontFamily: 'Roboto, sans-serif' }}>
                Password reset is secure and encrypted
              </span>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </>
  );
};

export default ForgotPasswordDialog;
