import logging
import sys
import os

def setup_logging():
    """Setup application logging configuration"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            # Add file handler if needed
            # logging.FileHandler('app.log')
        ]
    )
    
    # Add project path for imports
    sys.path.append(os.path.dirname(os.path.abspath(__file__ + "/..")))
    
    logger = logging.getLogger(__name__)
    logger.info("✅ Logging configured successfully")