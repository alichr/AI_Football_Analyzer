import os
import yaml
import torch
from ultralytics import YOLO
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    try:
        # Define paths
        data_dir = "data/processed/football-players-detection.v14i.yolov11"
        data_yaml = os.path.join(data_dir, "data.yaml")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"checkpoint/yolo11_football_{timestamp}"
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up logging to file
        file_handler = logging.FileHandler(os.path.join(output_dir, "training.log"))
        logger.addHandler(file_handler)
        
        # Check CUDA availability
        device = "0" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        # Load data configuration
        with open(data_yaml, 'r') as f:
            data_config = yaml.safe_load(f)
        
        logger.info(f"Training on dataset with {data_config['nc']} classes: {data_config['names']}")
        
        # Initialize YOLO11 model
        model = YOLO("yolo11x.pt")
        
        # Train the model using training data
        results = model.train(
            data=data_yaml,
            epochs=10,
            imgsz=640,
            batch=16,
            name=os.path.basename(output_dir),
            project=os.path.dirname(output_dir),
            patience=20,
            device=device,
            amp=True,
            dropout=0.1,
            augment=True,
            save=True,
            save_period=10,  # Save checkpoints every 10 epochs
            plots=True,
            pretrained=True,
            cos_lr=True,  # Use cosine learning rate scheduler
            lr0=0.01,  # Initial learning rate
            lrf=0.01,  # Final learning rate ratio
        )
         
        
        # Test the model on test set
        logger.info("Testing on test dataset...")
        test_metrics = model.val(data=data_yaml, split="test")
        logger.info(f"Test results: {test_metrics}")
        
        
        logger.info(f"Training completed. Models saved to {output_dir}")
        return results
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
