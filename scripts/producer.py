import logging
import time
import cv2
import numpy as np

from openfilter.filter_runtime.filter import FilterConfig, Filter, Frame

__all__ = ["MultiSourceProducerConfig", "MultiSourceProducer"]

logger = logging.getLogger(__name__)


class MultiSourceProducerConfig(FilterConfig):
    """
    Configuration for MultiSourceProducer that outputs multiple frame topics.
    """
    num_regions: int = 5                  # Number of text regions to create
    font_scale: float = 2.0               # Font scale for text
    font_thickness: int = 3               # Font thickness for text
    padding: int = 20                     # Padding around text regions


class MultiSourceProducer(Filter):
    """
    A filter that adds text regions to input video frames and outputs cropped frames:
    
    1. main - The input frame with all text regions added
    2. region_[N] - Cropped text regions from the frame (where N is region number)
    
    This demonstrates creating multiple different views of an input frame,
    each available as a separate topic for downstream filters to consume.
    """
    
    def setup(self, config: MultiSourceProducerConfig):
        """
        Initialize the filter with configuration.
        """
        logger.info("Setting up MultiSourceProducer")
        self.config = config
        self.frame_count = 0
        
        # Define text regions
        self.texts = [
            "Hello World",
            "OCR Test",
            "Sample Text",
            "Filter Example",
            "Multiple Regions"
        ]
        self.colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255)   # Magenta
        ]
        
        logger.info(f"MultiSourceProducer setup completed. Config: {config.__dict__}")

    def process(self, frames: dict[str, Frame]):
        """
        Process the input frame and produce multiple output topics.
        """
        # Get input frame
        input_frame = frames.get("input") or next(iter(frames.values()))
        if not input_frame or not input_frame.has_image:
            logger.warning("No valid input frame received")
            return {}
            
        # Increment frame counter
        self.frame_count += 1
        
        # Get the input image
        image = input_frame.rw_bgr.image.copy()
        height, width = image.shape[:2]
        
        # Calculate positions based on image size
        positions = [
            (width//4, height//4),        # Top-left
            (3*width//4, height//4),      # Top-right
            (width//4, 3*height//4),      # Bottom-left
            (3*width//4, 3*height//4),    # Bottom-right
            (width//2, height//2)         # Center
        ]
        
        # Draw text regions
        regions = []
        for i in range(self.config.num_regions):
            text = self.texts[i]
            color = self.colors[i]
            pos = positions[i]
            
            # Get text size
            (text_width, text_height), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, self.config.font_scale, self.config.font_thickness
            )
            
            # Draw rectangle background
            x1 = pos[0] - self.config.padding
            y1 = pos[1] - text_height - self.config.padding
            x2 = pos[0] + text_width + self.config.padding
            y2 = pos[1] + self.config.padding
            
            # Ensure coordinates are within image bounds
            x1 = max(0, min(x1, width-1))
            y1 = max(0, min(y1, height-1))
            x2 = max(0, min(x2, width-1))
            y2 = max(0, min(y2, height-1))
            
            # Draw background rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), (200, 200, 200), -1)
            
            # Draw text
            cv2.putText(
                image, text, pos,
                cv2.FONT_HERSHEY_SIMPLEX, self.config.font_scale,
                color, self.config.font_thickness
            )
            
            # Store region coordinates
            regions.append((x1, y1, x2, y2))
        
        # Dictionary to hold our output frames with different topics
        output_frames = {}
        
        # Add main frame (full image with text regions)
        output_frames["main"] = Frame(
            image, 
            {"meta": {"description": "Input frame with text regions", "frame_num": self.frame_count}}, 
            "BGR"
        )
        
        # Add cropped regions
        for i, (x1, y1, x2, y2) in enumerate(regions):
            # Crop the region
            cropped = image[y1:y2, x1:x2]
            
            # Add to output frames
            output_frames[f"region_{i}"] = Frame(
                cropped, 
                {
                    "meta": {
                        "description": f"Cropped region {i}",
                        "frame_num": self.frame_count,
                        "text": self.texts[i],
                        "region_id": i
                    }
                }, 
                "BGR"
            )
        
        return output_frames

    def shutdown(self):
        """
        Called once when the filter is shutting down.
        """
        logger.info("Shutting down MultiSourceProducer")
        logger.info(f"Processed {self.frame_count} frames")
        logger.info("MultiSourceProducer shutdown complete.")


if __name__ == "__main__":
    MultiSourceProducer.run() 