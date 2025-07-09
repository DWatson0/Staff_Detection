# AI Evaluation: Staff Detection Using Name Tag Recognition

## Tools & Technologies Used
- Python 3.12 (Anaconda/PyCharm)
- Label Studio – annotation
- Google Colab – model training
- YOLOv8 (Ultralytics) – object detection
- OpenCV – frame extraction & video processing
- Supervision – visual annotation

## Workflow
### a) Frame Extraction
- Used OpenCV to extract individual frames from the sample video.

### b) Frame Annotation
- Annotated frames manually with Label Studio.
- Only frames with visible name tags were labeled.
- Negative samples (without name tags) included to reduce false positives.

### c) Model Training
- Model: YOLOv8n in Google Colab
- Epochs: 40
- Image size: 960×960
- Dataset: 103 positive + 30 negative images, split 80%/20% into training/validation

### d) Model Evaluation and Refinement
- Evaluated `best.pt` model on the original video.
- Repeated steps b) and c) to iteratively improve the model.
- Final model performance:
  - Precision: **0.998**
  - Recall: **1**
  - mAP50: **0.995**
  - mAP50-95: **0.648**

### e) Annotated Video Output
- Processed the video frame-by-frame using OpenCV.
- Staff detected with name tags were enclosed in bounding boxes and labeled with:
  `"staff {confidence} (x1, y1) (x2, y2)"`
