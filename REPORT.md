# Report

## Initial Experimentation

Initial experimentation was done with **Grounding DINO** using a hosted notebook:  
[Zero-shot Object Detection with Grounding DINO (Colab)](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/zero-shot-object-detection-with-grounding-dino.ipynb)  

I copied it to my own drive and uploaded stills from the video to test object detection.  
This initial experimentation allowed me to tune the text prompts required for recognition.

---

## Lens Distortion Correction

I initially noticed the video had wide-angle barrel distortion, so I wanted to correct for this.  
I did not know the camera matrices, focal lengths, or have a checkerboard sample to calibrate from.  
I ran the experiment `manual_calibration.py` to manually find some *k1, k2* values that looked reasonable.  
This was done by eye, and values were recorded for use in a transform step of my processing pipeline.

---

## Grounding DINO Experiments

Some experiments were done running Grounding DINO locally as a processing step, first on some images and then on a video.  
Finally, it was added into the processing pipeline.

---

## Interaction Logging

A simple interaction logging process was developed using overlaps of detection boxes to determine interactions.  
For example:  
- Hand overlaps bottle = hand touches bottle  
- Bottle cap no longer overlaps bottle = bottle is opened  

---

## Future Directions

### Pipeline Structure
1. Undistort Image  
2. Detect Objects (boxes)  
3. Interaction Tracking (box overlap)  
4. Segment Anything (masks)  
5. HSV Thresholding & Beer-Lambert law to determine liquid depth & petri dish fill %  
6. Centroid extraction  
7. Tracking using StoneSoup extended Kalman tracker  

---

### Next Steps

- **Fix unsatisfactory behavior in interaction tracking**  
  Currently only seems to detect hand/bottle interactions. Tests indicate it should be working fine, but no time to investigate further.  

- **Develop Segment Anything Masks**  
  Segmentation masks were the original plan.  
  The Dockerfile builds Grounded SAM ([IDEA-Research/Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2)).  
  I began experimenting with it (`segment_anything.py`), but had to stop before fully integrating it.  
  Masks are required for the next phases.  

- **Develop Fluid Level Detection & Fill Percentage**  
  Using HSV filters from OpenCV on detection-masked video to isolate yellow liquid.  
  This would determine what percentage of a petri dish is filled.  

  Also planned to use a Beer-Lambert inspired formulation to estimate liquid depth in the bottle:

  $$
  \text{depth} = \frac{-\ln(\text{Intensity} / \text{Background\_Intensity})}{k}
  $$

  To estimate how many filled petri dishes are stacked:

  $$
  N = \frac{-\ln(\text{Intensity} / \text{Background})}{k \times \text{dish\_depth}}
  $$

  I started this with the `manual_hsv_tuner.py` experiment to tune filters but ran out of time.  
  I also intended to use filtering to detect spills (liquid outside bottle or dish).

- **Develop Centroid Extraction**  
  Fairly easy — find center of mass of object masks or just the center of bounding boxes. Needed for tracking.

- **Develop Kalman Tracker using StoneSoup library**  
  Up to now, detections are not associated between frames.  
  Intended to build an extended Kalman tracker to group detections over time and establish track probabilities.  
  This would increase pipeline reliability, allowing breaks in detection to be compensated for.  
  Driven by centroids and logits from Grounding DINO detections.  
  Tracking would create a belief state, tracking and identifying unique bottles/petri dishes across the video.  
  This would allow associating fill levels with the correct entities over time.

---

### Data Model (Pydantic)

```python
from pydantic import BaseModel, Field
from uuid import UUID, uuid4
import numpy as np

class Entity(BaseModel):
    entity_id: UUID = Field(default_factory=uuid4)
    pos_x: float
    pos_y: float

class Hand(Entity):
    pose_info: np.ndarray

class Bottle(Entity):
    stoppered: bool
    fill_level: float

class PetriDish(Entity):
    filled_pct: float
```

### API Development

I also wanted to develop the API into a service using **Socket.IO** so that live video can be streamed to the service, and logs, interactions, tracks, and entities can be streamed out on unique topics.

---

### Additional Notes

- I had a lot of housekeeping left to do, including reducing the Docker image size and trying to speed up the process.  
- Currently, the pipeline runs at around 6 FPS on my RTX 4070 Super.
