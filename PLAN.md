# Video Analysis Plan

## Tasks

- Identify different object types in the scene (petri dishes, bottles, hands).
- Determine interactions between objects (e.g., hand touches dish, dish is filled).
- Track the count of filled dishes as the scene unfolds.
- Identify any additional vision-based insights in the scene that you find compelling.

---

## Video Notes

- **Strong colors**: orange gloves, blue lids on some bottles.
- Bottle of pipettes in bottom left, along with a heating(?) device.
- Video is not calibrated — obvious barrel distortion.
- Petri dishes are added in stacks; hard to tell how many are in a stack initially.
- Operator presents bottle of brown liquid to camera after agitation.
- Petri dishes are filled with translucent yellow fluid.
- One stack is filled at once and then restacked.
- Petri dishes have lids that are removed prior to filling and replaced afterwards.
- Stacks contain **5 dishes**.
- In the second stack:
  - First two dishes are filled from top down.
  - Then method switches: top two dishes are removed, and bottom one is filled.
- A small spill occurs after filling the first dish from the second stack.
- Scientist seems agitated about something stuck to their hand.
- Third set of dishes is filled from bottom up.
- Final stack only has 2 dishes filled before running out of solution.
- Spill is wiped at the end.
- Some video compression artifacts.

---

## Thoughts

- Use neural networks (NN) like **Grounding DINO + Segment Anything Model (SAM)** for zero-shot object detection and segmentation.
- Track hand position/pose using **OpenCV hand tracking**.
- **Color filtering in HSV** could help detect:
  - Filled dishes (yellow hue).
  - Bottle fill level (attenuation).
- If petri dishes can be localized:
  - Determine fill percentage by yellow area coverage.
- Bottle fill estimation via attenuation model:
  - Liquid translucency decreases as bottle empties.
  - Use exponential decay model (inspired by Beer-Lambert law):

    ```
    Intensity = Background_Intensity * e^(-k * depth)
    depth = -ln(Intensity / Background_Intensity) / k
    ```

  - Determine `k` experimentally.
- Text markers on the bottle may assist in gauging liquid level.
- Detect bottle lid to infer whether bottle is stoppered.
- Dishes added in **stacks of 5** — can be used for inference.
- Maintain a data structure for dishes and their fill level.

### Dish Fill Detection

- Detecting fill state complicated by stacking behavior:
  - Filled dishes may be covered by empty ones.
- Use attenuation model to determine number of filled dishes in a stack:

  ```
  N = -ln(Intensity / Background) / (k * dish_depth)
  ```

---

## Tracking

- Track dishes between frames using:
  - **Extended Kalman Filter**
  - **Particle Filter**
- Track XY position of entities.
- Use tracker to disassociate observations from belief state.
- Challenge: tracking stacked dishes.

---

## Lens Correction

- Apply transform to correct barrel distortion.
- Use straight features (top and left edge of frame) for calibration.

---

## Interaction Logic

- Simple overlap rules:
  - Hand overlaps petri dish ⇒ touch.
  - Dish is filled ⇒ filled.

---

## Entity Definitions (Pydantic-style)

```python
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

---

## Pipeline

1. Video Frames  
2. Lens Correction  
3. Object Detection  
4. Segmentation (masking)  
5. Tracker Update  
6. Fill Level Analysis & Modelling  
7. Interaction Logic  
8. Output / Log / Relay  

---

## TODO

- [x] Install Python
- [x] Install VSCode
- [x] Install Docker
- [x] Install Git
- [x] Fork Repo
- [x] Add Boilerplate
- [x] Add runners (if time)
- [x] Add flake8, linting, PDOC scripts
- [ ] Do experiments
- [ ] Create pipeline
- [ ] Write README
- [ ] Dockerize
- [x] Test pipeline (if time)
- [ ] Build pipeline (if time)
- [ ] Add extras (if time)

---

## Experiments (Jupyter Notebooks)

- Use screenshots for quick iterations.
- Check Grounding DINO zero-shot detections.
- Check Segment Anything (SAM).
- Evaluate detection & segmentation (raw vs. corrected).
- Try OpenCV hand tracking.
- Create HSV filters for:
  - Liquid (petri dish fill).
- Determine filters for bottle level detection (attenuation model).
- Calibrate coefficient `k`:
  - For bottle.
  - For dishes (fit curves across 3 known stacks).
- Test tracking algorithms (ByteTrack, StoneSoup).
- Experiment with spill detection (if time).

---

## Final Integration

- **Dockerize**
- **Output** via `socket.io` with topics:
  - Interactions
  - Annotated video
  - Entity tracks
  - Warnings

- **Input**: Video stream  
  ⇒ May require a video streaming service

---

## Additional Notes

### Spill Detection

- Liquids are semi-transparent — tricky to detect.
- Possible techniques:
  - Glare/reflection detection
  - Optical flow (spreading liquid)
  - HSV color deviation from background
  - Ensemble detector
  - Neural net approach (e.g., DINO/SAM)
  - A spill vs liquid in an open topped dish look the same, other than one is within the bounds of a dish

> "What is a spill? A puddle of liquid outside an open topped container."  
> *(What is a man? A miserable little pile of secrets – Dracula)*

---

## Performance Goals

- Real-time inference preferred
- Bottleneck: neural net inference speed
  - GPU/TPU acceleration needed
- May need to downsample framerate

### Video Specs

- 1280x720  
- 30fps  
- 3000kbps datarate  
- Audio: discarded (128kbps stereo)

---

## Development Philosophy

- Prefer **TDD** and **unit tests** to validate pipeline
- May cut corners for faster iteration
- If DINO + SAM is too heavy:
  - Try YOLOv8 + edge detection within bounding boxes
