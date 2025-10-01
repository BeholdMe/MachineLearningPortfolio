from skimage.morphology import dilation, square
from skimage.measure import label, regionprops
import numpy as np

'''
### Explanation of Class
**Input:**
- $\alpha$ (alpha)
- $\tau$ (tau)
- $\delta$ (delta)
- `s`
- `N`

**Maintains:**
- Frame history `[t - 2, t - 1, t]`
- Tracked object candidates

**update(frame)**
- compute motion
  - `diff1` $= |t - t_1|$
  - `diff2` $= |t_1 - t_2|$
- `motion = np.minimum(diff1, diff2)`
- threshold of $\tau$ (tau) to reduce noise from differences
- Dilate with `9x9` kernel
- Connected components $\to$ centroids + bounding boxes
- Match new detections to existing filters
  - Use $distance < \delta$ (delta)
- Add new objects active for $\alpha$ (alpha) frames
- Remove objects unseen for $\alpha$ (alpha) frames
'''

class MotionDetector:
  def __init__(self, alpha=3, tau=25, delta=50, s=1, N=10):
    self.alpha = alpha
    self.tau = tau
    self.delta = delta
    self.skip = s
    self.max_objects = N

    # store last 3 greyscale frames
    self.prev_frames = []
    self.frame_count = 0

  def compute_motion(self, t, t1, t2):
    # get the diff1 (t - t1)
    diff1 = np.abs(t.astype(np.int16) - t1.astype(np.int16))
    # get the diff2 (t1 - t2)
    diff2 = np.abs(t1.astype(np.int16) - t2.astype(np.int16))

    # use minimum difference to keep noise down
    motion = np.minimum(diff1, diff2)

    # apply tau and convert to binary mask
    motion = (motion >= self.tau).astype(np.uint8) * 255
    return motion

  def dilate_and_label(self, motion_mask):
    # dialte motion w/9x9 square and label blobs
    dilated = dilation(motion_mask, square(9))
    labeled = label(dilated)
    return labeled

  def extract_candidates(self, labeled_image, min_area=100):
    # get region properties for each blob that has been labeled
    props = regionprops(labeled_image)
    candidates = []
    for prop in props:
      if prop.area >= min_area:
        # get centroid and bounding box
        y, x = prop.centroid
        bbox = prop.bbox
        candidates.append({
            'centroid': (int(x), int(y)),
            'bbox': bbox,
            'area': prop.area
        })
    return candidates

  def update(self, gray_frame):
    self.frame_count += 1
    self.prev_frames.append(gray_frame)

    # return if there is not enough frames yet
    if len(self.prev_frames) < 3:
      return []

    # keeps the 3 most recent frames only
    if len(self.prev_frames) > 3:
      self.prev_frames.pop(0)

    # skip the frame if not on skip interval
    if self.frame_count % self.skip != 0:
      return []

    # get the last 3 frames in order
    # compute motion mask, process blobs, and extract
    # blob candidates from regions that has been labeled
    f2, f1, f0 = self.prev_frames
    motion = self.compute_motion(f0, f1, f2)
    labeled = self.dilate_and_label(motion)
    candidates = self.extract_candidates(labeled)

    return candidates

'''
## Kalman Filter
The objects will be **tracked** using Kalman filters. Motion Detector will supply list of candidates for which each object currently tracked can be compared.

### Logic
1. If proposed object is active over $\alpha$ frame $\to$ added to list of currently tracked objects
2. If **distance** b/w (a) proposed object and (b) prediction of a filter $< \space \delta \space \to$ proposal is measurement for corresponding filter
3. If currently tracked object not updated when $\alpha$ frame $\to$ object is **inactive** $\therefore$ remove filter from current list of tracked objects

### Explanation on Class
This will initialize and define matrices A, H, Q, R, P, and x.

`predict()` function use $x = Ax$, $P = APA^T + Q$.

`update(y)` function use Kalman gain $K$ before updating $x$ and $P$

This will store history of positions for the trails.
'''
class KalmanFilter:
  def __init__(self, initial_position):
    # initial state [x, y, vx, vy]
    self.x = np.array([[initial_position[0]], [initial_position[1]], [0], [0]])

    # the transititon matrix for the state
    self.A = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    # the observation matrix
    self.H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])

    # the covariances
    # P will have large initial uncertainty
    # Q will process noise and R will measure noise
    self.P = np.eye(4) * 500
    self.Q = np.eye(4) * 1
    self.R = np.eye(2) * 10

    self.history = [initial_position]
    self.missed_frames = 0

  def predict(self):
    # use motion mdoel to predict next state
    # then predict next covar matrix, then extrac predicted position
    # finally, append to history
    self.x = self.A.dot(self.x)
    self.P = self.A.dot(self.P).dot(self.A.T) + self.Q
    pred = (self.x[0, 0], self.x[1, 0])
    self.history.append(pred)
    return pred

  def update(self, measurement):
    # get residual and residual covariance
    # find K (kalman gain)
    z = np.array([[measurement[0]], [measurement[1]]])
    y = z - (self.H.dot(self.x))
    S = self.H.dot(self.P).dot(self.H.T) + self.R
    K = self.P.dot(self.H.T).dot(np.linalg.inv(S))

    # update state w/the measruement and covar with K
    self.x = self.x + (K.dot(y))
    I = np.eye(self.A.shape[0])
    self.P = (I - (K.dot(self.H))).dot(self.P)

    # reset the missed frame counter
    self.missed_frames = 0

    # update the last history w/right state
    updates = (self.x[0, 0], self.x[1, 0])
    self.history[-1] = updates
    return updates

'''
## Tracking
Class to handle matching detections w/filters using **nearest neighbor**, initiliaze new filters for unmatched detections, remove filters that miss updates for too long, and return track trails for visualization.
'''
class Tracker:
  def __init__(self, alpha=3, delta=50, max_objects=10):
    self.alpha = alpha
    self.delta = delta
    self.max_objects = max_objects
    self.objects = []
    self.ids = []
    self.next_id = 0

  def distance(self, p1, p2):
    # compute distance b/w two points (Euclidian)
    return np.linalg.norm(np.array(p1) - np.array(p2))

  def step(self, detections):
    # keep track of which ids were updated and which already detected
    updated_ids = set()
    detected_already = set()

    # predict next positions for all current Kalman filters
    predictions = [kf.predict() for kf in self.objects]

    # Match each detection to closest prediction within delta threshold
    for i, pred in enumerate(predictions):
      min_distance = float('inf')
      best_idx = -1

      for j, detection in enumerate(detections):
        # if j already detected -> skip
        if j in detected_already:
          continue

        dist = self.distance(detection['centroid'], pred)
        if dist < min_distance:
          min_distance = dist
          best_idx = j

      if best_idx != -1 and min_distance < self.delta:
        # update matched Kalman filters
        self.objects[i].update(detections[best_idx]['centroid'])
        detected_already.add(best_idx)
        updated_ids.add(i)

    # increment missed frame counters for unmatched objects
    for i, kf in enumerate(self.objects):
      if i not in updated_ids:
        kf.missed_frames += 1

    # remove filters that exceeded alpha missed frames
    retained_objects = []
    retained_ids = []
    for i, kf in enumerate(self.objects):
      if kf.missed_frames < self.alpha:
        retained_objects.append(kf)
        retained_ids.append(self.ids[i])
    self.objects = retained_objects
    self.ids = retained_ids

    # add new detections as new Kalman filters if limit not reached
    for detection in detections:
      det_centroid = detection['centroid']
      is_new = True
      for kf in self.objects:
        kf_pos = (kf.x[0, 0], kf.x[1, 0])
        if self.distance(det_centroid, kf_pos) < self.delta:
          is_new = False
          break

      if is_new and len(self.objects) < self.max_objects:
        new_kf = KalmanFilter(det_centroid)
        self.objects.append(new_kf)
        self.ids.append(self.next_id)
        self.next_id += 1

  def get_tracks(self):
    # return a list of tracked object histories and their IDs
    return [
        {'id': obj_id,
        'trail': kf.history}
        for obj_id, kf in zip(self.ids, self.objects)
    ]