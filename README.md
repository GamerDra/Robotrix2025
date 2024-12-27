# Basket the Ball!!!


## Calculation of 3d using 2 images /Stereo Vision 

### Formula for Depth Calculation
---
The depth \( Z \) of an object in stereo vision is calculated as:

\[
Z = \frac{f \cdot B}{d}
\]

Where:
- \( Z \): Depth (distance of the object from the cameras)
- \( f \): Focal length of the camera lens
- \( B \): Baseline (distance between the two cameras)
- \( d \): Disparity (difference in the position of corresponding points in the left and right images)
The **focal length** of a camera lens can be calculated from the **field of view (FoV)** 

### Formula:

\[
f = \frac{W}{2 \cdot \tan\left(\frac{\text{FoV}}{2}\right)}
\]

Where:
- \( f \): Focal length (in millimeters)
- \( W \): Width of the camera sensor (in millimeters)
- \( \text{FoV} \): Field of view (in radians)

---

## Approches

### Using projection
Sample the ball's trajectory at any 2 points and then calculate the location where the ball would fall, and then move the net to that spot. <br/>
We had two options to perform the calculation:
one using quadratic and other using Kalman Filter.
We incorporated both the techniques, but the ball position calculation had errors which caused issues with net positioning. <br/>
We also had problems with accurately determining when to take the samples. <br/>
We ultimately decided to drop this idea

### Using ball centering
This is a very simple approach, we just always try to keep the ball in the center of the image (not exactly the center because the net is offset from the camera). <br/>
This is what we ended up using in the end. It's pretty accurate at getting to the ball but we consistently encounter an issue where the ball hits the blackboard and flies away due to the force. <br/>
Another issue is that if the camera loses track of the ball (if it goes higher than the camera's max vertical range for example), we have no way to position the net other than hoping it'll come back close to the net. This can be fixed with a few solutions (Take the direction of movement of the ball before it goes out of the frame and continue moving in that direction till it comes back in to frame, kinda like projection), but we didn't have time enough to implement that.


### Attempt at RL
we tried of using NEAT(NeuroEvolution of Augmenting Topologies) to allow perfect capturing of ball, irrespective of luck, unfortunately we lacked time to correctly implement this though we tried on a sample 

    
## Results
During our testing, we found that the ball kept hitting the board and bouncing off, due to miscalc in the ball position or lack of understanding of the projection.

We were consistently able to hit the basket though not hoop it more than once.
With more parameter refining certainly we can certainly increase its acccuracy.

### Videos
the drive folder containing [video](https://drive.google.com/drive/folders/1Os0rfMXUQQF5Q4ycMKVQgPddxUpDIJjj)

## Referenes
[basket](https://www.youtube.com/watch?v=xHWXZyfhQas)<br/>
[application of neat](https://www.youtube.com/watch?v=WSW-5m8lRMs)
