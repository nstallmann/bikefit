# Nick's Bikefit

Analyze your dynamic body position while riding a bike and get recommendations for adapting your bike geometry based on 
sports scientific research.

<img src="https://github.com/nstallmann/bikefit/blob/main/example/example.gif" height="280" /> <img src="https://github.com/nstallmann/bikefit/blob/main/example/measured_vs_recommended_angles.png" height="280" />

## Requirements

- **Python version:** 3.11.8
- See `requirements.txt` or `environment.yml` for additional dependencies.

## Video Recommendations

To achieve meaningful results, you will need a video of yourself riding your bike on an indoor trainer. Consider the 
following guidelines:

- **Warm-Up:** Warm up for a few minutes to better resemble your position during actual rides.
- **Content:** The video should only contain footage of you on the bike; edit it accordingly.
- **Duration:** A few pedal strokes are sufficient for analysis. Aim for approximately 10 seconds of video.
- **Resolution:** A resolution of 720p is sufficient. Higher resolutions will increase computing time without 
- significant benefits.

## How to Run

1. **Prepare Your Video:**
   - Ensure your video meets the recommendations above.
   - Note the path to your video file.

2. **Initialize the Script:**
   - Open the file `bikefit.py`.
   - Enter the path to your video file.

3. **Choose Your Bike Type:**
   - **Road Bike:**
     - Initialize a `RoadBike` object.
     - Choose a riding style by passing it as an argument. Available options are `RidingStyle.CASUAL`,
     - `RidingStyle.FITNESS`, and `RidingStyle.RACING`.
   - **Mountain Bike:**
     - Initialize a `MountainBike` object. No additional arguments are required.

4. **Run the Script:**
   - Upon running, a folder with the video's stem name will be generated.
   - Inside, you will find two files:
     - `bikefit_top_vs_bottom_stroke.jpg`: Shows your body position on the bike.
     - `measured_vs_recommended_angles.png`: Shows your measured angles within the recommended range.

## Approaches for adjusting your bike

- **Saddle Height**: The saddle height is probably the most important variable to get right. It can mainly be adjusted 
  based on your maximum knee angle. Lower the saddle to decrease the maximum knee angle or heighten it to increase it.
- **Cleat Position**: Your cleat position can be adjusted based on your minimum knee angle. In order to increase it,
  you can try to move your cleat back towards your ankle. Another approach is to try shorter cranks.
- **Shoulder/Arms**: Shoulder and arm position can be changed by various handlebar/stem combinations. If your arm is
  too stretched, consider a shorter stem or one with a higher rise. In order to increase elbow angle, you can choose a
  longer stem or one with higher rise.
