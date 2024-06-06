## Nick's Bikefit
With this project, you can analyze your dynamic body position while riding a bike. Your body position is analyzed and adaptations to your bike geometry are recommended based on sports scientific research.

![](https://github.com/nstallmann/bikefit/blob/main/example.gif)

## Requirements
* python version: 3.11.8
* see `requirements.txt` / `environment.yml`

## Video Recommendations
You will need a video of yourself riding your bike on an indoor trainer. For meaningful results, consider the following:
* It is recommended to warm up for a few minutes in order to resemble your position during actual rides better
* The video must only contain footage with you on the bike, cut the video accordingly
* A few pedal strokes are sufficient for analysis, aim for ~10s of video
* A video with 720p resolution is sufficient, higher resolution only leads to higher computing time

## How to Run
* Open the file `bikefit.py` and enter the path to your video
* Initialize a bike. You can choose between `RoadBike` and `MountainBike`. In case of `RoadBike`, choose a riding style by passing it as an argument. You can choose between `RidingStyle.CASUAL`, `RidingStyle.FITNESS` and `RidingStyle.RACING`. For `MountainBike`, no arguments are required.
* Upon running, a folder with the video's stem name will be generated. In ot, you will find the files `bikefit_top_vs_bottom_stroke.jpg` and `measured_vs_recommended_angles.png` showing your body position on the bike and how the measured angles are lying in the recommended range.
