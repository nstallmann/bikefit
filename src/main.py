from Constants.bike import RoadBike, RidingStyle
from Controller.controller import Controller

if __name__ == "__main__":
    controller = Controller(bike=RoadBike(riding_style=RidingStyle.RACING))
    controller.start()
