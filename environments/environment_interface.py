from abc import ABC, abstractmethod
from utils.data_saver import DataSaver
from PIL import Image, ImageDraw


class EnvironmentInterface(ABC):

    def __init__(self, logger, data_saver: DataSaver):
        self.logger = logger
        self.data_saver = data_saver
        self.temp_state_filename = "state.png"

    @abstractmethod
    def get_state(self):
        """
        Defines the current state of the environment
        """
        pass

    @abstractmethod
    def execute_action(self, action, step_id):
        """
        Executes the action in the environment
        """
        pass

    @abstractmethod
    def clean_up(self):
        """
        Cleans up the environment
        """
        pass

    def draw_and_save_click(self, coords):
        # Get set of marks image
        image = Image.open(self.temp_state_filename)
        # Draw a circle on the image at the coordinates
        draw = ImageDraw.Draw(image)

        pixel_ratio = self.get_pixel_ratio()
        xy = [
            (coords[0] * pixel_ratio) - 10,
            (coords[1] * pixel_ratio) - 10,
            (coords[0] * pixel_ratio) + 10,
            (coords[1] * pixel_ratio) + 10,
        ]
        draw.ellipse(xy, fill="red", width=1)
        self.data_saver.save_click_image(image)
