import requests
import base64
from PIL import Image
from io import BytesIO


class ModelClient:
    SERVER_URL = "http://treble.cs.cmu.edu:7111"

    def encode_image(self, image: Image):
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format="PNG")  # You can change the format as needed
        img_byte_arr = img_byte_arr.getvalue()
        return base64.b64encode(img_byte_arr).decode("utf-8")

    def query_model(self, pil_img, question):
        img_data = self.encode_image(pil_img)
        response = requests.post(
            self.SERVER_URL + "/infer", json={"image": img_data, "question": question}
        )
        return response.json()


if __name__ == "__main__":
    # Example usage
    image = Image.open(
        "/Users/waynechi/dev/gui-agent/bounding_boxes/datasets/vwa_crawls/vwa_crawl_20240214_220303/page_1_run_0/step_0/base.png"
    )
    question = "What is in the picture?"

    model_client = ModelClient()

    result = model_client.query_model(image, question)
    print(result)
