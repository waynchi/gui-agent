import os
import math
from definitions import PROJECT_DIR
from bounding_boxes.model_client import ModelClient
import pandas as pd
from io import StringIO
import csv
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import matplotlib.pyplot as plt

# CLASS_TYPE_TO_ID = {
#     "IMG": 0,
#     "A": 1,
#     "INPUT": 2,
#     "BUTTON": 3,
#     "TEXTAREA": 4,
#     "SELECT": 5,
#     "UL": 6,
#     "DIV": 7,
#     "OTHER": 8,
# }

# CLASS_TYPE_TO_ID = {"UI_ELEMENT": 0, "UI_ELEMENT": 1}
# CLASS_ID_TO_TYPE = {0: "UI_ELEMENT", 1: "UI_ELEMENT"}


class BoundingBox:
    def __init__(
        self,
        top,
        left,
        bottom,
        right,
        interactable=False,
        confidence=0.0,
        class_id=0,
        x_offset=0,
        y_offset=0,
        class_type="CUSTOM",
        alt_text="",  # For images
        text="",
        center_x=None,
        center_y=None,
    ):
        """
        Offsets should be precomputed for the coordinates. They are just stored here for reference / reconstruction if necessary.
        The offsets are only used when inputting the bounding box from a webpage since there are y offsets usually due to scrolling. This is only when constructing the GT.
        """
        self.top = top
        self.left = left
        self.bottom = bottom
        self.right = right
        if center_x is None:
            self.center_x = (left + right) / 2
        else:
            self.center_x = center_x
        if center_y is None:
            self.center_y = (top + bottom) / 2
        else:
            self.center_y = center_y
        self.interactable = interactable
        self.confidence = confidence
        self.class_id = class_id
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.class_type = class_type
        self.alt_text = alt_text
        self.text = text

    def is_visible(self):
        return self.interactable

    def to_dict(self):
        return {
            "top": self.top,
            "left": self.left,
            "bottom": self.bottom,
            "right": self.right,
            "center_x": self.center_x,
            "center_y": self.center_y,
            "interactable": self.interactable,
            "confidence": self.confidence,
            "x_offset": self.x_offset,
            "y_offset": self.y_offset,
            "class_id": self.class_id,
            "class_type": self.class_type,
            "alt_text": self.alt_text,  # For images
            "text": self.text,
        }

    @classmethod
    def from_dict(cls, bbox_dict):
        instance = cls(
            top=bbox_dict["top"],
            left=bbox_dict["left"],
            bottom=bbox_dict["bottom"],
            right=bbox_dict["right"],
            center_x=bbox_dict["center_x"],
            center_y=bbox_dict["center_y"],
            interactable=bbox_dict["interactable"],
            confidence=bbox_dict["confidence"],
            class_id=bbox_dict["class_id"],
            x_offset=bbox_dict["x_offset"],
            y_offset=bbox_dict["y_offset"],
            class_type=bbox_dict["class_type"],
            alt_text=bbox_dict["alt_text"],
            text=bbox_dict["text"],
        )
        return instance

    def distance_to(self, other):
        return math.sqrt(
            (self.center_x - other.center_x) ** 2
            + (self.center_y - other.center_y) ** 2
        )

    def distance_to_origin(self):
        return math.sqrt(self.center_x**2 + self.center_y**2)


def get_valid_bounding_boxes_from_csv_string(
    csv_string, x_offset=0, y_offset=0, env="vwa"
):
    """
    Get the valid bounding boxes for the environment
    """
    df = pd.read_csv(StringIO(csv_string), delimiter=",", quotechar='"')
    bounding_boxes = []
    if env == "vwa":
        CLASS_TYPE_TO_ID = {"IMG": 0, "A": 1}
    else:
        CLASS_TYPE_TO_ID = {"UI_ELEMENT": 0}

    for _, row in df.iterrows():
        class_type = row["Element"]
        if class_type not in CLASS_TYPE_TO_ID:
            # class_type = "OTHER"
            class_type = "UI_ELEMENT"

        class_id = CLASS_TYPE_TO_ID[class_type]

        bounding_box = BoundingBox(
            row["Top"] - y_offset,
            row["Left"] - x_offset,
            row["Bottom"] - y_offset,
            row["Right"] - x_offset,
            row["Interactable"],
            x_offset=x_offset,
            y_offset=y_offset,
            alt_text=row["Alt"],
            class_id=class_id,
            class_type=class_type,
            text=row["TextContent"],
        )
        bounding_boxes.append(bounding_box)

    bounding_boxes = [
        bounding_box for bounding_box in bounding_boxes if bounding_box.is_visible()
    ]

    return bounding_boxes


def get_csv_string_from_bounding_boxes(bounding_boxes):
    """
    Get the csv string from the bounding boxes
    """
    columns = [
        "ID",
        "Element",
        "Top",
        "Right",
        "Bottom",
        "Left",
        "Width",
        "Height",
        "Alt",
        "Class",
        "Id",
        "TextContent",
        "Interactable",
    ]
    output = StringIO()
    csv_writer = csv.writer(output, quoting=csv.QUOTE_ALL)
    csv_writer.writerow(columns)
    for i, bounding_box in enumerate(bounding_boxes):
        bounding_box_elements = [
            str(i + 1),
            bounding_box.class_type,
            bounding_box.top,
            bounding_box.right,
            bounding_box.bottom,
            bounding_box.left,
            bounding_box.right - bounding_box.left,
            bounding_box.bottom - bounding_box.top,
            bounding_box.alt_text,
            "",
            "",
            bounding_box.text,
            "true" if bounding_box.interactable else "false",
        ]
        csv_writer.writerow(bounding_box_elements)

    csv_string = output.getvalue()
    output.close()

    return csv_string


def compute_bounding_boxes_webui(
    model, image_path, confidence_threshold=0.1, merge_iou_thresholds=[0.1]
):
    image = Image.open(image_path).convert("RGB")
    image_transforms = transforms.ToTensor()
    image_input = image_transforms(image)
    out = model.model([image_input.to(model.device)])[0]
    boxes = out["boxes"].detach().cpu().numpy()
    scores = out["scores"].detach().cpu().numpy()
    labels = out["labels"].detach().cpu().numpy()
    # Create a list of bounding boxes based on these boxes
    bounding_boxes = []
    for i in range(boxes.shape[0]):
        if scores[i] < confidence_threshold:
            continue
        bounding_box = BoundingBox(
            left=boxes[i][0],
            top=boxes[i][1],
            right=boxes[i][2],
            bottom=boxes[i][3],
            interactable=True,
            confidence=scores[i],
            class_id=0,
            # class_id=labels[i],  # Not used currently
        )
        bounding_boxes.append(bounding_box)

    # Merge any bounding boxes with a significant overlap
    merged_boxes_dict = {}
    for iou_threshold in merge_iou_thresholds:
        merged_boxes = merge_bounding_boxes(bounding_boxes, iou_threshold=iou_threshold)
        merged_boxes_dict[iou_threshold] = merged_boxes

    draw = ImageDraw.Draw(image)
    for bounding_box in merged_boxes:
        draw.rectangle(
            [
                bounding_box.left,
                bounding_box.top,
                bounding_box.right,
                bounding_box.bottom,
            ],
            outline=(255, 0, 0),
        )

    return merged_boxes_dict


def merge_bounding_boxes(bounding_boxes, iou_threshold=0.5):
    merged_boxes = []
    while len(bounding_boxes) > 0:
        current_box = bounding_boxes[0]
        remaining_boxes = []
        for box in bounding_boxes[1:]:
            iou = calculate_iou(current_box, box)
            if iou >= iou_threshold:
                current_box = merge_two_boxes(current_box, box)
            else:
                remaining_boxes.append(box)
        merged_boxes.append(current_box)
        bounding_boxes = remaining_boxes
    return merged_boxes


def calculate_iou(box1, box2):
    intersection_left = max(box1.left, box2.left)
    intersection_top = max(box1.top, box2.top)
    intersection_right = min(box1.right, box2.right)
    intersection_bottom = min(box1.bottom, box2.bottom)

    intersection_area = max(0, intersection_right - intersection_left + 1) * max(
        0, intersection_bottom - intersection_top + 1
    )

    box1_area = (box1.right - box1.left + 1) * (box1.bottom - box1.top + 1)
    box2_area = (box2.right - box2.left + 1) * (box2.bottom - box2.top + 1)

    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou


def merge_two_boxes(box1, box2):
    merged_box = BoundingBox(
        left=min(box1.left, box2.left),
        top=min(box1.top, box2.top),
        right=max(box1.right, box2.right),
        bottom=max(box1.bottom, box2.bottom),
        interactable=True,
        confidence=max(box1.confidence, box2.confidence),
        class_id=0,
    )
    return merged_box


def rectangles_overlap(rect1, rect2, padding):
    """
    Check if two rectangles overlap.

    Args:
    rect1 (list): A list of four numbers [x1, y1, x2, y2] representing the first rectangle.
    rect2 (list): A list of four numbers [x1, y1, x2, y2] representing the second rectangle.
    padding (int): An extra padding to consider around each rectangle.

    Returns:
    bool: True if rectangles overlap, False otherwise.
    """

    # Expanding each rectangle by padding
    rect1_padded = [
        rect1[0] - padding,
        rect1[1] - padding,
        rect1[2] + padding,
        rect1[3] + padding,
    ]

    rect2_padded = [
        rect2[0] - padding,
        rect2[1] - padding,
        rect2[2] + padding,
        rect2[3] + padding,
    ]

    # Check if one rectangle is to the left of the other
    if rect1_padded[2] < rect2_padded[0] or rect2_padded[2] < rect1_padded[0]:
        return False

    # Check if one rectangle is above the other
    if rect1_padded[3] < rect2_padded[1] or rect2_padded[3] < rect1_padded[1]:
        return False

    return True


def add_margin(pil_img, padding):
    width, height = pil_img.size
    new_width = width + padding * 2
    new_height = height + padding * 2
    result = Image.new(pil_img.mode, (new_width, new_height), (255, 255, 255))
    result.paste(pil_img, (padding, padding))
    return result


def draw_bounding_boxes(
    data_string,
    screenshot_img,
    pixel_ratio,
    viewport_size=None,
    add_ids=True,
    bbox_color=None,
    min_width=8,
    min_height=8,
    bbox_padding=0,
    bbox_border=2,
    plot_ids=None,
    img_padding=0,
    window_bounds=None,
    add_coords=False,
    allow_interaction_with_text=False,
    use_tag=True,
    use_id=True,
):
    model_client = ModelClient()
    """
    min_width and min_height: Minimum dimensions of the bounding box to be plotted.
    """
    # Read CSV data
    df = pd.read_csv(StringIO(data_string), delimiter=",", quotechar='"')
    # Save df to a CSV file
    df["Area"] = df["Width"] * df["Height"]
    # df.sort_values(by='Area', ascending=True, inplace=True)  # Draw smaller boxes first?

    if window_bounds is None:
        window_bounds = {
            "upper_bound": 0,
            "left_bound": 0,
            "right_bound": screenshot_img.size[0],
            "lower_bound": screenshot_img.size[1],
        }

    # Remove bounding boxes that are clipped.
    b_x, b_y = (
        # -img_padding,
        # -img_padding,
        window_bounds["left_bound"],
        window_bounds["upper_bound"],
    )

    # For windows that have scrolling functionality
    if viewport_size is not None:
        df = df[
            (df["Bottom"] - b_y >= 0)
            & (df["Top"] - b_y <= viewport_size["height"])
            & (df["Right"] - b_x >= 0)
            & (df["Left"] - b_x <= viewport_size["width"])
        ]
        viewport_area = viewport_size["width"] * viewport_size["height"]
        max_bbox_area = 0.8 * viewport_area  # 80% of the viewport area
        # Filter out bounding boxes that are more than 80% of the viewport area
        df = df[df["Area"] <= max_bbox_area]

    # Open the screenshot image
    img = screenshot_img.copy()
    img = add_margin(img, img_padding)
    draw = ImageDraw.Draw(img)

    # Load a TTF font with a larger size
    font_path = os.path.join(PROJECT_DIR, "SourceCodePro-Semibold.ttf")
    font_size = 16
    padding = 2
    font = ImageFont.truetype(font_path, font_size)

    # Create a color cycle using one of the categorical color palettes in matplotlib
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    bbox_id2visid = {}
    bbox_id2desc = {}
    index = 0
    id2center = {}
    existing_text_rectangles = []
    text_to_draw = []
    # Provide [id] textContent inputs to the model as text.
    text_content_elements = []
    text_content_text = set()  # Store text of interactable elements

    filtered_ids = []

    # Iterate through each row in the CSV and draw bounding boxes
    for _, row in df.iterrows():
        if not row["Interactable"]:
            # TODO(jykoh): Move this to a function
            content = ""
            if row["Element"] == "IMG" and pd.notna(row["Alt"]):
                content += row["Alt"]
            if pd.notna(row["TextContent"]):
                content += (
                    row["TextContent"].strip().replace("\n", "").replace("\t", "")
                )[
                    :200
                ]  # Limit to 200 characters

            if content and not (
                content.startswith(".") and "{" in content
            ):  # Check if the text is a CSS selector
                if content not in text_content_text:
                    if allow_interaction_with_text:
                        unique_id = str(index + 1)
                        bbox_id2visid[row["ID"]] = (
                            unique_id  # map the bounding box ID to the unique character ID
                        )
                        filtered_ids.append(row["ID"])

                        top, right, bottom, left, width, height = (
                            row["Top"],
                            row["Right"],
                            row["Bottom"],
                            row["Left"],
                            row["Width"],
                            row["Height"],
                        )

                        left = left + img_padding - b_x
                        right = right + img_padding - b_x
                        top = top + img_padding - b_y
                        bottom = bottom + img_padding - b_y

                        id2center[unique_id] = (
                            (left + right) / (2 * pixel_ratio),
                            (bottom + top) / (2 * pixel_ratio),
                            width / pixel_ratio,
                            height / pixel_ratio,
                        )
                        static_text_id = unique_id
                    else:
                        static_text_id = ""
                    if use_tag:
                        tag_text = "[StaticText] "
                    else:
                        tag_text = ""
                    if use_id:
                        id_text = f"[{static_text_id}] "
                    else:
                        id_text = ""
                    if add_coords:
                        x = (row["Left"] + row["Right"]) / (2 * pixel_ratio)
                        y = (row["Top"] + row["Bottom"]) / (2 * pixel_ratio)
                        text_content_elements.append(
                            f"{id_text}{tag_text}[x:{x},y:{y}] [{content}]"
                        )
                    else:
                        text_content_elements.append(f"{id_text}{tag_text}[{content}]")
                    text_content_text.add(content)
            continue

        if (plot_ids is not None) and (row["ID"] not in plot_ids):
            continue

        unique_id = str(index + 1)
        bbox_id2visid[row["ID"]] = (
            unique_id  # map the bounding box ID to the unique character ID
        )
        filtered_ids.append(row["ID"])

        top, right, bottom, left, width, height = (
            row["Top"],
            row["Right"],
            row["Bottom"],
            row["Left"],
            row["Width"],
            row["Height"],
        )

        left = left + img_padding - b_x
        right = right + img_padding - b_x
        top = top + img_padding - b_y
        bottom = bottom + img_padding - b_y

        id2center[unique_id] = (
            (left + right) / (2 * pixel_ratio),
            (bottom + top) / (2 * pixel_ratio),
            width / pixel_ratio,
            height / pixel_ratio,
        )

        if width >= min_width and height >= min_height:
            # Get the next color in the cycle
            color = bbox_color or color_cycle[index % len(color_cycle)]
            draw.rectangle(
                [
                    left - bbox_padding,
                    top - bbox_padding,
                    right + bbox_padding,
                    bottom + bbox_padding,
                ],
                outline=color,
                width=bbox_border,
            )
            bbox_id2desc[row["ID"]] = color

            # Draw the text on top of the rectangle
            if add_ids:
                # Calculate text position and size
                text_positions = [
                    (left - font_size, top - font_size),  # Top-left corner
                    (
                        left,
                        top - font_size,
                    ),  # Top-left corner (a little to the right)
                    (right, top - font_size),  # Top-right corner
                    (
                        right - font_size - 2 * padding,
                        top - font_size,
                    ),  # Top-right corner (a little to the left)
                    (left - font_size, bottom),  # Bottom-left corner
                    (
                        left,
                        bottom,
                    ),  # Bottom-left corner (a little to the right)
                    (
                        right - font_size - 2 * padding,
                        bottom,
                    ),  # Bottom-right corner (a little to the left)
                    (
                        left,
                        bottom,
                    ),  # Bottom-left corner (a little to the right)
                    (
                        right - font_size - 2 * padding,
                        bottom,
                    ),  # Bottom-right corner (a little to the left)
                ]
                text_width = draw.textlength(unique_id, font=font)
                text_height = font_size  # Assume the text is one line

                if viewport_size is not None:
                    for text_position in text_positions:
                        new_text_rectangle = [
                            text_position[0] - padding,
                            text_position[1] - padding,
                            text_position[0] + text_width + padding,
                            text_position[1] + text_height + padding,
                        ]

                        # Check if the new text rectangle is within the viewport
                        if (
                            new_text_rectangle[0] >= 0
                            and new_text_rectangle[1] >= 0
                            and new_text_rectangle[2] <= viewport_size["width"]
                            and new_text_rectangle[3] <= viewport_size["height"]
                        ):
                            # If the rectangle is within the viewport, check for overlaps
                            overlaps = False
                            for existing_rectangle in existing_text_rectangles:
                                if rectangles_overlap(
                                    new_text_rectangle,
                                    existing_rectangle,
                                    padding * 2,
                                ):
                                    overlaps = True
                                    break

                            if not overlaps:
                                break
                        else:
                            # If the rectangle is outside the viewport, try the next position
                            continue
                else:
                    # If none of the corners work, move the text rectangle by a fixed amount
                    text_position = (
                        text_positions[0][0] + padding,
                        text_positions[0][1],
                    )
                    new_text_rectangle = [
                        text_position[0] - padding,
                        text_position[1] - padding,
                        text_position[0] + text_width + padding,
                        text_position[1] + text_height + padding,
                    ]

                existing_text_rectangles.append(new_text_rectangle)
                text_to_draw.append(
                    (new_text_rectangle, text_position, unique_id, color)
                )

                # TODO(jykoh): Move this to a function
                content = ""
                if row["Element"] == "IMG":
                    if pd.notna(row["Alt"]):
                        content += row["Alt"]
                    # else:
                    #     try:
                    #         cropped_image = screenshot_img.crop(
                    #             (left, top, right, bottom)
                    #         )
                    #         caption = model_client.query_model(
                    #             cropped_image, "What is in the picture?"
                    #         )
                    #         content += "description: {}".format(caption["answer"])
                    #     except Exception as e:
                    #         print("error: {}".format(e))
                if pd.notna(row["TextContent"]):
                    content += (
                        row["TextContent"].strip().replace("\n", "").replace("\t", "")
                    )[
                        :200
                    ]  # Limit to 200 characters

                if use_tag:
                    tag_text = f"[{row['Element']}] "
                else:
                    tag_text = ""
                if use_id:
                    id_text = f"[{unique_id}] "
                else:
                    id_text = ""
                if add_coords:
                    x = id2center[unique_id][0]
                    y = id2center[unique_id][1]
                    text_content_elements.append(
                        f"{id_text}{tag_text}[x:{x},y:{y}] [{content}]"
                    )
                else:
                    text_content_elements.append(f"{id_text}{tag_text}[{content}]")
                if content in text_content_text:
                    # Remove text_content_elements with content
                    text_content_elements = [
                        element
                        for element in text_content_elements
                        if element.strip() != content
                    ]
                text_content_text.add(content)

        index += 1

    filtered_df = df[df["ID"].isin(filtered_ids)]

    for text_rectangle, text_position, unique_id, color in text_to_draw:
        # Draw a background rectangle for the text
        draw.rectangle(text_rectangle, fill=color)
        draw.text(text_position, unique_id, font=font, fill="white")

    content_str = "\n".join(text_content_elements)
    return img, id2center, content_str, filtered_df
