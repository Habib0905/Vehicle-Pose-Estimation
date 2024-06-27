import argparse
import json
import os
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()

# Number of keypoints for car pose estimation
numOfKpts = 14  # Update this with the number of keypoints in your dataset

parser.add_argument(
    "--coco_json_path",
    default="G:/Desktop/root/Annotations/COCO/morewood1/car_keypoints_test.json",
    type=str,
    help="Input: COCO format JSON file containing car pose annotations",
)

parser.add_argument(
    "--yolo_save_root_dir",
    default="G:/Desktop/root/Annotations/YOLO/labels/morewood1",
    type=str,
    help="Specify where to save the output directory of labels in YOLO pose format",
)


def convert_keypoints_to_list(keypoints, img_width, img_height):
    keypoints_list = []
    for i in range(0, numOfKpts * 3, 3):
        x, y, v = keypoints[i], keypoints[i + 1], keypoints[i + 2]
        if v > 0:  # Check if keypoint is visible
            if x < 0 or y < 0 or x > img_width or y > img_height:  # Handle negative or out-of-bounds keypoints
                v = 0  # Set visibility flag to 0
            else:
                x /= img_width
                y /= img_height
            keypoints_list.extend([x, y, v])
        else:
            # Pad with zeros for keypoints that are not labeled
            keypoints_list.extend([0, 0, 0])
    return keypoints_list


def main(json_file, yolo_save_root_dir):
    os.makedirs(yolo_save_root_dir, exist_ok=True)

    data = json.load(open(json_file, "r"))

    id_map = {}  # Remap category IDs if necessary

    for category in data["categories"]:
        id_map[category["id"]] = len(id_map)

    images = {image["id"]: image for image in data["images"]}
    annotations_by_image_id = {}

    for annotation in data["annotations"]:
        image_id = annotation["image_id"]
        if image_id not in annotations_by_image_id:
            annotations_by_image_id[image_id] = []
        annotations_by_image_id[image_id].append(annotation)

    for image_id, annotations in annotations_by_image_id.items():
        image_info = images[image_id]
        img_filename = image_info["file_name"]
        img_width, img_height = image_info["width"], image_info["height"]

        # Adjust filename generation to remove first two characters and insert underscore after second character
        image_filename = image_info["file_name"]
        image_id_modified = str(image_id)[2:]  # Remove first two characters
        image_id_modified = image_id_modified[:2] + "_" + image_id_modified[2:]  # Insert underscore after second character
        label_filename = f"{image_id_modified}.txt"

        with open(os.path.join(yolo_save_root_dir, label_filename), "w") as label_file:
            for annotation in annotations:
                bbox = annotation["bbox"]
                x, y, w, h = bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2, bbox[2], bbox[3]
                x /= img_width
                y /= img_height
                w /= img_width
                h /= img_height

                keypoints = annotation.get("keypoints", [])  # Get keypoints or empty list if not present
                if len(keypoints) != numOfKpts * 3:  # Ensure correct number of keypoints
                    print(f"Warning: Incorrect number of keypoints for image {image_id}")
                    continue

                keypoints_list = convert_keypoints_to_list(keypoints, img_width, img_height)

                cls = id_map.get(annotation["category_id"], -1)  # Get class index or -1 if not found
                if cls == -1:
                    print(f"Warning: Category ID not found for annotation in image {image_id}")
                    continue

                if len(keypoints_list) != numOfKpts * 3:  # Ensure correct number of keypoints after conversion
                    print(f"Warning: Incorrect number of keypoints after conversion for image {image_id}")
                    continue

                line = f"{cls} {x} {y} {w} {h} {' '.join(map(str, keypoints_list))}\n"
                label_file.write(line)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args.coco_json_path, args.yolo_save_root_dir)
