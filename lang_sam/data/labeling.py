import json
import os

import fastdup
from torchvision import ops
from tqdm import tqdm

from lang_sam import LangSAM
from lang_sam.data.utils import get_image_paths
from lang_sam.data.utils import load_image
from lang_sam.data.utils import mask_to_polygon
from lang_sam.data.utils import results_to_df


def clean_results(data_path, results):
    df = results_to_df(results)
    df['split'] = 'train'
    df = df.drop_duplicates()
    fd = fastdup.create(work_dir=data_path)
    fd.run(annotations=df)
    outlier_df = fd.outliers()
    list_of_outliers = outlier_df[outlier_df.distance < 0.68].img_filename_outlier.tolist()
    stats_df = fd.img_stats()
    blurry_images = stats_df[stats_df['blur'] < 50]
    bright_images = stats_df[stats_df['mean'] > 220.5]
    dark_images = stats_df[stats_df['mean'] < 13]

    print(f"Broken: {len(list_of_broken_images)}")
    print(f"Duplicates: {len(list_of_duplicates)}")
    print(f"Outliers: {len(list_of_outliers)}")
    print(f"Dark: {len(list_of_dark_images)}")
    print(f"Bright: {len(list_of_bright_images)}")
    print(f"Blurry: {len(list_of_blurry_images)}")

    problem_images = list_of_duplicates + list_of_broken_images + list_of_outliers + list_of_dark_images + list_of_bright_images + list_of_blurry_images

    print(f"Total unique images: {len(set(problem_images))}")


def create_coco_dataset(results):
    images = []
    annotations_list = []
    categories = {}
    category_id = 1
    ann_id = 1

    for i, res in enumerate(results):
        images.append({
            "file_name": os.path.basename(res['image']['file_path']),
            "id": i,
            "width": res['image']['width'],
            "height": res['image']['height']
        })

        for ann in res['outputs']:
            polygon, box, label, score = ann
            # Get or assign category_id
            if label not in categories:
                categories[label] = category_id
                category_id += 1
            cat_id = categories[label]

            annotations_list.append({
                "id": ann_id,
                "image_id": i,
                "category_id": cat_id,
                "segmentation": polygon,
                "bbox": box.tolist(),
                "iscrowd": 0,
                "area": float(box[2] * box[3]),
                "score": score
            })

            ann_id += 1

    categories_list = [{"id": i, "name": cat} for i, cat in enumerate(categories, start=1)]
    coco_data = {
        "images": images,
        "annotations": annotations_list,
        "categories": categories_list,
    }

    return coco_data


def predict_images(image_paths, text_prompt, lang_sam_model):
    results = []
    for image_path in tqdm(image_paths):
        image_pil = load_image(image_path)
        masks, boxes, phrases, logits = lang_sam_model.predict(image_pil, text_prompt)
        boxes = ops.box_convert(boxes, "xyxy", "xywh").numpy()
        detections = parse_results(masks, boxes, phrases, logits)
        results.append({
            "image": {
                "file_path": image_path,
                "width": image_pil.width,
                "height": image_pil.height
            },
            "outputs": detections
        })
    return results


def parse_results(masks, boxes, phrases, logits):
    detections = []
    for mask, box, phrase, logit in zip(masks, boxes, phrases, logits):
        polygons = mask_to_polygon(mask)
        score = logit.item()
        label = phrase
        detections.append([polygons, box, label, score])
    return detections


def main(data_folder, text_prompt):
    image_paths = get_image_paths(data_folder)
    lang_sam_model = LangSAM()
    results = predict_images(image_paths, text_prompt, lang_sam_model)
    coco_data = create_coco_dataset(results)
    with open(os.path.join(data_folder, "coco.json"), "w") as f:
        json.dump(coco_data, f)


if __name__ == "__main__":
    data_folder = "./assets/"
    main(data_folder, 'food')
