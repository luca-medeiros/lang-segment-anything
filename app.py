import os
import json
import time
from io import BytesIO
from datetime import datetime

import gradio as gr
import requests
import numpy as np
from PIL import Image

from lang_sam import SAM_MODELS
from lang_sam.server import PORT, server


def mask_to_polygon(mask):
    """Convert a binary mask to polygon representation."""
    import cv2
    # Ensure mask is binary and in uint8 format
    if mask.dtype != np.uint8:
        binary_mask = (mask > 0).astype(np.uint8) * 255
    else:
        binary_mask = mask
    
    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Convert contours to polygons
    polygons = []
    for contour in contours:
        # Flatten the contour and convert to list
        polygon = contour.flatten().tolist()
        # Only add polygons with enough points
        if len(polygon) >= 6:  # At least 3 points (x,y pairs)
            polygons.append(polygon)
    
    return polygons

def save_mask_images(masks, image_path, output_dir="annotations/masks"):
    """Save individual mask images for visualization and further processing."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a unique base name
    image_filename = os.path.basename(image_path)
    base_name = os.path.splitext(image_filename)[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    mask_paths = []
    for i, mask in enumerate(masks):
        # Convert mask to binary image
        binary_mask = (mask > 0).astype(np.uint8) * 255
        mask_img = Image.fromarray(binary_mask)
        
        # Save mask
        mask_path = os.path.join(output_dir, f"{base_name}_mask_{i}_{timestamp}.png")
        mask_img.save(mask_path)
        mask_paths.append(mask_path)
    
    return mask_paths

def save_coco_annotations(image_path, masks, boxes, scores, labels, output_dir="annotations"):
    """Save annotations in COCO format with proper segmentation masks."""
    import cv2
    os.makedirs(output_dir, exist_ok=True)
    
    # Load image to get dimensions
    img = Image.open(image_path)
    width, height = img.size
    
    # Create a unique filename based on the original image name
    image_filename = os.path.basename(image_path)
    base_name = os.path.splitext(image_filename)[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create COCO format data
    coco_data = {
        "info": {
            "description": f"Lang-SAM annotations for {image_filename}",
            "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "images": [{
            "id": 1,
            "file_name": image_filename,
            "width": width,
            "height": height
        }],
        "annotations": [],
        "categories": []
    }
    
    # Add categories based on unique labels
    unique_labels = set(labels)
    category_map = {}
    for i, label in enumerate(unique_labels):
        category_id = i + 1
        coco_data["categories"].append({
            "id": category_id,
            "name": label,
            "supercategory": "object"
        })
        category_map[label] = category_id
    
    # Save mask images for visualization
    mask_paths = save_mask_images(masks, image_path)
    
    # Add annotations
    for i, (mask, box, score, label) in enumerate(zip(masks, boxes, scores, labels)):
        # Convert mask to polygon
        polygons = mask_to_polygon(mask)
        
        if not polygons:  # If no valid polygons, use bounding box
            x1, y1, x2, y2 = box
            polygons = [[float(x1), float(y1), float(x2), float(y1), 
                         float(x2), float(y2), float(x1), float(y2)]]
        
        # Calculate area from mask
        area = float(np.sum(mask > 0))
        
        # Get bounding box in COCO format [x, y, width, height]
        x1, y1, x2, y2 = box
        coco_bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
        
        coco_data["annotations"].append({
            "id": i + 1,
            "image_id": 1,
            "category_id": category_map.get(label, 1),
            "segmentation": polygons,
            "area": area,
            "bbox": coco_bbox,
            "iscrowd": 0,
            "score": float(score),
            "mask_path": mask_paths[i] if i < len(mask_paths) else None
        })
    
    # Save to file
    output_file = os.path.join(output_dir, f"{base_name}_coco_{timestamp}.json")
    with open(output_file, "w") as f:
        json.dump(coco_data, f, indent=2)
    
    return output_file

def save_yolo_annotations(image_path, masks, boxes, scores, labels, output_dir="annotations"):
    """Save annotations in YOLO format with segmentation support."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load image to get dimensions
    img = Image.open(image_path)
    width, height = img.size
    
    # Create a unique filename based on the original image name
    image_filename = os.path.basename(image_path)
    base_name = os.path.splitext(image_filename)[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create a class mapping file
    unique_labels = list(set(labels))
    class_map = {label: i for i, label in enumerate(unique_labels)}
    
    # Save class mapping
    classes_file = os.path.join(output_dir, f"{base_name}_classes_{timestamp}.txt")
    with open(classes_file, "w") as f:
        for label in unique_labels:
            f.write(f"{label}\n")
    
    # Create YOLO annotation file (standard format)
    yolo_file = os.path.join(output_dir, f"{base_name}_yolo_{timestamp}.txt")
    
    # Create YOLO segmentation file (YOLO v8 format)
    yolo_seg_file = os.path.join(output_dir, f"{base_name}_yolo_seg_{timestamp}.txt")
    
    # Save both standard YOLO and segmentation YOLO formats
    with open(yolo_file, "w") as f_box, open(yolo_seg_file, "w") as f_seg:
        for mask, box, score, label in zip(masks, boxes, scores, labels):
            # Get class ID
            class_id = class_map[label]
            
            # Convert box coordinates to YOLO format: [class_id, x_center, y_center, width, height]
            x1, y1, x2, y2 = box
            x_center = (x1 + x2) / 2 / width
            y_center = (y1 + y2) / 2 / height
            box_width = (x2 - x1) / width
            box_height = (y2 - y1) / height
            
            # Write standard YOLO format (bounding box)
            f_box.write(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")
            
            # Get polygon points for segmentation
            polygons = mask_to_polygon(mask)
            if polygons:
                # Use the first polygon (usually the largest)
                polygon = polygons[0]
                
                # Convert polygon points to normalized coordinates
                normalized_polygon = []
                for i in range(0, len(polygon), 2):
                    if i+1 < len(polygon):
                        x = polygon[i] / width
                        y = polygon[i+1] / height
                        normalized_polygon.extend([x, y])
                
                # Write YOLO segmentation format
                # Format: class_id x1 y1 x2 y2 ... xn yn
                seg_line = f"{class_id}"
                for point in normalized_polygon:
                    seg_line += f" {point:.6f}"
                f_seg.write(seg_line + "\n")
            else:
                # If no polygon, use the bounding box as fallback
                f_seg.write(f"{class_id} {x1/width:.6f} {y1/height:.6f} {x2/width:.6f} {y1/height:.6f} "
                           f"{x2/width:.6f} {y2/height:.6f} {x1/width:.6f} {y2/height:.6f}\n")
    
    # Also save mask images
    mask_dir = os.path.join(output_dir, "masks")
    os.makedirs(mask_dir, exist_ok=True)
    
    for i, mask in enumerate(masks):
        # Convert mask to binary image
        binary_mask = (mask > 0).astype(np.uint8) * 255
        mask_img = Image.fromarray(binary_mask)
        
        # Save mask
        mask_path = os.path.join(mask_dir, f"{base_name}_mask_{i}_{timestamp}.png")
        mask_img.save(mask_path)
    
    return yolo_seg_file, classes_file

def inference(sam_type, box_threshold, text_threshold, image, text_prompt, save_format="none"):
    """Gradio function that makes a request to the /predict LitServe endpoint."""
    url = f"http://localhost:{PORT}/predict"  # Adjust port if needed
    
    # Check if image is valid
    if not image or not os.path.exists(image):
        print(f"Invalid image path: {image}")
        return None, "Invalid image path"
    
    # Get the raw prediction results directly from the model
    try:
        # Load the image
        image_pil = Image.open(image).convert("RGB")
        
        # Initialize the model locally for direct access
        from lang_sam import LangSAM
        model = LangSAM(sam_type=sam_type)
        
        # Make prediction
        results = model.predict(
            images_pil=[image_pil],
            texts_prompt=[text_prompt],
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )[0]  # Get the first result
        
        # Draw the results on the image
        from lang_sam.utils import draw_image
        image_array = np.asarray(image_pil)
        output_image = draw_image(
            image_array,
            results["masks"],
            results["boxes"],
            results["scores"],
            results["labels"]
        )
        output_image = Image.fromarray(np.uint8(output_image)).convert("RGB")
        
        # Save annotations if requested
        annotation_path = ""
        if save_format != "none" and len(results["masks"]) > 0:
            masks = results["masks"]
            boxes = results["boxes"]
            scores = results["scores"]
            labels = results["labels"]
            
            if save_format == "coco":
                annotation_path = save_coco_annotations(image, masks, boxes, scores, labels)
                print(f"Saved COCO annotations to {annotation_path}")
            elif save_format == "yolo":
                yolo_path, classes_path = save_yolo_annotations(image, masks, boxes, scores, labels)
                annotation_path = f"YOLO annotations: {yolo_path}, Classes: {classes_path}"
                print(f"Saved YOLO annotations to {yolo_path} and classes to {classes_path}")
        
        return output_image, annotation_path
        
    except Exception as e:
        import traceback
        print(f"Error in inference: {e}")
        print(traceback.format_exc())
        return None, f"Error: {str(e)}"


with gr.Blocks(title="lang-sam") as blocks:
    with gr.Row():
        sam_model_choices = gr.Dropdown(choices=list(SAM_MODELS.keys()), label="SAM Model", value="sam2.1_hiera_small")
        box_threshold = gr.Slider(minimum=0.0, maximum=1.0, value=0.3, label="Box Threshold")
        text_threshold = gr.Slider(minimum=0.0, maximum=1.0, value=0.25, label="Text Threshold")
    with gr.Row():
        image_input = gr.Image(type="filepath", label="Input Image")
        output_image = gr.Image(type="pil", label="Output Image")
    
    with gr.Row():
        text_prompt = gr.Textbox(lines=1, label="Text Prompt")
        save_format = gr.Dropdown(
            choices=["none", "coco", "yolo"], 
            label="Save Annotations Format", 
            value="none",
            info="Select format to save annotations"
        )
    
    submit_btn = gr.Button("Run Prediction")
    annotation_output = gr.Textbox(label="Annotation Path", interactive=False)

    submit_btn.click(
        fn=inference,
        inputs=[sam_model_choices, box_threshold, text_threshold, image_input, text_prompt, save_format],
        outputs=[output_image, annotation_output],
    )

    examples = [
        [
            "sam2.1_hiera_small",
            0.32,
            0.25,
            os.path.join(os.path.dirname(__file__), "assets", "fruits.jpg"),
            "kiwi. watermelon. blueberry.",
        ],
        [
            "sam2.1_hiera_small",
            0.3,
            0.25,
            os.path.join(os.path.dirname(__file__), "assets", "car.jpeg"),
            "wheel.",
        ],
        [
            "sam2.1_hiera_small",
            0.3,
            0.25,
            os.path.join(os.path.dirname(__file__), "assets", "food.jpg"),
            "food.",
        ],
    ]

    gr.Examples(
        examples=examples,
        inputs=[sam_model_choices, box_threshold, text_threshold, image_input, text_prompt],
        outputs=output_image,
    )

server.app = gr.mount_gradio_app(server.app, blocks, path="/gradio")

if __name__ == "__main__":
    print(f"Starting LitServe and Gradio server on port {PORT}...")
    server.run(port=PORT)
