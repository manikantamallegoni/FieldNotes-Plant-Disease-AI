from pathlib import Path
import ast
import io
import os
import httpx

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import onnxruntime as ort


# ==================================================
# APP SETUP
# ==================================================

app = FastAPI(title="Plant Disease AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent


# ==================================================
# EXISTING V3 CLASSIFICATION MODEL
# ==================================================

MODEL_PATH = BASE_DIR / "fine_tuned_v3_classweights.keras"

if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"V3 model not found: {MODEL_PATH}")

model = tf.keras.models.load_model(str(MODEL_PATH))

class_names = [
    "Apple_Black_rot",
    "Apple_Cedar_apple_rust",
    "Apple_Powdery_mildew",
    "Apple_healthy",
    "Apple_scab",
    "Cherry_(including_sour)_Powdery_mildew",
    "Cherry_(including_sour)_healthy",
    "Corn_(maize)_Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)_Common_rust",
    "Corn_(maize)_Northern_Leaf_Blight",
    "Corn_(maize)_healthy",
    "Cotton_leaf_diseased",
    "Cotton_leaf_fresh",
    "Grape_Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange_Haunglongbing_(Citrus_greening)",
    "Orange__Black_Rot",
    "Orange__Canker",
    "Orange__Healthy",
    "Peach_Bacterial_spot",
    "Peach_healthy",
    "Pepper,_bell_Bacterial_spot",
    "Pepper,_bell_healthy",
    "Potato_Early_blight",
    "Potato_Late_blight",
    "Potato_healthy",
    "Squash_Powdery_mildew",
    "Strawberry_Leaf_scorch",
    "Strawberry_healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato_Target_Spot",
    "Tomato_Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato_Tomato_mosaic_virus",
    "Tomato_healthy",
    "Wheat_healthy",
    "Wheat_leaf_rust",
    "Wheat_nitrogen_deficiency",
]

model_output_classes = model.output_shape[-1]
if model_output_classes is not None and int(model_output_classes) != len(class_names):
    raise RuntimeError(
        f"V3 model outputs {model_output_classes} classes, but "
        f"{len(class_names)} class names are configured."
    )


# ==================================================
# PADDY CLASSIFICATION MODEL
# ==================================================

PADDY_MODEL_PATH = BASE_DIR / "paddy_classifier.onnx"

if not PADDY_MODEL_PATH.is_file():
    raise FileNotFoundError(f"Paddy model not found: {PADDY_MODEL_PATH}")

paddy_session = ort.InferenceSession(
    str(PADDY_MODEL_PATH),
    providers=["CPUExecutionProvider"],
)

paddy_input_name = paddy_session.get_inputs()[0].name

paddy_class_names = [
    "Bacterial Leaf Blight",
    "Brown Spot",
    "Healthy Rice Leaf",
    "Leaf Blast",
    "Leaf scald",
    "Sheath Blight",
]


# ==================================================
# CHILLI CLASSIFICATION MODEL
# ==================================================

CHILLI_MODEL_PATH = BASE_DIR / "chilli_classifier.onnx"

if not CHILLI_MODEL_PATH.is_file():
    raise FileNotFoundError(f"Chilli model not found: {CHILLI_MODEL_PATH}")

chilli_session = ort.InferenceSession(
    str(CHILLI_MODEL_PATH),
    providers=["CPUExecutionProvider"],
)

chilli_input_name = chilli_session.get_inputs()[0].name

chilli_class_names = [
    "Chilli __Whitefly",
    "Chilli __Yellowish",
    "Chilli__Anthracnos",
    "Chilli__Damping_Off",
    "Chilli__Leaf_Curl_Virus",
    "Chilli__Leaf_Spot",
    "Chilli__Veinal_Mottle_Virus",
    "Chilli___healthy",
]


# ==================================================
# YOLO LEAF DETECTION MODEL
# ==================================================

DETECTOR_MODEL_PATH = BASE_DIR / "plant_leaf_detector.onnx"

if not DETECTOR_MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Leaf detector model not found: {DETECTOR_MODEL_PATH}"
    )

detector_session = ort.InferenceSession(
    str(DETECTOR_MODEL_PATH),
    providers=["CPUExecutionProvider"],
)

detector_input = detector_session.get_inputs()[0]
detector_input_name = detector_input.name

DETECTOR_SIZE = 640

detector_metadata = (
    detector_session.get_modelmeta().custom_metadata_map
)

try:
    detector_class_names = ast.literal_eval(
        detector_metadata["names"]
    )
except (KeyError, ValueError, SyntaxError) as exc:
    raise RuntimeError(
        "Could not read YOLO class names from ONNX metadata."
    ) from exc

if isinstance(detector_class_names, dict):
    detector_class_names = [
        detector_class_names[i]
        for i in range(len(detector_class_names))
    ]

if not detector_class_names:
    raise RuntimeError("YOLO detector has no class names.")

detector_output_shape = detector_session.get_outputs()[0].shape


# ==================================================
# CROP MAPPING FOR V3 MODEL
# ==================================================

crop_prefixes = {
    # These crop groups are present in the 44-class V3 classifier.
    "apple": "Apple_",
    "cherry": "Cherry_",
    "corn": "Corn_(maize)_",
    "maize": "Corn_(maize)_",
    "cotton": "Cotton_",
    "grape": "Grape",
    "orange": "Orange",
    "peach": "Peach_",
    "pepper": "Pepper,_bell",
    "bell pepper": "Pepper,_bell",
    "capsicum": "Pepper,_bell",
    "potato": "Potato_",
    "squash": "Squash_",
    "strawberry": "Strawberry_",
    "tomato": "Tomato_",
    "wheat": "Wheat_",
}


# ==================================================
# SHARED CLASSIFICATION HELPERS
# ==================================================

def get_prediction_status(confidence: float) -> str:
    if confidence >= 0.75:
        return "prediction_available"
    if confidence >= 0.45:
        return "possible_disease_retake_advised"
    return "uncertain_please_retake_clearer_image"


def get_top_predictions(names, scores, count=3):
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)

    if len(names) != len(scores):
        raise ValueError(
            f"Class-name count ({len(names)}) does not match "
            f"score count ({len(scores)})."
        )

    if scores.size == 0 or not np.all(np.isfinite(scores)):
        raise ValueError("Invalid classification scores.")

    top_ids = np.argsort(scores)[::-1][:min(count, len(scores))]

    return [
        {
            "prediction": names[int(i)],
            "confidence": float(scores[int(i)]),
        }
        for i in top_ids
    ]


def outputs_to_probabilities(raw_scores):
    """Keep valid probability outputs; otherwise apply softmax."""
    scores = np.asarray(raw_scores, dtype=np.float32).reshape(-1)

    if scores.size == 0 or not np.all(np.isfinite(scores)):
        raise ValueError("Model returned invalid prediction scores.")

    already_probabilities = (
        np.all(scores >= 0.0)
        and np.all(scores <= 1.0)
        and np.isclose(float(np.sum(scores)), 1.0, atol=0.02)
    )

    if already_probabilities:
        return scores

    shifted = scores - np.max(scores)
    exp_scores = np.exp(shifted)
    total = float(np.sum(exp_scores))

    if total <= 0.0 or not np.isfinite(total):
        raise ValueError("Could not convert scores to probabilities.")

    return exp_scores / total


def prepare_onnx_image(image: Image.Image) -> np.ndarray:
    """Preprocess an image for the 224x224 classification ONNX models."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = image.resize((224, 224))

    image_array = np.asarray(image, dtype=np.float32) / 255.0
    image_array = np.transpose(image_array, (2, 0, 1))

    return np.expand_dims(image_array, axis=0).astype(np.float32)


# ==================================================
# CLASSIFICATION FUNCTIONS
# ==================================================

def predict_paddy(image: Image.Image):
    image_array = prepare_onnx_image(image)

    outputs = paddy_session.run(
        None,
        {paddy_input_name: image_array},
    )[0]

    probabilities = outputs_to_probabilities(outputs[0])

    return get_top_predictions(paddy_class_names, probabilities)


def predict_chilli(image: Image.Image):
    image_array = prepare_onnx_image(image)

    outputs = chilli_session.run(
        None,
        {chilli_input_name: image_array},
    )[0]

    probabilities = outputs_to_probabilities(outputs[0])

    return get_top_predictions(chilli_class_names, probabilities)


def predict_existing_crop(image: Image.Image, selected_crop: str):
    # Preserve original V3 preprocessing: RGB, 224x224, values 0..255.
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = image.resize((224, 224))

    image_array = np.asarray(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array, verbose=0)[0]

    if selected_crop:
        prefix = crop_prefixes.get(selected_crop)

        if prefix is None:
            supported = ", ".join(sorted(crop_prefixes.keys()))
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Crop '{selected_crop}' is not supported by "
                    f"the V3 classifier. Supported crops: {supported}, "
                    "paddy, rice, chilli, chili, mirchi."

                ),
            )

        valid_ids = [
            i for i, name in enumerate(class_names)
            if name.startswith(prefix)
        ]

        if not valid_ids:
            raise HTTPException(
                status_code=400,
                detail=f"No model classes found for '{selected_crop}'.",
            )

        return get_top_predictions(
            [class_names[i] for i in valid_ids],
            predictions[valid_ids],
        )

    return get_top_predictions(class_names, predictions)


# ==================================================
# YOLO DETECTION HELPERS
# ==================================================

def prepare_detector_image(image: Image.Image):
    """
    Resize and pad an RGB image to 640x640.

    Returns:
        input_tensor, scale, left_padding, top_padding,
        original_width, original_height
    """
    image = ImageOps.exif_transpose(image).convert("RGB")

    original_width, original_height = image.size

    if original_width <= 0 or original_height <= 0:
        raise ValueError("Image dimensions are invalid.")

    scale = min(
        DETECTOR_SIZE / original_width,
        DETECTOR_SIZE / original_height,
    )

    resized_width = max(1, round(original_width * scale))
    resized_height = max(1, round(original_height * scale))

    resized = image.resize(
        (resized_width, resized_height),
        Image.Resampling.BILINEAR,
    )

    canvas = Image.new(
        "RGB",
        (DETECTOR_SIZE, DETECTOR_SIZE),
        (114, 114, 114),
    )

    pad_x = (DETECTOR_SIZE - resized_width) // 2
    pad_y = (DETECTOR_SIZE - resized_height) // 2

    canvas.paste(resized, (pad_x, pad_y))

    array = np.asarray(canvas, dtype=np.float32) / 255.0
    array = np.transpose(array, (2, 0, 1))
    array = np.expand_dims(array, axis=0).astype(np.float32)

    return (
        array,
        scale,
        pad_x,
        pad_y,
        original_width,
        original_height,
    )


def box_iou(box, boxes):
    """Calculate intersection-over-union for xyxy boxes."""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    intersection = (
        np.maximum(0, x2 - x1) *
        np.maximum(0, y2 - y1)
    )

    area_a = (
        max(0, box[2] - box[0]) *
        max(0, box[3] - box[1])
    )

    area_b = (
        np.maximum(0, boxes[:, 2] - boxes[:, 0]) *
        np.maximum(0, boxes[:, 3] - boxes[:, 1])
    )

    union = area_a + area_b - intersection

    return intersection / np.maximum(union, 1e-7)


def run_leaf_detection(
    image: Image.Image,
    confidence_threshold: float = 0.35,
    iou_threshold: float = 0.45,
):
    (
        prepared,
        scale,
        pad_x,
        pad_y,
        original_width,
        original_height,
    ) = prepare_detector_image(image)

    raw_outputs = detector_session.run(
        None,
        {detector_input_name: prepared},
    )

    if not raw_outputs:
        raise ValueError("YOLO returned no output.")

    predictions = np.asarray(raw_outputs[0])

    # Expected YOLO11 output: [1, 34, 8400] for 30 classes.
    if predictions.ndim == 3:
        predictions = predictions[0]

    if predictions.ndim != 2:
        raise ValueError(
            f"Unexpected detector output shape: {predictions.shape}"
        )

    expected_features = 4 + len(detector_class_names)

    if predictions.shape[0] == expected_features:
        predictions = predictions.T
    elif predictions.shape[1] != expected_features:
        raise ValueError(
            f"Detector output shape {predictions.shape} does not match "
            f"{len(detector_class_names)} classes."
        )

    if predictions.shape[0] == 0:
        return []

    # YOLO output is centre_x, centre_y, width, height, class scores.
    boxes_xywh = predictions[:, :4]
    class_scores = predictions[:, 4:]

    if not np.all(np.isfinite(boxes_xywh)):
        raise ValueError("Detector returned invalid box coordinates.")

    # This exported model has one score per class and no objectness column.
    class_ids = np.argmax(class_scores, axis=1)
    scores = class_scores[np.arange(len(class_scores)), class_ids]

    valid = (
        np.isfinite(scores)
        & (scores >= confidence_threshold)
        & (scores <= 1.0)
        & (boxes_xywh[:, 2] > 0)
        & (boxes_xywh[:, 3] > 0)
    )

    boxes_xywh = boxes_xywh[valid]
    class_ids = class_ids[valid]
    scores = scores[valid]

    if len(scores) == 0:
        return []

    boxes = np.empty_like(boxes_xywh, dtype=np.float32)

    boxes[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    boxes[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    boxes[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
    boxes[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2

    # Class-aware non-maximum suppression.
    keep = []

    for class_id in np.unique(class_ids):
        class_indices = np.where(class_ids == class_id)[0]

        order = class_indices[
            np.argsort(scores[class_indices])[::-1]
        ]

        while len(order) > 0:
            current = int(order[0])
            keep.append(current)

            if len(order) == 1:
                break

            overlaps = box_iou(boxes[current], boxes[order[1:]])

            order = order[1:][overlaps < iou_threshold]

    # Limit the number of boxes returned per image.
    keep = sorted(
        keep,
        key=lambda index: float(scores[index]),
        reverse=True,
    )[:30]

    detections = []

    for index in keep:
        x1, y1, x2, y2 = boxes[index]

        # Map coordinates back to the original, unpadded image.
        x1 = (x1 - pad_x) / scale
        y1 = (y1 - pad_y) / scale
        x2 = (x2 - pad_x) / scale
        y2 = (y2 - pad_y) / scale

        x1 = float(np.clip(x1, 0, original_width))
        y1 = float(np.clip(y1, 0, original_height))
        x2 = float(np.clip(x2, 0, original_width))
        y2 = float(np.clip(y2, 0, original_height))

        if x2 <= x1 or y2 <= y1:
            continue

        class_id = int(class_ids[index])

        detections.append({
            "class_id": class_id,
            "label": str(detector_class_names[class_id]),
            "confidence": float(scores[index]),
            "box": {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            },
        })

    return detections


# ==================================================
# ROUTES
# ==================================================

@app.get("/")
def home():
    return {
        "message": "Plant Disease AI API is running!",
        "models": [
            "Existing V3 crop classifier",
            "Paddy ONNX classifier",
            "Chilli ONNX classifier",
            "YOLO11s leaf detector",
        ],
        "endpoints": ["/", "/health", "/predict", "/detect", "/treatment", "/docs"],
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "v3_model_loaded": model is not None,
        "paddy_model_loaded": paddy_session is not None,
        "chilli_model_loaded": chilli_session is not None,
        "detector_model_loaded": detector_session is not None,
        "detector_class_count": len(detector_class_names),
        "treatment_configured": bool(
            os.getenv("TAVILY_API_KEY") and os.getenv("OPENROUTER_API_KEY")
        ),
        "endpoints": ["/", "/health", "/predict", "/detect", "/treatment", "/docs"],
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    crop: str = Form(default=""),
):
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")

    try:
        image = Image.open(io.BytesIO(contents))
        image.load()
        image = ImageOps.exif_transpose(image).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file. Please select a valid image.",
        )

    selected_crop = crop.strip().lower()

    try:
        if selected_crop in {"paddy", "rice"}:
            top_3 = predict_paddy(image)
            crop_name = "Paddy"

        elif selected_crop in {"chilli", "chili", "mirchi"}:
            top_3 = predict_chilli(image)
            crop_name = "Chilli"

        else:
            top_3 = predict_existing_crop(image, selected_crop)

            if selected_crop:
                crop_name = (
                    "Bell Pepper"
                    if selected_crop == "bell pepper"
                    else selected_crop.title()
                )
            else:
                predicted_label = top_3[0]["prediction"]
                first_token = predicted_label.split("_")[0].strip()
                crop_name = {
                    "Apple": "Apple",
                    "Cherry": "Cherry",
                    "Corn": "Corn",
                    "Cotton": "Cotton",
                    "Grape": "Grape",
                    "Orange": "Orange",
                    "Peach": "Peach",
                    "Pepper,": "Bell Pepper",
                    "Potato": "Potato",
                    "Squash": "Squash",
                    "Strawberry": "Strawberry",
                    "Tomato": "Tomato",
                    "Wheat": "Wheat",
                }.get(first_token, first_token)

        if not top_3:
            raise ValueError("The model returned no predictions.")

        top_prediction = top_3[0]["prediction"]
        top_confidence = float(top_3[0]["confidence"])

        return {
            "crop": crop_name,
            "disease": top_prediction,
            "confidence": top_confidence,
            "status": get_prediction_status(top_confidence),
            "top_3_predictions": top_3,
        }

    except HTTPException:
        raise
    except Exception as exc:
        print(
            f"Prediction failed for crop "
            f"'{selected_crop or 'auto'}': {exc}"
        )
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check the API terminal for the error.",
        )


@app.post("/detect")
async def detect_leaves(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(default=0.35),
):
    """
    Detect leaves or leaf conditions using YOLO.

    Returns boxes in original-image pixel coordinates.
    """
    if not 0.05 <= confidence_threshold <= 0.95:
        raise HTTPException(
            status_code=400,
            detail="Confidence threshold must be between 0.05 and 0.95.",
        )

    contents = await file.read()

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="The uploaded image is empty.",
        )

    try:
        image = Image.open(io.BytesIO(contents))
        image.load()
        image = ImageOps.exif_transpose(image).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file. Upload a valid image.",
        )

    try:
        detections = run_leaf_detection(
            image,
            confidence_threshold=confidence_threshold,
        )

        return {
            "detector": "YOLO11s",
            "image_width": image.width,
            "image_height": image.height,
            "detection_count": len(detections),
            "detections": detections,
        }

    except Exception as exc:
        print(f"Leaf detection failed: {exc}")
        raise HTTPException(
            status_code=500,
            detail="Leaf detection failed. Check the API terminal.",
        )
class TreatmentRequest(BaseModel):
    crop: str = Field(min_length=1, max_length=80)
    disease: str = Field(min_length=1, max_length=160)


@app.post("/treatment")
async def get_treatment(request: TreatmentRequest):
    tavily_key = os.getenv("TAVILY_API_KEY")
    openrouter_key = os.getenv("OPENROUTER_API_KEY")

    if not tavily_key or not openrouter_key:
        raise HTTPException(
            status_code=503,
            detail=(
                "Treatment guidance is not configured. Set both "
                "TAVILY_API_KEY and OPENROUTER_API_KEY on the backend."
            ),
        )

    crop = request.crop.strip()
    disease = request.disease.strip()
    if not crop or not disease:
        raise HTTPException(
            status_code=400,
            detail="Crop and disease are required.",
        )

    search_query = (
        f"{crop} {disease} disease management guidance India "
        "(site:icar.org.in OR site:agritech.tnau.ac.in "
        "OR site:agritech.telangana.gov.in)"
    )

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(45.0)) as client:
            search_response = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": tavily_key,
                    "query": search_query,
                    "search_depth": "basic",
                    "max_results": 5,
                    "include_answer": False,
                },
            )
            search_response.raise_for_status()
            search_data = search_response.json()
            results = search_data.get("results", [])

            if not results:
                return {
                    "crop": crop,
                    "disease": disease,
                    "treatment": (
                        "No relevant agricultural sources were found for "
                        "this crop and condition. Do not apply a treatment "
                        "based only on the model prediction. Consult a local "
                        "agricultural extension officer or qualified expert."
                    ),
                    "sources": [],
                }

            source_context = "\n\n".join(
                (
                    f"Title: {item.get('title', '')}\n"
                    f"URL: {item.get('url', '')}\n"
                    f"Content: {str(item.get('content', ''))[:2500]}"
                )
                for item in results
            )

            ai_response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {openrouter_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://fieldnotes.local",
                    "X-Title": "FieldNotes Plant Health",
                },
                json={
                    "model": "openai/gpt-oss-20b",
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are an agricultural information assistant. "
                                "Use only the supplied search results for "
                                "crop-specific disease-management claims. "
                                "Return concise, readable Markdown with clear "
                                "headings and bullet points; do not use Markdown "
                                "tables. Separate evidence, practical non-chemical "
                                "management, uncertainty, and safety notes. "
                                "Never invent pesticide products, doses, mixing "
                                "instructions, application rates, or schedules. "
                                "Do not recommend applying a chemical based only "
                                "on an AI prediction. Say when evidence is missing "
                                "or contradictory, and direct the reader to local "
                                "agricultural extension guidance and the product "
                                "label where relevant."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Crop: {crop}\n"
                                f"Predicted condition: {disease}\n\n"
                                f"Retrieved agricultural sources:\n{source_context}\n\n"
                                "Summarize only the guidance supported by these "
                                "sources. Include uncertainty and safety "
                                "precautions. Do not treat the model output as a "
                                "confirmed diagnosis."
                            ),
                        },
                    ],
                    "temperature": 0.2,
                    "max_tokens": 700,
                },
            )
            ai_response.raise_for_status()
            ai_data = ai_response.json()
            choices = ai_data.get("choices", [])
            if not choices:
                raise HTTPException(
                    status_code=502,
                    detail="The AI provider returned no treatment summary.",
                )

            treatment = (
                choices[0].get("message", {}).get("content", "") or ""
            ).strip()
            if not treatment:
                raise HTTPException(
                    status_code=502,
                    detail="The AI provider returned an empty treatment summary.",
                )

            sources = [
                {
                    "title": item.get("title") or "Agricultural source",
                    "url": item.get("url") or "",
                    "snippet": str(item.get("content") or "")[:1000],
                }
                for item in results
            ]

            return {
                "crop": crop,
                "disease": disease,
                "treatment": treatment,
                "sources": sources,
            }

    except HTTPException:
        raise
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail="The agricultural search or AI service timed out. Please retry.",
        )
    except httpx.HTTPStatusError as exc:
        print(f"Treatment provider returned HTTP {exc.response.status_code}")
        raise HTTPException(
            status_code=502,
            detail="An external agricultural information service returned an error.",
        )
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
        print(f"Treatment request failed: {exc}")
        raise HTTPException(
            status_code=502,
            detail="Could not retrieve or format treatment guidance. Please retry.",
        )

