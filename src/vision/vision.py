"""
Enterprise Vision Module
MobileNetV2 Feature Extractor + Binary or Multi-Class Classifier Support
Fully Backward Compatible
"""

import os
import re
import numpy as np
from typing import Dict, List
from PIL import Image
import io
import base64
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.config import IMAGE_CATEGORIES, CONFIDENCE_THRESHOLD

try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    from tensorflow.keras.models import Model
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ TensorFlow not installed.")


class VisionAnalyzer:

    def __init__(self):
        self.model_dir = os.path.join(os.path.dirname(__file__), "model")
        self.models = {}
        self.model_product_map = {}
        self.categories = IMAGE_CATEGORIES
        self.confidence_threshold = CONFIDENCE_THRESHOLD
        self.image_size = (224, 224)

        self.feature_extractor = None

        if TENSORFLOW_AVAILABLE:
            self._initialize_mobilenet()

        self._load_models()

    # ============================================================
    # MobileNetV2 Feature Extractor
    # ============================================================
    def _initialize_mobilenet(self):
        print("🔄 Initializing MobileNetV2...")

        base_model = MobileNetV2(
            weights="imagenet",
            include_top=False,
            input_shape=(224, 224, 3)
        )

        self.feature_extractor = Model(
            inputs=base_model.input,
            outputs=base_model.output
        )

        print("✅ MobileNetV2 Ready\n")

    # ============================================================
    # Normalize Product Name
    # ============================================================
    def _normalize_product_name(self, product_name: str) -> str:
        if not product_name:
            return ""
        name = product_name.lower().strip()
        name = re.sub(r"[^\w\s]", "", name)
        name = re.sub(r"\s+", " ", name)
        return name

    # ============================================================
    # Load Classifiers
    # ============================================================
    def _load_models(self):

        if not TENSORFLOW_AVAILABLE:
            return

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir, exist_ok=True)
            return

        files = [f for f in os.listdir(self.model_dir) if f.endswith(".h5")]

        for filename in files:
            if "_good_bad_classifier.h5" not in filename:
                continue

            try:
                path = os.path.join(self.model_dir, filename)
                product_name = filename.replace("_good_bad_classifier.h5", "")
                norm = self._normalize_product_name(product_name)

                model = load_model(path)

                self.models[product_name] = model
                self.model_product_map[norm] = product_name

                print(f"✅ Loaded classifier: {filename}")

            except Exception as e:
                print(f"❌ Failed loading {filename}: {e}")

    # ============================================================
    # Model Finder
    # ============================================================
    def _find_model_for_product(self, product_id=None, product_name=None):

        if not self.models:
            return None

        if product_name:
            norm = self._normalize_product_name(product_name)

            if norm in self.model_product_map:
                key = self.model_product_map[norm]
                return self.models[key]

        return list(self.models.values())[0]

    # ============================================================
    # Feature Extraction
    # ============================================================
    def _extract_features(self, img_array):
        processed = preprocess_input(img_array)
        features = self.feature_extractor.predict(processed, verbose=0)
        features = np.mean(features, axis=(1, 2))
        return features

    # ============================================================
    # Main Analysis
    # ============================================================
    def analyze_image(
        self,
        image_input,
        product_id=None,
        product_name=None,
        order_id=None,
        model_name=None
    ) -> Dict:

        if not TENSORFLOW_AVAILABLE:
            return {"error": "TensorFlow not available"}

        img = self._load_image(image_input)
        if img is None:
            return {"error": "Invalid image"}

        if model_name and model_name in self.models:
            model = self.models[model_name]
        else:
            model = self._find_model_for_product(product_id, product_name)

        if not model:
            return {"error": "No classifier model available"}

        img_array = self._preprocess_image(img)
        features = self._extract_features(img_array)
        predictions = model.predict(features, verbose=0)

        result = self._process_predictions(predictions)

        result.update({
            "product_id": product_id,
            "product_name": product_name,
            "order_id": order_id,
            "model_used": "MobileNetV2 + Classifier"
        })

        return result

    # ============================================================
    # SAFE Prediction Processing (Binary + Multi-Class Support)
    # ============================================================
    def _process_predictions(self, predictions):

        predictions = np.array(predictions)

        # -------- BINARY CLASSIFIER --------
        if predictions.shape[1] == 1:
            prob = float(predictions[0][0])

            # Convert sigmoid output to two-class probability
            probs = [1 - prob, prob]

            idx = int(prob >= 0.5)
            confidence = max(probs)

            categories = self.categories[:2] if len(self.categories) >= 2 else ["negative", "positive"]

            all_preds = {
                categories[0]: probs[0],
                categories[1]: probs[1]
            }

            category = categories[idx]

        # -------- MULTI-CLASS --------
        else:
            idx = int(np.argmax(predictions[0]))
            confidence = float(predictions[0][idx])

            usable_categories = self.categories[:predictions.shape[1]]

            all_preds = {
                usable_categories[i]: float(predictions[0][i])
                for i in range(predictions.shape[1])
            }

            category = usable_categories[idx]

        return {
            "category": category,
            "confidence": confidence,
            "meets_threshold": confidence >= self.confidence_threshold,
            "all_predictions": all_preds,
            "recommendation": self._get_recommendation(category, confidence)
        }

    # ============================================================
    # Recommendation
    # ============================================================
    def _get_recommendation(self, category, confidence):

        if confidence < self.confidence_threshold:
            return f"Low confidence ({confidence:.1%}). Please verify image."

        recommendations = {
            "damaged_product": "Product appears damaged. Eligible for return.",
            "wrong_product": "Wrong product delivered. Eligible for replacement.",
            "good": "Product looks fine.",
            "bad": "Product appears defective."
        }

        return recommendations.get(category, "Issue detected.")

    # ============================================================
    # Image Loader
    # ============================================================
    def _load_image(self, image_input):
        try:
            if isinstance(image_input, str):
                if os.path.exists(image_input):
                    return Image.open(image_input).convert("RGB")
                elif image_input.startswith("data:image"):
                    image_bytes = base64.b64decode(image_input.split(",")[1])
                    return Image.open(io.BytesIO(image_bytes)).convert("RGB")
                else:
                    image_bytes = base64.b64decode(image_input)
                    return Image.open(io.BytesIO(image_bytes)).convert("RGB")

            elif isinstance(image_input, bytes):
                return Image.open(io.BytesIO(image_input)).convert("RGB")

            elif isinstance(image_input, Image.Image):
                return image_input.convert("RGB")

        except:
            return None

        return None

    # ============================================================
    # Preprocess
    # ============================================================
    def _preprocess_image(self, img):
        img = img.resize(self.image_size)
        arr = np.array(img).astype("float32")
        arr = np.expand_dims(arr, axis=0)
        return arr

    # ============================================================
    # Utility
    # ============================================================
    def get_available_models(self) -> List[str]:
        return list(self.models.keys())

    def get_model_info(self) -> Dict:
        return {
            "total_models": len(self.models),
            "models": list(self.models.keys()),
            "model_directory": self.model_dir,
            "feature_extractor": "MobileNetV2",
            "supports_binary": True,
            "supports_multiclass": True
        }

    def get_categories(self) -> List[str]:
        return self.categories


vision_analyzer = VisionAnalyzer()