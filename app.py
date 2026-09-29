from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

import os
import uuid
import gc

import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input


# =========================================================
# TensorFlow Configuration
# =========================================================

# Disable GPU usage on Render.
try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass

# Limit CPU thread usage.
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)


# =========================================================
# Flask Application
# =========================================================

app = Flask(__name__)


# =========================================================
# Upload Configuration
# =========================================================

UPLOAD_FOLDER = "static/uploads"

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Maximum upload size = 10 MB
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


# =========================================================
# Model Configuration
# =========================================================

CLASSIFIER_MODEL_PATH = "models/best_b1_model.keras"

VALIDATOR_MODEL_PATH = "models/lung_ct_validator.keras"

IMG_SIZE = 224

VALIDATOR_THRESHOLD = 0.50


# =========================================================
# Classes
# =========================================================

classes = [
    "adenocarcinoma",
    "large.cell.carcinoma",
    "normal",
    "squamous.cell.carcinoma"
]


# =========================================================
# Memory Cleanup
# =========================================================

def clear_tensorflow_memory():

    gc.collect()

    try:
        tf.keras.backend.clear_session()
    except Exception:
        pass

    gc.collect()


# =========================================================
# Image Preparation
# =========================================================

def prepare_image(filepath):

    img = image.load_img(
        filepath,
        target_size=(IMG_SIZE, IMG_SIZE)
    )

    img_array = image.img_to_array(
        img
    )

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    img_array = preprocess_input(
        img_array
    )

    return img_array


# =========================================================
# Load Validator
# =========================================================

def load_validator():

    print("Loading Lung CT validator...")

    validator = tf.keras.models.load_model(
        VALIDATOR_MODEL_PATH,
        compile=False
    )

    print("Lung CT validator loaded.")

    return validator


# =========================================================
# Load Classifier
# =========================================================

def load_classifier():

    print("Loading EfficientNetB1 classifier...")

    classifier = tf.keras.models.load_model(
        CLASSIFIER_MODEL_PATH,
        compile=False
    )

    print("EfficientNetB1 classifier loaded.")

    return classifier


# =========================================================
# Home
# =========================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# =========================================================
# Prediction
# =========================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    # -----------------------------------------------------
    # Check image
    # -----------------------------------------------------

    if "image" not in request.files:

        return render_template(
            "index.html",
            error="No image was uploaded."
        )

    file = request.files["image"]

    if file.filename == "":

        return render_template(
            "index.html",
            error="Please select an image."
        )

    # -----------------------------------------------------
    # Patient Information
    # -----------------------------------------------------

    patient_name = request.form.get(
        "patient_name",
        ""
    ).strip()

    patient_age = request.form.get(
        "patient_age",
        ""
    ).strip()

    patient_place = request.form.get(
        "patient_place",
        ""
    ).strip()

    # -----------------------------------------------------
    # File Validation
    # -----------------------------------------------------

    allowed_extensions = {
        "png",
        "jpg",
        "jpeg"
    }

    original_filename = secure_filename(
        file.filename
    )

    if "." not in original_filename:

        return render_template(
            "index.html",
            error="Invalid image file."
        )

    extension = (
        original_filename
        .rsplit(".", 1)[-1]
        .lower()
    )

    if extension not in allowed_extensions:

        return render_template(
            "index.html",
            error="Only PNG, JPG and JPEG images are allowed."
        )

    # -----------------------------------------------------
    # Unique Filename
    # -----------------------------------------------------

    unique_filename = (
        uuid.uuid4().hex
        + "."
        + extension
    )

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        unique_filename
    )

    # -----------------------------------------------------
    # Save Image
    # -----------------------------------------------------

    try:

        file.save(filepath)

    except Exception as e:

        print(
            "File save error:",
            repr(e)
        )

        return render_template(
            "index.html",
            error="The uploaded image could not be saved."
        )

    # -----------------------------------------------------
    # Prepare Image
    # -----------------------------------------------------

    try:

        img_array = prepare_image(
            filepath
        )

    except Exception as e:

        print(
            "Image processing error:",
            repr(e)
        )

        try:

            if os.path.exists(filepath):
                os.remove(filepath)

        except Exception:
            pass

        return render_template(
            "index.html",
            error="The uploaded file could not be processed as an image."
        )

    # =====================================================
    # STEP 1
    # LOAD VALIDATOR ONLY
    # =====================================================

    validator = None

    try:

        validator = load_validator()

        validator_prediction = validator(
            img_array,
            training=False
        )

        validator_prediction = np.asarray(
            validator_prediction
        )

        not_lung_ct_score = float(
            validator_prediction[0][0]
        )

        print(
            "Validator score:",
            not_lung_ct_score
        )

    except Exception as e:

        print(
            "Validator error:",
            repr(e)
        )

        validator = None

        clear_tensorflow_memory()

        return render_template(
            "index.html",
            error="The image validation process failed. Please try again."
        )

    # =====================================================
    # IMPORTANT:
    # DELETE VALIDATOR BEFORE LOADING B1
    # =====================================================

    del validator
    del validator_prediction

    clear_tensorflow_memory()

    # =====================================================
    # STEP 1 RESULT
    # NON-LUNG CT IMAGE
    # =====================================================

    if not_lung_ct_score >= VALIDATOR_THRESHOLD:

        image_path = (
            "/static/uploads/"
            + unique_filename
        )

        del img_array

        clear_tensorflow_memory()

        return render_template(
            "index.html",
            unsupported_image=True,
            validator_score=round(
                not_lung_ct_score * 100,
                2
            ),
            image_path=image_path,
            patient_name=patient_name,
            patient_age=patient_age,
            patient_place=patient_place
        )

    # =====================================================
    # STEP 2
    # LOAD B1 CLASSIFIER ONLY
    # =====================================================

    classifier = None
    predictions = None

    try:

        classifier = load_classifier()

        predictions = classifier(
            img_array,
            training=False
        )

        predictions = np.asarray(
            predictions
        )[0]

        print(
            "Classification probabilities:",
            predictions
        )

    except Exception as e:

        print(
            "Classification error:",
            repr(e)
        )

        classifier = None
        predictions = None

        del img_array

        clear_tensorflow_memory()

        return render_template(
            "index.html",
            error="The classification process failed. Please try again."
        )

    # -----------------------------------------------------
    # Get Prediction
    # -----------------------------------------------------

    index = int(
        np.argmax(
            predictions
        )
    )

    confidence = float(
        predictions[index] * 100
    )

    prediction_name = classes[index]

    image_path = (
        "/static/uploads/"
        + unique_filename
    )

    # =====================================================
    # IMPORTANT:
    # DELETE B1 MODEL AFTER PREDICTION
    # =====================================================

    del classifier
    del predictions
    del img_array

    clear_tensorflow_memory()

    # =====================================================
    # Return Result
    # =====================================================

    return render_template(
        "index.html",
        prediction=prediction_name,
        confidence=round(
            confidence,
            2
        ),
        image_path=image_path,
        patient_name=patient_name,
        patient_age=patient_age,
        patient_place=patient_place,
        unsupported_image=False
    )


# =========================================================
# File Too Large
# =========================================================

@app.errorhandler(413)
def request_entity_too_large(error):

    return render_template(
        "index.html",
        error="Image is too large. Please upload an image smaller than 10 MB."
    ), 413


# =========================================================
# Health Check
# =========================================================

@app.route("/health")
def health():

    return {
        "status": "ok",
        "service": "LungScopeAI"
    }, 200


# =========================================================
# Start Application
# =========================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(
            os.environ.get(
                "PORT",
                5000
            )
        ),
        debug=False
    )