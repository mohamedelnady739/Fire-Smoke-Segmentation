import streamlit as st
from PIL import Image
from ultralytics import YOLO
import os
import tempfile


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Fire & Smoke Dashboard",
    page_icon="🔥",
    layout="wide"
)


# =========================================================
# CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #080b10 0%,
        #10151c 50%,
        #0b0f14 100%
    );
    color: #f5f5f5;
}

.dashboard-title {
    font-size: 42px;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: 5px;
}

.subtitle {
    color: #9da7b3;
    font-size: 16px;
    margin-bottom: 30px;
}

h2 {
    color: #ffffff !important;
    margin-top: 30px;
}

h3 {
    color: #e6edf3 !important;
}

.card {
    padding: 24px;
    border-radius: 18px;
    background: linear-gradient(
        145deg,
        #151b23,
        #10151c
    );
    border: 1px solid #29313b;
    text-align: center;
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.25);
    transition: 0.25s ease;
}

.card:hover {
    border-color: #ff6b35;
    transform: translateY(-3px);
}

.label {
    color: #9da7b3;
    font-size: 15px;
    font-weight: 500;
}

.value {
    font-size: 32px;
    font-weight: 800;
    color: #ffffff;
    margin-top: 6px;
}

.fire-card {
    padding: 25px;
    border-radius: 18px;
    background: linear-gradient(
        145deg,
        #291313,
        #171014
    );
    border: 1px solid #8f2d1f;
    text-align: center;
    box-shadow: 0 8px 25px rgba(255, 69, 0, 0.12);
}

.fire-title {
    color: #ff7043;
    font-size: 22px;
    font-weight: 800;
}

.fire-value {
    color: #ffb199;
    font-size: 28px;
    font-weight: 700;
    margin-top: 8px;
}

.smoke-card {
    padding: 25px;
    border-radius: 18px;
    background: linear-gradient(
        145deg,
        #121c28,
        #10151c
    );
    border: 1px solid #31536d;
    text-align: center;
    box-shadow: 0 8px 25px rgba(70, 150, 200, 0.10);
}

.smoke-title {
    color: #72b7e6;
    font-size: 22px;
    font-weight: 800;
}

.smoke-value {
    color: #b9ddf5;
    font-size: 28px;
    font-weight: 700;
    margin-top: 8px;
}

img {
    border-radius: 12px;
}

[data-testid="stFileUploader"] {
    background: #11171f;
    border: 1px dashed #394553;
    border-radius: 15px;
    padding: 10px;
}

hr {
    border-color: #29313b;
}

.stButton > button {
    width: 100%;
    border-radius: 12px;
    border: 1px solid #ff6b35;
    background: #ff6b35;
    color: white;
    font-weight: 700;
    padding: 10px;
}

.stButton > button:hover {
    background: #e85b2a;
    border-color: #e85b2a;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL
# =========================================================

MODEL_PATH = "models/best.pt"


@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="dashboard-title">🔥 Fire & Smoke Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Fire and smoke segmentation using YOLO</div>',
    unsafe_allow_html=True
)


# =========================================================
# IMAGE SEGMENTATION
# =========================================================

st.markdown("## 🖼️ Image Segmentation")

if not os.path.exists(MODEL_PATH):

    st.error(
        "Model not found. Please place best.pt inside models/."
    )

else:

    model = load_model()

    uploaded_image = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"],
        key="image_upload"
    )

    if uploaded_image is not None:

        image = Image.open(uploaded_image).convert("RGB")

        st.markdown("### Selected Image")

        st.image(
            image,
            use_container_width=True
        )

        if st.button("🔥 Run Segmentation"):

            results = model.predict(
                source=image,
                conf=0.25,
                verbose=False
            )

            result = results[0]

            # =============================================
            # COUNT CLASSES
            # =============================================

            fire_count = 0
            smoke_count = 0

            if result.boxes is not None:

                for cls in result.boxes.cls:

                    class_id = int(cls)

                    if class_id == 0:
                        fire_count += 1

                    elif class_id == 1:
                        smoke_count += 1

            # =============================================
            # SEGMENTED IMAGE
            # =============================================

            segmented_image = result.plot()

            # =============================================
            # RESULTS
            # =============================================

            st.markdown("## 🔍 Segmentation Result")

            col1, col2 = st.columns(2)

            with col1:

                st.markdown("### Original")

                st.image(
                    image,
                    use_container_width=True
                )

            with col2:

                st.markdown("### Segmented")

                st.image(
                    segmented_image,
                    channels="BGR",
                    use_container_width=True
                )

            # =============================================
            # DETECTION COUNTS
            # =============================================

            st.markdown("## 🔥 Detection Results")

            c1, c2 = st.columns(2)

            with c1:

                st.markdown(
                    f"""
                    <div class="fire-card">
                        <div class="fire-title">🔥 Fire</div>
                        <div class="fire-value">
                            {fire_count} detected
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with c2:

                st.markdown(
                    f"""
                    <div class="smoke-card">
                        <div class="smoke-title">💨 Smoke</div>
                        <div class="smoke-value">
                            {smoke_count} detected
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    else:

        st.info("Upload an image to start segmentation.")


# =========================================================
# DIVIDER
# =========================================================

st.divider()


# =========================================================
# MODEL METRICS
# =========================================================

st.markdown("## 🎯 Segmentation Performance")

c1, c2, c3, c4 = st.columns(4)

metrics = [
    ("Precision", "71.17%"),
    ("Recall", "70.89%"),
    ("mAP50", "69.86%"),
    ("mAP50-95", "47.43%")
]

for col, (name, value) in zip(
    [c1, c2, c3, c4],
    metrics
):

    with col:

        st.markdown(
            f"""
            <div class="card">
                <div class="label">{name}</div>
                <div class="value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True
        )


# =========================================================
# FIRE / SMOKE CLASSES
# =========================================================

st.markdown("## 🔥 Detection Classes")

c1, c2 = st.columns(2)

with c1:

    st.markdown(
        """
        <div class="fire-card">
            <div class="fire-title">🔥 Fire</div>
            <div class="fire-value">
                Fire Segmentation
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with c2:

    st.markdown(
        """
        <div class="smoke-card">
            <div class="smoke-title">💨 Smoke</div>
            <div class="smoke-value">
                Smoke Segmentation
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# VALIDATION PLOTS
# =========================================================

st.markdown("## 📈 Validation Analysis")

val_folder = "/content/runs/segment/val"

plots = [
    ("PR Curve", "PR_curve.png"),
    ("Precision Curve", "P_curve.png"),
    ("Recall Curve", "R_curve.png"),
    ("F1 Curve", "F1_curve.png"),
    ("Confusion Matrix", "confusion_matrix.png")
]

for title, filename in plots:

    path = os.path.join(
        val_folder,
        filename
    )

    if os.path.exists(path):

        st.markdown(f"### {title}")

        st.image(
            path,
            use_container_width=True
        )


# =========================================================
# VALIDATION IMAGES
# =========================================================

st.markdown("## 🖼️ Validation Samples")

pred_folder = "/content/runs/segment/predict"

if os.path.exists(pred_folder):

    images = [
        x for x in os.listdir(pred_folder)
        if x.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]

    cols = st.columns(3)

    for i, image_name in enumerate(images[:9]):

        with cols[i % 3]:

            st.image(
                os.path.join(
                    pred_folder,
                    image_name
                ),
                use_container_width=True
            )

else:

    st.info(
        "Validation prediction images are not available yet."
    )

