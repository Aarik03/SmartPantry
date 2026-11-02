import spaces
import gradio as gr
import pandas as pd
import joblib

# Load the trained machine learning model
model = joblib.load("SmartPantry_Model.pkl")


# Prediction function
@spaces.GPU(duration=10)
def smartpantry_app(food_type, storage_type, days_stored,
                    quantity_remaining, temperature, humidity):

    new_food = pd.DataFrame([{
        "Food_Type": food_type,
        "Storage_Type": storage_type,
        "Days_Stored": days_stored,
        "Quantity_Remaining": quantity_remaining,
        "Temperature": temperature,
        "Humidity": humidity
    }])

    prediction = model.predict(new_food)[0]
    probabilities = model.predict_proba(new_food)[0]
    confidence = max(probabilities) * 100

    if prediction == "High":
        icon = "🔴"
        recommendation = """
**Prioritize checking this item.**

Review its storage history, packaging and product label.
Do not consume food that appears spoiled.
"""

    elif prediction == "Medium":
        icon = "🟡"
        recommendation = """
**Consider checking this item soon.**

Review its storage conditions and plan its use appropriately.
"""

    else:
        icon = "🟢"
        recommendation = """
**The model estimates a lower waste risk.**

Continue appropriate storage and regular checks.
"""

    result = f"""
# {icon} SmartPantry Prediction

## Estimated Waste Risk: {prediction}

**Model confidence:** {confidence:.2f}%

---

### Food Details

| Parameter | Input |
|---|---|
| Food Type | {food_type} |
| Storage Method | {storage_type} |
| Days Stored | {days_stored} |
| Quantity Remaining | {quantity_remaining}% |
| Temperature | {temperature}°C |
| Humidity | {humidity}% |

---

### Recommendation

{recommendation}

---

*This is a simple ML model trained on sample data to estimate food waste risk, not actual food freshness or safety.*
"""

    return result


# Custom interface styling
custom_css = """
.gradio-container {
    max-width: 1150px !important;
    margin: auto !important;
}

#main-header {
    background: linear-gradient(120deg, #123c32, #24745b);
    color: white;
    padding: 32px;
    border-radius: 18px;
    margin-bottom: 20px;
    text-align: center;
}

#main-header h1 {
    font-size: 38px;
    margin-bottom: 8px;
    color: white;
}

#main-header p {
    font-size: 15px;
    color: #e3f4eb;
}

#predict-button {
    background: #217653 !important;
    color: white !important;
    border-radius: 12px !important;
    font-weight: bold !important;
    min-height: 48px !important;
}

#result-panel {
    border: 1px solid #d5e9dc;
    border-radius: 15px;
    padding: 18px;
}
"""


# Create Gradio application
with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="green",
        secondary_hue="emerald"
    ),
    css=custom_css,
    title="SmartPantry - Food Waste Predictor"
) as demo:

    gr.HTML("""
    <div id="main-header">
        <h1>🥬 SmartPantry</h1>
        <p>Smart Food Inventory & Waste Risk Prediction System</p>
        <p>Make informed food inventory decisions with Machine Learning.</p>
    </div>
    """)

    gr.Markdown("""
    ## Check Your Food Waste Risk

    Enter the food details below to receive an estimated risk category.
    """)

    with gr.Row():

        with gr.Column(scale=1):

            gr.Markdown("### 🥗 Food Information")

            food_type = gr.Dropdown(
                choices=[
                    "Milk", "Bread", "Vegetables", "Fruits",
                    "Meat", "Cooked Food", "Leafy Vegetables"
                ],
                value="Milk",
                label="Food Type"
            )

            storage_type = gr.Dropdown(
                choices=[
                    "Refrigerator",
                    "Room Temperature",
                    "Freezer"
                ],
                value="Refrigerator",
                label="Storage Method"
            )

            days_stored = gr.Slider(
                minimum=1,
                maximum=14,
                value=5,
                step=1,
                label="Days Stored"
            )

            quantity_remaining = gr.Slider(
                minimum=5,
                maximum=100,
                value=40,
                step=5,
                label="Quantity Remaining (%)"
            )

            temperature = gr.Slider(
                minimum=-10,
                maximum=40,
                value=6,
                step=1,
                label="Temperature (°C)"
            )

            humidity = gr.Slider(
                minimum=35,
                maximum=95,
                value=70,
                step=1,
                label="Humidity (%)"
            )

            predict_button = gr.Button(
                "🔍 Predict Waste Risk",
                elem_id="predict-button"
            )

        with gr.Column(scale=1):

            gr.Markdown("### 📊 Prediction Results")

            result_output = gr.Markdown(
                value="""
## Welcome to SmartPantry!

Enter your food details and click
**Predict Waste Risk**.

Your estimated risk category and recommendation
will appear here.
""",
                elem_id="result-panel"
            )

    gr.Markdown("---")

    gr.Markdown("""
    ### About SmartPantry

    SmartPantry is an educational machine learning prototype
    that classifies estimated food waste risk into Low, Medium
    and High categories.

    **Technologies:** Python · Pandas · NumPy · Scikit-learn · Gradio

    *This application does not determine actual freshness,
    expiry or food safety.*
    """)

    predict_button.click(
        fn=smartpantry_app,
        inputs=[
            food_type,
            storage_type,
            days_stored,
            quantity_remaining,
            temperature,
            humidity
        ],
        outputs=result_output
    )


# Launch application
demo.launch()

