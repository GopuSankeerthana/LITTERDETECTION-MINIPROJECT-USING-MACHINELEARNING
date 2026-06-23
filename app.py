from __future__ import annotations

import os
import gradio as gr
from PIL import Image
from detect import DEFAULT_THRESHOLD, LitterDetector

detector = LitterDetector(threshold=DEFAULT_THRESHOLD)

def analyze_image(image: Image.Image, threshold: float):
    if image is None:
        return None, "Please upload an image to analyze."

    try:
        detector.threshold = threshold
        result = detector.detect(image)
        return result.annotated_image, result.summary
    except Exception as e:
        return None, f"Error during detection: {str(e)}"

def build_app():
    with gr.Blocks(title="Litter Detection") as demo:
        gr.Markdown(
            "# 🚯 Litter Detection System\n\n"
            "Upload an image to detect litter using a trained model."
        )

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="pil", label="Upload Image")
                threshold = gr.Slider(
                    minimum=0.1,
                    maximum=0.95,
                    value=DEFAULT_THRESHOLD,
                    step=0.05,
                    label="Confidence Threshold",
                )
                analyze_btn = gr.Button("Detect Litter", variant="primary")

            with gr.Column():
                output_image = gr.Image(type="pil", label="Detection Result")
                output_text = gr.Textbox(label="Analysis Summary", lines=10)

        analyze_btn.click(
            fn=analyze_image,
            inputs=[input_image, threshold],
            outputs=[output_image, output_text],
        )

        sample_path = os.path.join(os.path.dirname(__file__), "test_images", "image12.jpg")

        if os.path.exists(sample_path):
            gr.Examples(
                examples=[[sample_path, DEFAULT_THRESHOLD]],
                inputs=[input_image, threshold],
                outputs=[output_image, output_text],
                fn=analyze_image,
                cache_examples=False,
            )

    return demo

if __name__ == "__main__":
    print("Loading model...")
    detector.load()
    print("Model loaded successfully.")

    app = build_app()

    app.launch(
    server_name="127.0.0.1",
    server_port=7863,
    share=False,
    show_error=True
)