import gradio as gr
from inference import classify_image

interface = gr.Interface(fn=classify_image, inputs="image", outputs="text")
interface.launch(share=True)

