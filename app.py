import gradio as gr
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

SYSTEM_PROMPT = """You are an expert technical analyst.

You will analyze TradingView-style candlestick chart screenshots.

Respond with:

- Overall trend
- Main pattern(s) (wedges, triangles, ranges, etc.)
- Key support/resistance zones (rough)
- Momentum and volatility notes
- Bias (LONG / SHORT / WAIT)
- One short explanation in plain English

Do NOT predict or guarantee future prices. Be probabilistic.
"""

def build_prompt(timeframe):
    tf = timeframe.strip() if timeframe else "unknown timeframe"
    return f"{SYSTEM_PROMPT}\n\nTimeframe: {tf}\n\nAnalyze the chart image."

def analyze(image, timeframe):
    if image is None:
        return "Upload a chart screenshot."

    prompt = build_prompt(timeframe)

    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)

    generated = model.generate(
        **inputs,
        max_new_tokens=400,
        do_sample=True,
        temperature=0.4,
        top_p=0.9
    )

    output = processor.batch_decode(generated, skip_special_tokens=True)[0]
    return output


demo = gr.Interface(
    fn=analyze,
    inputs=[
        gr.Image(label="Upload TradingView Chart Screenshot", type="pil"),
        gr.Textbox(label="Timeframe (e.g. 5m, 1h, 4h, D)", placeholder="Optional")
    ],
    outputs=gr.Markdown(label="Analysis"),
    title="AI Trading Pattern Explainer",
    description="Upload a chart screenshot and get an AI analysis using a free open-source vision LLM."
)

demo.launch()

