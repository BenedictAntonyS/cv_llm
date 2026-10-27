import os
import json
import urllib.request
import urllib.error

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "llama-3.3-70b-versatile"
FALLBACK_MODEL = "llama-3.1-8b-instant"

SYSTEM_PROMPT_ANALYST = """You are Dr. NeuroCell AI, an expert Senior Computational Oncologist and Bio-imaging Specialist.
You specialize in Neuroblastoma cell dynamics (SH-SY5Y cell line, LIVECell dataset), deep learning time-lapse forecasting (GANs, ConvLSTM),
and quantitative cellular metrics (cell confluence %, mitotic index, proliferation velocity, occlusion sensitivity).

Your role:
1. Analyze quantitative metrics computed from microscopy frames and GAN forecasts.
2. Provide a rigorous, academic yet accessible clinical and cellular dynamics evaluation.
3. Highlight growth trajectory, cell density changes, morphological state, and biological implications.
4. Provide actionable insights for downstream experiments (e.g., drug efficacy assays, colony formation, doubling time).
Format your report with clean Markdown, bullet points, and key findings.
"""

SYSTEM_PROMPT_CHAT = """You are Dr. NeuroCell AI, an interactive AI research copilot for the Neuroblastoma Cell Growth Forecasting project.
You have deep expertise in:
- Neuroblastoma biology (pediatric sympathetic nervous system cancer, SH-SY5Y cell culture, differentiation).
- The LIVECell dataset (large-scale label-free phase-contrast microscopy benchmark).
- Deep learning architectures: Generative Adversarial Networks (GANs), spatiotemporal generator, discriminator, SSIM loss.
- Time-lapse frame prediction and occlusion sensitivity (why early vs late frames dictate forecasting accuracy).

Be helpful, concise, scientifically accurate, and encouraging. Answer researchers' questions about the model, findings, or cell biology.
"""

def call_groq_api(messages, model=DEFAULT_MODEL, temperature=0.7, max_tokens=1024):
    """Call Groq API using urllib for zero-dependency reliability."""
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
        "User-Agent": "NeuroCell-GAN-Forecaster/1.0"
    }
    
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(GROQ_API_URL, data=data, headers=headers, method="POST")
    
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            res_body = response.read().decode("utf-8")
            res_json = json.loads(res_body)
            return {
                "success": True,
                "content": res_json["choices"][0]["message"]["content"],
                "model": res_json.get("model", model)
            }
    except urllib.error.HTTPError as e:
        error_msg = e.read().decode("utf-8")
        # Try fallback model if default model isn't available
        if model != FALLBACK_MODEL:
            try:
                return call_groq_api(messages, model=FALLBACK_MODEL, temperature=temperature, max_tokens=max_tokens)
            except Exception:
                pass
        return {
            "success": False,
            "error": f"HTTP {e.code}: {error_msg}"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def generate_growth_analysis(metrics_data):
    """
    Generate automated bio-oncology report based on forecasted metrics.
    metrics_data: dict with confluence, cell_count, delta, occlusion_weights, etc.
    """
    user_prompt = f"""Please provide a Comprehensive Bio-Oncological Growth Analysis Report for this Neuroblastoma forecasting cycle:

Observed Quantitative Data:
- Initial Frame Confluence: {metrics_data.get('confluence_t0', 'N/A')}%
- Current Sequence Confluence (T4): {metrics_data.get('confluence_t4', 'N/A')}%
- GAN Forecasted Confluence (T5 / Next Step): {metrics_data.get('confluence_forecast', 'N/A')}%
- Net Confluence Change (Δ%): {metrics_data.get('confluence_delta', 'N/A')}%
- Estimated Detected Cells: {metrics_data.get('estimated_cells', 'N/A')}
- Proliferation State: {metrics_data.get('growth_state', 'Exponential Proliferation')}
- GAN Structural Similarity (SSIM vs prior): {metrics_data.get('ssim', 0.88):.3f}
- Occlusion Temporal Impact: Frame 1 (35.8%), Frame 2 (5.9%), Frame 3 (5.1%), Frame 4 (3.0%)

Please structure your report into:
1. Executive Cellular Summary
2. Quantitative Dynamics & Proliferation Assessment
3. GAN Temporal Weighting & Occlusion Interpretation (explain why Frame 1 carries ~35.8% impact)
4. Recommended Next Experimental Steps (e.g. passage timing, cytotoxic assay, drug testing)
"""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_ANALYST},
        {"role": "user", "content": user_prompt}
    ]
    
    return call_groq_api(messages, temperature=0.6, max_tokens=1200)

def chat_with_llm(history, user_message, context_summary=None):
    """
    Interactive Q&A chat with Dr. NeuroCell AI.
    history: list of {"role": "user"|"assistant", "content": str}
    """
    sys_content = SYSTEM_PROMPT_CHAT
    if context_summary:
        sys_content += f"\n\nCurrent Active Experiment Context:\n{context_summary}"
        
    messages = [{"role": "system", "content": sys_content}]
    for msg in history[-10:]: # keep last 10 messages for context
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_message})
    
    return call_groq_api(messages, temperature=0.7, max_tokens=1000)
