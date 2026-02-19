import gradio as gr
import boto3
import json
import os
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")
os.environ['AWS_ACCESS_KEY_ID'] = os.getenv('AWS_ACCESS_KEY')
os.environ['AWS_SECRET_ACCESS_KEY'] = os.getenv('AWS_SECRET_ACCESS_KEY')

# ================================================================
# LOAD MODELS
# ================================================================
print("Loading models...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
qdrant = QdrantClient(url="http://localhost:6333")
bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name="ap-southeast-2"
)
COLLECTION = "metricon_rag"
print("✅ Models loaded!")

# ================================================================
# RAG FUNCTION
# ================================================================
def rag_answer(question, history, top_k, show_sources):
    try:
        query_vec = embedder.encode(question).tolist()
        results = qdrant.query_points(
            collection_name=COLLECTION,
            query=query_vec,
            limit=int(top_k)
        )

        if not results.points:
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": "⚠️ No relevant information found. Please contact Metricon on 1300 786 773."})
            return history, ""

        context_chunks = []
        sources = []
        for p in results.points:
            context_chunks.append(
                f"[{p.payload['category']}] {p.payload['title']}\n{p.payload['text']}"
            )
            sources.append(
                f"📄 **{p.payload['title']}**\n"
                f"📂 {p.payload['source']} | 🏷️ {p.payload['category']} | ⭐ {p.payload['importance']}/10"
            )

        context = "\n\n".join(context_chunks)

        prompt = f"""
You are a friendly and professional AI assistant for Metricon Homes Australia.
Answer the customer's question using ONLY the provided context.
Be warm, helpful and concise. Use dot points where appropriate.
If the answer is not in the context, say "I don't have that information. Please contact Metricon on 1300 786 773 or visit metricon.com.au"

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
        response = bedrock.invoke_model(
            modelId="amazon.nova-micro-v1:0",
            body=json.dumps({
                "messages": [{"role": "user", "content": [{"text": prompt}]}],
                "inferenceConfig": {"maxTokens": 400, "temperature": 0.2}
            })
        )
        output = json.loads(response["body"].read())
        answer = output["output"]["message"]["content"][0]["text"]

        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})

        sources_text = "\n\n".join(sources) if show_sources else ""
        return history, sources_text

    except Exception as e:
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": f"❌ Error: {str(e)}"})
        return history, ""

# ================================================================
# GRADIO UI
# ================================================================
with gr.Blocks() as demo:

    gr.HTML("""
    <div style="background: linear-gradient(135deg, #1a3c5e, #2d6a9f);
    border-radius: 20px; padding: 30px; text-align: center; margin-bottom: 20px;
    animation: glow 3s infinite;">
        <div style="font-size:3em;">🏡</div>
        <h1 style="color:white; margin:10px 0; font-size:2.5em; font-weight:700;">
            Metricon AI Assistant
        </h1>
        <p style="color:rgba(255,255,255,0.8); font-size:16px; margin:5px 0;">
            Your intelligent guide to building your dream home in Australia
        </p>
        <div style="margin-top:15px;">
            <span style="background:rgba(255,255,255,0.15); border:1px solid rgba(255,255,255,0.3);
            padding:5px 12px; border-radius:20px; font-size:13px; color:white; margin:3px;">
            ⚡ AWS Bedrock Nova</span>
            <span style="background:rgba(255,255,255,0.15); border:1px solid rgba(255,255,255,0.3);
            padding:5px 12px; border-radius:20px; font-size:13px; color:white; margin:3px;">
            🔍 Qdrant Vector Search</span>
            <span style="background:rgba(255,255,255,0.15); border:1px solid rgba(255,255,255,0.3);
            padding:5px 12px; border-radius:20px; font-size:13px; color:white; margin:3px;">
            🧠 Agentic RAG</span>
        </div>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="💬 Chat with Metricon AI",
                height=450
            )

            with gr.Row():
                msg = gr.Textbox(
                    placeholder="💬 Ask anything about building with Metricon...",
                    label="",
                    scale=4,
                    container=False
                )
                send_btn = gr.Button("Send 🚀", scale=1, variant="primary")

            gr.HTML("<p style='font-weight:600; margin:15px 0 8px 0;'>💡 Quick Questions:</p>")
            with gr.Row():
                btn1 = gr.Button("🏗️ Building process?", size="sm")
                btn2 = gr.Button("💰 Costs involved?", size="sm")
                btn3 = gr.Button("🏠 Why Metricon?", size="sm")
            with gr.Row():
                btn4 = gr.Button("🔑 First home buyer?", size="sm")
                btn5 = gr.Button("📅 How long to build?", size="sm")
                btn6 = gr.Button("💳 Finance options?", size="sm")

        with gr.Column(scale=1):
            gr.HTML("<p style='font-weight:600;'>⚙️ Settings</p>")
            top_k = gr.Slider(minimum=1, maximum=7, value=3, step=1, label="Top-K Results")
            show_sources = gr.Checkbox(label="Show Sources", value=True)
            gr.HTML("<hr>")
            gr.HTML("""
            <p style='font-weight:600;'>📊 System Status</p>
            <p style='color:green;'>✅ AWS Bedrock Connected</p>
            <p style='color:green;'>✅ Qdrant Running</p>
            <p style='color:blue;'>📄 Metricon PDFs Indexed</p>
            """)
            gr.HTML("<hr>")
            clear_btn = gr.Button("🗑️ Clear Chat", variant="stop")
            gr.HTML("<hr>")
            sources_box = gr.Markdown(value="Sources will appear here...")

    # ================================================================
    # EVENT HANDLERS
    # ================================================================
    def respond(message, history, top_k, show_sources):
        return rag_answer(message, history, top_k, show_sources)

    def clear():
        return [], ""

    def set_question(q, history, top_k, show_sources):
        return rag_answer(q, history, top_k, show_sources)

    send_btn.click(
        respond,
        inputs=[msg, chatbot, top_k, show_sources],
        outputs=[chatbot, sources_box]
    ).then(lambda: "", outputs=msg)

    msg.submit(
        respond,
        inputs=[msg, chatbot, top_k, show_sources],
        outputs=[chatbot, sources_box]
    ).then(lambda: "", outputs=msg)

    clear_btn.click(clear, outputs=[chatbot, sources_box])

    btn1.click(lambda h, k, s: set_question("What is the building process at Metricon?", h, k, s), inputs=[chatbot, top_k, show_sources], outputs=[chatbot, sources_box])
    btn2.click(lambda h, k, s: set_question("What are the costs involved in building?", h, k, s), inputs=[chatbot, top_k, show_sources], outputs=[chatbot, sources_box])
    btn3.click(lambda h, k, s: set_question("Why should I choose Metricon?", h, k, s), inputs=[chatbot, top_k, show_sources], outputs=[chatbot, sources_box])
    btn4.click(lambda h, k, s: set_question("What options are available for first home buyers?", h, k, s), inputs=[chatbot, top_k, show_sources], outputs=[chatbot, sources_box])
    btn5.click(lambda h, k, s: set_question("How long does it take to build a Metricon home?", h, k, s), inputs=[chatbot, top_k, show_sources], outputs=[chatbot, sources_box])
    btn6.click(lambda h, k, s: set_question("What finance options are available?", h, k, s), inputs=[chatbot, top_k, show_sources], outputs=[chatbot, sources_box])

# ================================================================
# LAUNCH
# ================================================================
if __name__ == "__main__":
    demo.launch(server_port=7860, share=False)