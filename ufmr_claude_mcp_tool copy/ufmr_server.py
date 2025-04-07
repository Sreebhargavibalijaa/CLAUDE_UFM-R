from flask import Flask, request, jsonify
from PIL import Image
import torch
import torchvision.transforms as T
import base64
import io
import os

from agent import FederatedUFMSystem
from utils import plot_patch_overlay_on_image  # should return a heatmap image path or PIL image

app = Flask(__name__)

@app.route('/functions/diagnose', methods=['POST'])
def diagnose():
    print("✅ /functions/diagnose endpoint hit")
    try:
        patient_id = request.form['patient_id'].strip()
        age = int(request.form['age'])
        bp = int(request.form['bp'])
        hr = int(request.form['hr'])
        report = request.form['report']
        xray_file = request.files['xray_image']
        print(f"Received patient_id={patient_id}, age={age}, bp={bp}, hr={hr}, report={report}")

        # Preprocess image
        img = Image.open(xray_file.stream).convert("RGB")
        transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])
        img_tensor = transform(img).unsqueeze(0)
        tab_tensor = torch.tensor([[age, bp, hr]], dtype=torch.float32)

        # Run federated model
        fed = FederatedUFMSystem(num_agents=3)
        results = fed.run_all(tab_tensor, report, img_tensor, patient_id)

        decisions, confidences = zip(*[(r[1], r[2]) for r in results])
        majority_label = max(set(decisions), key=decisions.count)
        majority_diagnosis = "Pneumonia" if majority_label.lower() != "normal" else "Normal"

        # Fetch memory from agent-1
        agent_1 = fed.agents[0]
        memory = agent_1.memory[patient_id]
        print(f"📦 Memory for patient_id={patient_id} fetched")

        # Generate and encode heatmap overlay image
        overlay_path = plot_patch_overlay_on_image(memory["img_contribs"], 10, 10, img_tensor)
        with open(overlay_path, "rb") as f:
            encoded_overlay = base64.b64encode(f.read()).decode("utf-8")

        # Token attention
        tokenizer = agent_1.tokenizer
        tokens = tokenizer.convert_ids_to_tokens(
            tokenizer(report, return_tensors="pt", padding="max_length", truncation=True, max_length=64)["input_ids"][0]
        )
        scores = memory["attn"][0].detach().cpu().numpy().flatten()
        top_tokens = sorted([
            (tokens[i], float(scores[i])) for i in range(min(len(tokens), len(scores))) if tokens[i] != "<pad>"
        ], key=lambda x: x[1], reverse=True)[:10]

        # Tabular feature contributions
        tab_contribs = [
            float(val.flatten()[0].item() if val.numel() > 1 else val.item())
            for val in memory["tab_contribs"]
        ]

        print("✅ Inference complete, sending response")
        return jsonify({
            "majority_diagnosis": majority_diagnosis,
            "agent_predictions": [
                {"agent": r[0], "label": r[1], "probability": round(r[2], 4)} for r in results
            ],
            "agent_1_diagnosis": {
                "label": "Pneumonia" if confidences[0] > 0.5 else "Normal",
                "confidence": round(confidences[0], 4),
                "top_text_tokens": top_tokens,
                "tabular_contributions": {
                    "Age": round(tab_contribs[0], 4),
                    "BP": round(tab_contribs[1], 4),
                    "HR": round(tab_contribs[2], 4)
                },
                "patch_overlay_base64": encoded_overlay
            }
        })

    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("🚀 Starting Flask server on port 5050")
    app.run(port=5050)
