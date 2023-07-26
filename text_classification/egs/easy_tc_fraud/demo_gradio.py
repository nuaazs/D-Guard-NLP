import gradio as gr
from dguard_nlp.interface.pretrained import load_by_name,inference

embedding_model, classifier = load_by_name('bert_cos_b1_entropyloss',device='cuda:0')
def classify_text(text):
    results = inference(embedding_model, classifier, [text], print_result=False)
    label = "涉诈" if results[0][0] == 1 else "非涉诈"
    confidence = results[0][1]
    return f"分类结果：{label}，置信度：{confidence}"

inputs = gr.inputs.Textbox(lines=5, placeholder="输入文本")
output = gr.outputs.Textbox(label="分类结果")


iface = gr.Interface(fn=classify_text, inputs=inputs, outputs=output, title="反电话诈骗-文本二分类模型演示页面")
iface.launch(share=True, server_name="0.0.0.0", server_port=8987)
