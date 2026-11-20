import gradio as gr

from demo.pipeline import LABELS, EmotionDemo

model = EmotionDemo()


def run(audio_path, text):
    if not audio_path or not text or not text.strip():
        raise gr.Error("Please provide both a speech clip and its transcript.")
    return model.predict(audio_path, text)


app = gr.Interface(
    fn=run,
    inputs=[gr.Audio(type="filepath", label="Speech clip (upload or record; first 12 s are used)"),
            gr.Textbox(label="Transcript of the clip", lines=2)],
    outputs=gr.Label(num_top_classes=len(LABELS), label="Predicted emotion"),
    title="Multimodal Speech Emotion Recognition (MELD)",
    description="Frozen WavLM (audio) + frozen RoBERTa (text) features, MLP head trained on MELD. "
                "Trained on TV-dialogue clips from Friends, so expect weaker results on other domains.",
)

if __name__ == "__main__":
    app.launch()
