from transformers import BartTokenizer, BartForConditionalGeneration
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/")
@app.route("/home")
def home():
    return render_template('index.html')


@app.route("/index", methods=['POST',"GET"])
def index():
    output = request.form.to_dict()
    youtube_video = output["link"]
    #print(youtube_video)
    #youtube_video = "https://www.youtube.com/watch?v=A4OmtyaBHFE"
    video_id = youtube_video.split("=")[1]

    YouTubeTranscriptApi.get_transcript(video_id)
    transcript = YouTubeTranscriptApi.get_transcript(video_id)

    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

    result = ""
    for i in transcript:
        result += ' ' + i['text']
    print(result)
    #print(len(result))

    ARTICLE_TO_SUMMARIZE = result

    inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors="pt")

    # Generate Summary
    summary_ids = model.generate(inputs["input_ids"], num_beams=2, min_length=200, max_length=1000)
    ans = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    #print(ans)
    #print(len(ans))
    model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-one-to-many-mmt")
    tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-one-to-many-mmt", src_lang="en_XX")
    model_inputs = tokenizer(ans, return_tensors="pt")

    generated_tokens1 = model.generate(
        **model_inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id["ta_IN"]
    )
    translation1 = tokenizer.batch_decode(generated_tokens1, skip_special_tokens=True)
    generated_tokens2 = model.generate(
        **model_inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id["hi_IN"]
    )
    translation2 = tokenizer.batch_decode(generated_tokens2, skip_special_tokens=True)
    print(translation1)
    print(translation2)
    return render_template('index.html', ans = ans, trans1 = translation1[0], trans2 = translation2[0])

if __name__ == "__main__":
    app.run(debug=True, port=5001)
