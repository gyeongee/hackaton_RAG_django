import os
import uuid
import ast
import numpy as np
import pandas as pd
from numpy.linalg import norm
from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings
import openai
from .video_pipeline import VideoMAE_ONNX_Pipeline

# -----------------------------
# OpenAI API 키
# -----------------------------
with open(os.path.join(settings.BASE_DIR, "video_app/key/openai_api_key.txt"), "r", encoding="utf-8") as f:
    api_key = f.read().strip()

client = openai.OpenAI(api_key=api_key)

# -----------------------------
# CSV embedding 불러오기
# -----------------------------
CSV_DIR = os.path.join(settings.BASE_DIR, "video_app/csv_data")
EMBEDDING_FILE = os.path.join(CSV_DIR, "embeddings_df.csv")

embeddings_df = pd.read_csv(EMBEDDING_FILE)
embeddings_df['embedding'] = embeddings_df['embedding'].apply(lambda x: np.array(ast.literal_eval(x)))

# -----------------------------
# 코사인 유사도
# -----------------------------
def cos_sim(A, B):
    return np.dot(A, B) / (norm(A) * norm(B))

# -----------------------------
# 질문 처리
# -----------------------------
def process_question_with_embeddings(question):
    question_emb = np.array(client.embeddings.create(
        model="text-embedding-ada-002",
        input=question
    ).data[0].embedding)

    embeddings_df['similarity'] = embeddings_df['embedding'].apply(lambda x: cos_sim(question_emb, x))
    top3 = embeddings_df.sort_values("similarity", ascending=False).head(3)

    system_message = "주어진 문서를 참고하여 질문에 답변하세요.\n\n"
    for i, row in top3.iterrows():
        system_message += f"문서{i+1} ({row['csv_file']}): {row['text']}\n\n"

    user_message = f"질문: {question}"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        temperature=0.4,
        max_tokens=500
    )

    answer = response.choices[0].message.content
    return {
        "answer": answer,
        "top3_csv_files": top3['csv_file'].tolist(),
        "top3_texts": top3['text'].tolist(),
        "top3_similarities": top3['similarity'].tolist()
    }

# -----------------------------
# AJAX 뷰
# -----------------------------
def ask_question(request):
    question = request.GET.get("q")
    if not question:
        return JsonResponse({"error": "No question provided."}, status=400)
    result = process_question_with_embeddings(question)
    return JsonResponse(result)

# -----------------------------
# video 페이지 뷰
# -----------------------------
onnx_path = os.path.join(settings.BASE_DIR, "anomalous_behavior_video_cls_model", "video_cls_model.onnx")
id2label = {
    0: "assult", 1: "datefight", 2: "robbery", 3: "burglary", 4: "trespass",
    5: "wander", 6: "vandalism", 7: "fight", 8: "dump", 9: "swoon", 10: "kidnap"
}
pipeline = VideoMAE_ONNX_Pipeline(onnx_path=onnx_path, id2label=id2label)

def video_page(request):
    result = None
    video_file_url = None

    if request.method == "POST" and request.FILES.get("video_file"):
        uploaded_file = request.FILES["video_file"]
        filename = f"{uuid.uuid4().hex}_{uploaded_file.name}"
        file_path = os.path.join(settings.MEDIA_ROOT, filename)
        os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
        with open(file_path, "wb+") as f:
            for chunk in uploaded_file.chunks():
                f.write(chunk)
        result = pipeline(file_path)
        video_file_url = settings.MEDIA_URL + filename

    return render(request, "video_app/video.html", {"result": result, "video_file_url": video_file_url})






