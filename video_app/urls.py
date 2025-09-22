from django.urls import path
from . import views

app_name = "video_app"

urlpatterns = [
    path("", views.video_page, name="video_page"),          # /video/ → 영상 업로드 페이지
    path("ask/", views.ask_question, name="ask_question"),  # /video/ask/ → CSV Q&A AJAX
]
