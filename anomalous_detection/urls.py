#urls.py
from django.urls import path, include
from django.views.generic import RedirectView

urlpatterns = [
    path('video/', include('video_app.urls')),  # /video/ 경로 연결
    path('', RedirectView.as_view(url='/video/', permanent=False)),  # 루트 URL 접속 시 /video/로 리다이렉트
]
