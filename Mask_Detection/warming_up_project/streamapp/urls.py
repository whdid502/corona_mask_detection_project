from django.urls import path, include
from streamapp import views


urlpatterns = [
    path('', views.index, name='index'),
    path('yolo_feed', views.yolo_feed, name='yolo_feed'),
    # path('mask_feed', views.mask_feed, name='mask_feed'),
    ]
