from django.shortcuts import render
from django.http.response import StreamingHttpResponse
from streamapp.camera import yoloDetect
# from streamapp.camera import MaskDetect
# Create your views here.


def index(request):
	return render(request, 'streamapp/home.html')


def gen(camera):
	while True:
		frame = camera.get_frame()
		yield (b'--frame\r\n'
				b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


def yolo_feed(request):
	return StreamingHttpResponse(gen(yoloDetect()),
					content_type='multipart/x-mixed-replace; boundary=frame')

# def mask_feed(request):
#     	return StreamingHttpResponse(gen(MaskDetect()),
# 					content_type='multipart/x-mixed-replace; boundary=frame')