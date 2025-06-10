from flask import Flask, render_template, Response, request, jsonify
from AirDrawingCanvas import AirDrawingCanvas
from LucasKanadeMotionDetection import OpticalFlowVisualizer
from Filters import FilterCamera
from face_detection import FaceDetection
from volume_control import VolumeControl
from mouse_control import MouseControl
import os

app = Flask(__name__)
drawing_app = AirDrawingCanvas()
flow_app = OpticalFlowVisualizer()
filter_app = FilterCamera()
face_app = FaceDetection()
volume_app = VolumeControl()
mouse_app = MouseControl()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed_drawing')
def video_feed_drawing():
    return Response(drawing_app.generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_flow')
def video_feed_flow():
    return Response(flow_app.generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_filter')
def video_feed_filter():
    return Response(filter_app.generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_face')
def video_feed_face():
    return Response(face_app.generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_volume')
def video_feed_volume():
    return Response(volume_app.generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_mouse')
def video_feed_mouse():
    return Response(mouse_app.generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_filter_mode', methods=['POST'])
def set_filter_mode():
    filter_mode = request.json.get("mode", "original")
    filter_app.set_filter_mode(filter_mode)
    return jsonify({"success": True})

@app.route('/save_canvas', methods=['POST'])
def save_canvas():
    filename = drawing_app.save_canvas()
    return jsonify({"success": True, "filename": filename}) if filename else jsonify({"success": False})

@app.route('/stop_camera_drawing', methods=['POST'])
def stop_camera_drawing():
    drawing_app.stop_camera()
    return jsonify({"success": True})

@app.route('/stop_camera_flow', methods=['POST'])
def stop_camera_flow():
    flow_app.stop_camera()
    return jsonify({"success": True})

@app.route('/stop_camera_filter', methods=['POST'])
def stop_camera_filter():
    filter_app.stop_camera()
    return jsonify({"success": True})

@app.route('/stop_camera_face', methods=['POST'])
def stop_camera_face():
    face_app.stop_camera()
    return jsonify({"success": True})

@app.route('/stop_camera_volume', methods=['POST'])
def stop_camera_volume():
    volume_app.stop_camera()
    return jsonify({"success": True})

@app.route('/stop_camera_mouse', methods=['POST'])
def stop_camera_mouse():
    mouse_app.stop_camera()
    return jsonify({"success": True})

@app.route('/stop_camera_<feature>', methods=['POST'])
def stop_camera(feature):
    if feature == 'drawing':
        drawing_app.stop_camera()
    elif feature == 'flow':
        flow_app.stop_camera()
    elif feature == 'filter':
        filter_app.stop_camera()
    elif feature == 'face':
        face_app.stop_camera()
    elif feature == 'volume':
        volume_app.stop_camera()
    elif feature == 'mouse':
        mouse_app.stop_camera()

    return jsonify({"success": True})


@app.route('/start_camera_<feature>', methods=['POST'])
def start_camera(feature):
    if feature == 'drawing':
        drawing_app.stop_camera()  # Stop and reinitialize drawing camera
        drawing_app.start_camera()
    elif feature == 'flow':
        flow_app.stop_camera()
        flow_app.start_camera()
    elif feature == 'filter':
        filter_app.stop_camera()
        filter_app.start_camera()
    elif feature == 'face':
        face_app.stop_camera()
        face_app.start_camera()
    elif feature == 'volume':
        volume_app.stop_camera()
        volume_app.start_camera()
    elif feature == 'mouse':
        mouse_app.stop_camera()
        mouse_app.start_camera()

    return jsonify({"success": True})



if __name__ == '__main__':
    os.makedirs("saved_drawings", exist_ok=True)
    app.run(debug=True)
