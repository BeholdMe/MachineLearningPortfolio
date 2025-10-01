import sys
import random
import argparse
import numpy
import cv2
from tracking import MotionDetector, KalmanFilter, Tracker
numpy.float = numpy.float64
numpy.int = numpy.int_

from PySide6 import QtCore, QtWidgets, QtGui
from skvideo.io import vread


class QtDemo(QtWidgets.QWidget):
    def __init__(self, frames, grey=False):
        super().__init__()

        self.button = QtWidgets.QPushButton("Next Frame")
        print("Help")
        self.frames = frames
        self.grey = grey
        self.current_frame = 0

        # set up the motion detector
        self.motion_detector = MotionDetector(alpha=3, tau=25, delta=50)
        self.tracker = Tracker(alpha=3, delta=30)

        # Configure image label
        self.img_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.update_image(0)

        '''
        h, w, c = self.frames[0].shape
        if c == 1:
            img = QtGui.QImage(self.frames[0], w, h, QtGui.QImage.Format_Grayscale8)
        else:
            img = QtGui.QImage(self.frames[0], w, h, QtGui.QImage.Format_RGB888)
        self.img_label.setPixmap(QtGui.QPixmap.fromImage(img))
        '''

        #print("[INFO] Image shape: ", self.frames[0].shape)

        # Configure slider
        self.frame_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.frame_slider.setTickInterval(1)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(self.frames.shape[0]-1)

        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.addWidget(self.img_label)
        self.layout.addWidget(self.button)
        self.layout.addWidget(self.frame_slider)

        # Connect functions
        self.button.clicked.connect(self.on_click)
        self.frame_slider.sliderMoved.connect(self.on_move)

    def update_image(self, index):
        frame = self.frames[index]
        self.current_frame = index

        #convert if needed (probably not but just in case)
        gray = frame
        if frame.ndim == 3 and frame.shape[2] == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        #gray = frame if self.grey else cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        #detect the motion, convert to displayable image, and draw the boxes
        detections = self.motion_detector.update(gray)
        print(f"Detected: {detections}")
        self.tracker.step(detections)
        tracks = self.tracker.get_tracks()

        if len(frame.shape) == 2 or frame.shape[2] == 1:
            disp_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            disp_frame = frame.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            cv2.rectangle(disp_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        for track in tracks:
            pts = track['trail']
            for i in range(1, len(pts)):
                pt1 = (int(pts[i - 1][0]), int(pts[i - 1][1]))
                pt2 = (int(pts[i][0]), int(pts[i][1]))
                cv2.line(disp_frame, pt1, pt2, (255, 0, 0), 2)
            last_pt = pts[-1]
            cv2.putText(disp_frame, f"ID {track['id']}", (int(last_pt[0]), int(last_pt[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        h, w, _ = disp_frame.shape
        img = QtGui.QImage(disp_frame.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
        self.img_label.setPixmap(QtGui.QPixmap.fromImage(img))

    @QtCore.Slot()
    def on_click(self):
        if self.current_frame >= self.frames.shape[0] - 1:
            return
        self.current_frame += 1
        self.frame_slider.setValue(self.current_frame)
        self.update_image(self.current_frame)
        '''
        if self.current_frame == self.frames.shape[0]-1:
            return
        h, w, c = self.frames[self.current_frame].shape
        if c == 1:
            img = QtGui.QImage(self.frames[self.current_frame], w, h, QtGui.QImage.Format_Grayscale8)
        else:
            img = QtGui.QImage(self.frames[self.current_frame], w, h, QtGui.QImage.Format_RGB888)
        self.img_label.setPixmap(QtGui.QPixmap.fromImage(img))
        self.current_frame += 1
        '''

    @QtCore.Slot()
    def on_move(self, pos):
        self.update_image(pos)
        self.frame_slider.setValue(pos)
        '''
        self.current_frame = pos
        h, w, c = self.frames[self.current_frame].shape
        if c == 1:
            img = QtGui.QImage(self.frames[self.current_frame], w, h, QtGui.QImage.Format_Grayscale8)
        else:
            img = QtGui.QImage(self.frames[self.current_frame], w, h, QtGui.QImage.Format_RGB888)
        self.img_label.setPixmap(QtGui.QPixmap.fromImage(img))
        '''


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Demo for loading video with Qt5.")
    parser.add_argument("video_path", metavar='PATH_TO_VIDEO', type=str)
    parser.add_argument("--num_frames", metavar='n', type=int, default=-1)
    parser.add_argument("--grey", metavar='True/False', type=str, default=False)
    args = parser.parse_args()

    gray = args.grey

    num_frames = args.num_frames

    if num_frames > 0:
        frames = vread(args.video_path, num_frames=num_frames, as_grey=args.grey)
    else:
        frames = vread(args.video_path, as_grey=args.grey)
    

    print("Loaded video with shape:", frames.shape)

    app = QtWidgets.QApplication([])

    widget = QtDemo(frames, grey=gray)
    widget.resize(800, 600)
    widget.show()

    sys.exit(app.exec_())
