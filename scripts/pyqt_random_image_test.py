from helpers import *
import numpy as np
import cv2
import time
import sys
import threading
import torch
import concurrent.futures
from PIL import Image
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QSlider, QPushButton
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread


model = init_sd()

prompt1 = '''A surreal, stylistic australian shepherd catching an ethereal glowing fish with \
    its mouth from a majestic psychedelic river in the colorful countryside, mountainous background, beautiful \
    sunny day, symmetrical face, beautiful eyes, detailed eyes, detailed paws, \
    symmetrical legs, Salvador Dali, surreal.'''
prompt2 = '''A tiger stalking its prey in the woods at dusk, dark, eerie, creepy, beautiful and majestic, ethereal, \
    surreal and psychedelic, in the style of Bosch, during dusk, big moon, detailed paws.'''
options = get_options_dict(model, rand_seed=True)

ucs, cs, text_enc_time = encode_text(model, prompt1, options)
encoded_samples1, sample_enc_time = encode_samples(model, ucs, cs, options)

ucs, cs, text_enc_time = encode_text(model, prompt2, options)
encoded_samples2, sample_enc_time = encode_samples(model, ucs, cs, options)

dream_steps = 35
diff_vec = (encoded_samples2[0] - encoded_samples1[0]) / dream_steps
all_encoded_samples = [encoded_samples1[0] + diff_vec * i for i in range(dream_steps)]
all_encoded_samples = torch.cat(all_encoded_samples, dim=0)

slider_values = {'latent_mean': 0., 'latent_var': 1.}
slider_lock = threading.Lock()
buffer_lock = threading.Lock()
frame_buffer = []

class ImageDisplay(QWidget):
    update_image = pyqtSignal(QImage)

    def __init__(self):
        super().__init__()
        self.image_label = QLabel(self)

        self.latent_mean = QSlider(Qt.Horizontal, self)
        self.latent_mean.setMinimum(-100)
        self.latent_mean.setMaximum(100)
        self.latent_mean.setValue(0)
        self.latent_mean.valueChanged.connect(self.slider_changed)
        self.label_mean = QLabel("0.50", self)

        self.latent_var = QSlider(Qt.Horizontal, self)
        self.latent_var.setMinimum(-100)
        self.latent_var.setMaximum(100)
        self.latent_var.setValue(1)
        self.latent_var.valueChanged.connect(self.slider_changed)
        self.label_var = QLabel("0.50", self)

        self.button = QPushButton("Clear Frame Buffer", self)
        self.button.clicked.connect(self.onButtonClicked)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.latent_mean)
        layout.addWidget(self.latent_var)
        self.setLayout(layout)

        self.update_image.connect(self.setImage)

    @pyqtSlot(QImage)
    def setImage(self, image):
        pixmap = QPixmap.fromImage(image)
        self.image_label.setPixmap(pixmap)

    def slider_changed(self, value):
        global slider_values
        with slider_lock:
            slider_values['latent_mean'] = self.latent_mean.value()
            slider_values['latent_var'] = self.latent_var.value()

            self.label_mean.setText(f"Mean: {self.latent_mean.value() / 10.:.2f}")
            self.label_var.setText(f"Variance: {self.latent_var.value():.2f}")

    def onButtonClicked(self):
        global buffer_lock
        global frame_buffer
        with buffer_lock:
            frame_buffer = frame_buffer[-10:]


def convert_array_to_qimage(array):
    height, width, channels = array.shape
    bytes_per_line = channels * width
    qimage = QImage(array.tobytes(), width, height, bytes_per_line, QImage.Format_RGB888)
    return qimage


def decode_dream(model, encoded_sample):
    with torch.no_grad(), \
        options['precision_scope']("cuda"), \
        model.ema_scope():
            x_samples = model.decode_first_stage(encoded_sample)[0]
            x_samples = torch.clamp((x_samples + 1.0) / 2.0 * 255.0, min=0.0, max=255.0)
    return x_samples


def resize_image(image, new_size, interpolation=cv2.INTER_CUBIC):
    return cv2.resize(image, new_size, interpolation=interpolation)

def resize_images_parallel(image_batch, new_size, interpolation=cv2.INTER_CUBIC):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(resize_image, img, new_size, interpolation) for img in image_batch]
        return [future.result() for future in concurrent.futures.as_completed(futures)]
    

def interpolate_frames_linear(prev_frame, frame, num_inter=3):
    if prev_frame is None or frame is None:
        return frame.unsqueeze(0)
    
    C, H, W = frame.shape
    output = torch.empty((num_inter + 1, C, H, W), dtype=frame.dtype)
    output[-1] = frame

    for i in range(num_inter):
        alpha = (i + 1) / (num_inter + 1)
        interpolated_frame = (1 - alpha) * prev_frame + alpha * frame
        output[i] = interpolated_frame

    return output


def dream_loop(all_encoded_samples, clean_freq = 4):
    global buffer_lock
    global frame_buffer
    global slider_values

    dream_times = list()
    frame_times = list()
    processing_times = list()
    cleaning_times = list()
    prev_dream = None

    while True:
        for i in range(0, dream_steps-1, 1):
            start_frame = time.time()
            start_dreaming = time.time()
            dream = decode_dream(model, all_encoded_samples[i:i+1])
            end_dreaming = time.time()

            start_process = time.time()
            frames = interpolate_frames_linear(prev_dream, dream)
            prev_dream = dream
            frames = frames.byte().permute(0, 2, 3, 1).to('cpu', non_blocking=False).numpy()
            frames = resize_images_parallel(frames, (3840, 2160))
            frames = [convert_array_to_qimage(frame) for frame in frames]
            with buffer_lock:
                frame_buffer.extend(frames)
            end_process = time.time()

            start_cleaning = time.time()
            if i % clean_freq == 0:
                torch.cuda.empty_cache()
            end_cleaning = time.time()
            end_frame = time.time()

            dream_times.append(end_dreaming - start_dreaming)
            cleaning_times.append(end_cleaning - start_cleaning)
            processing_times.append(end_process - start_process)
            frame_times.append(end_frame - start_frame)

        with slider_lock:
            current_slider_mean = float(slider_values['latent_mean'])
            current_slider_var = float(slider_values['latent_var'])
        cur_latent = all_encoded_samples[-1].unsqueeze(0)
        shape = cur_latent.shape
        # std_dev = np.random.randint(4, 5)
        random_latent = torch.randn(shape, dtype=torch.float32) * 0.1 #+ current_slider_mean / 10.
        random_latent = random_latent.to(options['device'])
        diff_vec = (random_latent - cur_latent) / dream_steps
        all_encoded_samples = [cur_latent + diff_vec * i for i in range(dream_steps)]
        all_encoded_samples = torch.cat(all_encoded_samples, dim=0)
        
        print(f'Average dream time: {sum(dream_times) / len(dream_times)}')
        print(f'Average processing time: {sum(processing_times) / len(processing_times)}')
        print(f'Average cleaning time: {sum(cleaning_times) / len(cleaning_times)}')
        print(f'Average time per frame: {sum(frame_times) / len(frame_times)}')


def display_frames(display_widget, display_rate = 26):
    global buffer_lock
    global frame_buffer

    while True:
        frame = None
        with buffer_lock:
            if frame_buffer:
                frame = frame_buffer.pop(0)
        
        if frame is not None:
            display_widget.update_image.emit(frame)
            print(len(frame_buffer))
        else:
            print('no frame!')

        time.sleep(1 / display_rate)


def start_gui_and_processing():
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)

    display_widget = ImageDisplay()
    display_widget.show()

    dream_thread = threading.Thread(target=dream_loop, args=(all_encoded_samples,))
    dream_thread.start()

    display_thread = threading.Thread(target=display_frames, args=(display_widget,))
    display_thread.start()

    sys.exit(app.exec_())


if __name__ == '__main__':
    start_gui_and_processing()