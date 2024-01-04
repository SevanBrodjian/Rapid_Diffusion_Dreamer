from helpers import *
import numpy as np
import cv2
import time
import sys
import threading
import torch
import concurrent.futures
from PIL import Image
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout
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

dream_steps = 20
diff_vec = (encoded_samples2[0] - encoded_samples1[0]) / dream_steps
all_encoded_samples = [encoded_samples1[0] + diff_vec * i for i in range(75)]
all_encoded_samples = torch.cat(all_encoded_samples, dim=0)


class ImageDisplay(QWidget):
    update_image = pyqtSignal(QImage)

    def __init__(self):
        super().__init__()
        self.image_label = QLabel(self)
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        self.setLayout(layout)
        self.update_image.connect(self.setImage)

    @pyqtSlot(QImage)
    def setImage(self, image):
        pixmap = QPixmap.fromImage(image)
        self.image_label.setPixmap(pixmap)


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


def dream_loop(all_encoded_samples, frame_buffer, buffer_lock, clean_freq = 4):
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

        for i in range(dream_steps-1, 1, -1):
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
        
        print(f'Average dream time: {sum(dream_times) / len(dream_times)}')
        print(f'Average processing time: {sum(processing_times) / len(processing_times)}')
        print(f'Average cleaning time: {sum(cleaning_times) / len(cleaning_times)}')
        print(f'Average time per frame: {sum(frame_times) / len(frame_times)}')


def display_frames(display_widget, frame_buffer, buffer_lock, display_rate = 30):
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

    frame_buffer = []
    buffer_lock = threading.Lock()

    dream_thread = threading.Thread(target=dream_loop, args=(all_encoded_samples, frame_buffer, buffer_lock))
    dream_thread.start()

    display_thread = threading.Thread(target=display_frames, args=(display_widget, frame_buffer, buffer_lock))
    display_thread.start()

    sys.exit(app.exec_())


if __name__ == '__main__':
    start_gui_and_processing()