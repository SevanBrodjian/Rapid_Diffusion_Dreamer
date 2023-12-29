from helpers import *
import numpy as np
import cv2
import time
import sys
import threading
import torch
import gc
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread


esrgan_model = initialize_esrgan()
model = init_sd()


prompt3 = '''A realistic, highly-detailed australian shepherd catching a fish with \
    its mouth from a river in the countryside, mountainous background, beautiful \
    sunny day, symmetrical face, beautiful eyes, detailed eyes, detailed paws, \
    symmetrical legs, realistic fur, high-resolution.'''
prompt1 = '''A suureal, stylistic australian shepherd catching an ethereal glowing fish with \
    its mouth from a majestic psychedelic river in the colorful countryside, mountainous background, beautiful \
    sunny day, symmetrical face, beautiful eyes, detailed eyes, detailed paws, \
    symmetrical legs, Salvador Dali, surreal.'''
prompt2 = '''A tiger stalking its prey in the woods at dusk, dark, eerie, creepy, beautiful and majestic, ethereal, \
    surreal and psychedelic, in the style of Bosch, during dusk, big moon, detailed paws.'''
options = get_options_dict(model)


ucs, cs, text_enc_time = encode_text(model, prompt1, options)
encoded_samples1, sample_enc_time = encode_samples(model, ucs, cs, options)

ucs, cs, text_enc_time = encode_text(model, prompt2, options)
encoded_samples2, sample_enc_time = encode_samples(model, ucs, cs, options)


dream_step_size = 35
diff_vec = (encoded_samples2[0] - encoded_samples1[0]) / dream_step_size


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


def upscale_image_nn(array, new_size):
    return cv2.resize(array, new_size, interpolation=cv2.INTER_NEAREST)

def bilinear_interpolation(array, new_size):
    return cv2.resize(array, new_size, interpolation=cv2.INTER_LINEAR)

def bicubic_interpolation(array, new_size):
    return cv2.resize(array, new_size, interpolation=cv2.INTER_CUBIC)


def convert_array_to_qimage(array):
    array = bicubic_interpolation(array, (3840, 2160))
    height, width, channels = array.shape
    bytes_per_line = channels * width
    array = array.astype(np.uint8)
    qimage = QImage(array.tobytes(), width, height, bytes_per_line, QImage.Format_RGB888)
    return qimage


def generate_next_frames(model, encoded_samples):
    start_time = time.time()
    with torch.no_grad(), \
        options['precision_scope']("cuda"), \
        model.ema_scope():
            x_samples = model.decode_first_stage(encoded_samples)
            x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
    end_time = time.time()
    return x_samples, (end_time - start_time)


def dream_thread(display_widget):
    dream_times = list()
    frame_times = list()
    cleaning_times = list()
    display_times = list()
    encoded_dream_all_samples = [encoded_samples1[0] + diff_vec * i for i in range(75)]
    encoded_dream_all_samples = torch.cat(encoded_dream_all_samples, dim=0)
    batch_size = 1
    clean_freq = 2

    while True:
        for i in range(0, dream_step_size, batch_size):
            start_time = time.time()
            dream_samples, dream_gen_time = generate_next_frames(model, encoded_dream_all_samples[i:i+1])
            dream_samples = (dream_samples * 255).to(torch.uint8)
            dream_samples = dream_samples.permute(0, 2, 3, 1).to('cpu', non_blocking=True).numpy()

            start_display = time.time()
            for dream in dream_samples:
                qimage = convert_array_to_qimage(dream)
                display_widget.update_image.emit(qimage)
            end_display = time.time()

            if (i / batch_size) % clean_freq == 0:
                start_cleaning = time.time()
                # torch.cuda.empty_cache()
                gc.collect()
                end_cleaning = time.time()

            end_time = time.time()
            dream_times.append(dream_gen_time)
            display_times.append(end_display - start_display)
            cleaning_times.append(end_cleaning - start_cleaning)
            frame_times.append(end_time - start_time)
        for i in range(dream_step_size-1, 0, -batch_size):
            start_time = time.time()
            dream_samples, dream_gen_time = generate_next_frames(model, encoded_dream_all_samples[i:i+1])
            dream_samples = (dream_samples * 255).to(torch.uint8)
            dream_samples = dream_samples.permute(0, 2, 3, 1).to('cpu', non_blocking=False).numpy()

            start_display = time.time()
            for dream in dream_samples:
                qimage = convert_array_to_qimage(dream)
                display_widget.update_image.emit(qimage)
            end_display = time.time()

            if (i / batch_size) % clean_freq == 0:
                start_cleaning = time.time()
                # torch.cuda.empty_cache()
                gc.collect()
                end_cleaning = time.time()

            end_time = time.time()
            dream_times.append(dream_gen_time)
            display_times.append(end_display - start_display)
            cleaning_times.append(end_cleaning - start_cleaning)
            frame_times.append(end_time - start_time)
        print(f'Average dream time: {sum(dream_times) / len(dream_times)}')
        print(f'Average display time: {sum(display_times) / len(display_times)}')
        print(f'Average cleaning time: {sum(cleaning_times) / len(cleaning_times)}')
        print(f'Average time per frame: {sum(frame_times) / len(frame_times)}')


def start_gui_and_processing():
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)

    display_widget = ImageDisplay()
    display_widget.show()

    thread = threading.Thread(target=dream_thread, args=(display_widget,))
    thread.start()

    app.exec_()

    display_widget.deleteLater()


if __name__ == '__main__':
    start_gui_and_processing()