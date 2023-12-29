from helpers import *
import numpy as np

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

dream_step_size = 75
diff_vec = (encoded_samples2[0] - encoded_samples1[0]) / dream_step_size

dream_times = list()
frame_times = list()
q_flag=False
while True:
    for i in range(75):
        start_time = time.time()
        encoded_dream_sample = encoded_samples1[0] + diff_vec * i
        dream_sample, dream_gen_time = decode_imgs(model, encoded_dream_sample.unsqueeze(0), options)
        dream = dream_sample[0][0]
        dream = (dream * 255).to(torch.uint8)
        dream = dream.permute(1, 2, 0).cpu().numpy()

        image_bgr = cv2.cvtColor(dream.astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imshow('Image', image_bgr)
        if cv2.waitKey(4) == ord('q'):
            q_flag = True
            break

        end_time = time.time()
        dream_times.append(dream_gen_time)
        frame_times.append(end_time - start_time)

        print(f'Average dream time: {sum(dream_times) / len(dream_times)}')
        print(f'Average fps: {sum(frame_times) / len(frame_times)}')
        
    if q_flag:
        break


cv2.destroyAllWindows()