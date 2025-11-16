import cv2
from mtcnn import MTCNN
import sys, os.path
import json
from keras import backend as K
import tensorflow as tf
import warnings                     # ✅ NEW: suppress cleanup warnings
from tqdm import tqdm               # ✅ NEW: progress bar

# ✅ Suppress noisy cleanup warnings
warnings.filterwarnings("ignore", category=ResourceWarning)

print(tf.__version__)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

base_path = '.\\train_sample_videos\\'

def get_filename_only(file_path):
    file_basename = os.path.basename(file_path)
    filename_only = file_basename.split('.')[0]
    return filename_only

with open(os.path.join(base_path, 'metadata.json')) as metadata_json:
    metadata = json.load(metadata_json)
    print(len(metadata))

for filename in metadata.keys():
    tmp_path = os.path.join(base_path, get_filename_only(filename))
    print('Processing Directory: ' + tmp_path)

    frame_images = [x for x in os.listdir(tmp_path) if os.path.isfile(os.path.join(tmp_path, x))]
    frame_images = sorted(frame_images)[:50]     # ✅ Limit to first 50 frames for speed

    faces_path = os.path.join(tmp_path, 'faces')
    print('Creating Directory: ' + faces_path)
    os.makedirs(faces_path, exist_ok=True)
    print('Cropping Faces from Images...')

    for frame in tqdm(frame_images):             # ✅ Add progress bar
        print('Processing ', frame)
        detector = MTCNN()

        # ✅ Check if image is readable before converting
        img_path = os.path.join(tmp_path, frame)
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            print(f"Warning: Could not read {img_path}")
            continue

        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = detector.detect_faces(image)
        print('Face Detected: ', len(results))
        count = 0

        for result in results:
            bounding_box = result['box']
            print(bounding_box)
            confidence = result['confidence']
            print(confidence)

            if len(results) < 2 or confidence > 0.95:
                margin_x = bounding_box[2] * 0.3
                margin_y = bounding_box[3] * 0.3
                x1 = max(int(bounding_box[0] - margin_x), 0)
                x2 = min(int(bounding_box[0] + bounding_box[2] + margin_x), image.shape[1])
                y1 = max(int(bounding_box[1] - margin_y), 0)
                y2 = min(int(bounding_box[1] + bounding_box[3] + margin_y), image.shape[0])
                print(x1, y1, x2, y2)

                crop_image = image[y1:y2, x1:x2]
                new_filename = '{}-{:02d}.png'.format(os.path.join(faces_path, get_filename_only(frame)), count)
                count += 1
                cv2.imwrite(new_filename, cv2.cvtColor(crop_image, cv2.COLOR_RGB2BGR))
            else:
                print('Skipped a face..')