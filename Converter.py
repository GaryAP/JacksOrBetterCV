import os
import json
import tensorflow as tf
from PIL import Image
import io
from object_detection.utils import dataset_util


def create_tf_example(image, annotations, image_dir):
    filename = image['file_name']
    image_id = image['id']
    image_path = os.path.join(image_dir, filename)

    # Debugging statement
    print(f"Processing image: {image_path}")

    with tf.io.gfile.GFile(image_path, 'rb') as fid:
        encoded_image_data = fid.read()
    image = Image.open(io.BytesIO(encoded_image_data))
    width, height = image.size

    image_format = b'jpeg' if filename.lower().endswith('.jpg') else b'png'

    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for ann in annotations:
        if ann['image_id'] == image_id:
            xmin = ann['bbox'][0] / width
            xmax = (ann['bbox'][0] + ann['bbox'][2]) / width
            ymin = ann['bbox'][1] / height
            ymax = (ann['bbox'][1] + ann['bbox'][3]) / height

            xmins.append(xmin)
            xmaxs.append(xmax)
            ymins.append(ymin)
            ymaxs.append(ymax)
            classes_text.append(str(ann['category_id']).encode('utf8'))
            classes.append(ann['category_id'])

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(str(image_id).encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def create_tf_record(output_filename, image_dir, annotations_path):
    with open(annotations_path) as f:
        coco = json.load(f)

    images = coco['images']
    annotations = coco['annotations']

    writer = tf.io.TFRecordWriter(output_filename)

    for image in images:
        image_annotations = [ann for ann in annotations if ann['image_id'] == image['id']]
        tf_example = create_tf_example(image, image_annotations, image_dir)
        writer.write(tf_example.SerializeToString())

    writer.close()


# Update these paths to your dataset
# train_image_dir = 'Z:\\441\\JacksOrBetterCV\\Data\\train'
val_image_dir = 'Z:\\441\\JacksOrBetterCV\\Data\\valid'
# train_annotations_path = 'Z:\\441\\JacksOrBetterCV\\Data\\train\\_annotations.coco.json'
val_annotations_path = 'Z:\\441\\JacksOrBetterCV\\Data\\valid\\_annotations.coco.json'

# Convert training and validation datasets
# create_tf_record('train.record', train_image_dir, train_annotations_path)
create_tf_record('val.record', val_image_dir, val_annotations_path)
