from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt
import os
from pathlib import Path
import numpy as np
from argparse import ArgumentParser
from copy import deepcopy
import torch

fnt = ImageFont.truetype(R"arial.ttf", 80)

def plot_rectangle(image, boxes, names=None):

    draw_image = deepcopy(image)
    draw = ImageDraw.Draw(draw_image)
    for index, box in enumerate(boxes):
        draw.rectangle(box.tolist(), width=3, outline=(255, 0, 0, 255))
        if names is not None:
            draw.text(box, names[index], font=fnt, fill=(255, 255, 255, 255), anchor='lb')

    plt.imshow(draw_image)
    plt.show()

def make_reference_images(image, boxes):
    
    ret_list = []
    for box in boxes:
        box = box.tolist()
        center = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
        shape = np.array([box[2] - box[0], box[3] - box[1]])
        shape = shape * 2
        tl = center - shape / 2
        br = center + shape / 2

        new_rect = np.concatenate([tl ,br])
        new_rect[[0, 2]] = np.clip(new_rect[[0, 2]], 0, image.width)
        new_rect[[1, 3]] = np.clip(new_rect[[1, 3]], 0, image.height)

        new_cropped_face = image.crop(new_rect.tolist())
        ret_list.append(new_cropped_face)

    return ret_list


class Classifier:
    def __init__(self):
        self.mtcnn = MTCNN(device="cpu", keep_all=True, select_largest=False,)
        self.resnet = InceptionResnetV1(pretrained='vggface2', device="cpu").eval()

        self.embeddings = []

    def make_ref_db(self, path):
        directory = Path(path)
        for name in os.listdir(path):
            for filename in os.listdir(directory / name):
                filename = Path(filename)
                with Image.open(directory / name / filename) as image:
                    _, _, faces = self.detect_image(image)
                    if faces.shape[0] > 1:
                        return None

                    embedding = self.resnet(faces.cpu()).cpu()

                    self.embeddings.append((name, embedding))

    def detect_image(self, image : Image):
        if image.mode != 'RGB':
            image = image.convert('RGB')

        boxes, score = self.mtcnn.detect(image, landmarks=False)
        faces = self.mtcnn.extract(image, boxes, None)

        return boxes, score, faces

    def classify_faces(self, faces):
        embeddings = self.resnet(faces.cpu()).cpu()
        nb_faces = embeddings.shape[0]
        names = [None for _ in range(nb_faces)]
        for face_i in range(nb_faces):
            min_distance = 1e36
            for emb in self.embeddings:
                value = torch.dist(embeddings[face_i, :], emb[1])
                if value.data < min_distance and value.data > 0:
                    min_distance=value
                    names[face_i] = emb[0]

        return names 

          
if __name__ == "__main__":

    argument_parser = ArgumentParser(prog="Face extractor")
    argument_parser.add_argument("-r", "--reference_folder", help="Reference folded containing images of single faces with associated name")
    argument_parser.add_argument("-i", "--input_folder", required=True, help="Folder of images to extract faces from")
    argument_parser.add_argument("-o", "--output_folder", required=True, help="Folder containing the extracted faces")
    argument_parser.add_argument("-d", "--display", help="Display the detected faces on images", action='store_true')


    args = argument_parser.parse_args()

    classify = args.reference_folder != None
    input_folder = Path(args.input_folder)
    output_folder = Path(args.output_folder)
    display = args.display

    classifier = Classifier()
    if classify:
        classifier.make_ref_db(args.reference_folder)
        for emb in classifier.embeddings:
            os.makedirs(output_folder/emb[0], exist_ok=True)
    else:
        os.makedirs(output_folder)
    
    for file_name in os.listdir(args.input_folder):
        print(f"Opening image {file_name}")
        names = None
        with Image.open(input_folder / file_name) as image:
            boxes, score, faces = classifier.detect_image(image)
            print(f"Found {len(boxes)} face{'s' if len(boxes) > 1 else ''}")
            if classify:
                names = classifier.classify_faces(faces)

            if args.display:
                plot_rectangle(image, boxes, names)

            reference_faces = make_reference_images(image, boxes)
            for face_i, ref_face in enumerate(reference_faces):
                output_file_name = Path(file_name).stem + f"_{face_i}.png"
                if classify:
                    output_file_name = names[face_i] + "/" + output_file_name
                
                ref_face.save(output_folder / output_file_name)

