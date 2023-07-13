from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt
import os
from pathlib import Path
import numpy as np
from argparse import ArgumentParser
from copy import deepcopy
import torch
from tqdm import tqdm
import pickle

fnt = ImageFont.truetype(R"arial.ttf", 80)

def plot_rectangle(image, boxes, names=None):
    draw_image = deepcopy(image)
    draw = ImageDraw.Draw(draw_image)
    for index, box in enumerate(boxes):
        draw.rectangle(box.tolist(), width=3, outline=(255, 0, 0, 255))
        if names is not None:
            draw.text(box, names[index], font=fnt, fill=(255, 255, 255, 255), anchor='lb')

    plt.imshow(draw_image)
    plt.pause(2)

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

        self.embeddings = {}

    def make_ref_db(self, path):
        directory = Path(path)
        db_file_path = directory / "db.pt"

        if db_file_path.exists():
            print("Loading existing db")
            with open(db_file_path, 'rb') as db_file:
                self.embeddings = torch.load(db_file)
            return

        temp_embeddings = []
        for name in tqdm(os.listdir(path), desc = "Folders"):
            file_names = list(os.listdir(directory / name))

            for filename in tqdm(file_names, desc = name):
                filename = Path(filename)
                with Image.open(directory / name / filename) as image:
                    _, _, faces = self.detect_image(image)
                    if faces.shape[0] > 1:
                        continue

                    embedding = self.resnet(faces.cpu()).cpu()

                    temp_embeddings.append((name, embedding))

        emb_tensor = torch.empty((len(temp_embeddings), 512))
        names = []

        for index, (name, emb) in enumerate(temp_embeddings):
            emb_tensor[index, :] = emb
            names.append(name)

        self.embeddings = {"names" : names, "embeddings" : emb_tensor}

        with open(db_file_path, 'wb') as db_file:
            torch.save(self.embeddings, db_file)


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
        distances = [None for _ in range(nb_faces)]
        for face_i in range(nb_faces):
            min_distance = 1e36
            distances = torch.linalg.norm(embeddings[face_i, :] - self.embeddings["embeddings"], dim=1)
            min_index = torch.argmin(distances).data

            names[face_i] = self.embeddings["names"][min_index]
            distances[face_i] = distances[min_index].data

        return names, distances 

def crop(args):
    classify = args.reference != None
    input_folder = Path(args.input_folder)
    output_folder = Path(args.output_folder)
    display = args.display

    output_file_path = output_folder / "log.txt"
    output_file = open(output_file_path, 'w')

    classifier = Classifier()
    if classify:
        print("Processing reference database...")
        classifier.make_ref_db(args.reference)
        for name in classifier.embeddings["names"]:
            os.makedirs(output_folder/name, exist_ok=True)
    else:
        os.makedirs(output_folder)
    

    file_names = list(os.listdir(args.input_folder))

    print("Processing images...")
    progress_bar = tqdm(file_names, unit='im')
    for file_name in progress_bar:
        progress_bar.set_description(desc=file_name)
        names = None
        with Image.open(input_folder / file_name) as image:
            boxes, score, faces = classifier.detect_image(image)
            if boxes is None:
                output_file.write(f"{file_name} : 0\n")
                continue
            output_file.write(f"{file_name} : {len(boxes)}\n")

            if classify:
                names, distances = classifier.classify_faces(faces)
                good_indices = []
                for index, (name, distance) in enumerate(zip(names, distances)):
                    output_file.write(f"\t{name} : {distance}\n")
                    if distance > args.confidence_threshold:
                        pass
                    else :
                        good_indices.append(index)

            if args.display:
                plot_rectangle(image, boxes, names)

            reference_faces = make_reference_images(image, boxes)
            for face_i, ref_face in enumerate(reference_faces):
                output_file_name = Path(file_name).stem + f"_{face_i}.png"
                if classify and face_i in good_indices:
                    output_file_name = names[face_i] + "/" + output_file_name
                ref_face.save(output_folder / output_file_name)

def sort(args):
    print("Sorting is not implemented yet")
    return
    
    if args.reference == None:
        print("Path to network required")

    input_folder = Path(args.input_folder)
    output_folder = Path(args.output_folder)
    display = args.display

    classifier = Classifier()
    classifier.load_state_dict(torch.load(args.reference))

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
                names, distances = classifier.classify_faces(faces)
                good_indices = []
                for index, (name, distance) in enumerate(zip(names, distances)):
                    print(f"{name} with distance {distance}")
                    if distance > args.confidence_threshold:
                        print("Rejecting")
                    else :
                        good_indices.append(index)
                        

            if args.display:
                plot_rectangle(image, boxes, names)

            reference_faces = make_reference_images(image, boxes)
            for face_i, ref_face in enumerate(reference_faces):
                output_file_name = Path(file_name).stem + f"_{face_i}.png"
                if classify and face_i in good_indices:
                    output_file_name = names[face_i] + "/" + output_file_name
                ref_face.save(output_folder / output_file_name)


if __name__ == "__main__":

    argument_parser = ArgumentParser(prog="Face extractor")
    argument_parser.add_argument("-r", "--reference", help="Reference folder containing images of single faces with associated name")
    argument_parser.add_argument("-i", "--input_folder", required=True, help="Folder of images to extract faces from")
    argument_parser.add_argument("-o", "--output_folder", required=True, help="Folder containing the extracted faces")
    argument_parser.add_argument("-d", "--display", help="Display the detected faces on images", action='store_true')
    argument_parser.add_argument("-t", "--confidence_threshold", help="Maximum embedding distance to reference images", default=0.8)
    argument_parser.add_argument("-m", "--mode", required=True, choices=['crop', 'sort'])

    args = argument_parser.parse_args()
    if args.mode == 'sort':
        sort(args)
    elif args.mode == 'crop':
        crop(args)

   