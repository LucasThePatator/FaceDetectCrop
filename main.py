from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image, ImageDraw, ImageFont, ImageFile
from matplotlib import pyplot as plt
import os
from pathlib import Path
import numpy as np
from argparse import ArgumentParser
from copy import deepcopy
import torch
from tqdm import tqdm
import pickle
import glob
import shutil

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

def make_reference_images(image, boxes, crop_factor): 
    ret_list = []
    for box in boxes:
        box = box.tolist()
        center = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
        shape = np.array([box[2] - box[0], box[3] - box[1]])
        shape = shape * crop_factor
        tl = center - shape / 2
        br = center + shape / 2

        new_rect = np.concatenate([tl ,br])
        new_rect[[0, 2]] = np.clip(new_rect[[0, 2]], 0, image.width)
        new_rect[[1, 3]] = np.clip(new_rect[[1, 3]], 0, image.height)

        new_cropped_face = image.crop(new_rect.tolist())
        ret_list.append(new_cropped_face)

    return ret_list


class Classifier:
    def __init__(self, mode="crop"):
        self.mode = mode
        if mode == "crop":
            self.mtcnn = MTCNN(device="cpu", keep_all=True, select_largest=False)
        else:
            self.mtcnn = MTCNN(device="cpu", keep_all=False, select_largest=True)

        self.resnet = InceptionResnetV1(pretrained='vggface2', device="cpu").eval()
        self.embeddings = {}

        ImageFile.LOAD_TRUNCATED_IMAGES 

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

def crop(args, crop_factor):
    classify = args.reference != None
    input_folder = Path(args.input_folder)
    output_folder = Path(args.output_folder)
    display = args.display

    classifier = Classifier()
    os.makedirs(output_folder, exist_ok=True)
    if classify:
        print("Processing reference database...")
        classifier.make_ref_db(args.reference)
        for name in classifier.embeddings["names"]:
            os.makedirs(output_folder/name, exist_ok=True)

    log_file_path = output_folder / "log.txt"
    log_file = open(log_file_path, 'w', encoding="utf-8")

    file_names = list(os.listdir(args.input_folder))

    print("Processing images...")
    progress_bar = tqdm(list(input_folder.rglob("*.*")))
    for file in progress_bar:
        progress_bar.set_description(desc=file.name)
        file = Path(os.path.join(*file.parts[1:]))
        names = None
        with Image.open(input_folder / file) as image:
            boxes, score, faces = classifier.detect_image(image)
            if boxes is None:
                log_file.write(f"{file} : 0\n")
                continue

            log_file.write(f"{file} : {len(boxes)}\n")

            if classify:
                names, distances = classifier.classify_faces(faces)
                good_indices = []
                for index, (name, distance) in enumerate(zip(names, distances)):
                    log_file.write(f"\t{name} : {distance}\n")
                    if distance > args.confidence_threshold:
                        pass
                    else :
                        good_indices.append(index)

            if args.display:
                plot_rectangle(image, boxes, names)

            reference_faces = make_reference_images(image, boxes, crop_factor=crop_factor)
            for face_i, ref_face in enumerate(reference_faces):
                output_file_name = file.with_stem(file.stem + f"_{face_i}")
                if classify and face_i in good_indices:
                    output_file_name = Path(names[face_i]) / output_file_name

                output_file_name = output_folder / output_file_name
                os.makedirs(output_file_name.parent, exist_ok=True)
                ref_face.save(output_file_name)

    log_file.flush()

def sort(args):  
    if args.reference == None:
        print("Path to db required")

    input_folder = Path(args.input_folder)
    output_folder = Path(args.output_folder)
    display = args.display

    classifier = Classifier(mode="sort")

    print("Processing reference database...")
    classifier.make_ref_db(args.reference)
    os.makedirs(output_folder, exist_ok=True)
    for name in classifier.embeddings["names"]:
        os.makedirs(output_folder/name, exist_ok=True)
    
    os.makedirs(output_folder/"unsorted", exist_ok=True)

    log_file_path = output_folder / "log.txt"
    log_file = open(log_file_path, 'w', encoding="utf-8")
    
    progress_bar = tqdm(list(input_folder.rglob("*.*")))
    for file in progress_bar:
        if file.suffix not in [".png", ".tiff", ".tif", ".jpeg", ".jpg", ".webp", ".bmp", ".img"]:
            log_file.write(f"{file} is not an image\n")
            continue

        progress_bar.set_description(desc=file.name)
        file = file.relative_to(input_folder)
        with Image.open(input_folder / file) as image:

            boxes, score, faces = classifier.detect_image(image)
            if args.display:
                plot_rectangle(image, boxes, names)

            if boxes is None:
                log_file.write(f"{file} has no face\n")
                current_name="unsorted"
                current_distance=0.0
            else:
                faces = torch.unsqueeze(faces, 0)
                names, distances = classifier.classify_faces(faces)

                current_name=names[0]
                current_distance=distances[0]
                if distances[0] > args.confidence_threshold:
                    log_file.write(f"{file}\t{names[0]}\t{distances[0]}\trejected\n")
                    current_name="unsorted"
                    current_distance=0.0

            log_file.write(f"{file}\t{current_name}\t{current_distance}\n")
            output_file_name = Path(current_name) / file
            output_file_name = output_folder / output_file_name
            os.makedirs(output_file_name.parent, exist_ok=True)
            shutil.move(input_folder / file, output_file_name)

    log_file.flush()

if __name__ == "__main__":

    argument_parser = ArgumentParser(prog="Face extractor")
    argument_parser.add_argument("-r", "--reference", help="Reference folder containing images of single faces with associated name")
    argument_parser.add_argument("-i", "--input_folder", required=True, help="Folder of images to extract faces from")
    argument_parser.add_argument("-o", "--output_folder", required=True, help="Folder containing the extracted faces")
    argument_parser.add_argument("-d", "--display", help="Display the detected faces on images", action='store_true')
    argument_parser.add_argument("-t", "--confidence_threshold", help="Maximum embedding distance to reference images", default=0.8, type=float)
    argument_parser.add_argument("-m", "--mode", required=True, choices=['crop', 'sort'])
    argument_parser.add_argument("--crop_factor", default=2.5, type=float, help="Factor for the crop size, 1 is strictly the face")


    args = argument_parser.parse_args()
    if args.mode == 'sort':
        sort(args)
    elif args.mode == 'crop':
        crop(args,crop_factor=args.crop_factor)

   