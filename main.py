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
    def __init__(self):

        self.mtcnn = MTCNN(device="cpu", keep_all=True, select_largest=False)

        self.resnet = InceptionResnetV1(pretrained='vggface2', device="cpu").eval()
        self.embeddings = {}

        ImageFile.LOAD_TRUNCATED_IMAGES = True

    def make_ref_db(self, path, update : bool = False):
        directory = Path(path)
        db_file_path = directory / "db.pt"

        log_file = open(directory / "reference_log.txt",'w', encoding="utf-8")

        update_db = False
        if db_file_path.exists():
            print("Loading existing db")
            with open(db_file_path, 'rb') as db_file:
                self.embeddings = torch.load(db_file)
            
            if not update:
                return
            
            update_db = True

        temp_embeddings = []
        for name in tqdm(os.listdir(path), desc = "Folders"):
            current_dir = Path(directory / name)
            if not current_dir.is_dir():
                continue

            file_names = list(os.listdir(current_dir))

            for filename in tqdm(file_names, desc = name):
                filename = Path(filename)
                log_file.write(f"{current_dir}/{filename}")

                with Image.open(directory / name / filename) as image:
                    _, _, faces = self.detect_image(image)
                    if faces is None :
                        log_file.write(f" : no face : skip\n")
                        continue
                    if faces.shape[0] > 1:
                        log_file.write(f" : {faces.shape[0]} faces : skip\n")
                        continue

                    embedding = self.resnet(faces.cpu()).cpu()
                    if update_db:
                        min_distance = 1e36
                        distances = torch.linalg.norm(embedding - self.embeddings["embeddings"], dim=1)
                        min_index = torch.argmin(distances).data

                        if distances[min_index].data < 1e-5:
                            log_file.write(f" : already in db\n")
                            continue
                    
                    log_file.write(f" : added to db\n")
                    temp_embeddings.append((name, embedding))

                log_file.flush()

        emb_tensor = torch.empty((len(temp_embeddings), 512))
        names = []

        for index, (name, emb) in enumerate(temp_embeddings):
            emb_tensor[index, :] = emb
            names.append(name)

        if update_db:
            self.embeddings["names"].extend(names)
            self.embeddings["embeddings"] = torch.cat((self.embeddings["embeddings"], emb_tensor))
        else :
            self.embeddings = {"names" : emb_tensor, "embeddings" : emb_tensor}

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
        out_distances = [None for _ in range(nb_faces)]
        for face_i in range(nb_faces):
            min_distance = 1e36
            distances = torch.linalg.norm(embeddings[face_i, :] - self.embeddings["embeddings"], dim=1)
            min_index = torch.argmin(distances).data

            names[face_i] = self.embeddings["names"][min_index]
            out_distances[face_i] = distances[min_index].data

        return names, out_distances 

def crop(args, classifier, crop_factor):
    classify = args.reference != None
    input_folder = Path(args.input_folder)
    output_folder = Path(args.output_folder)
    display = args.display

    os.makedirs(output_folder, exist_ok=True)

    log_file_path = output_folder / "log.txt"
    log_file = open(log_file_path, 'w', encoding="utf-8")

    file_names = list(os.listdir(args.input_folder))

    print("Processing images...")
    progress_bar = tqdm(list(input_folder.rglob("*.*")))
    for file in progress_bar:
        if file.suffix.lower() not in [".png", ".tiff", ".tif", ".jpeg", ".jpg", ".webp", ".bmp", ".img"]:
            log_file.write(f"{file} is not an image\n")
            continue
        
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

def sort(args, classifier):  
    if args.reference == None:
        print("Path to db required")

    input_folder = Path(args.input_folder)
    output_folder = Path(args.output_folder)
    display = args.display

    os.makedirs(output_folder, exist_ok=True)

    log_file_path = output_folder / "log.txt"
    log_file = open(log_file_path, 'w', encoding="utf-8")
    
    progress_bar = tqdm(list(input_folder.rglob("*.*")))
    for file in progress_bar:
        log_file.write(f"{file}")
        if file.suffix.lower() not in [".png", ".tiff", ".tif", ".jpeg", ".jpg", ".webp", ".bmp", ".img"]:
            log_file.write(f" is not an image\n")
            continue

        progress_bar.set_description(desc=file.name)
        file = file.relative_to(input_folder)
        with Image.open(input_folder / file) as image:

            boxes, score, faces = classifier.detect_image(image)
            if args.display:
                plot_rectangle(image, boxes, names)

            if boxes is None:
                log_file.write(f" has no face\n")
                output_subfolder="unsorted"
                current_distance=0.0
            else:
                names, distances = classifier.classify_faces(faces)
                output_subfolder = "+".join(set(names))

                if max(distances) > args.confidence_threshold:
                    log_file.write(f" rejected\n  with distance {max(distances)}")
                    output_subfolder="unsorted"
                    current_distance=0.0
            
            log_file.write(f"\t{output_subfolder}\n")
            output_file_name = Path(output_subfolder) / file
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
    argument_parser.add_argument("--update_db",  action="store_true",  help="If it exists, updates the db with all the images in the reference folder")


    args = argument_parser.parse_args()

    classifier = Classifier()

    if args.reference != None:
        print("Processing reference database...")
        classifier.make_ref_db(args.reference, update=args.update_db)
        output_folder = Path(args.output_folder)
        os.makedirs(output_folder, exist_ok=True)
        for name in classifier.embeddings["names"]:
            os.makedirs(output_folder/name, exist_ok=True)
    
    os.makedirs(output_folder/"unsorted", exist_ok=True)

    if args.mode == 'sort':
        sort(args, classifier)
    elif args.mode == 'crop':
        crop(args, classifier, crop_factor=args.crop_factor)

   