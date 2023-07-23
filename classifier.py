import os
from pathlib import Path

import torch
from tqdm import tqdm
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image, ImageFile
import csv

image_extensions = [".png", ".tiff", ".tif", ".jpeg", ".jpg", ".webp", ".bmp", ".img"]

class Classifier:
    def __init__(self):

        self.mtcnn = MTCNN(device="cpu", keep_all=True, select_largest=False)

        self.resnet = InceptionResnetV1(pretrained='vggface2', device="cpu").eval()
        self.embeddings = {}

        ImageFile.LOAD_TRUNCATED_IMAGES = True

    def load_reference_db(self, path : Path):
        if path.exists():
            print("Loading existing db")
            with open(path, 'rb') as db_file:
                self.embeddings = torch.load(db_file)
              
        else:
            print("Reference db doesn't exist")
            exit(-1)

    def make_ref_db(self, path : Path, update : bool = False):
        if update:
            db_file_path = Path(path)
            directory = db_file_path.parent
        else :
            directory = Path(path)
            db_file_path = directory / "db.pt"

        labels_file = open(directory / "labels.csv", mode = "r", newline='', encoding="utf-8")

        labels_reader = csv.reader(labels_file, delimiter=';')
        row_count = sum(1 for row in labels_reader)
        labels_file.seek(0)
        labels_reader = csv.reader(labels_file, delimiter=';')

        log_file = open(directory / "reference_log.txt", mode = 'w', encoding="utf-8")

        if update:
            self.load_reference_db(db_file_path)       

        first_name = True
        for row in tqdm(labels_reader, desc = "Folders", total=row_count):
            temp_embeddings = []
            current_dir = Path(directory / row[0])
            name = row[1]

            if not current_dir.is_dir():
                continue

            file_names = list(os.listdir(current_dir))

            for filename in tqdm(file_names, desc = f"{name} @ {row[0]}"):
                filename = Path(filename)
                log_file.write(f"{current_dir}/{filename}")

                image_path = directory / name / filename
                if image_path.suffix.lower() not in image_extensions:
                    log_file.write(f" is not an image\n")
                    continue

                with Image.open(image_path) as image:
                    _, _, faces = self.detect_image(image)
                    if faces is None :
                        log_file.write(f" : no face : skip\n")
                        continue
                    if faces.shape[0] > 1:
                        log_file.write(f" : {faces.shape[0]} faces : skip\n")
                        continue

                    embedding = self.resnet(faces.cpu()).cpu()
                    if update:
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

            if update or not first_name:
                self.embeddings["names"].extend(names)
                self.embeddings["embeddings"] = torch.cat((self.embeddings["embeddings"], emb_tensor))
            else :
                self.embeddings = {"names" : names, "embeddings" : emb_tensor}

            with open(db_file_path, 'wb') as db_file:
                torch.save(self.embeddings, db_file)

            first_name = False


    def detect_image(self, image : Image):
        if image.mode != 'RGB':
            image = image.convert('RGB')

        try:
            boxes, score = self.mtcnn.detect(image, landmarks=False)
        except:
            return None, None, None         

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

    def make_output_folders(self, output_folder : Path): 
        output_folder = Path(output_folder)
        os.makedirs(output_folder, exist_ok=True)
        for name in self.embeddings["names"]:
            os.makedirs(output_folder / name, exist_ok=True)
        
        os.makedirs(output_folder / "unsorted", exist_ok=True)