
import shutil
import os
from argparse import ArgumentParser
from pathlib import Path
from copy import deepcopy
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt
from classifier import Classifier

fnt = ImageFont.truetype(R"arial.ttf", 80)
image_extensions = [".png", ".tiff", ".tif", ".jpeg", ".jpg", ".webp", ".bmp", ".img", ".jfif"]

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

def crop(args, classifier, crop_factor):
    classify = args.reference is not None
    input_folder = Path(args.input_folder)
    output_folder = Path(args.output_folder)
    display = args.display

    classifier.make_output_folders(output_folder)

    log_file_path = output_folder / "log.txt"
    log_file = open(log_file_path, 'w', encoding="utf-8")

    file_names = list(os.listdir(args.input_folder))

    print("Processing images...")
    progress_bar = tqdm(list(input_folder.rglob("*.*")))
    for file in progress_bar:
        if file.suffix.lower() not in image_extensions:
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
                else :
                    output_file_name = Path("unsorted") / output_file_name

                output_file_name = output_folder / output_file_name
                os.makedirs(output_file_name.parent, exist_ok=True)
                ref_face.save(output_file_name)

        log_file.flush()

def sort(args, classifier):  

    if args.reference is None:
        print("Path to db required")

    input_folder = Path(args.input_folder)
    output_folder = Path(args.output_folder)
    display = args.display

    classifier.make_output_folders(output_folder)

    log_file_path = output_folder / "log.txt"
    log_file = open(log_file_path, 'w', encoding="utf-8")
    
    progress_bar = tqdm(list(input_folder.rglob("*.*")))
    for file in progress_bar:
        log_file.write(f"{file}")
        if file.suffix.lower() not in image_extensions:
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
                names = sorted(set(names))
                output_subfolder = "+".join(names)

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

def main():
    argument_parser = ArgumentParser(prog="Face extractor")
    argument_parser.add_argument("-r", "--reference", help="Reference folder containing images of single faces with associated name")
    argument_parser.add_argument("-i", "--input_folder", required=False, help="Folder of images to extract faces from")
    argument_parser.add_argument("-o", "--output_folder", required=False, help="Folder containing the extracted faces")
    argument_parser.add_argument("-d", "--display", help="Display the detected faces on images", action='store_true')
    argument_parser.add_argument("-t", "--confidence_threshold", help="Maximum embedding distance to reference images", default=0.8, type=float)
    argument_parser.add_argument("-m", "--mode", required=True, choices=['crop', 'sort', 'update_db', 'make_db'])
    argument_parser.add_argument("--crop_factor", default=2.5, type=float, help="Factor for the crop size, 1 is strictly the face")

    args = argument_parser.parse_args()
    classifier = Classifier()

    if args.mode == 'sort' or args.mode == 'crop':
        classifier.load_reference_db(Path(args.reference))

    if args.mode == 'sort':
        if args.input_folder is None or args.output_folder is None:
            print("Input and output folders required for sorting")
        sort(args, classifier)

    elif args.mode == 'crop':
        if args.input_folder is None or args.output_folder is None:
            print("Input and output folders required for cropping")
        crop(args, classifier, crop_factor=args.crop_factor)

    elif args.mode == 'update_db':
        print("Processing reference database...")
        classifier.make_ref_db(args.reference, update=True)

    elif args.mode == 'make_db':
        print("Processing reference database...")
        classifier.make_ref_db(args.reference, update=False)

if __name__ == "__main__":
    main()
    
   