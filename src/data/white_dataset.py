import os
import shutil
import argparse
from rembg import remove
from PIL import Image

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def process_real_world_image(input_path, output_path):
    """
    Removes the background from an image and places the object on a white background.
    """
    img = Image.open(input_path).convert("RGBA")
    img_no_bg = remove(img)

    if img_no_bg.mode != "RGBA":
        img_no_bg = img_no_bg.convert("RGBA")

    bg = Image.new("RGB", img_no_bg.size, (255, 255, 255))
    alpha = img_no_bg.split()[3]

    bg.paste(img_no_bg, mask=alpha)
    bg.save(output_path)


def main(args):
    input_root = args.input_root
    output_root = args.output_root

    ensure_dir(output_root)

    for class_name in os.listdir(input_root):
        class_input_dir = os.path.join(input_root, class_name)
        if not os.path.isdir(class_input_dir):
            continue

        class_output_dir = os.path.join(output_root, class_name)
        ensure_dir(class_output_dir)

        for subfolder in ["default", "real_world"]:
            sub_input_dir = os.path.join(class_input_dir, subfolder)
            if not os.path.isdir(sub_input_dir):
                continue

            sub_output_dir = os.path.join(class_output_dir, subfolder)
            ensure_dir(sub_output_dir)

            for filename in os.listdir(sub_input_dir):
                if not filename.lower().endswith(IMAGE_EXTENSIONS):
                    continue

                input_path = os.path.join(sub_input_dir, filename)
                output_path = os.path.join(sub_output_dir, filename)

                if subfolder == "default":
                    shutil.copy2(input_path, output_path)
                    print(f"[COPY] {input_path} -> {output_path}")
                else:
                    try:
                        process_real_world_image(input_path, output_path)
                        print(f"[OK]   {input_path} -> {output_path}")
                    except Exception as e:
                        print(f"[ERR]  {input_path}: {e}")

    print("\n🎉 Done! Output dataset created in:", output_root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess a waste dataset by removing background from real_world images."
    )
    parser.add_argument("--input_root", type=str, required=True,
                        help="Path to the original dataset root")
    parser.add_argument("--output_root", type=str, required=True,
                        help="Path to the output dataset root")

    args = parser.parse_args()
    main(args)
