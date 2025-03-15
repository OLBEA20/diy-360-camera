from PIL import Image, ImageDraw


def generate_stitch_mask():
    radius = 1430
    size = (radius * 2, radius * 2)

    # Create a black background image (RGBA for transparency support)
    image = Image.new("RGB", size, "black")

    draw = ImageDraw.Draw(image)
    draw.ellipse((0, 0, size[0], size[1]), fill="white")
    draw.ellipse((20, 20, size[0] - 20, size[1] - 20), fill="black")

    image.save("stitch_mask.png")  # Save to a file


if __name__ == "__main__":
    generate_stitch_mask()
