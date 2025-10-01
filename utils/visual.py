from PIL import Image, ImageDraw, ImageFont

def make_grid_with_caption(images: list[Image.Image], caption: str, grid_size=(1, 5), padding=20, bg_color=(255, 255, 255)) -> Image.Image:
    """ 
    Отрисовываем сетку мзображений с подписью
    
    Args:
        images (list[Image.Image]): список изображений
        caption: (str): подпись
        grid_size (tuple[int, int]): расположение на сетке
        padding (int): обрамление
        bg_color (tuple): цвет фона
    
    Return:
        grid_img (list[Image.Image]): изображение сетки
    """
    rows, cols = grid_size
    img_w, img_h = images[0].size
    
    # Шрифт
    font = ImageFont.truetype("DejaVuSans.ttf", 100)
    
    # Определяем размер текста
    dummy_img = Image.new("RGB", (10, 10))
    draw = ImageDraw.Draw(dummy_img)
    bbox = draw.textbbox((0, 0), caption, font=font)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    
    # Размер итогового полотна
    grid_width = cols * img_w + (cols + 1) * padding
    grid_height = rows * img_h + (rows + 1) * padding + text_h + padding
    
    grid_img = Image.new("RGB", (grid_width, grid_height), bg_color)
    draw = ImageDraw.Draw(grid_img)
    
    # Рисуем картинки
    for idx, img in enumerate(images):
        r, c = divmod(idx, cols)
        x = padding + c * (img_w + padding)
        y = padding + r * (img_h + padding)
        grid_img.paste(img, (x, y))
    
    # Подпись снизу
    text_x = (grid_width - text_w) // 2
    text_y = grid_height - text_h - padding
    draw.text((text_x, text_y), caption, font=font, fill=(0, 0, 0))
    
    return grid_img