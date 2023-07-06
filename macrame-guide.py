import cv2
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
def plot_image(img, figsize=(15, 9), show_grid=False, offset=0, save_filename=None):
    fig = plt.figure(figsize=figsize)
    
    if show_grid:
        ax = fig.add_subplot(111)
        ax.margins(x=0, y=0)

        grid_points = [i - 0.5 + offset for i in range(img.shape[0])]
        ax.xaxis.set_ticks(grid_points)
        ax.yaxis.set_ticks(grid_points)
        ax.grid(True)
    
    plt.imshow(img)

    if save_filename:
        plt.savefig(f'{save_filename}.jpg')

def find_nearest_color(pallete, color):
    """
    Given a pallete and a color C, find the color in pallete which is closer (in terms of euclidean distance) to C,
    using RGB pixel values.
    
    Args:
        pallete (list): List of colors (RGB) that compose the pallete
        color (tuple): RGB color representation
    
    Returns:
        tuple: Color (RGB) in pallete which is closer to color given as parameter
    
    """
    min_dist = 2 ** 16
    ans = (0, 0, 0)
    b = np.array(color)
    
    for c in pallete:
        a = np.array(c)
        dist_ab = np.linalg.norm(a - b)
        if dist_ab < min_dist:
            
            min_dist = dist_ab
            ans = c
            
    return ans

def resize_image_and_match_colors_pallete(img, size, pallete):
    """
    Resize image using size parameter as dimensions, after resizing match each pixel of the image to the nearest color (euclidian distance) in pallete
    Important: If you are enlarging the image, you should prefer to use INTER_LINEAR or INTER_CUBIC interpolation. If you are shrinking the image (this case!), you should prefer to use INTER_AREA interpolation.
    https://stackoverflow.com/questions/23853632/which-kind-of-interpolation-best-for-resizing-image
    
    Args:
        img (numpy.ndarray): Image as a 3-dimensional array (RGB)
        size (tuple): Size (height, widht) to downscale the image
        pallete (list): List of colors (RGB) that compose the pallete to match colors after resize
    
    Returns:
        img (numpy.ndarray): Image resized and using only colors that are in pallete as a 3-dimensional array (RGB)
    """
        
    img_resized = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    H, W, _ = img_resized.shape
    
    for i in range(H):
        for j in range(W):
            b, g, r = img_resized[i, j]
            nearest_color = find_nearest_color(pallete, (b, g, r))
            img_resized[i, j] = nearest_color
    
    return img_resized

def get_primary_masks(img, pallete):
    """
    Create multiple masks, one for each color present in image. Each masks only have two colors, the color of interest for that mask
    and the white color (255, 255, 255) for pixels with a different color
    
    Args:
        img (numpy.ndarray): Image as a 3-dimensional array (RGB)
        pallete (list): List of colors (RGB) that compose the pallete
    
    Returns:
        list: List of tuples (mask, number of pixels that are present in mask, RGB color used to build the mask)
    """
    H, W, _ = img.shape
    output = list()
    
    for p in pallete:
        b_, g_, r_ = p
        img_ = np.zeros((H, W, 3), np.uint8)
        number_of_pixels = 0
        
        for i in range(0, H):
            for j in range(0, W):
                if np.all(img[i, j] == p):
                    img_[i, j] = p
                    number_of_pixels += 1
                else:
                    img_[i, j] = [255, 255, 255]
                    
        output.append((img_, number_of_pixels, p))

    output = sorted(output, key=lambda x: x[1], reverse=True)
        
    return output

def build_component(img, mask_boolean, position, color):
    """
    Recursively build a component that is stored in mask_boolean 2d-array.
    Each time pixel (x, y) of the image matches the color of the component being built, four new calls (one for each position [up, down, left, right]) are made.
    If mask_boolean in x, y coordinate is True, means that pixel has already been included in component and shouldn't be called anymore. 
    
    Args:
        img (numpy.ndarray): Image as a 3-dimensional array (RGB)
        mask_boolean (numpy.ndarray): Mask as a 2-dimensional array (HxW) used to store component
        position (tuple): Coordinate (x, y) that determine the position being executed
        color (tuple): RGB color representation
    
    Returns:
        void: The resultant component is stored in mask_boolean variable
    """
    x, y = position # (x=row, y=column)
    
    # if x or y are out of bounds, return
    if x < 0 or y < 0 or x == img.shape[0] or y == img.shape[1]:
        return
    
    # if x, y coordinate was already included, return
    if mask_boolean[position]:
        return
    
    if np.all(img[position] == color):
        mask_boolean[position] = 1
        
        build_component(img, mask_boolean, (x-1, y), color)
        build_component(img, mask_boolean, (x+1, y), color)
        build_component(img, mask_boolean, (x, y-1), color)
        build_component(img, mask_boolean, (x, y+1), color)
        
    return

def get_components(mask, color):
    """
    Get all components that forms a mask of color = c. This functions makes uses of the recursive function build_component.
    
    Args:
        mask (numpy.ndarray): Mask of color = c
        color (tuple): RGB color representation
        
    Returns:
        list: List of components that compose the mask
    """
    H, W, _ = mask.shape
    components = [] # list that store each mask's components
       
    # create a list to store all pixels coordinates that aren't covered by any component
    remaining_pixels = []   
    
    # initially all pixels that have color=c are included in the list
    for i in range(H):
        for j in range(W):
            if np.all(mask[i, j] == color):
                remaining_pixels.append((i, j))
    
    # the remaining pixels list must be emptied to guarantee that all components of the mask are created
    while remaining_pixels != []:
        component = np.zeros((H, W, 1), np.uint8) # create an auxiliar mask to store component
        
        # each time the recursivelly calls of this function ends, a new components has been created
        build_component(mask, component, position=remaining_pixels[0], color=color)
        components.append(component)
                
        # pixels that are already visited should be removed from list
        for i in range(H):
            for j in range(W):
                if component[i, j]:
                    remaining_pixels.remove((i, j))
    
    return components

def find_initial_component(components):
    """
    Returns index of initial component. 
    The initial component is that have pixel in position (0, 0) with value different of white (255, 255, 255)
    
    Args:
        components (list): List of masks (numpy.ndarray)
    
    Returns:
        int: Index of initial mask, returns -1 if there's no possible initial mask in list
    """
    for i, component in enumerate(components):
        if np.all(component[0, 0] != (255, 255, 255)):
            return i
    
    return -1

def get_sub_component(component, current_answer):
    H, W, _ = component.shape
    sub_component = np.zeros((H, W, 1), np.uint8)

    for i in range(H):
        for j in range(W):
            if component[i, j]:
                if i-1 < 0:
                    sub_component[i, j] = 1
                    current_answer[i, j] = 1
                
                else:
                    if current_answer[i-1, j]:
                        sub_component[i, j] = 1
                        current_answer[i, j] = 1
                    
                    else:
                        return sub_component, current_answer
    
    return sub_component, current_answer

def get_remaining_part_of_sub_component(component, sub_component):
    H, W, _ = component.shape
    remaining_component = np.zeros((H, W, 1), np.uint8)
    exists_remaining_component = False
    
    for i in range(H):
        for j in range(W):
            if component[i, j] and not sub_component[i, j]:
                remaining_component[i, j] = 1
                exists_remaining_component = True
    
    return remaining_component, exists_remaining_component

def get_idx_component_by_position(components, position):
    for i, component in enumerate(components):
        if component[position]:
            return i
        
    return -1

def get_most_upper_left_not_setted_pixel(mask):
    H, W, _ = mask.shape
    
    for i in range(H):
        for j in range(W):
            if not mask[i, j]:
                return (i, j)
            
    return None
def create_rgb_mask(binary_mask, color, default_color=[255, 255, 255]):
    H, W, _ = mask.shape
    img_ = np.zeros((H, W, 3), np.uint8)

    for i in range(H):
        for j in range(W):
            if binary_mask[i, j] == 1:
                img_[i, j] = color
            else:
                img_[i, j] = default_color

    return img_
def get_pixel_art_img(filename, img_size, pallete):
    # read img
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # resize and match colors pallete
    img_resized = resize_image_and_match_colors_pallete(img, img_size, pallete)
    
    return img_resized  
(1) Get pixel art image
filename = 'samples/meisje_met_de_parel_cropped.jpg'
# filename = 'samples/vangogh_cropped.jpeg'
pixel_art_img_size = (10, 10)
pallete = [[255, 255, 153], [51, 153, 255], [0, 0, 0], [255, 178, 102]] # define colors in pallete

img = get_pixel_art_img(filename, pixel_art_img_size, pallete)
print(img.shape)

print('(1) Resized image with pallete colors:')
plot_image(img, show_grid=True)

(2) Get primary masks
# Create all primary masks
masks_outputs = get_primary_masks(img, pallete)
masks = [mask for mask, _, _ in masks_outputs]

print('(2) Primary masks:')

for mask, number_of_pixels, color in masks_outputs:
    print(f'\tColor: {color} - #Pixels: {number_of_pixels}')
    plot_image(mask, figsize=(10, 5))

(3) Get mask's components
def create_rgb_mask(binary_mask, color, default_color=[255, 255, 255]):
    H, W, _ = mask.shape
    img_ = np.zeros((H, W, 3), np.uint8)

    for i in range(H):
        for j in range(W):
            if binary_mask[i, j] == 1:
                img_[i, j] = color
            else:
                img_[i, j] = default_color

    return img_
all_components = []

cc = 0

for mask, _, color in masks_outputs:
    components = get_components(mask, color)
    
    print(f'Component of color={color}')
    
    plot_image(mask, figsize=(10, 5), show_grid=True, offset=1, save_filename=f'outputs/mask_{cc}.jpg')
    plt.show()

    ccc = 0
    
    for component in components:
        all_components.append(component)
        # component_ = cv2.bitwise_and(img, img, mask=component)
        # plt.imshow(component)
        # plt.show()
        component_rgb = create_rgb_mask(component, color)
        plot_image(component_rgb, show_grid=True, save_filename=f'outputs/component_{cc}_{ccc}.jpg')

        ccc += 1

    cc += 1

(4) Compose guide using components, according to macrame rules
def colorize_component(component, pixel_art_img, default_color=[255, 255, 255]):
    H, W, _ = component.shape
    img_ = np.zeros((H, W, 3), np.uint8)

    for i in range(H):
        for j in range(W):
            if component[i, j] == 1:
                img_[i, j] = pixel_art_img[i, j]
            else:
                img_[i, j] = default_color

    return img_
def f(components):
    H, W, _ = components[0].shape
    answer = np.zeros((H, W, 1), np.uint8)
    answer_sequence = []
    
    # get initial component (component that have a setted color in position (0, 0))
    idx = find_initial_component(components)
    print(idx)
    
    while components != []:
        print(len(components))
        # since component[idx] is gonna be processed, remove it from list of components
        current_component = components.pop(idx)
        
        print('Current component:')
        current_component_rgb = colorize_component(current_component, img)
        plot_image(current_component_rgb, show_grid=True)
        plt.show()
        
        # get sub_component (part of component that can be included in answer)
        sub_component, answer = get_sub_component(current_component, answer)
        
        print('Sub-component:')
        sub_component_rgb = colorize_component(sub_component, img)
        plot_image(sub_component_rgb, show_grid=True)
        plt.show()
        
        # include sub_component in answer
        answer_sequence.append(sub_component)
        
        print('Current answer:')
        answer_rgb = colorize_component(answer, img)
        plot_image(answer_rgb, show_grid=True)
        plt.show()
        
        # verify if exists remaining sub component
        remaing_component, exists_remaining_component = get_remaining_part_of_sub_component(current_component, sub_component)
        
        # if so, included in list of components to be processed in laterly
        if exists_remaining_component:
            print('Current remaining sub-component:')
            remaing_component_rgb = colorize_component(remaing_component, img)
            plot_image(remaing_component_rgb, show_grid=True)
            plt.show()
            components.append(remaing_component)
        
        # discover which position must be used in sequence
        position = get_most_upper_left_not_setted_pixel(answer)
        
        if position:
            # get idx of next component according to position above
            idx = get_idx_component_by_position(components, position)
            print('Next component:')
            next_component_rgb = colorize_component(components[idx], img)
            plot_image(next_component_rgb, show_grid=True)
            plt.show()

#         break
        
    return answer_sequence
all_components = []

for mask, _, color in masks_outputs:
    components = get_components(mask, color)
    
    for component in components:
        all_components.append(component)

answer = f(all_components)

