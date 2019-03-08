import numpy as np

def rgb2gray(rgb):
    '''
    Parameters
    ----------
    rgb : shape (height, width, 3)

    Returns
    -------
    gray : shape (height, width)
    '''
    return rgb[:, :, 0] * 0.299 + rgb[:, :, 1] * 0.578 + rgb[:, :, 2] * 0.114

def Hog(image, orientations, cells_per_block, pixels_per_cell, stride):
    '''
    Parameters
    ----------
    image : shape (height, width)
            gray image
    orientations : Number of orientation bins
    cells_per_block : Number of cells in each block
    pixels_per_cell : Size (in pixels) of a cell
    stride : stride size

    Returns
    -------
    feature : shape (n_feature,)
              HOG descriptor for the image
    '''
    gx = np.apply_along_axis(np.convolve, 1, image, [-1, 0, 1], mode='same')
    gy = np.apply_along_axis(np.convolve, 1, image.T, [1, 0, -1], mode='same').T
    gxy = np.sqrt(gx ** 2 + gy ** 2)
    alpha = np.rad2deg(np.arctan2(gy, gx)) % 360
    
    bins = alpha // (360 / orientations)

    image_height, image_width = image.shape
    cell_height, cell_width = pixels_per_cell

    cells_hog_height, cells_hog_width = (image_height + cell_height - 1) // cell_height, (image_width + cell_width - 1) // cell_width
    cells_hog = np.zeros((cells_hog_height, cells_hog_width, orientations))
    for i in range(cells_hog_height):
        for j in range(cells_hog_width):
            bin_cell = bins[cell_height*i:cell_height*i+cell_height, cell_width*j:cell_width*j+cell_width]
            g_cell = gxy[cell_height*i:cell_height*i+cell_height, cell_width*j:cell_width*j+cell_width]
            for k in range(orientations):
                cells_hog[i, j, k] = np.sum(g_cell[np.nonzero(bin_cell == k)])
    
    block_height, block_width = cells_per_block
    blocks_hog_height, blocks_hog_width = (cells_hog_height - block_height) // stride + 1, (cells_hog_width - block_width) // stride + 1
    blocks_hog = np.zeros((blocks_hog_height, blocks_hog_width, orientations * block_height * block_width))
    for i in range(blocks_hog_height):
        for j in range(blocks_hog_width):
            block = cells_hog[stride*i:stride*i+block_height:, stride*j:stride*j+block_width].ravel()
            blocks_hog[i, j] = block / (np.linalg.norm(block) + 1e-8)
    
    return blocks_hog.ravel()