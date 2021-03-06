'''
Problem Set 3 for Stanford University's course Computer Vision, From 3D Reconstruction to Recognition: https://web.stanford.edu/class/cs231a/syllabus.html

Adapted from the starter code provided by the university. 
- Implemented: carve() and form_initial_voxels()
- extended get_voxel_bounds(),
- and modified main().

'''

import numpy as np
import scipy.io as sio
import argparse
from camera import Camera
# make sure to pip install open3d for visualisations : http://www.open3d.org/docs/release/getting_started.html
import open3d as o3d

# A very simple, but useful method to take the difference between the
# first and second element (usually for 2D vectors)
def diff(x):
    return x[1] - x[0]


'''
FORM_INITIAL_VOXELS  create a basic grid of voxels ready for carving

Arguments:
    xlim - The limits of the x dimension given as [xmin xmax]

    ylim - The limits of the y dimension given as [ymin ymax]

    zlim - The limits of the z dimension given as [zmin zmax]

    num_voxels - The approximate number of voxels we desire in our grid

Returns:
    voxels - An ndarray of size (N, 3) where N is approximately equal the 
        num_voxels of voxel locations.

    voxel_size - The distance between the locations of adjacent voxels
        (a voxel is a cube)

Our initial voxels will create a rectangular prism defined by the x,y,z
limits. Each voxel will be a cube, so you'll have to compute the
approximate side-length (voxel_size) of these cubes, as well as how many
cubes you need to place in each dimension to get around the desired
number of voxel. This can be accomplished by first finding the total volume of
the voxel grid and dividing by the number of desired voxels. This will give an
approximate volume for each cubic voxel, which you can then use to find the 
side-length. The final "voxels" output should be a ndarray where every row is
the location of a voxel in 3D space.
'''
def form_initial_voxels(xlim, ylim, zlim, num_voxels):
    # TODO: Implement this method!
    # volume of voxel grid = x*y*z
    x = (xlim[-1]-xlim[0])
    y = (ylim[-1]-ylim[0])
    z = (zlim[-1]-zlim[0])

    total_volume = x*y*z
    voxel_volume = total_volume/num_voxels
    
    # find side length of one voxel
    voxel_size = voxel_volume**(1/3)
    
    # find number of voxels per dim
    unit_num_voxels = int(num_voxels**(1/3))

    xloc = np.linspace(xlim[0], xlim[0]+(unit_num_voxels*voxel_size), unit_num_voxels)
    yloc = np.linspace(ylim[0], ylim[0]+(unit_num_voxels*voxel_size), unit_num_voxels)
    zloc = np.linspace(zlim[0], zlim[0]+(unit_num_voxels*voxel_size), unit_num_voxels)

    xv,yv,zv = np.meshgrid(xloc,yloc,zloc)
    voxels = np.vstack((xv.flatten(), yv.flatten(), zv.flatten())).T
    return voxels, voxel_size




'''
GET_VOXEL_BOUNDS: Gives a nice bounding box in which the object will be carved
from. We feed these x/y/z limits into the construction of the inital voxel
cuboid. 

Arguments:
    cameras - The given data, which stores all the information
        associated with each camera (P, image, silhouettes, etc.)

    estimate_better_bounds - a flag that simply tells us whether to set tighter
        bounds. We can carve based on the silhouette we use.

    num_voxels - If estimating a better bound, the number of voxels needed for
        a quick carving.

Returns:
    xlim - The limits of the x dimension given as [xmin xmax]

    ylim - The limits of the y dimension given as [ymin ymax]

    zlim - The limits of the z dimension given as [zmin zmax]

The current method is to simply use the camera locations as the bounds. In the
section underneath the TODO, please implement a method to find tigther bounds
by doing a quick carving of the object on a grid with very few voxels. From this coarse carving,
we can determine tighter bounds. Of course, these bounds may be too strict, so we should have 
a buffer of one voxel_size around the carved object. 
'''
def get_voxel_bounds(cameras, estimate_better_bounds = False, num_voxels = 4000):
    camera_positions = np.vstack([c.T for c in cameras])
    xlim = [camera_positions[:,0].min(), camera_positions[:,0].max()]
    ylim = [camera_positions[:,1].min(), camera_positions[:,1].max()]
    zlim = [camera_positions[:,2].min(), camera_positions[:,2].max()]

    # For the zlim we need to see where each camera is looking. 
    camera_range = 0.6 * np.sqrt(diff( xlim )**2 + diff( ylim )**2)
    for c in cameras:
        viewpoint = c.T - camera_range * c.get_camera_direction()
        zlim[0] = min( zlim[0], viewpoint[2] )
        zlim[1] = max( zlim[1], viewpoint[2] )

    # Move the limits in a bit since the object must be inside the circle
    xlim = xlim + diff(xlim) / 4 * np.array([1, -1])
    ylim = ylim + diff(ylim) / 4 * np.array([1, -1])

    if estimate_better_bounds:
        # TODO: Implement this method!
        # perform quick carving
        # This part is simply to test forming the initial voxel grid
        voxels, voxel_size = form_initial_voxels(xlim, ylim, zlim, num_voxels)

        # Result after all carvings
        # print(len(cameras))
        for c in cameras:
            voxels = carve(voxels, c)
        
        # use this single voxel as potential better bounds. Add 1.5 units on either side.
        xlim = [voxels[:,0]- 1.5*voxel_size, voxels[:,0]+1.5*voxel_size]
        ylim = [voxels[:,1]- 1.5*voxel_size, voxels[:,1]+1.5*voxel_size]
        zlim = [voxels[:,2]- 1.5*voxel_size, voxels[:,2]+1.5*voxel_size]
        # raise Exception('Not Implemented Error')
    return xlim, ylim, zlim
    

'''
CARVE: carves away voxels that are not inside the silhouette contained in 
    the view of the camera. The resulting voxel array is returned.

Arguments:
    voxels - an Nx3 matrix where each row is the location of a cubic voxel

    camera - The camera we are using to carve the voxels with. Useful data
        stored in here are the "silhouette" matrix, "image", and the
        projection matrix "P". 

Returns:
    voxels - a subset of the argument passed that are inside the silhouette
'''
def carve(voxels, camera):
    # TODO: Implement this method!
    h,w = camera.silhouette.shape
    
    ones = np.ones((voxels.shape[0], 1))
    voxels_ext = np.append(voxels,ones, axis=1)    
    
    # project from 3D to 2D
    uvs = camera.P@voxels_ext.T

    # normalise
    uvs /= uvs[2,:]

    # is voxel within the bounds of image?
    within_x = np.logical_and(uvs[0, :] >= 0, uvs[0, :] < w)
    within_y = np.logical_and(uvs[1, :] >= 0, uvs[1, :] < h)

    within_frame = np.logical_and(within_x, within_y)
    
    # store in frame voxel indices
    idx = np.where(within_frame)[0]

    # drop z 
    sub_uvs = uvs[:2, idx]

    # type as int - to be used as indices
    sub_uvs = sub_uvs.astype(int)
    
    # resulting voxels where silhouette is 1
    results = (camera.silhouette[sub_uvs[1, :], sub_uvs[0, :]]==1)
    
    # only consider these voxels colliding with silhouettes from results above
    idx = idx[results]

    return voxels[idx,:]
    # raise Exception('Not Implemented Error')


'''
ESTIMATE_SILHOUETTE: Uses a very naive and color-specific heuristic to generate
the silhouette of an object

Arguments:
    im - The image containing a known object. An ndarray of size (H, W, C).

Returns:
    silhouette - An ndarray of size (H, W), where each pixel location is 0 or 1.
        If the (i,j) value is 0, then that pixel location in the original image 
        does not correspond to the object. If the (i,j) value is 1, then that
        that pixel location in the original image does correspond to the object.
'''
def estimate_silhouette(im):
    return np.logical_and(im[:,:,0] > im[:,:,2], im[:,:,0] > im[:,:,1] )


if __name__ == '__main__':
    estimate_better_bounds = True
    use_true_silhouette = True

    # load image data, silhouettes, and intrinsic matrices.
    frames = sio.loadmat('frames.mat')['frames'][0]
    
    # show sample images
    cameras = [Camera(x) for x in frames]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3))
    plt.suptitle('Sample views of images from dataset')
    for i, ax in enumerate(axs):
        ax.imshow(cameras[i].image)
    plt.show()

    fig, axs = plt.subplots(1, 2, figsize=(10, 3))
    plt.suptitle('Sample silhouettes from dataset')
    for i, ax in enumerate(axs):
        ax.imshow(cameras[i].silhouette)
    plt.show()

    # Generate the silhouettes based on a color heuristic
    # TODO: test other heuristics
    if not use_true_silhouette: # test true silhouettes be passed as input?
        for i, c in enumerate(cameras):
            c.true_silhouette = c.silhouette
            c.silhouette = estimate_silhouette(c.image)
            if i == 0:
                plt.figure()
                plt.subplot(121)
                plt.imshow(c.true_silhouette, cmap = 'gray')
                plt.title('True Silhouette')
                plt.subplot(122)
                plt.imshow(c.silhouette, cmap = 'gray')
                plt.title('Estimated Silhouette')
                plt.show()

    # Generate the voxel grid
    # You can reduce the number of voxels for faster debugging, but
    # make sure you use the full amount for your final solution
    num_voxels = 6e6
    xlim, ylim, zlim = get_voxel_bounds(cameras, estimate_better_bounds)

    # Form the initial voxel grid
    voxels, voxel_size = form_initial_voxels(xlim, ylim, zlim, num_voxels)

    # Test the initial carving
    # voxels = carve(voxels, cameras[0])

    # Result after all carvings
    for c in cameras:
        voxels = carve(voxels, c)  

    point_cloud= np.array(voxels)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    o3d.visualization.draw_geometries([pcd], width=960, height=540)   

    # TODO: refine and save mesh 