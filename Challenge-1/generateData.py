from scipy import misc, ndimage
import numpy as np



def generate_data(N_samples = 5000, exist_prob = 0.5):

    np.random.seed(seed = 0)

    N_obj = 40
    N_fig = 100

    image = misc.imread('jungle.jpg')
    obj   = misc.imread('lego.png')

    obj   = misc.imresize( obj, (N_obj, N_obj) )
    h,w,_ = image.shape

    x = np.zeros((N_fig, N_fig, N_samples))
    y = np.zeros((N_samples, 4))


    for i in range(0, N_samples) :

        h_obj_b = int(np.random.uniform(0, h - N_fig ))
        w_obj_b = int(np.random.uniform(0, w - N_fig))
        subimage = image[h_obj_b:h_obj_b+N_fig,w_obj_b:w_obj_b+N_fig,:].astype(np.float64)

        if np.random.uniform(0,1) < exist_prob :

            rotation = np.random.uniform(0, 2*np.pi)
            rotate_obj = ndimage.rotate(obj, rotation / np.pi * 360)

            h_obj, w_obj, _ = rotate_obj.shape

            mask = rotate_obj[:,:,3:4]/255
            mask = np.stack( [mask,mask,mask], axis = 2)[:,:,:,0]

            h_b = int(np.random.uniform(0, N_fig - h_obj))
            w_b = int(np.random.uniform(0, N_fig - w_obj))

            subimage[h_b:h_b+h_obj,w_b:w_b+w_obj,:] *= (1-mask)
            subimage[h_b:h_b+h_obj,w_b:w_b+w_obj,:] += rotate_obj[:,:,:3] * mask

            y[i,0] = 1
            y[i,1] = rotation
            y[i,2] = h_b + h_obj/2
            y[i,3] = w_b + w_obj/2

        x[:,:,i] = (np.sum(subimage, axis = 2)/3).astype(np.uint8)


    return x,y



x,y = generate_data(N_samples = 5000, exist_prob = 0.5)
np.save('lego_class_labels',y)
np.save('lego_class_data',x)

x,y = generate_data(N_samples = 5000, exist_prob = 1)

np.save('lego_est_labels',y)
np.save('lego_est_data',x)
