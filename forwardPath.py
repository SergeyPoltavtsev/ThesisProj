import tensorflow as tf
import numpy as np
import os
import utils as ut
import models as models

layers_names = ['pool5', 'fc7']

def spectrumToArrays(device, params, img_path, output_folder, file_name):
    g = tf.Graph()
    content_image = ut.read_image(img_path)
    with g.device(device), g.as_default(), tf.Session(graph=g, config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        print "Load content values..."
        image = tf.constant(content_image)
        model = models.getModel(image, params)
        content_image_y_val = [sess.run(y_l) for y_l in model.y()]  # sess.run(y_l) is a constant numpy array

        for i in range(len(content_image_y_val)):
            output_path = output_folder + layers_names[i] + "/"
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            np.save(output_path + file_name, content_image_y_val[i])   

    #cleanUp
    del image
    del content_image
    del model
    del content_image_y_val
    del g
    
def getPath(output_folder, file_name, chunk):
    split1 = file_name.split('.', 1)
    file_name = split1[0]

    file_name_split = file_name.split('-', 1)
    speaker = file_name_split[0]


    output_path = output_folder + speaker + "/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
      
    name = file_name.split('.', 1)[0] + "_" + chunk.split('.', 1)[0]
    
    return (output_path, name)