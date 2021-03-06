import numpy as np

LEFT =1
RIGHT = 2
STRAIGHT = 0
ACCELERATE =3
BRAKE = 4
ACCELERATE_LEFT = 5
ACCELERATE_RIGHT = 6
BRAKE_LEFT = 7
BRAKE_RIGHT = 8

def one_hot(labels):
    """
    this creates a one hot encoding from a flat vector:
    i.e. given y = [0,2,1]
     it creates y_one_hot = [[1,0,0], [0,0,1], [0,1,0]]
    """
    classes = np.unique(labels)
    n_classes = 9 #classes.size
    one_hot_labels = np.zeros(labels.shape + (n_classes,))
    for c in classes:
        one_hot_labels[labels == c, c] = 1.0
    return one_hot_labels

def rgb2gray(rgb):
    """ 
    this method converts rgb images to grayscale.
    """
    gray = np.dot(rgb[...,:3], [0.2125, 0.7154, 0.0721])
    return gray.astype('float32') 


def action_to_id(a):
    """ 
    this method discretizes the actions.
    Important: this method only works if you recorded data pressing only one key at a time!
    """
    if all(a == [-1.0, 0.0, 0.0]): return LEFT               # LEFT: 1
    elif all(a == [1.0, 0.0, 0.0]): return RIGHT             # RIGHT: 2
    elif all(a == [0.0, 1.0, 0.0]): return ACCELERATE        # ACCELERATE: 3
    elif all(a == [0.0, 0.0, 0.2]): return BRAKE             # BRAKE: 4
    elif all(a == [-1.0, 1.0, 0.0]): return ACCELERATE_LEFT            # ACCELERATE_LEFT : 5
    elif all(a == [1.0, 1.0, 0.0]): return ACCELERATE_RIGHT            # ACCELERATE_RIGHT : 6
    elif all(a == [-1.0, 0.0, 0.2]): return BRAKE_LEFT             # BRAKE_LEFT: 7
    elif all(a == [1.0, 0.0, 0.2]): return BRAKE_RIGHT             # BRAKE_RIGHT: 8
    else:       
        return STRAIGHT                                      # STRAIGHT = 0
    
def unhot_to_action(y):
    a_index = np.argmax(y)
    a = [0.0, 0.0, 0.0, 0.0]
    if a_index in [LEFT,  ACCELERATE_LEFT,  BRAKE_LEFT]: a[0] = -1.0
    if a_index in [RIGHT,  ACCELERATE_RIGHT,  BRAKE_RIGHT]: a[0] = 1.0
    if a_index in [ACCELERATE,  ACCELERATE_LEFT,  ACCELERATE_RIGHT]: a[1] = 1.0
    if a_index in [BRAKE,  BRAKE_LEFT,  BRAKE_RIGHT]: a[2] = 0.2
    
    return a
