from helper import *

def main():

    # call image to use
    my_image = 'data/trap.jpg'

    # load yolov3 model
    model = load_model('model.h5')

    # define parameters for the model
    image_size = (416, 416)  # expected input shape for the model   (default 416x416)
    anchors = [[116,90, 156,198, 373,326], [30,61, 62,45, 59,119], [10,13, 16,30, 33,23]]
    class_threshold = 0.4 # for trap picture

    detection(my_image, model, anchors, class_threshold, image_size)


if __name__ == '__main__':
    main()

