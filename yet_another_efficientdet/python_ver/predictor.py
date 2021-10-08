import numpy as np
from utils.utils import preprocess, invert_affine, postprocess, STANDARD_COLORS, standard_to_bgr, get_index_label, plot_one_box

class Predictor:
    def __init__(self, openvinonet, image_size, input, output):
        self.openvinonet = openvinonet
        self.image_size = image_size
        self.input = input
        self.output = output
    def predict(self, image):
        #imageの処理あってる?
        ori_img, framed_img, framed_meta = preprocess(image)
        res = self.openvinonet.infer(inputs={self.input: framed_img})
        #TODO res合わせる
        return ori_img