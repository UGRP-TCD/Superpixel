import matlab.engine
import numpy as np
from PIL import Image
import time


if __name__ == "__main__":
    st_engine = time.time()
    eng = matlab.engine.start_matlab()
    eng.cd(r'C:/Users/ksw/Desktop/UGRP/Superpixel-main/matlab', nargout=0)
    st_func = time.time()

    img_path = '12003.jpg'
    pic = np.array(eng.main(img_path))
    et = time.time()

    print('Total Time: %0.2fs' %(et - st_engine))
    print('Func Time: %0.2fs' %(et - st_func))
    
    image = Image.fromarray(pic)
    image.show()