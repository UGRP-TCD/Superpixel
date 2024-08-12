import matlab.engine
import numpy as np
from PIL import Image
import time


if __name__ == "__main__":
    st_engine = time.time()
    eng = matlab.engine.start_matlab()
    eng.cd(r'C:/Users/ksw/Desktop/UGRP/Superpixel-main/matlab', nargout=0)
    
    img_path = 'C:/Users/ksw/Desktop/UGRP/Superpixel-main/matlab/12003.jpg'
    img = Image.open(img_path)

    down_factor = 0.5
    down_w = int(img.width * down_factor)
    down_h = int(img.height * down_factor)
    down_img = np.array(img.resize((down_w, down_h), Image.Resampling.LANCZOS), dtype=np.double)

    print('Original img shape: ', np.array(img).shape)
    print('Downsampling img shape: ', down_img.shape)

    st_func = time.time()
    pic = np.array(eng.main(down_img))
    et = time.time()

    print('Total Time: %0.2fs' %(et - st_engine))
    print('Func Time: %0.2fs' %(et - st_func))
    
    image = Image.fromarray(pic)
    image.show()