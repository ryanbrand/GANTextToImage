import os
from random import shuffle

if __name__=='__main__':
    os.chdir('/home/n/Documents/code/python_sandbox/Drori_GANs_UI/deeplearn_flask_server/static')
    jpgs = os.listdir()
    shuffle(jpgs)

    for filename in jpgs:
        if 'jpg' in filename:
            if 'tree' not in filename:
                print(filename)
                os.system('cp -f '+filename+' /home/n/Documents/code/python_sandbox/Drori_GANs_UI/deeplearn_flask_server/static/tree.jpg')
                break

