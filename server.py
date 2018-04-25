#!/usr/bin/env python2.7

'''
#TODO:  still writing to file?
        5.  I edited a different show.html and server.py somewhere
'''

import os
import subprocess
import utils
from flask import Flask, request, render_template, g, redirect, Response, send_from_directory

tmpl_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
app = Flask(__name__,
         template_folder=tmpl_dir,
         static_folder=static_dir)

@app.route('/')
def index():
    return render_template("show.html")

@app.route('/generator', methods=['POST','GET'])
def generator():
    # save typed-in text
    text = request.form['input']

    # change the image in the background
    print('start')
    cmd = ["/home/ubuntu/GANTextToImage/test_scripts/test_birds_server.sh",text]
    ret_val = subprocess.call(cmd)
    print('return val of '+cmd[0]+' '+cmd[1]+' is '+str(ret_val))
    print('end')

    # clean out old generated.jpg
    subprocess.call(['sudo', 'rm', '/home/ubuntu/GANTextToImage/static/generated.jpg'])
    # convert Bryan's png to generated.jpg
    subprocess.call(['convert', '/home/ubuntu/GANTextToImage/static/single_samples_256_sentence0.png', '/home/ubuntu/GANTextToImage/static/generated.jpg'])
    # NOTE: pngs don't show with my method for some reason.  I don't know why, but I care more about the code working than understanding every tiny detail

    return render_template('show.html')

if __name__ == "__main__":
    HOST='0.0.0.0'
    PORT=6007
    app.run(host=HOST, port=PORT)








































