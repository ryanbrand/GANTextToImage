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

@app.route('/generator', methods=['POST'])
def generator():
    # save typed-in text
    text = request.form['input']
    filename = "/home/ubuntu/icml2016/scripts/cub_queries.txt"
    with open(filename, "a+") as f:
        f.write(text + "\n")

    # change the image in the background
    print('start')
    subprocess.call("./scripts/demo_cub.sh", shell=True)
    # TODO: should be subprocess.call(absolute/path/to/demo_cub.sh
    print('end')

    # TODO:  generalize this img_name to automatically figure out where the generated image is

    return render_template('show.html')

if __name__ == "__main__":
    HOST='0.0.0.0'
    PORT=6007
    app.run(host=HOST, port=PORT)








































