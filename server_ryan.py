#!/usr/bin/env python2.7

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
  text = request.form['input_text']
  print(text)
  print "start"
  cmd = ["/home/ubuntu/GANTextToImage/test_scripts/test_birds_server.sh", "\"" + text + "\""]
  #return_value = subprocess.check_call(cmd, shell=True)
  return_value = subprocess.call(cmd)
  
  #output = p.stdout.read()
  print return_value
  print "end"
  return render_template("index2.html")

if __name__ == "__main__":
    HOST='0.0.0.0'
    PORT=6007
    app.run(host=HOST, port=PORT)








































