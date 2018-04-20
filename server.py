#!/usr/bin/env python2.7

import os
import subprocess
from flask import Flask, request, render_template, g, redirect, Response

tmpl_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
app = Flask(__name__, template_folder=tmpl_dir)

@app.route('/')
def index():
  print request.args
  return render_template("index.html")

@app.route('/generator', methods=['POST'])
def generator():
  text = request.form['input_text']
  print(text)
  text_file = open("/home/ubuntu/icml2016/scripts/cub_queries.txt", "w")
  text_file.write(text + "\n")
  text_file.close()
  print "start"
  subprocess.call("/home/ubuntu/icml2016/scripts/demo_cub.sh", shell=True)
  print "end"
  return render_template("index2.html")

if __name__ == "__main__":
  import click

  @click.command()
  @click.option('--debug', is_flag=True)
  @click.option('--threaded', is_flag=True)
  @click.argument('HOST', default='0.0.0.0')
  @click.argument('PORT', default=6007, type=int)
  def run(debug, threaded, host, port):
    """
    This function handles command line parameters.
    Run the server using:
        python server.py
    Show the help text using:
        python server.py --help
    """

    HOST, PORT = host, port
    print "running on %s:%d" % (HOST, PORT)
    app.run(host=HOST, port=PORT, debug=debug, threaded=threaded)


  run()
