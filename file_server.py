#! /usr/bin/python

"""Extend Python's built in HTTP server
based on https://floatingoctothorpe.uk/2017/receiving-files-over-http-with-python.html I think.

This server will handle GET (with no auth), PUT, and DELETE. 
"""
import os
import time

from absl import flags
from absl import app

FLAGS = flags.FLAGS
flags.DEFINE_integer('port', help='address ip:port of experiment manager server',
                    default=8072)
flags.DEFINE_string('password', help='address ip:port of experiment manager server',
                    default='very_secure')

hostName = "0.0.0.0"

try:
  import http.server as server
  from http.server import BaseHTTPRequestHandler, HTTPServer
except ImportError:
  # Handle Python 2.x
  import SimpleHTTPServer as server

class HTTPRequestHandler(server.SimpleHTTPRequestHandler):
  """Extend SimpleHTTPRequestHandler to handle PUT and DELETE with some auth"""

  def MagicWord(self):
    if "magicpass" not in self.headers or self.headers['magicpass']!=FLAGS.password:
      self.send_response(403, 'Go away, jeez.')
      self.end_headers()
      return False
    return True
    
  def Overwrite(self):
    return "overwrite" in self.headers and self.headers['overwrite']=="true"
        
  def do_PUT(self):
    if not self.MagicWord():
      return
    """Save a file following a HTTP PUT request"""
    filename = self.path

    # Don't overwrite files
    if os.path.exists(filename) and not self.Overwrite():
      self.send_response(409, 'File Exists')
      self.end_headers()
      reply_body = '"%s" already exists\n' % filename
      self.wfile.write(reply_body.encode('utf-8'))
      return

    file_length = int(self.headers['Content-Length'])

    if not os.path.exists(os.path.dirname(filename)):
      os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as output_file:
      output_file.write(self.rfile.read(file_length))
    self.send_response(201, 'Created')
    self.end_headers()
    reply_body = 'Saved "%s"\n' % filename
    self.wfile.write(reply_body.encode('utf-8'))

  def do_DELETE(self):
    if not self.MagicWord():
      return
    filename = self.path

    if os.path.exists(filename):
      os.remove(filename)
      self.send_response(200, 'Deleted')
      self.end_headers()
    else:
      self.send_response(404, 'missing')
      self.end_headers()

def main(argv):
  myServer = HTTPServer((hostName, FLAGS.port), HTTPRequestHandler)
  print(time.asctime(), "Server Starts - %s:%s" % (hostName, FLAGS.port))
  try:
    myServer.serve_forever()
  except KeyboardInterrupt:
    pass

  myServer.server_close()
  print(time.asctime(), "Server Stops - %s:%s" % (hostName, FLAGS.port))
            
if __name__ == '__main__':
   app.run(main)
