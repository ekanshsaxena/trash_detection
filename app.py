import caffe
import os
from pylab import *
import sys
from PIL import Image
import numpy as np
#from flask_ngrok import run_with_ngrok
from flask import Flask,request, jsonify, send_file,render_template, send_from_directory

import cv2

# %matplotlib inline

mean_filename='garbnet_mean.binaryproto'
deploy_filename = 'deploy_garbnet.prototxt'
caffemodel_file = 'garbnet_fcn.caffemodel'

proto_data = open(mean_filename, "rb").read()
a = caffe.io.caffe_pb2.BlobProto.FromString(proto_data)
mean  = caffe.io.blobproto_to_array(a)[0]

net = caffe.Net(deploy_filename,caffemodel_file,caffe.TEST)

app = Flask(__name__)
#run_with_ngrok(app)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def index():
  return render_template("upload.html")

@app.route("/upload",methods=['POST'])
def upload():
  if request.method=="POST":

      target = os.path.join(APP_ROOT, 'image/')
      print(target)
      if not os.path.isdir(target):
              os.mkdir(target)
      else:
          print("Couldn't create upload directory: {}".format(target))
      print(request.files.getlist("file"))
      for upload in request.files.getlist("file"):
          print(upload)
          print("{} is the file name".format(upload.filename))
          filename = upload.filename
          destination = "/".join([target, filename])
          print ("Accept incoming file:", filename)
          print ("Save it to:", destination)
          upload.save(destination)
      images=[]
      names=[]
      folder='image'
      example_image = folder+'/'+filename
      input_image = Image.open(example_image)
      images.append(input_image)
      names.append(filename)
      """
      folder='image'
      imageNames=None
      images = []
      names = []
      files = os.listdir(folder)
      total = len(files)
      print ('Total images in folder', total, folder)
      for i in os.listdir(folder):
          try:
              if imageNames is None or i in imageNames:
                  example_image = folder+'/'+i
                  input_image = Image.open(example_image)
                  images.append(input_image)
                  names.append(i)
          except:
              pass
      """   




      #--------------------------------------------------------------------------------
      size=4
      thresh=0.999
      output_folder='trashed_image'
      for i in range(len(images)):
          w,h = images[i].size
          if w<h:
              test_image= images[i].resize((int(227*size),int((227*h*size)/w))) #227x227 is input for regular CNN
          else:
              test_image= images[i].resize((int((227*w*size)/h),int(227*size)))
        
          in_ = np.array(test_image,dtype = np.float32)
          in_ = in_[:,:,::-1]
          in_ -= np.array(mean.mean(1).mean(1))
          in_ = in_.transpose((2,0,1))

          net.blobs['data'].reshape(1,*in_.shape)
          net.blobs['data'].data[...] = in_
          net.forward()
          probMap =net.blobs['prob'].data[0,1]
          print (names[i]+'...',)
          if len(np.where(probMap>thresh)[0]) > 0:
                print ('Garbage!')
          else:
                print ('Not Garbage!')
          kernel = np.ones((6,6),np.uint8)
          wt,ht = test_image.size
          out_bn = np.zeros((ht,wt),dtype=uint8)

          for h in range(probMap.shape[0]):
                for k in range(probMap.shape[1]):
                      if probMap[h,k] > thresh:
                            x1 = h*62 #stride 2 at fc6_gb_conv equivalent to 62 pixels stride in input
                            y1 = k*62
                            for hoff in range(x1,227+x1):
                                  if hoff < out_bn.shape[0]:
                                      for koff in range(y1,227+y1):
                                          if koff < out_bn.shape[1]:
                                              out_bn[hoff,koff] = 255
          edge = cv2.Canny(out_bn,200,250)
          box = cv2.dilate(edge,kernel,iterations = 3)

          or_im_ar = np.array(test_image)
          or_im_ar[:,:,1] = (or_im_ar[:,:,1] | box)
          or_im_ar[:,:,2] = or_im_ar[:,:,2] * box + or_im_ar[:,:,2]
          or_im_ar[:,:,0] = or_im_ar[:,:,0] * box + or_im_ar[:,:,0]

          out_= Image.fromarray(or_im_ar)
          out_.save(output_folder + '/output_' + names[i])
          #specify 'input' folder containing images for prediction
          #images,names = gatherImages('input')
          #specify 'output' folder to store segmented predictions
          #getPredictionsFor(images,names,4,0.999,'output')
          #return send_file(output_folder + '/output_' + names[i])
          new_name='output_' + names[i]
          return render_template("complete_display_image.html", image_name=new_name)


@app.route('/trashed_image/<filename>')
def send_image(filename):
    return send_from_directory("trashed_image", filename)
    #return send_file("/home/ekansh/Desktop/trash/trashed_image"+filename)

if __name__ == "__main__":
    app.run()
