import re
from django.shortcuts import render
from django.http import HttpResponse
from . forms import RequestDemo, ImageHorizontal
from . models import DemoRequests, OpticaiData
import string
import random
import numpy as np
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.relpath(__file__)))
import torch
from torchvision import transforms
import warnings
import cv2
import numpy as np
import time
from torch import nn
from torch.autograd import Variable
from PIL import Image
import smtplib, ssl
from email.message import EmailMessage
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
warnings.filterwarnings("ignore")

# Create your views here.

def request_demonstration(request):

    request_demo = RequestDemo()

    context = {'request_demo': request_demo}

    if request.method == 'POST':

        request_demo = RequestDemo(request.POST)

        if request_demo.is_valid():
            email = request.POST['email']
            letters = string.ascii_uppercase
            email_id = str(np.random.randint(1000000)).join(random.choice(letters) for i in range(2))
            new_data = DemoRequests(email_id=email_id, email=email)
            new_data.save()
           #send access email 
            message = MIMEMultipart()
            message['Subject'] = 'OpticAI Demo Access'
            TEXT = u'<a href="http://127.0.0.1:8000/opticai-demo#' +str(email_id).lower()+'%'+u'">Click here</a> to access OpticAI Demo. <br><br><br><br> Thank you!<br><br><b>NileAGI Team</b> <br> Making Sense of AI for Humanity '
            body_content = TEXT
            message.attach(MIMEText(body_content, "html"))
            msg_body = message.as_string()
            smtpserver = None
            fromaddr = None
            from_password = None
            email_split = str(email).split('@')
            if email_split[1] != 'nileagi':
                smtpserver = smtplib.SMTP('smtp.gmail.com', 587)
                fromaddr = 'zephania@beem.africa'
                from_password = '@Nsom4zrts@Beem'
                # fromaddr = os.environ['FROM_GMAIL']
                # from_password = os.environ['FROM_PASSWORD_GMAIL']
            if email_split[1] == 'nileagi':
                smtpserver = smtplib.SMTP('mail.nileagi.com', 25)
                fromaddr = 'info@nileagi.com'
                from_password = 'info@nileagi'
                # fromaddr = os.environ['FROM_MAIL']
                # from_password = os.environ['FROM_PASSWORD_MAIL']
            # smtpserver.set_debuglevel(1)

            toaddr = email
            print(toaddr)
            # Create a secure SSL context
            context = ssl.create_default_context()
            success_msg = None
            fail_msg = None
            try:
                smtpserver.ehlo()
                smtpserver.starttls(context=context)
                smtpserver.ehlo()
                smtpserver.login(fromaddr, from_password)
                smtpserver.sendmail(fromaddr, toaddr, msg_body)
                success_msg = 'Request successfully submitted, check your email. Thank you!'
            except smtplib.SMTPException as e:
                fail_msg = 'Request failed, try again. Thank you!'
                print(e)
                # raise smtplib.SMTPException("Error sending email")
            context = {'request_demo': request_demo, 'success_msg': success_msg, 'fail_msg': fail_msg}
            return render(request, 'opticai/pages/request_demo_response.html', context=context) 
        else:
            return render(request, 'opticai/pages/request_demo.html', context=context)  
    else:
        return render(request, 'opticai/pages/request_demo.html', context=context)

def about_opticai(request):
    return render(request, 'opticai/pages/about-opticai.html')


def opticai(request):

    image_horizontal = ImageHorizontal()

    context = {'image_horizontal': image_horizontal}

    if request.method == 'POST' and request.FILES['image_file']:

        image_horizontal = ImageHorizontal(request.POST, request.FILES)

        if image_horizontal.is_valid():

                image_path = request.FILES['image_file']

                image_name = image_path.name

                # print('Image name: ', image_name)

                image_name = str(image_name).replace(' ', '_')

                if image_name.endswith(".jpg") or image_name.endswith(".png"):
                    import string
                    letters = string.ascii_uppercase
                    import random
                    image_id = str(np.random.randint(1000000)).join(random.choice(letters) for i in range(2))
                    new_file = OpticaiData(image_id=image_id, filepaths=image_path, filename=image_name)

                    new_file.save()

                    # import all import libraries

                    # create a dataset transformer
                    transformer_input = transforms.Compose([
                        transforms.Resize((224,224)),
                        transforms.CenterCrop((224,224)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

                    """load image, returns tensor"""
                    image_path=os.path.join(BASE_DIR,'media/images/'+image_name)
                    # print("Image path: ", image_path)
                    image = Image.open(image_path)
                    # img = cv2.imread(image_path)
                    # cv2.imshow("", img)
                    # cv2.waitKey(0) 
                    # cv2.destroyAllWindows() 
                    image = transformer_input(image).float()
                    image = Variable(image, requires_grad=True)

                    since_time = time.time();
                    # load the saved model
                    loaded_model = torch.load(os.path.join(BASE_DIR,'models/model_resnet152.pth'), map_location='cpu')
                    loaded_model.eval()

                    output_single = loaded_model(image.view(1, 3, 224, 224))
                    output_single_probability = torch.softmax(output_single, dim=1)
                    prediction_proba,prediction=torch.max(output_single_probability, 1)
                    
                    # labels dictionary
                    labels_dict = {'grade_0': 0, 'grade_1': 1, 'grade_2': 2, 'grade_3': 3, 'grade_4': 4}
                    probabilities = output_single_probability.detach().numpy()[0]

                    prob=[]
                    for i in probabilities:
                        prob.append(i)

                    initial_pred = ''
                    pred = ''
                    pred_index = 0
                    for class_name, class_index in labels_dict.items():  # for name, age in dictionary.iteritems():  (for Python 2.x)
                        if class_index == prediction:
                            initial_pred = class_name
                            if prediction_proba.item() >= 0:
                                 pred = class_name    
                                 pred_index = class_index
                            else:
                                pred = 'Undetermined'      
                    # print("Probabilities: ", probabilities)  
                    # print("Initial Prediction: ", initial_pred)       
                    # print("Class: ", pred, " | Probabilty: ", prediction_proba.item() )
                    time_elapse = time.time() - since_time
                    # print("Time elapse: ", time_elapse)
                    # getting all the objects of hotel.
                    image = OpticaiData.objects.get(image_id=image_id)
                    context = {'image_horizontal': image_horizontal,'prediction':pred, 'proba': prediction_proba.item(),
                    'pred_index': pred_index, 'probabilities': prob, 'image_path':image_name,'image_name':image_name.split('.')[-2].upper(), 'image_id':image_id, 'image':image}
                    return render(request, 'opticai/pages/opticai-demo.html', context=context)    

                else:

                    format_message = "Unsupported format, supported format are .png and .jpg "

                    context = {'image_horizontal': image_horizontal,'format_massage': format_message}

                    return render(request, 'opticai/pages/opticai-demo.html', context=context)

        else:
            return render(request, template_name="opticai/pages/opticai-demo.html", context=context)

    return render(request, template_name="opticai/pages/opticai-demo.html", context=context)