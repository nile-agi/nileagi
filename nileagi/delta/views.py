from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import SearchData, FileData, SearchHistory
from .serializers import SearchDataSerializer, FileDataSerializer
from .rag_handlers import search_bot, pure_llm_local, pure_llm_api
from .vision_search import ollama_call_image_to_text
from .text_search import ollama_call_text_to_text
from django.http import JsonResponse
from django.http import HttpResponseServerError, HttpResponse
import os
import json
from django.shortcuts import render

BASE_DIR = os.path.dirname(os.path.dirname(os.path.relpath(__file__)))

class SearchAPI(APIView):
    
    def get(self, request, *args, **kwargs):
        api_data_entries = SearchData.objects.filter(source_type='chat')
        serializer = SearchDataSerializer(api_data_entries, many=True)
        return Response(serializer.data)

    def post(self, request, *args, **kwargs):
        method = 'ollma'
        prompt = request.data['prompt']
        # APIData.objects.create(source_type='text-to-text', data_content=request.data.get('api_data'))
        # return Response({'message': 'Text-to-text data collected successfully'}, status=status.HTTP_201_CREATED)
        # print(request.data)

        response = None
        if method == 'rag':
           response = search_bot(prompt=prompt, scope='local')
           # for output_item in output:
           #     response = str(response).join(output_item)
        #    print(response)

        if method == 'llm':
           response_data = pure_llm_api(prompt=prompt)
           response = response_data.choices[0].message.content

        if method == 'ollma':
            # print("Here is my prompt: ", prompt)
            response = ollama_call_text_to_text(user_input=prompt)
            # print(response)
           
        return Response({'message': 'Search data collected successfully','response':response}, status=status.HTTP_201_CREATED)
    
    
    def search_engine(request):
        try:
            if request.method == 'POST':
                
                    file_path = request.FILES.get('file') 

                    prompt= request.POST.get('prompt')
                    
                    ip_address = get_client_ip(request)
                    
                    # SearchHistory.objects.create(ip_address=ip_address, query=prompt)

                    if file_path != None:
                        
                        image_extensions = ('.png', '.jpg', '.jpeg')

                        
                        if str(file_path.name).lower().endswith(image_extensions):
                            
                            # print("Processing Image")
                            
                            file_name = file_path.name

                            new_file = FileData(filepath=file_path, filename=file_name)

                            new_file.save()       

                            # time.sleep(30)

                            data = FileData.objects.order_by('-upload_date').first()


                            response = ollama_call_image_to_text(image_path=data.filepath,prompt=prompt)
                            
                            serializer = FileDataSerializer(data, many = False)
                            
                            image_data = serializer.data

                            # print(serializer.data)
                            
                            
                            return JsonResponse({'image_path':image_data['filepath'],'success': True, 'message': 'Image successful processed','output':response,'query':prompt,'image':True}, safe=False)

                        else:
                            return JsonResponse({'success': False, 'message': 'Unsupported file format'})
                    else:
                        # print("Processing text", prompt)
                        response = ollama_call_text_to_text(user_input=prompt)
                        # print(response)
                        return JsonResponse({'success': True, 'message': 'Query sucessful processed', 'output': response, 'query':prompt,'image':False,'history':get_search_history(ip_address)})
            else:
               ip_address = get_client_ip(request)
               return render(request, template_name='delta/pages/delta.html', context={'history':get_search_history(ip_address)})
        except Exception as e:
            print("Exception:", e)
            return JsonResponse({'success': False, 'message': 'Internal Server Error'})
        
        
def get_search_history(ip_address):
    # Retrieve search history for the given IP address
    history = SearchHistory.objects.filter(ip_address=ip_address).order_by('-timestamp')
    return [{'query': entry.query, 'timestamp': entry.timestamp} for entry in history]

def get_client_ip(request):
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip