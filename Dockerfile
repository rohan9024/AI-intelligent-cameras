FROM python 3.11
COPY . /LicensePlateRecognition
WORKDIR /LicensePlateRecognition
RUN pip install -r requirements.txt
EXPOSE ssh sushi@34.93.112.153
CMD python ./LicensePlateRecognition/main_2_indian_num_plate.py