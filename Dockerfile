FROM pytorch/pytorch

VOLUME /mnt/input
VOLUME /mnt/output

# Install python dependencies
ADD requirements.txt /
RUN pip install -r /requirements.txt

# Download pretrained weights
ADD models.py /
RUN python /models.py

COPY . /app
WORKDIR /app

CMD ["sh", "run.sh"]