#Stage 1: Builder
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu20.04 AS Builder

ENV PATH=/opt/miniconda3/bin:$PATH
ARG PATH=/opt/miniconda3/bin:$PATH

# USER root
RUN apt-get update && apt-get upgrade -y

WORKDIR /usr/local/app
#install miniconda
RUN apt-get install -y wget
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py312_24.3.0-0-Linux-x86_64.sh \
    && bash Miniconda3-py312_24.3.0-0-Linux-x86_64.sh -b -p /opt/miniconda3 \
    && rm Miniconda3-py312_24.3.0-0-Linux-x86_64.sh

# Stage 2: Final
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu20.04
WORKDIR /usr/local/app
COPY --from=builder /opt/miniconda3 /opt/miniconda3
#install opencv dependencies
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
#setup conda env
RUN ls -l /opt/miniconda3/bin/activate
ENV PATH=/opt/miniconda3/bin:$PATH
ARG PATH=/opt/miniconda3/bin:$PATH
COPY environment.yml ./
RUN conda env create -f environment.yml
# RUN /opt/miniconda3/bin/conda init && . ~/.bashrc && conda activate computer_vision_prep
RUN apt-get update && apt-get install libegl1 -y && apt-get install libopengl0

# Copy in the source code
COPY ./receive_video.py .
RUN mkdir models
COPY ./models/yolo_bounding_box.py ./models
COPY ./models/__init__.py ./models

# Setup an app user so the container doesn't run as the root user
# RUN useradd app
# USER app

CMD /opt/miniconda3/bin/conda init && . ~/.bashrc && conda activate basketball_tracker_env && python receive_video.py -ip 0.0.0.0 -port 9999 -model yolov9c