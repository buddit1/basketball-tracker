# Basketball Tracker

A two part application to track a basketball through a video or livestream using a [pretrained yolo model](https://docs.ultralytics.com/). The client 
application provides a video, either through a camera or a file and the backend performs model inference. 

#### Example Output

![Example GIF](./images/example_video.gif)

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Roadmap](#roadmap)



## Installation
A conda environment file, `environment.yml` is included that has dependencies for both the front and backend. Assuming conda is installed ([installation instructions](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)) then the environment can be built with 
```
conda env create -f environment.yml
```

A docker file is also provided to run the backend. Assuming docker is installed, it can be built with 
```
docker build -t <tag-name> .
```

Once the image is built it can be run with 
```
docker run --network host --rm --gpus all -v /tmp/.X11-unix/:/tmp/.X11-unix --env DISPLAY=$DISPLAY <tag-name>
```

Note that the `-v` and `--env` flags are needed to enable displaying the video from within the 
docker container.

## Usage 

### Backend

The backend is implemented in the file `receive_video.py` Minimum usage looks like

```
python receive_video.py -ip <ip address> -port <port number> -model <yolo-variant e.g. yolov9c>
```

The full set of arguments are detailed in the table below. 

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Simple HTML Table</title>
    <style>
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            padding: 8px 12px;
            border: 1px solid #ddd;
            text-align: left;
        }
        th {
        }
    </style>
</head>
<body>
    <table>
        <thead>
            <tr>
                <th>Argument</th>
                <th>required (T/F)</th>
                <th>description</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>-h, --help</td>
                <td>False</td>
                <td>Show help message</td>
            </tr>
            <tr>
                <td>-ip</td>
                <td>True</td>
                <td>IP address to receive video at. e.g. 0.0.0.0</td>
            </tr>
            <tr>
                <td>-port </td>
                <td> True </td>
                <td> Port number to listen to </td>
            </tr>
            <tr>
                <td>-model </td>
                <td> True </td>
                <td> YOLO variant to use. One of: yolov9c,yolov9e,yolov8n,yolov8s,yolov8m,yolov8l,yolov8x
                </td>
            </tr>
            <tr>
                <td>--trajectory-length </td>
                <td> False </td>
                <td> How many frames to keep displaying basketball locations for after location. Default 10
                </td>
            </tr>
            <tr>
                <td>--denoise-method </td>
                <td> False </td>
                <td> Denoising method to use. Options: None, weiner, richardson-lucy. Default None. Experimentally can slightly improve detection at cost of frame rate. 
                </td>
            </tr>
            <tr>
                <td>--richardson-lucy-iters </td>
                <td> False </td>
                <td> How many iterations to use if using Richardson-Lucy denoising. Increasing significantly reduces frame rate for marginal improvements. Default 2. 
                </td>
            </tr>
            <tr>
                <td>--denoise-kernel-size </td>
                <td> False </td>
                <td> Kernel dimension for denoising method. Only accepts a single integer and uses a square kernel. Default 5.
                </td>
            </tr>
            <tr>
                <td>--weiner-noise </td>
                <td> False </td>
                <td> Noise parameter for weiner denoising. Optional but empirically necessary to avoid large artifacts in dark regions when using weiner denoising. Should be between 0 and 1. 
                </td>
            </tr>
            <tr>
                <td>--output </td>
                <td> False </td>
                <td> Output file location to save video with ball trajectories overlaid. Only tested using .mp4 extension.
                </td>
            </tr>
            <tr>
                <td>--no-display </td>
                <td> False </td>
                <td> Optional flag to prevent displaying video. 
                </td>
            </tr>
        </tbody>
    </table>
</body>
</html>


## Client

The client is implemented in `send_video.py` and can be run with

```
python send_video.py -backend-ip <ip address> -port <backend port> 
```

The above usage will look for a webcam if available and stream video from it. To read a video from a file use the `--file` argument. The full set of arguments are detailed below. 



<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Simple HTML Table</title>
    <style>
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            padding: 8px 12px;
            border: 1px solid #ddd;
            text-align: left;
        }
        th {
        }
    </style>
</head>
<body>
    <table>
        <thead>
            <tr>
                <th>Argument</th>
                <th>required (T/F)</th>
                <th>description</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>-h, --help</td>
                <td>False</td>
                <td>Show help message</td>
            </tr>
            <tr>
                <td>-backend-ip</td>
                <td>True</td>
                <td>IP address to send video to. Corresponds to -ip argument for backend. </td>
            </tr>
            <tr>
                <td>-port </td>
                <td> True </td>
                <td> Port number to send video to. Corresponds to -port argument for backend. </td>
            </tr>
            <tr>
                <td>--jpeg-quality </td>
                <td> False </td>
                <td> Parameter passed to opencv video compression. Should be between 0 and 100. Lower numbers lead to smaller packets but lossier compression. Default 95
                </td>
            </tr>
            <tr>
                <td>--image-size </td>
                <td> False </td>
                <td> Length of longest side of of image after resizing. The size of the shorter dimension is set to a multiple of 32 while approximately maintaining the original aspect ratio. Default 640
                </td>
            </tr>
        </tbody>
    </table>
</body>
</html>



## Roadmap 

### Custom Dataset and Model Finetuning

Currently detection is done with a pretrained yolo model by only using the class sports ball. This leads to false positives, particularly when bright green or yellow shoes are present in a frame. The model also has significantly worse performance when the ball is far away or moving rapidly. Finetuning the model on a basketball specific dataset would likely improve this.

#### Failure Example
![Shoe False Positive Gif](./images/shoe_false_pos_video.gif)


### Activity Recognition

Several datasets exist which annotate basketball videos with several action categories, e.g. dribbling. Integrating these into finetuning could allow for more interesting applications than just tracking a basketball. Long term goal is to automatically extract statistics like shooting percentage per team or per player etc. 